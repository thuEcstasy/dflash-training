"""
DFlash Data Pipeline
====================

Key concepts from the paper (§3.3):

1.  Random anchor sampling
    - For each response, randomly sample M anchor positions.
    - Each anchor is the first (clean) token of a masked block.
    - The next (block_size - 1) positions become the prediction targets.
    - Blocks must be non-overlapping and fully within the response.

2.  Block-diagonal attention mask
    - Draft tokens attend bidirectionally *within* their block.
    - No cross-block token-to-token attention.
    - All draft tokens attend to the *full* context KV (from KV injection).
    - The context KV covers the original sequence length L and is always
      unmasked (handled inside DFlashAttention, not here).

3.  Position-dependent loss decay  (paper eq. 1)
    w_k = exp(-(k-1) / gamma),  k = 1 .. block_size-1
    Weights sum is normalised so loss scale is comparable across
    different gamma values and block sizes.

Training sample format (jsonl):
    {
      "messages": [
          {"role": "system",   "content": "..."},
          {"role": "user",     "content": "..."},
          {"role": "assistant","content": "..."}
      ]
    }
"""

import json
import random
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .config import DFlashConfig


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DFlashDataset(Dataset):
    """
    Loads tokenised SFT conversations.

    Each item stores:
      - input_ids : (L,)          full token sequence (prompt + response)
      - loss_mask : (L,)  bool    True for *response* positions only
                                  (anchors and targets are drawn from these)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 4096,
        cache_path: Optional[str] = None,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict] = []

        cached = Path(cache_path) if cache_path else None
        if cached and cached.exists():
            print(f"[DFlashDataset] Loading tokenised cache from {cached}")
            import pickle
            with open(cached, "rb") as f:
                self.samples = pickle.load(f)
            return

        print(f"[DFlashDataset] Tokenising {data_path} ...")
        raw = [json.loads(l) for l in Path(data_path).read_text().splitlines()
               if l.strip()]

        for item in raw:
            sample = self._encode(item)
            if sample is not None:
                self.samples.append(sample)

        print(f"[DFlashDataset] {len(self.samples)} sequences loaded.")

        if cached:
            import pickle
            cached.parent.mkdir(parents=True, exist_ok=True)
            with open(cached, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"[DFlashDataset] Cache saved to {cached}")

    def _encode(self, item: Dict) -> Optional[Dict]:
        messages = item.get("messages") or item.get("conversations")
        if messages is None:
            return None

        # Normalise conversation keys (support "from"/"value" style too)
        normalised = []
        for m in messages:
            role    = m.get("role") or (
                "user" if m.get("from") in ("human", "user") else "assistant"
            )
            content = m.get("content") or m.get("value", "")
            normalised.append({"role": role, "content": content})

        # Apply chat template; tokenise
        try:
            text = self.tokenizer.apply_chat_template(
                normalised,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return None

        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]   # (L,)
        L = len(input_ids)
        if L < 8:
            return None

        # Build response mask: mark tokens that belong to assistant turns.
        # We re-tokenise prompt-only to find the boundary.
        prompt_messages = [m for m in normalised if m["role"] != "assistant"]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: use the last 50 % of the sequence as "response"
            loss_mask = torch.zeros(L, dtype=torch.bool)
            loss_mask[L // 2:] = True
            return {"input_ids": input_ids, "loss_mask": loss_mask}

        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt"
        )["input_ids"][0]
        prompt_len = min(len(prompt_ids), L)

        loss_mask = torch.zeros(L, dtype=torch.bool)
        loss_mask[prompt_len:] = True

        # Need at least block_size response tokens to form one block
        if loss_mask.sum() < 8:
            return None

        return {"input_ids": input_ids, "loss_mask": loss_mask}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Collator  (block sampling + attention mask construction)
# ─────────────────────────────────────────────────────────────────────────────

class DFlashCollator:
    """
    Converts a list of dataset samples into a training batch with:
      - padded input_ids for the full sequences (used by the target model)
      - block_input_ids : (B, M*block_size)  draft model input
          [anchor, MASK, MASK, …, anchor, MASK, …]
      - block_labels    : (B, M*block_size)  ground-truth token ids
          (-100 for anchor positions, token id for masked positions)
      - block_position_ids : (B, M*block_size)  real positions in original seq
      - block_attn_mask : (B, 1, M*block_size, M*block_size + L_ctx)
          block-diagonal + full context column
      - loss_weights    : (M*(block_size-1),)  per-position decay weights

    MASK token: tokenizer.mask_token_id if available, else tokenizer.unk_token_id.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DFlashConfig,
    ):
        self.tokenizer   = tokenizer
        self.config      = config
        self.block_size  = config.block_size
        self.M           = config.blocks_per_sequence
        self.max_seq_len = config.max_seq_len

        # MASK token id for diffusion input
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        elif tokenizer.unk_token_id is not None:
            self.mask_token_id = tokenizer.unk_token_id
        else:
            self.mask_token_id = 0

        # Pre-compute loss decay weights w_k = exp(-(k-1)/γ), k=1..block_size-1
        gamma     = config.loss_decay_gamma
        k_indices = torch.arange(1, self.block_size, dtype=torch.float32)  # k=1..B-1
        weights   = torch.exp(-(k_indices - 1) / gamma)
        # Normalise so mean weight = 1 (keeps loss scale independent of gamma)
        self.loss_weights = weights / weights.mean()  # (block_size-1,)

    # ── Anchor sampling ───────────────────────────────────────────────────────

    def _sample_anchors(
        self, loss_mask: torch.Tensor
    ) -> Optional[List[int]]:
        """
        Randomly sample M non-overlapping anchor positions from response tokens.
        Each block spans [anchor, anchor+1, ..., anchor+block_size-1].

        Returns sorted list of M anchor positions, or None if not enough tokens.
        """
        # Valid anchor positions: response token whose block fits within seq
        L = len(loss_mask)
        valid = [
            i for i in range(L - self.block_size + 1)
            if loss_mask[i].item()  # anchor must be a response token
        ]
        if len(valid) < self.M:
            return None

        # Sample M non-overlapping anchors
        sampled: List[int] = []
        attempts = 0
        while len(sampled) < self.M and attempts < 200:
            attempts += 1
            a = random.choice(valid)
            # Check non-overlap with existing anchors
            overlap = any(
                abs(a - s) < self.block_size for s in sampled
            )
            if not overlap:
                sampled.append(a)
        if len(sampled) < self.M:
            return None
        return sorted(sampled)

    # ── Attention mask construction ───────────────────────────────────────────

    def _build_block_attn_mask(
        self, M: int, B: int, L_ctx: int
    ) -> torch.Tensor:
        """
        Build (M*B, L_ctx + M*B) additive attention mask.

        Query tokens (M*B) attend to:
          - Context KV:  ALL L_ctx positions → 0 (unmasked)
          - Token KV:    only tokens within the SAME block → 0 / -inf
        """
        S = M * B
        # -inf for forbidden positions, 0 for allowed
        mask = torch.full((S, L_ctx + S), float("-inf"))
        # All queries can see the full context KV (first L_ctx columns)
        mask[:, :L_ctx] = 0.0
        # Block-diagonal: within-block bidirectional attention
        for m in range(M):
            start = m * B
            end   = start + B
            mask[start:end, L_ctx + start : L_ctx + end] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, L_ctx+S)

    # ── Main collate ──────────────────────────────────────────────────────────

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad full sequences for the target model
        input_ids_list  = [s["input_ids"]  for s in batch]
        loss_mask_list  = [s["loss_mask"]  for s in batch]

        max_len = min(max(len(x) for x in input_ids_list), self.max_seq_len)
        pad_id  = self.tokenizer.pad_token_id or 0

        # Pad full sequences (right-padding)
        full_input_ids  = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        full_attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids_list):
            n = min(len(ids), max_len)
            full_input_ids[i, :n] = ids[:n]
            full_attn_mask[i, :n] = 1

        # ── Sample blocks for each sequence ──────────────────────────────────
        block_input_ids_batch  = []
        block_labels_batch     = []
        block_position_ids_batch = []
        valid_indices: List[int] = []

        for i, (ids, lm) in enumerate(zip(input_ids_list, loss_mask_list)):
            anchors = self._sample_anchors(lm[:max_len])
            if anchors is None:
                continue   # skip sequences with insufficient response tokens
            valid_indices.append(i)

            bids   = []   # block input ids (with MASK)
            blabels = []  # ground-truth ids
            bpos   = []   # position ids

            for a in anchors:
                for k in range(self.block_size):
                    pos = a + k
                    true_id = ids[pos].item() if pos < len(ids) else pad_id
                    if k == 0:
                        # anchor: feed clean token, no loss
                        bids.append(true_id)
                        blabels.append(self.IGNORE_INDEX)
                    else:
                        # masked position: feed MASK token, compute loss
                        bids.append(self.mask_token_id)
                        blabels.append(true_id)
                    bpos.append(pos)

            block_input_ids_batch.append(torch.tensor(bids,    dtype=torch.long))
            block_labels_batch.append(   torch.tensor(blabels, dtype=torch.long))
            block_position_ids_batch.append(torch.tensor(bpos, dtype=torch.long))

        if not valid_indices:
            # Degenerate batch – return empty tensors (training loop should skip)
            return {}

        # Stack block tensors
        block_input_ids   = torch.stack(block_input_ids_batch)    # (B', M*bs)
        block_labels      = torch.stack(block_labels_batch)
        block_position_ids = torch.stack(block_position_ids_batch)

        # Subset full sequences to valid_indices
        full_input_ids  = full_input_ids[valid_indices]
        full_attn_mask  = full_attn_mask[valid_indices]

        # Attention mask: (1, 1, M*bs, max_len + M*bs)
        block_attn_mask = self._build_block_attn_mask(
            M=self.M, B=self.block_size, L_ctx=max_len
        )

        return {
            # For target model (feature extraction)
            "full_input_ids":   full_input_ids,          # (B', L)
            "full_attn_mask":   full_attn_mask,          # (B', L)
            # For draft model (block diffusion)
            "block_input_ids":     block_input_ids,      # (B', M*bs)
            "block_labels":        block_labels,         # (B', M*bs)
            "block_position_ids":  block_position_ids,   # (B', M*bs)
            "block_attn_mask":     block_attn_mask,      # (1, 1, M*bs, L+M*bs)
            # Loss weights
            "loss_weights": self.loss_weights,           # (bs-1,)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss computation with position-dependent decay
# ─────────────────────────────────────────────────────────────────────────────

def compute_dflash_loss(
    logits: torch.Tensor,          # (B, M*block_size, vocab)
    labels: torch.Tensor,          # (B, M*block_size)
    loss_weights: torch.Tensor,    # (block_size-1,)
    block_size: int,
    M: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    DFlash block-diffusion cross-entropy loss with position-dependent decay.

    For position k within a block (k = 1..block_size-1):
        loss_k = CE(logit_k, label_k) * w_k
        w_k    = exp(-(k-1) / gamma)  [pre-computed and stored in loss_weights]

    The anchor position (k=0) is excluded from the loss.
    """
    B, S, V = logits.shape
    device = logits.device
    loss_weights = loss_weights.to(device)

    total_loss   = torch.tensor(0.0, device=device, requires_grad=True)
    total_weight = torch.tensor(0.0, device=device)

    for m in range(M):
        block_start = m * block_size
        # Skip anchor (k=0), iterate over k=1..block_size-1
        for k in range(1, block_size):
            pos = block_start + k
            # logits at this position: (B, V)
            log_k   = logits[:, pos, :]
            label_k = labels[:, pos]
            w_k     = loss_weights[k - 1]   # scalar

            # Compute CE, ignoring padding / anchor positions
            valid_mask = label_k != ignore_index
            if valid_mask.sum() == 0:
                continue

            ce = F.cross_entropy(
                log_k[valid_mask],
                label_k[valid_mask],
                reduction="mean",
            )
            total_loss   = total_loss + w_k * ce
            total_weight = total_weight + w_k

    if total_weight > 0:
        total_loss = total_loss / total_weight
    return total_loss
