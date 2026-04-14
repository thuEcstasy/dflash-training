"""
DFlash Draft Model
==================
Architecture (paper §3.2):

  1.  ContextFusion
      - Uniformly sample K hidden states from the frozen target model
      - Project: concat[h_l1, h_l2, ..., h_lK]  (K·H_t) → H_d  via a small MLP

  2.  DFlash transformer layers  (num_draft_layers = 3)
      - Each layer is identical to one target-model transformer layer in structure
      - KV Injection: context features are projected to extra K and V tensors
        and *concatenated* before the softmax so every query attends to both
        the regular token KV and the context KV
      - Embeddings and LM head are **shared / frozen** from the target model

  3.  Block-diffusion at inference:
      - In a single forward pass, the model predicts all 15 masked tokens
        given one anchor token and the injected context

The injection mechanism (different from EAGLE-3 which only injects at the
first layer):
    K_full = cat([K_ctx, K_tok], dim=-2)   # context KV prepended
    V_full = cat([V_ctx, V_tok], dim=-2)
    attn   = softmax(Q @ K_full^T / √d) @ V_full
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Context fusion
# ─────────────────────────────────────────────────────────────────────────────

class ContextFusion(nn.Module):
    """
    Fuses K uniformly-sampled hidden states from the target model into a
    single context embedding per token position.

    input  : list of K tensors, each (B, L, H_target)
    output : (B, L, H_draft)
    """

    def __init__(self, target_hidden_size: int, draft_hidden_size: int,
                 num_sample_layers: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_sample_layers * target_hidden_size, draft_hidden_size,
                      bias=False),
            nn.SiLU(),
            nn.Linear(draft_hidden_size, draft_hidden_size, bias=False),
        )

    def forward(self, hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        # hidden_states_list: K × (B, L, H_t) → (B, L, K·H_t)
        fused = torch.cat(hidden_states_list, dim=-1)
        return self.proj(fused)          # (B, L, H_d)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Draft attention with KV injection
# ─────────────────────────────────────────────────────────────────────────────

class DFlashAttention(nn.Module):
    """
    Multi-head attention for the draft model.

    Compared with standard attention, each layer receives an extra
    `context` tensor (B, L, H_draft) that is projected to additional K, V
    entries and prepended before the main softmax.  This gives every
    query persistent access to the target-model context regardless of
    draft depth.
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int,
                 max_position_embeddings: int = 32768,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # KV injection projections (from context → draft KV space)
        self.k_ctx_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_ctx_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        # Q/K norms (Qwen3-style)
        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)

        self.rotary_emb = Qwen3RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.scaling = head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, S, H)
        context: torch.Tensor,                # (B, L, H)  target context (full seq)
        position_ids: torch.Tensor,           # (B, S)
        attention_mask: Optional[torch.Tensor] = None,   # (B, 1, S, S+L)
        output_attentions: bool = False,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        L = context.shape[1]

        # ── Q, K, V for draft tokens ────────────────────────────────────────
        Q = self.q_proj(hidden_states)   # (B, S, nh·hd)
        K = self.k_proj(hidden_states)   # (B, S, nk·hd)
        V = self.v_proj(hidden_states)

        Q = Q.view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Q/K norms
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # RoPE for draft tokens
        cos, sin = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # ── KV from context (injected) ──────────────────────────────────────
        # Context covers the full input length (same as target sequence)
        K_ctx = self.k_ctx_proj(context)  # (B, L, nk·hd)
        V_ctx = self.v_ctx_proj(context)
        K_ctx = K_ctx.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_ctx = V_ctx.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # No RoPE on context KV (they represent global context, not positional)

        # ── Concatenate: context KV prepended to token KV ──────────────────
        K_full = torch.cat([K_ctx, K], dim=2)   # (B, nk, L+S, hd)
        V_full = torch.cat([V_ctx, V], dim=2)

        # GQA expansion
        K_full = repeat_kv(K_full, self.num_kv_groups)   # (B, nh, L+S, hd)
        V_full = repeat_kv(V_full, self.num_kv_groups)

        # ── Attention ───────────────────────────────────────────────────────
        # Q: (B, nh, S, hd), K_full: (B, nh, L+S, hd)
        attn_weights = torch.matmul(Q, K_full.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # additive mask (0/-inf)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

        out = torch.matmul(attn_weights, V_full)          # (B, nh, S, hd)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Draft transformer layer
# ─────────────────────────────────────────────────────────────────────────────

class DFlashLayer(nn.Module):
    """One transformer block for the draft model."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 rms_norm_eps: float = 1e-6,
                 max_position_embeddings: int = 32768,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.self_attn = DFlashAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )
        # SwiGLU FFN
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, context, position_ids, attention_mask)
        hidden_states = residual + h

        # FFN with residual
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        hidden_states = residual + h
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full draft model
# ─────────────────────────────────────────────────────────────────────────────

class DFlashDraftModel(nn.Module):
    """
    DFlash draft model.

    Trainable components:
      - ContextFusion  (project K target hidden states → draft context)
      - num_draft_layers × DFlashLayer  (transformer + KV injection)

    Frozen (shared with target):
      - embed_tokens  (token embedding)
      - lm_head       (output projection + vocab logits)
      - norm          (final RMSNorm before lm_head)

    Usage during training:
        context = model.fuse_context(target_hidden_list)      # step 1
        logits  = model(input_ids, context, position_ids, mask)  # step 2
    """

    def __init__(self, target_model: PreTrainedModel, draft_config: "DFlashConfig"):
        super().__init__()
        cfg = target_model.config

        # ── Frozen components (shared weight tensors, no copy) ──────────────
        self.embed_tokens = target_model.model.embed_tokens
        self.norm         = target_model.model.norm
        self.lm_head      = target_model.lm_head

        # Freeze them
        for p in [*self.embed_tokens.parameters(),
                  *self.norm.parameters(),
                  *self.lm_head.parameters()]:
            p.requires_grad_(False)

        hidden_size       = cfg.hidden_size
        intermediate_size = cfg.intermediate_size
        num_heads         = cfg.num_attention_heads
        num_kv_heads      = cfg.num_key_value_heads
        head_dim          = getattr(cfg, "head_dim", hidden_size // num_heads)
        rms_norm_eps      = cfg.rms_norm_eps
        max_pos           = cfg.max_position_embeddings
        rope_theta        = getattr(cfg, "rope_theta", 10000.0)

        # ── Trainable: context fusion ────────────────────────────────────────
        self.context_fusion = ContextFusion(
            target_hidden_size=hidden_size,
            draft_hidden_size=hidden_size,
            num_sample_layers=draft_config.num_target_sample_layers,
        )

        # ── Trainable: draft transformer layers ──────────────────────────────
        self.layers = nn.ModuleList([
            DFlashLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_pos,
                rope_theta=rope_theta,
            )
            for _ in range(draft_config.num_draft_layers)
        ])

        self.draft_config = draft_config

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse_context(
        self, hidden_states_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        hidden_states_list: K tensors of shape (B, L, H_target)
        returns context: (B, L, H_draft)
        """
        return self.context_fusion(hidden_states_list)

    def forward(
        self,
        input_ids: torch.Tensor,              # (B, S) – masked sequence
        context: torch.Tensor,               # (B, L, H) – fused target context
        position_ids: torch.Tensor,          # (B, S)
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, S, L+S)
    ) -> torch.Tensor:
        """Returns logits (B, S, vocab_size)."""
        hidden_states = self.embed_tokens(input_ids)   # (B, S, H)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, context, position_ids, attention_mask
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)           # (B, S, V)
        return logits

    def trainable_parameters(self):
        """Yields only the parameters that should be updated."""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Helper: uniformly sample target layer indices
# ─────────────────────────────────────────────────────────────────────────────

def get_target_layer_indices(num_target_layers: int,
                             num_sample_layers: int) -> List[int]:
    """
    Return K layer indices uniformly spaced across [0, num_target_layers).
    e.g. 32 layers, 8 samples → [0, 4, 8, 12, 16, 20, 24, 28]
    """
    step = num_target_layers / num_sample_layers
    return [int(step * i) for i in range(num_sample_layers)]


def extract_target_hidden_states(
    target_model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> List[torch.Tensor]:
    """
    Run target model in no-grad mode and collect hidden states at layer_indices.

    Returns a list of K tensors, each (B, L, H_target).
    """
    target_model.eval()
    with torch.no_grad():
        outputs = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    # outputs.hidden_states: tuple of (num_layers+1) tensors including embedding layer
    # index 0 = embedding output, index i = after layer i-1
    all_hidden = outputs.hidden_states   # len = num_layers + 1
    return [all_hidden[i + 1] for i in layer_indices]   # skip embedding layer
