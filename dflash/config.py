"""
DFlash Draft Model Training Configuration
Paper: https://arxiv.org/abs/2602.06036
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DFlashConfig:
    # ── Target model ──────────────────────────────────────────────────────────
    target_model_name_or_path: str = "Qwen/Qwen3-8B"

    # ── Draft model architecture ──────────────────────────────────────────────
    num_draft_layers: int = 3         # 3 layers as per paper
    block_size: int = 16              # total tokens per block (1 anchor + 15 to predict)
    # Uniformly sample K layers from target to build context features.
    # e.g. for a 32-layer model with K=8: sample layers [0,4,8,12,16,20,24,28]
    num_target_sample_layers: int = 8

    # ── Block diffusion ───────────────────────────────────────────────────────
    blocks_per_sequence: int = 4      # M masked blocks per training sequence
    # loss decay:  w_k = exp(-(k-1) / loss_decay_gamma),  k = 1..block_size-1
    # SpecForge community uses gamma=7.0; paper notes exponential decay
    loss_decay_gamma: float = 7.0

    # ── Training hyper-params ─────────────────────────────────────────────────
    learning_rate: float = 6e-4
    min_lr: float = 6e-5              # cosine schedule minimum
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.04
    num_epochs: int = 6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data_path: str = "./cache/dataset/train_regen.jsonl"
    max_seq_len: int = 4096
    num_workers: int = 4

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints/dflash"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    resume_from_checkpoint: Optional[str] = None

    # ── Distributed / precision ───────────────────────────────────────────────
    # "bf16" | "fp16" | "fp32"
    mixed_precision: str = "bf16"
    # tensor parallel shards (requires target model to support TP)
    tp_size: int = 1

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    use_flex_attention: bool = True   # PyTorch ≥ 2.4 block-sparse mask

    @property
    def draft_tokens(self) -> int:
        """Number of tokens to predict per block (excludes anchor)."""
        return self.block_size - 1

    @property
    def total_block_tokens(self) -> int:
        """Total draft tokens per training sequence."""
        return self.blocks_per_sequence * self.block_size
