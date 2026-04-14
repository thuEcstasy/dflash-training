"""
DFlash Draft Model – Training Entry Point
==========================================

Training loop overview:
  For each batch:
  1. Run the FROZEN target model on full_input_ids → collect K hidden states
  2. Fuse hidden states via ContextFusion → context (B, L, H)
  3. Forward draft model with block inputs + context + block-diagonal mask
  4. Compute DFlash loss (CE + positional decay)
  5. Backprop only through draft model parameters

Launch single-GPU:
    python -m dflash.train --config configs/qwen3-8b.yaml

Launch multi-GPU (DeepSpeed / torchrun):
    torchrun --nproc_per_node 8 -m dflash.train --config configs/qwen3-8b.yaml

Requires: transformers, accelerate, torch >= 2.4, datasets, pyyaml
"""

import os
import math
import time
import json
import logging
import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import DFlashConfig
from .model import (
    DFlashDraftModel,
    get_target_layer_indices,
    extract_target_hidden_states,
)
from .data import DFlashDataset, DFlashCollator, compute_dflash_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    draft_model: DFlashDraftModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: DFlashConfig,
    accelerator: Accelerator,
):
    """Saves trainable weights + optimizer state."""
    if not accelerator.is_main_process:
        return
    ckpt_dir = Path(config.output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap from DDP/FSDP wrapper
    model = accelerator.unwrap_model(draft_model)

    # Save only trainable parameters (context_fusion + layers)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("embed_tokens")
        and not k.startswith("norm.")
        and not k.startswith("lm_head")
    }
    torch.save(trainable_state, ckpt_dir / "draft_model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save({"step": step, "scheduler": scheduler.state_dict()},
               ckpt_dir / "training_state.pt")
    # Save config
    (ckpt_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2)
    )
    logger.info(f"Checkpoint saved at step {step} → {ckpt_dir}")


def load_checkpoint(
    draft_model: DFlashDraftModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_dir: str,
    accelerator: Accelerator,
) -> int:
    """Loads checkpoint; returns the step it was saved at."""
    ckpt = Path(checkpoint_dir)
    model = accelerator.unwrap_model(draft_model)
    state = torch.load(ckpt / "draft_model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded draft weights. Missing: {missing}, Unexpected: {unexpected}")
    optimizer.load_state_dict(
        torch.load(ckpt / "optimizer.pt", map_location="cpu")
    )
    training_state = torch.load(ckpt / "training_state.pt", map_location="cpu")
    scheduler.load_state_dict(training_state["scheduler"])
    return training_state["step"]


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: DFlashConfig):
    # ── Accelerator ──────────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.output_dir,
    )
    accelerator.init_trackers("dflash_training")

    set_seed(config.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info(f"[DFlash] Training with config:\n{json.dumps(dataclasses.asdict(config), indent=2)}")

    # ── Load tokeniser + target model ────────────────────────────────────────
    logger.info(f"Loading target model: {config.target_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.target_model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        config.target_model_name_or_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32,
        trust_remote_code=True,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    num_target_layers = target_model.config.num_hidden_layers
    layer_indices = get_target_layer_indices(
        num_target_layers, config.num_target_sample_layers
    )
    logger.info(f"Target layer indices for feature extraction: {layer_indices}")

    # ── Build draft model ────────────────────────────────────────────────────
    draft_model = DFlashDraftModel(target_model, config)
    logger.info(
        f"Draft model trainable params: "
        f"{draft_model.num_trainable_params() / 1e6:.1f} M"
    )

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    collator = DFlashCollator(tokenizer, config)

    dataset = DFlashDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        cache_path=str(Path(config.output_dir) / "dataset_cache.pkl"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(draft_model.trainable_parameters()),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    total_steps     = steps_per_epoch * config.num_epochs
    warmup_steps    = int(total_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ───────────────────────────────────────────────────
    (
        draft_model,
        target_model,
        optimizer,
        dataloader,
        scheduler,
    ) = accelerator.prepare(
        draft_model, target_model, optimizer, dataloader, scheduler
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    if config.resume_from_checkpoint:
        global_step = load_checkpoint(
            draft_model, optimizer, scheduler,
            config.resume_from_checkpoint, accelerator
        )
        logger.info(f"Resumed from step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    for epoch in range(config.num_epochs):
        draft_model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if not batch:   # degenerate batch from collator
                continue

            with accelerator.accumulate(draft_model):

                # ── 1. Extract target hidden states (no grad) ────────────
                full_input_ids = batch["full_input_ids"]
                full_attn_mask = batch["full_attn_mask"]

                target_hiddens = extract_target_hidden_states(
                    accelerator.unwrap_model(target_model),
                    full_input_ids,
                    full_attn_mask,
                    layer_indices,
                )   # list of K × (B, L, H)

                # ── 2. Fuse context ──────────────────────────────────────
                context = accelerator.unwrap_model(draft_model).fuse_context(
                    target_hiddens
                )   # (B, L, H)

                # ── 3. Forward draft model ───────────────────────────────
                block_input_ids    = batch["block_input_ids"]
                block_position_ids = batch["block_position_ids"]
                block_attn_mask    = batch["block_attn_mask"]

                logits = draft_model(
                    input_ids=block_input_ids,
                    context=context,
                    position_ids=block_position_ids,
                    attention_mask=block_attn_mask,
                )   # (B, M*block_size, V)

                # ── 4. Loss ──────────────────────────────────────────────
                loss = compute_dflash_loss(
                    logits=logits,
                    labels=batch["block_labels"],
                    loss_weights=batch["loss_weights"],
                    block_size=config.block_size,
                    M=config.blocks_per_sequence,
                )

                # ── 5. Backward ──────────────────────────────────────────
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(draft_model.parameters()),
                        config.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().item()

            # Log every N gradient-sync steps
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config.logging_steps == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / max(1, config.logging_steps)
                    lr_now   = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | LR {lr_now:.2e} | "
                        f"Elapsed {elapsed:.1f}s"
                    )
                    accelerator.log(
                        {"train/loss": avg_loss, "train/lr": lr_now},
                        step=global_step,
                    )
                    running_loss = 0.0
                    t0 = time.time()

                if global_step % config.save_steps == 0:
                    save_checkpoint(
                        draft_model, optimizer, scheduler,
                        global_step, config, accelerator
                    )

        logger.info(f"Epoch {epoch+1} finished.")

    # Final save
    save_checkpoint(
        draft_model, optimizer, scheduler,
        global_step, config, accelerator
    )
    accelerator.end_training()
    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> DFlashConfig:
    parser = argparse.ArgumentParser(description="Train a DFlash draft model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Allow overriding any config field from the command line
    for f in dataclasses.fields(DFlashConfig):
        if f.name == "config":
            continue
        parser.add_argument(
            f"--{f.name.replace('_', '-')}",
            type=type(f.default) if not isinstance(f.default, dataclasses.MISSING.__class__) else str,
            default=None,
            dest=f.name,
            help=f"Override {f.name}",
        )

    args = parser.parse_args()

    # Build config from YAML (or defaults), then apply CLI overrides
    if args.config:
        with open(args.config) as fp:
            yaml_dict = yaml.safe_load(fp)
        config = DFlashConfig(**yaml_dict)
    else:
        config = DFlashConfig()

    for f in dataclasses.fields(DFlashConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(config, f.name, v)

    return config


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
