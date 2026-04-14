"""
Step 4: Export trained DFlash draft model to HuggingFace format.

After training, merges the draft weights with the target model's embeddings
and saves a self-contained HuggingFace model that can be used with:
  - z-lab/dflash inference code
  - SGLang (via --speculative-draft-model-path)
  - vLLM (with DFlash support)

Usage:
    python scripts/export_draft_model.py \
        --checkpoint ./checkpoints/qwen3-8b-dflash-b16/checkpoint-10000 \
        --target-model Qwen/Qwen3-8B \
        --output-dir ./exported/qwen3-8b-dflash-b16 \
        --config configs/qwen3-8b.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.config import DFlashConfig
from dflash.model import DFlashDraftModel, get_target_layer_indices


def export(
    checkpoint_dir: str,
    target_model_name: str,
    output_dir: str,
    config: DFlashConfig,
):
    print(f"[export] Loading target model: {target_model_name}")
    tokenizer    = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    print("[export] Building draft model ...")
    draft_model = DFlashDraftModel(target_model, config)

    # Load trained weights
    ckpt_path = Path(checkpoint_dir) / "draft_model.pt"
    print(f"[export] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = draft_model.load_state_dict(state, strict=False)
    print(f"  Missing keys   : {missing}")
    print(f"  Unexpected keys: {unexpected}")

    # Collect full state dict (including frozen shared weights)
    full_state = draft_model.state_dict()

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(full_state, out / "pytorch_model.bin")
    tokenizer.save_pretrained(out)

    # Save DFlash metadata (needed by inference backends)
    meta = {
        "model_type":             "dflash_draft",
        "target_model":           target_model_name,
        "num_draft_layers":       config.num_draft_layers,
        "block_size":             config.block_size,
        "num_target_sample_layers": config.num_target_sample_layers,
        "target_layer_indices":   get_target_layer_indices(
            target_model.config.num_hidden_layers,
            config.num_target_sample_layers
        ),
        "hidden_size":            target_model.config.hidden_size,
        "vocab_size":             target_model.config.vocab_size,
    }
    (out / "dflash_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] Done! Saved to {output_dir}")
    print(f"[export] DFlash metadata:\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--config",       required=True)
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg_dict = yaml.safe_load(fp)
    config = DFlashConfig(**cfg_dict)
    config.target_model_name_or_path = args.target_model

    export(args.checkpoint, args.target_model, args.output_dir, config)
