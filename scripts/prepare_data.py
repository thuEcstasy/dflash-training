"""
Step 1: Download and prepare seed dataset.

Paper setting (DFlash §4):
  ~800K samples = Nemotron Post-Training Dataset V2  (~780K)
                + CodeAlpaca                          (~20K)

Modes:
  --paper          Mix Nemotron V2 + CodeAlpaca to match paper (~800K total)
  --dataset NAME   Single dataset (nemotron / codealpaca / sharegpt / perfectblend)

Output: ./cache/dataset/dflash_paper_train.jsonl  (messages format, shuffled)

Usage:
    # Paper setting (recommended):
    python scripts/prepare_data.py --paper --output-dir ./cache/dataset

    # Single dataset:
    python scripts/prepare_data.py --dataset nemotron --output-dir ./cache/dataset

    # Quick smoke-test (1000 samples):
    python scripts/prepare_data.py --paper --max-samples 1000

Note: set HF_ENDPOINT=https://hf-mirror.com before running if on a CN server.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from datasets import load_dataset

# ── Mirror: honour HF_ENDPOINT if set (e.g. https://hf-mirror.com) ──────────
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "nemotron": {
        "hf_path":  "nvidia/Nemotron-Post-Training-Dataset-v2",
        "hf_split": "train",
        "converter": "nemotron",
    },
    "codealpaca": {
        "hf_path":  "sahil2801/CodeAlpaca-20k",
        "hf_split": "train",
        "converter": "alpaca",
    },
    "sharegpt": {
        "hf_path":  "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "hf_split": "train",
        "converter": "sharegpt",
    },
    "perfectblend": {
        "hf_path":  "eigen-ai-labs/perfectblend-qwen3-8b-regen-demo",
        "hf_split": "train",
        "converter": "messages",
    },
}

# Paper mix: (dataset_name, max_samples)
# The paper only states "~800K total from Nemotron V2 + CodeAlpaca",
# without specifying the exact split.
# CodeAlpaca is ~20K in total, so we take all of it;
# the remainder (~780K) is sampled from Nemotron V2.
# Adjust NEMOTRON_CAP if you want a different total size.
NEMOTRON_CAP = 780_000
PAPER_MIX = [
    ("nemotron",   NEMOTRON_CAP),
    ("codealpaca", -1),          # -1 = all (~20K)
]

# ─────────────────────────────────────────────────────────────────────────────
# Format converters → unified {"messages": [...]} format
# ─────────────────────────────────────────────────────────────────────────────

def convert_nemotron(item: Dict) -> Optional[Dict]:
    """
    Nemotron V2 uses OpenAI messages format: item["messages"] is a list of
    {"role": "system"/"user"/"assistant", "content": "..."}.
    Some rows may have a "conversations" key instead (older format).
    """
    msgs = item.get("messages") or item.get("conversations")
    if not msgs:
        return None
    # Normalise "conversations" style  {"from": ..., "value": ...}
    normalised = []
    for m in msgs:
        role    = m.get("role") or ("user" if m.get("from") in ("human", "user") else "assistant")
        content = m.get("content") or m.get("value", "")
        if role in ("system", "user", "assistant") and content:
            normalised.append({"role": role, "content": content})
    if not normalised:
        return None
    # Must have at least one user + one assistant turn
    roles = {m["role"] for m in normalised}
    if "user" not in roles or "assistant" not in roles:
        return None
    return {"messages": normalised}


def convert_alpaca(item: Dict) -> Optional[Dict]:
    instruction  = item.get("instruction", "").strip()
    input_text   = item.get("input", "").strip()
    output       = item.get("output", "").strip()
    if not instruction or not output:
        return None
    user_content = instruction + ("\n\n" + input_text if input_text else "")
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def convert_sharegpt(item: Dict) -> Optional[Dict]:
    messages = []
    for turn in item.get("conversations", []):
        role = "user" if turn.get("from") in ("human", "user") else "assistant"
        content = turn.get("value", "").strip()
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None
    return {"messages": messages}


def convert_messages(item: Dict) -> Optional[Dict]:
    msgs = item.get("messages")
    if not msgs:
        return None
    return {"messages": msgs}


CONVERTERS = {
    "nemotron":  convert_nemotron,
    "alpaca":    convert_alpaca,
    "sharegpt":  convert_sharegpt,
    "messages":  convert_messages,
}

# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def iter_dataset(name: str, max_samples: int) -> Iterator[Dict]:
    """Yields converted {"messages": [...]} dicts from one source dataset."""
    cfg       = DATASETS[name]
    converter = CONVERTERS[cfg["converter"]]

    print(f"  Loading {cfg['hf_path']} (split={cfg['hf_split']}) ...")
    ds = load_dataset(
        cfg["hf_path"],
        split=cfg["hf_split"],
        trust_remote_code=True,
    )
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    n_ok = n_skip = 0
    for item in ds:
        converted = converter(item)
        if converted is None:
            n_skip += 1
            continue
        n_ok += 1
        yield converted
    print(f"    → {n_ok:,} kept, {n_skip:,} skipped")

# ─────────────────────────────────────────────────────────────────────────────
# Prepare functions
# ─────────────────────────────────────────────────────────────────────────────

def prepare_paper_mix(output_dir: str, max_samples: int, seed: int):
    """
    Reproduces the paper's ~800K training set:
      Nemotron Post-Training V2  (780K)  +  CodeAlpaca  (~20K)
    Shuffles the merged list before writing.
    """
    print("[prepare_data] Building paper mix: Nemotron V2 + CodeAlpaca")
    all_samples: List[Dict] = []

    for ds_name, cap in PAPER_MIX:
        effective_cap = cap if max_samples < 0 else min(cap, max_samples // len(PAPER_MIX))
        all_samples.extend(iter_dataset(ds_name, effective_cap))

    # Global cap for quick tests
    if max_samples > 0:
        all_samples = all_samples[:max_samples]

    print(f"  Total before shuffle: {len(all_samples):,}")
    random.seed(seed)
    random.shuffle(all_samples)

    out_path = Path(output_dir) / "dflash_paper_train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        for item in all_samples:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[prepare_data] Wrote {len(all_samples):,} samples → {out_path}")


def prepare_single(dataset_name: str, output_dir: str, max_samples: int):
    out_path = Path(output_dir) / f"{dataset_name}_train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fp:
        for item in iter_dataset(dataset_name, max_samples):
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"[prepare_data] Wrote {n:,} samples → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paper",   action="store_true",
                       help="Paper setting: mix Nemotron V2 + CodeAlpaca (~800K)")
    group.add_argument("--dataset", choices=list(DATASETS.keys()),
                       help="Single dataset mode")
    parser.add_argument("--output-dir",  default="./cache/dataset")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Cap total samples (use for smoke-tests, e.g. 1000)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if args.paper:
        prepare_paper_mix(args.output_dir, args.max_samples, args.seed)
    else:
        prepare_single(args.dataset, args.output_dir, args.max_samples)
