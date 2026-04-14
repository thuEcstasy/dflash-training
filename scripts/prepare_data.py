"""
Step 1: Download and prepare seed dataset.

Supported seed datasets:
  - sharegpt           (anon8231489123/ShareGPT_Vicuna_unfiltered)
  - nemotron           (nvidia/Nemotron-Post-Training-Dataset-v2)
  - codealpaca         (sahil2801/CodeAlpaca-20k)
  - perfectblend       (HuggingFace dataset used in SpecForge community)

Output: ./cache/dataset/{name}_train.jsonl  (messages format)

Usage:
    python scripts/prepare_data.py --dataset perfectblend --output-dir ./cache/dataset
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset


DATASETS = {
    "sharegpt": {
        "hf_path":    "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "hf_split":   "train",
        "converter":  "sharegpt",
    },
    "nemotron": {
        "hf_path":    "nvidia/Nemotron-Post-Training-Dataset-v2",
        "hf_split":   "train",
        "converter":  "messages",
    },
    "codealpaca": {
        "hf_path":    "sahil2801/CodeAlpaca-20k",
        "hf_split":   "train",
        "converter":  "alpaca",
    },
    "perfectblend": {
        "hf_path":    "eigen-ai-labs/perfectblend-qwen3-8b-regen-demo",
        "hf_split":   "train",
        "converter":  "messages",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Format converters → unified {"messages": [...]} format
# ─────────────────────────────────────────────────────────────────────────────

def convert_sharegpt(item: Dict) -> Dict:
    messages = []
    for turn in item.get("conversations", []):
        role = "user" if turn["from"] in ("human", "user") else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}


def convert_alpaca(item: Dict) -> Dict:
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    output      = item.get("output", "")
    user_content = instruction + ("\n" + input_text if input_text else "")
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def convert_messages(item: Dict) -> Dict:
    """Pass-through for datasets already in messages format."""
    return {"messages": item["messages"]}


CONVERTERS = {
    "sharegpt": convert_sharegpt,
    "alpaca":   convert_alpaca,
    "messages": convert_messages,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def prepare(dataset_name: str, output_dir: str, max_samples: int = -1):
    cfg = DATASETS[dataset_name]
    print(f"[prepare_data] Loading {cfg['hf_path']} ({cfg['hf_split']}) ...")
    ds = load_dataset(cfg["hf_path"], split=cfg["hf_split"])

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    converter = CONVERTERS[cfg["converter"]]
    out_path  = Path(output_dir) / f"{dataset_name}_train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path, "w") as fp:
        for item in ds:
            try:
                converted = converter(item)
                if not converted["messages"]:
                    continue
                fp.write(json.dumps(converted, ensure_ascii=False) + "\n")
                n_written += 1
            except Exception as e:
                print(f"  Skipping item: {e}")

    print(f"[prepare_data] Wrote {n_written} samples → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--output-dir", default="./cache/dataset")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Limit for quick tests (-1 = all)")
    args = parser.parse_args()
    prepare(args.dataset, args.output_dir, args.max_samples)
