# ── Must run before any huggingface_hub / datasets import ────────────────────
import os

# 1. Mirror
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _HF_MIRROR

# 2. Cache dir → large disk instead of home partition
_HF_CACHE = os.environ.get("HF_HUB_CACHE", "/mnt/data/szf_temp/cache/hf_hub")
os.environ["HF_HUB_CACHE"]          = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_CACHE   # older env var name
os.makedirs(_HF_CACHE, exist_ok=True)

# 3. Patch huggingface_hub constants (endpoint cached at import time)
import huggingface_hub
import huggingface_hub.constants as _hf_constants
_hf_constants.ENDPOINT       = _HF_MIRROR
_hf_constants.HF_HUB_CACHE   = _HF_CACHE

# 4. Disable SSL verification — works for requests, httpx, and urllib3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # urllib / older libs
os.environ["CURL_CA_BUNDLE"]    = ""   # curl-based clients
os.environ["REQUESTS_CA_BUNDLE"] = ""  # requests
os.environ["HTTPX_VERIFY"]       = "0" # httpx env var

# Monkey-patch httpx.Client so every instance has verify=False
try:
    import httpx
    _orig_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        kwargs.setdefault("timeout", 120)
        _orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init
except ImportError:
    pass

print(f"[mirror] HF_ENDPOINT  → {_HF_MIRROR}")
print(f"[cache]  HF_HUB_CACHE → {_HF_CACHE}")
print(f"[ssl]    verify=False")
# ─────────────────────────────────────────────────────────────────────────────

"""
Step 1: Download and prepare seed dataset.

Paper setting (DFlash §4):
  ~800K samples = Nemotron Post-Training Dataset V1  (~780K)
                + CodeAlpaca                          (~20K)

Modes:
  --paper          Mix Nemotron V1 + CodeAlpaca to match paper (~800K total)
  --dataset NAME   Single dataset (nemotron / codealpaca / sharegpt / perfectblend)

Output: ./cache/dataset/dflash_paper_train.jsonl  (messages format, shuffled)

Usage:
    # Paper setting (recommended):
    python scripts/prepare_data.py --paper --output-dir ./cache/dataset

    # Single dataset:
    python scripts/prepare_data.py --dataset nemotron --output-dir ./cache/dataset

    # Quick smoke-test (1000 samples):
    python scripts/prepare_data.py --paper --max-samples 1000
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "nemotron": {
        # v2 is gated
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
# wget-based downloader (bypasses Python SSL issues entirely)
# ─────────────────────────────────────────────────────────────────────────────

import subprocess
import requests

def _wget_download(url: str, dest: str) -> bool:
    """Download url → dest using wget (no SSL verify). Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return True
    cmd = ["wget", "-q", "--no-check-certificate", "-O", dest, url]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        print(f"  [wget] failed: {url}\n  {ret.stderr.decode()[:200]}")
        return False
    return True


def _list_repo_files(hf_path: str, split: str = "train") -> List[str]:
    """
    Returns parquet file URLs for a HF dataset repo using the Hub API.
    Falls back to a single-shard guess if the API call fails.
    """
    mirror   = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    api_url  = f"{mirror}/api/datasets/{hf_path}/parquet/{split}"
    try:
        resp = requests.get(api_url, verify=False, timeout=30)
        if resp.ok:
            files = resp.json()
            return [f["url"].replace("https://huggingface.co", mirror) for f in files]
    except Exception as e:
        print(f"  [warn] API call failed: {e}")

    # Fallback: try the tree endpoint
    tree_url = f"{mirror}/api/datasets/{hf_path}/tree/main/data"
    try:
        resp = requests.get(tree_url, verify=False, timeout=30)
        if resp.ok:
            files = [
                f"{mirror}/datasets/{hf_path}/resolve/main/{f['path']}"
                for f in resp.json()
                if f["path"].endswith(".parquet") and split in f["path"]
            ]
            if files:
                return files
    except Exception as e:
        print(f"  [warn] tree API failed: {e}")

    return []


def _load_with_wget(hf_path: str, split: str, local_base: str) -> "datasets.Dataset":
    """Download parquet files via wget and load locally."""
    import datasets as _datasets

    urls     = _list_repo_files(hf_path, split)
    if not urls:
        raise RuntimeError(f"Could not find any parquet files for {hf_path}/{split}")

    print(f"  Found {len(urls)} parquet shard(s), downloading via wget ...")
    local_dir = os.path.join(local_base, hf_path.replace("/", "--"), split)
    os.makedirs(local_dir, exist_ok=True)

    local_files = []
    for url in urls:
        fname = url.split("/")[-1].split("?")[0]
        dest  = os.path.join(local_dir, fname)
        print(f"    {fname} ...", end=" ", flush=True)
        ok = _wget_download(url, dest)
        print("OK" if ok else "FAILED")
        if ok:
            local_files.append(dest)

    if not local_files:
        raise RuntimeError("All downloads failed.")

    return _datasets.load_dataset("parquet", data_files=local_files, split="train")


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def iter_dataset(name: str, max_samples: int) -> Iterator[Dict]:
    """Yields converted {"messages": [...]} dicts from one source dataset."""
    cfg       = DATASETS[name]
    converter = CONVERTERS[cfg["converter"]]
    local_base = os.environ.get("HF_HUB_CACHE", "/mnt/data/szf_temp/cache/hf_hub")

    print(f"  Loading {cfg['hf_path']} (split={cfg['hf_split']}) ...")
    try:
        ds = _load_with_wget(cfg["hf_path"], cfg["hf_split"], local_base)
    except Exception as e:
        print(f"  [wget] failed ({e}), falling back to load_dataset ...")
        ds = load_dataset(cfg["hf_path"], split=cfg["hf_split"])
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
      Nemotron Post-Training V1  (780K)  +  CodeAlpaca  (~20K)
    Shuffles the merged list before writing.
    """
    print("[prepare_data] Building paper mix: Nemotron V1 + CodeAlpaca")
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
