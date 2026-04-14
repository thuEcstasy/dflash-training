"""
Step 2: Regenerate assistant responses using the target model.

WHY: DFlash trains with teacher forcing on ground-truth tokens.
At inference, the draft model must predict the *target model's* output
distribution.  If we train on responses written by humans or a different
model, the distribution gap hurts acceptance length.

SOLUTION: Use SGLang to re-generate the assistant turns with the SAME
target model that will be used for speculative decoding.  The regenerated
responses are then used as training targets.

Usage:
    # Start the SGLang server first:
    python -m sglang.launch_server \
        --model Qwen/Qwen3-8B  \
        --dtype bfloat16       \
        --mem-frac 0.8         \
        --port 30000

    # Then run this script:
    python scripts/regenerate_data.py \
        --input  ./cache/dataset/perfectblend_train.jsonl   \
        --output ./cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
        --model  Qwen/Qwen3-8B \
        --server-address localhost:30000 \
        --concurrency 128 \
        --max-tokens 8192 \
        --temperature 0.7 \
        --is-reasoning-model   # for Qwen3 thinking mode
"""

import os
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _HF_MIRROR

import huggingface_hub
import huggingface_hub.constants as _hf_constants
_hf_constants.ENDPOINT = _HF_MIRROR
huggingface_hub.ENDPOINT = _HF_MIRROR          # type: ignore[attr-defined]
print(f"[mirror] HF_ENDPOINT → {_HF_MIRROR}")

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Async regeneration via OpenAI-compatible SGLang endpoint
# ─────────────────────────────────────────────────────────────────────────────

async def regenerate_one(
    session: aiohttp.ClientSession,
    server_url: str,
    model_name: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    is_reasoning_model: bool,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Regenerates the assistant response for one conversation."""
    # Keep only the turns up to (not including) the last assistant turn
    prompt_messages = []
    for m in messages:
        if m["role"] == "assistant":
            break
        prompt_messages.append(m)

    if not prompt_messages:
        return None

    payload = {
        "model": model_name,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Qwen3 thinking mode requires extra parameter
    if is_reasoning_model:
        payload["extra_body"] = {"enable_thinking": True}

    url = f"{server_url}/v1/chat/completions"
    async with semaphore:
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(f"HTTP {resp.status}: {text[:200]}")
                    return None
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return None


def build_new_messages(original: List[Dict], new_content: str) -> List[Dict]:
    """Replace the last assistant turn with new_content."""
    new_messages = []
    replaced = False
    for m in reversed(original):
        if m["role"] == "assistant" and not replaced:
            new_messages.insert(0, {"role": "assistant", "content": new_content})
            replaced = True
        else:
            new_messages.insert(0, m)
    if not replaced:
        new_messages.append({"role": "assistant", "content": new_content})
    return new_messages


async def regenerate_dataset(
    input_path: str,
    output_path: str,
    server_address: str,
    model_name: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    is_reasoning_model: bool,
    max_samples: int = -1,
    chunk_size: int = 2000,   # process this many samples at a time to bound memory
):
    # ── Resume: find already-completed indices ────────────────────────────────
    done_indices: set = set()
    out_file = Path(output_path)
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "_idx" in item:
                        done_indices.add(item["_idx"])
                except json.JSONDecodeError:
                    pass
        logger.info(f"Resuming: {len(done_indices)} samples already done.")

    # ── Read input (streaming to avoid loading 800K items at once) ────────────
    all_lines = Path(input_path).read_text().splitlines()
    if max_samples > 0:
        all_lines = all_lines[:max_samples]
    total = len(all_lines)

    todo = [
        (idx, json.loads(line))
        for idx, line in enumerate(all_lines)
        if line.strip() and idx not in done_indices
    ]
    logger.info(f"Total: {total} | Todo: {len(todo)} | Concurrency: {concurrency}")

    server_url = f"http://{server_address}"
    semaphore  = asyncio.Semaphore(concurrency)
    t0         = time.time()
    n_done     = len(done_indices)

    # ── Process in chunks to keep memory bounded ──────────────────────────────
    async with aiohttp.ClientSession() as session:
        with open(output_path, "a") as out_fp:
            for chunk_start in range(0, len(todo), chunk_size):
                chunk = todo[chunk_start : chunk_start + chunk_size]

                # Schedule this chunk's requests concurrently
                futures = [
                    asyncio.ensure_future(
                        regenerate_one(
                            session, server_url, model_name,
                            item["messages"], max_tokens, temperature,
                            is_reasoning_model, semaphore,
                        )
                    )
                    for _, item in chunk
                ]

                # Collect results as they finish
                for (idx, item), fut in zip(chunk, asyncio.as_completed(futures)):
                    new_content = await fut
                    if new_content is None:
                        continue

                    new_messages = build_new_messages(item["messages"], new_content)
                    out_fp.write(json.dumps(
                        {"messages": new_messages, "_idx": idx},
                        ensure_ascii=False,
                    ) + "\n")
                    out_fp.flush()
                    n_done += 1

                elapsed = time.time() - t0
                throughput = n_done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {n_done}/{total} | "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"Throughput: {throughput:.1f} samples/s | "
                    f"ETA: {(total - n_done) / max(throughput, 1e-6):.0f}s"
                )

    logger.info(f"Regeneration complete → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--model",          default="Qwen/Qwen3-8B")
    parser.add_argument("--server-address", default="localhost:30000")
    parser.add_argument("--concurrency",    type=int, default=128)
    parser.add_argument("--max-tokens",     type=int, default=8192)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--max-samples",    type=int, default=-1)
    args = parser.parse_args()

    asyncio.run(regenerate_dataset(
        input_path=args.input,
        output_path=args.output,
        server_address=args.server_address,
        model_name=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        is_reasoning_model=args.is_reasoning_model,
        max_samples=args.max_samples,
    ))
