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
):
    # Load existing output to allow resumption
    done_indices = set()
    out_file = Path(output_path)
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                item = json.loads(line)
                if "_idx" in item:
                    done_indices.add(item["_idx"])
        logger.info(f"Resuming: {len(done_indices)} samples already done.")

    raw = [
        json.loads(l) for l in Path(input_path).read_text().splitlines()
        if l.strip()
    ]
    if max_samples > 0:
        raw = raw[:max_samples]

    server_url = f"http://{server_address}"
    semaphore  = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, item in enumerate(raw):
            if idx in done_indices:
                continue
            tasks.append((idx, item, regenerate_one(
                session, server_url, model_name,
                item["messages"], max_tokens, temperature,
                is_reasoning_model, semaphore,
            )))

        logger.info(f"Regenerating {len(tasks)} samples with concurrency={concurrency}")
        t0 = time.time()

        with open(output_path, "a") as out_fp:
            pending = [(idx, item, asyncio.ensure_future(coro))
                       for idx, item, coro in tasks]
            for i, (idx, item, fut) in enumerate(pending):
                new_content = await fut
                if new_content is None:
                    continue

                # Replace last assistant turn with regenerated content
                new_messages = []
                last_assistant_replaced = False
                for m in reversed(item["messages"]):
                    if m["role"] == "assistant" and not last_assistant_replaced:
                        new_messages.insert(0, {"role": "assistant", "content": new_content})
                        last_assistant_replaced = True
                    else:
                        new_messages.insert(0, m)
                if not last_assistant_replaced:
                    new_messages.append({"role": "assistant", "content": new_content})

                out_fp.write(json.dumps({
                    "messages": new_messages,
                    "_idx": idx,
                }, ensure_ascii=False) + "\n")

                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"Progress: {i+1}/{len(pending)} | "
                        f"Elapsed: {elapsed:.0f}s | "
                        f"Throughput: {(i+1)/elapsed:.1f} samples/s"
                    )

    logger.info(f"Regeneration complete. Output: {output_path}")


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
