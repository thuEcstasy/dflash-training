#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# DFlash Draft Model Training Script
#
# Full pipeline:
#   Step 0: Install dependencies
#   Step 1: Download + prepare seed dataset
#   Step 2: Start SGLang server + regenerate responses with target model
#   Step 3: Train the DFlash draft model
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=${MODEL:-"Qwen/Qwen3-8B"}
CONFIG=${CONFIG:-"configs/qwen3-8b.yaml"}
NUM_GPUS=${NUM_GPUS:-8}
SEED_DATASET=${SEED_DATASET:-"perfectblend"}
SGLANG_PORT=${SGLANG_PORT:-30000}
REASONING_MODEL_FLAG=${REASONING_MODEL_FLAG:-"--is-reasoning-model"}  # Remove for non-thinking models

# ── Step 0: Install dependencies ─────────────────────────────────────────────
echo "=== [0/3] Installing dependencies ==="
pip install -q \
    torch>=2.4.0 \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    datasets \
    pyyaml \
    aiohttp \
    tensorboard

# Optional: install SGLang for data regeneration
# pip install sglang[all]

# ── Step 1: Prepare seed data ─────────────────────────────────────────────────
echo "=== [1/3] Preparing seed dataset: $SEED_DATASET ==="
python scripts/prepare_data.py \
    --dataset "$SEED_DATASET" \
    --output-dir ./cache/dataset

# ── Step 2: Regenerate data with target model ─────────────────────────────────
echo "=== [2/3] Starting SGLang server for data regeneration ==="

# Launch SGLang server in background
python -m sglang.launch_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --mem-frac 0.8 \
    --port "$SGLANG_PORT" &
SGLANG_PID=$!

# Wait for server to be ready
echo "Waiting for SGLang server to start..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        echo "SGLang server ready."
        break
    fi
    sleep 5
done

echo "=== Regenerating training data ==="
python scripts/regenerate_data.py \
    --input  "./cache/dataset/${SEED_DATASET}_train.jsonl" \
    --output "./cache/dataset/${SEED_DATASET}_regen.jsonl" \
    --model  "$MODEL" \
    --server-address "localhost:${SGLANG_PORT}" \
    --concurrency 128 \
    --max-tokens 8192 \
    --temperature 0.7 \
    $REASONING_MODEL_FLAG

# Stop SGLang server
kill $SGLANG_PID || true
echo "SGLang server stopped."

# ── Step 3: Train draft model ─────────────────────────────────────────────────
echo "=== [3/3] Training DFlash draft model ==="

# Use Accelerate for multi-GPU training
# Alternatively: deepspeed --num_gpus $NUM_GPUS -m dflash.train --config $CONFIG

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    -m dflash.train \
    --config "$CONFIG"

echo "=== Training complete! ==="
echo "Checkpoints saved to: $(grep output_dir $CONFIG | awk '{print $2}')"
