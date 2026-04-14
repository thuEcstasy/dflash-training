#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# DFlash Draft Model – One-shot Training Script
#
# Paper setting: Nemotron Post-Training V2 + CodeAlpaca (~800K total)
#
# Usage:
#   bash scripts/train.sh                        # defaults below
#   MODEL=Qwen/Qwen3-8B NUM_GPUS=4 bash scripts/train.sh
#
# Override any variable on the command line:
#   MODEL          target model (HF name or local path)
#   CONFIG         path to yaml config
#   NUM_GPUS       number of GPUs for training
#   SGLANG_PORT    port for the temporary SGLang regen server
#   IS_REASONING   1 = Qwen3 thinking mode, 0 = standard chat model
#   MAX_TOKENS     max tokens for regen (increase for reasoning models)
#   TEMPERATURE    sampling temperature for regen
#   CACHE_DIR      where to store datasets and pip cache
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── User-configurable variables ───────────────────────────────────────────────
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
CONFIG=${CONFIG:-"configs/qwen3-8b.yaml"}
NUM_GPUS=${NUM_GPUS:-8}
SGLANG_PORT=${SGLANG_PORT:-30000}
IS_REASONING=${IS_REASONING:-1}           # 1 for Qwen3 thinking, 0 otherwise
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.7}
CACHE_DIR=${CACHE_DIR:-"./cache"}

# Fixed paths (derived from paper dataset names)
SEED_FILE="${CACHE_DIR}/dataset/dflash_paper_train.jsonl"
REGEN_FILE="${CACHE_DIR}/dataset/dflash_paper_regen.jsonl"

# ── CN mirror for HuggingFace ─────────────────────────────────────────────────
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export HF_HUB_CACHE="${CACHE_DIR}/hf_hub"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
echo "[env] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE}"

mkdir -p "${CACHE_DIR}/dataset" "${HF_HUB_CACHE}" "${PIP_CACHE_DIR}"

# ── Step 0: Install dependencies ─────────────────────────────────────────────
echo ""
echo "=== [0/4] Installing dependencies ==="
pip install -q --cache-dir "${PIP_CACHE_DIR}" \
    "torch>=2.4.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.30.0" \
    "datasets>=2.20.0" \
    "pyyaml" \
    "aiohttp" \
    "tensorboard"

# SGLang is required for data regeneration
if ! python -c "import sglang" 2>/dev/null; then
    echo "[warn] sglang not found – installing (this may take a while) ..."
    pip install -q --cache-dir "${PIP_CACHE_DIR}" "sglang[all]"
fi

# ── Step 1: Prepare seed data (paper mix) ────────────────────────────────────
echo ""
echo "=== [1/4] Preparing paper-mix seed dataset ==="
if [ -f "${SEED_FILE}" ]; then
    echo "  [skip] ${SEED_FILE} already exists."
else
    python scripts/prepare_data.py \
        --paper \
        --output-dir "${CACHE_DIR}/dataset"
fi

# ── Step 2: Regenerate responses with target model ───────────────────────────
echo ""
echo "=== [2/4] Regenerating responses with ${MODEL} ==="

if [ -f "${REGEN_FILE}" ]; then
    DONE=$(wc -l < "${REGEN_FILE}")
    TOTAL=$(wc -l < "${SEED_FILE}")
    echo "  [resume] ${DONE}/${TOTAL} samples already regenerated."
fi

# Start SGLang server
echo "  Starting SGLang server on port ${SGLANG_PORT} ..."
python -m sglang.launch_server \
    --model "${MODEL}" \
    --dtype bfloat16 \
    --mem-frac 0.85 \
    --port "${SGLANG_PORT}" \
    > "${CACHE_DIR}/sglang.log" 2>&1 &
SGLANG_PID=$!

# Ensure server is killed on exit (even on error)
trap 'echo "Stopping SGLang server (pid ${SGLANG_PID}) ..."; kill ${SGLANG_PID} 2>/dev/null || true' EXIT

# Wait for server (up to 5 min)
echo "  Waiting for SGLang server to become ready ..."
READY=0
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        READY=1
        echo "  SGLang server ready (${i}×5s)."
        break
    fi
    sleep 5
done
if [ "${READY}" -eq 0 ]; then
    echo "[error] SGLang server did not start within 5 minutes."
    echo "        Check ${CACHE_DIR}/sglang.log for details."
    exit 1
fi

# Build reasoning flag
REASONING_FLAG=""
if [ "${IS_REASONING}" -eq 1 ]; then
    REASONING_FLAG="--is-reasoning-model"
fi

python scripts/regenerate_data.py \
    --input  "${SEED_FILE}" \
    --output "${REGEN_FILE}" \
    --model  "${MODEL}" \
    --server-address "localhost:${SGLANG_PORT}" \
    --concurrency 128 \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    ${REASONING_FLAG}

# Kill SGLang (trap will also handle it, this is just explicit)
kill "${SGLANG_PID}" 2>/dev/null || true
trap - EXIT
echo "  SGLang server stopped."

# ── Step 3: Patch config to point at regenerated data ────────────────────────
echo ""
echo "=== [3/4] Patching config train_data_path ==="
# Use Python to update the yaml field safely (avoids sed quoting issues)
python - <<EOF
import yaml, pathlib
cfg_path = pathlib.Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["train_data_path"] = "${REGEN_FILE}"
cfg_path.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False))
print(f"  train_data_path → ${REGEN_FILE}")
EOF

# ── Step 4: Train draft model ─────────────────────────────────────────────────
echo ""
echo "=== [4/4] Training DFlash draft model (${NUM_GPUS} GPUs) ==="

accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    -m dflash.train \
    --config "${CONFIG}"

echo ""
echo "=== Training complete! ==="
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['output_dir'])")
echo "Checkpoints saved to: ${OUTPUT_DIR}"
