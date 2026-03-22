#!/bin/bash
set -e

# ─────────────────────────────────────
# Resolve project root (CRITICAL)
# ─────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Running from: $(pwd)"

# ─────────────────────────────────────
# Configurable parameters
# ─────────────────────────────────────
RUN_ID=${RUN_ID:-$(date +%s)}

MODEL_CONFIG=${MODEL_CONFIG:-configs/model_390M.json}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_390M.json}

# Single GPU — set which device to use (default: GPU 0)
# Override with: CUDA_DEVICE=1 bash scripts/train.sh
CUDA_DEVICE=${CUDA_DEVICE:-0}

export RUN_ID
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo ""
echo "================================="
echo " Librarian Base v1 Training"
echo "================================="
echo "RUN_ID:  $RUN_ID"
echo "GPU:     $CUDA_DEVICE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Model:   $MODEL_CONFIG"
echo "Train:   $TRAIN_CONFIG"

# ─────────────────────────────────────
# Logging — tee everything to file
# ─────────────────────────────────────
mkdir -p logs
exec > >(tee logs/run_$RUN_ID.log) 2>&1

# ─────────────────────────────────────
# Environment sanity checks
# ─────────────────────────────────────
echo ""
echo "Checking environment..."

python - <<PYEOF
import torch
assert torch.cuda.is_available(), "CUDA not available"
props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1e9
print(f"CUDA OK:  {props.name}")
print(f"VRAM:     {total_gb:.1f} GB")
assert total_gb >= 20, f"Expected >=20GB VRAM, got {total_gb:.1f}GB"

import datasets
print(f"datasets: {datasets.__version__}")
PYEOF

# ─────────────────────────────────────
# .env — load HF_TOKEN if present
# ─────────────────────────────────────
if [ -f .env ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
    echo "HF_TOKEN: ${HF_TOKEN:+set (${#HF_TOKEN} chars)}"
else
    echo "WARNING: no .env file found — HF_TOKEN may not be set"
fi

# ─────────────────────────────────────
# Helper
# ─────────────────────────────────────
run_step () {
    STEP_NAME=$1
    CMD=$2

    echo ""
    echo "Step: $STEP_NAME"
    echo "---------------------------------"

    eval $CMD
}

# ─────────────────────────────────────
# Pipeline
# ─────────────────────────────────────

run_step "Download datasets" \
"python src/data/download.py"

run_step "Clean datasets" \
"python src/data/clean.py"

[ -f data/cleaned/merged_train.txt ] || {
    echo "ERROR: merged_train.txt missing after clean step"
    exit 1
}

run_step "Train tokenizer" \
"python tokenizer/train_tokenizer.py"

[ -f tokenizer/tokenizer.json ] || {
    echo "ERROR: tokenizer/tokenizer.json missing"
    exit 1
}

run_step "Tokenize dataset" \
"python src/data/tokenizer.py"

run_step "Pack tokens" \
"python src/data/pack.py"

[ -f data/tokenized/train_packed.bin ] || {
    echo "ERROR: train_packed.bin missing after pack step"
    exit 1
}

# ─────────────────────────────────────
# Training — single GPU, plain python
# (no torchrun — trainer.py has no DDP)
# ─────────────────────────────────────
RESUME=""

if [ -f checkpoints/latest.pt ]; then
    echo ""
    echo "Found checkpoints/latest.pt — resuming..."
    RESUME="--resume checkpoints/latest.pt"
fi

run_step "Model training" \
"python train.py \
  --model_config $MODEL_CONFIG \
  --train_config $TRAIN_CONFIG \
  $RESUME"

echo ""
echo "================================="
echo " Training Complete"
echo "================================="

# ─────────────────────────────────────
# Evaluation
# ─────────────────────────────────────
BEST_CKPT=$(ls -t checkpoints/best/*.pt 2>/dev/null | head -n 1)

if [ -n "$BEST_CKPT" ]; then
    echo "Evaluating: $BEST_CKPT"
    python src/evaluation/eval_runner.py \
        --checkpoint "$BEST_CKPT"
else
    echo "No checkpoint found, skipping eval"
fi
