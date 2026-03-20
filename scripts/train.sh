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
NUM_GPUS=${NUM_GPUS:-8}

MODEL_CONFIG=${MODEL_CONFIG:-configs/model_130M.json}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_130M.json}

export RUN_ID

echo ""
echo "================================="
echo " Librarian Base v1 Training"
echo "================================="
echo "RUN_ID: $RUN_ID"
echo "GPUs:   $NUM_GPUS"
echo "Model:  $MODEL_CONFIG"
echo "Train:  $TRAIN_CONFIG"

# ─────────────────────────────────────
# Logging
# ─────────────────────────────────────
mkdir -p logs
exec > >(tee logs/run_$RUN_ID.log) 2>&1

# ─────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────
echo ""
echo "Checking environment..."

python - <<EOF
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("CUDA OK:", torch.cuda.get_device_name(0))
EOF

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

# hard check
[ -f data/cleaned/merged_train.txt ] || {
  echo "ERROR: merged_train.txt missing"; exit 1;
}

run_step "Train tokenizer" \
"python tokenizer/train_tokenizer.py"

[ -f tokenizer/tokenizer.json ] || {
  echo "ERROR: tokenizer missing"; exit 1;
}

run_step "Tokenize dataset" \
"python src/data/tokenizer.py"

run_step "Pack tokens" \
"python src/data/pack.py"

[ -f data/tokenized/train_packed.bin ] || {
  echo "ERROR: packed dataset missing"; exit 1;
}

# ─────────────────────────────────────
# Training (with resume support)
# ─────────────────────────────────────
RESUME=""

if [ -f checkpoints/latest.pt ]; then
  echo "Resuming from checkpoint..."
  RESUME="--resume checkpoints/latest.pt"
fi

run_step "Model training" \
"torchrun --nproc_per_node=$NUM_GPUS train.py \
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
