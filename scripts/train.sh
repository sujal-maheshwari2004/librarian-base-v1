#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Running from: $(pwd)"

RUN_ID=${RUN_ID:-$(date +%s)}
MODEL_CONFIG=${MODEL_CONFIG:-configs/model_390M.json}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_390M.json}
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

mkdir -p logs
exec > >(tee logs/run_$RUN_ID.log) 2>&1

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

if [ -f .env ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
    echo "HF_TOKEN: ${HF_TOKEN:+set (${#HF_TOKEN} chars)}"
else
    echo "WARNING: no .env file found"
fi

run_step () {
    STEP_NAME=$1
    CMD=$2
    echo ""
    echo "Step: $STEP_NAME"
    echo "---------------------------------"
    eval $CMD
}

run_step "Download datasets"  "python -m src.data.download"
run_step "Clean datasets"     "python -m src.data.clean"

[ -f data/cleaned/merged_train.txt ] || {
    echo "ERROR: merged_train.txt missing after clean step"
    exit 1
}

run_step "Train tokenizer"    "python -m tokenizer.train_tokenizer"

[ -f tokenizer/tokenizer.json ] || {
    echo "ERROR: tokenizer/tokenizer.json missing"
    exit 1
}

run_step "Tokenize dataset"   "python -m src.data.tokenizer"
run_step "Pack tokens"        "python -m src.data.pack"

[ -f data/tokenized/train_packed.bin ] || {
    echo "ERROR: train_packed.bin missing after pack step"
    exit 1
}

# ── Carve validation split from train if not already present ──────────
if [ ! -f data/tokenized/validation_packed.bin ]; then
    echo ""
    echo "Step: Carve validation split"
    echo "---------------------------------"
    python - <<PYEOF
import numpy as np

SEQ_LEN  = 512
VAL_FRAC = 0.005   # 0.5% → ~12,500 sequences at 2.5M total

data  = np.memmap("data/tokenized/train_packed.bin", dtype=np.uint16, mode="r")
n_seq = len(data) // SEQ_LEN

val_seq    = max(1, int(n_seq * VAL_FRAC))
val_tokens = val_seq * SEQ_LEN

print(f"Total sequences : {n_seq:,}")
print(f"Val  sequences  : {val_seq:,}  ({VAL_FRAC*100:.1f}%)")
print(f"Train sequences : {n_seq - val_seq:,}")

val   = np.array(data[-val_tokens:])
train = np.array(data[:-val_tokens])

val.tofile("data/tokenized/validation_packed.bin")
train.tofile("data/tokenized/train_packed.bin")

print("validation_packed.bin written")
print("train_packed.bin  trimmed")
PYEOF
else
    echo "validation_packed.bin already exists — skipping split"
fi

# ── Resume if latest checkpoint exists ───────────────────────────────
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

BEST_CKPT=$(ls -t checkpoints/best/*.pt 2>/dev/null | head -n 1)
if [ -n "$BEST_CKPT" ]; then
    echo "Evaluating: $BEST_CKPT"
    python -m src.evaluation.eval_runner --checkpoint "$BEST_CKPT"
else
    echo "No checkpoint found, skipping eval"
fi
