#!/bin/bash

set -e  # exit immediately on error

echo "================================="
echo " Librarian Base v1 Training"
echo "================================="

# ─────────────────────────────────────
# Shared RUN_ID
# ─────────────────────────────────────
export RUN_ID=$(date +%s)

echo ""
echo "RUN_ID: $RUN_ID"

# helper function for step execution
run_step () {
    STEP_NAME=$1
    CMD=$2

    echo ""
    echo "Step: $STEP_NAME"
    echo "---------------------------------"

    eval $CMD

    if [ $? -ne 0 ]; then
        echo "FAILED at $STEP_NAME"
        exit 1
    fi
}

# ─────────────────────────────────────
# Step 1 — Download datasets
# ─────────────────────────────────────
run_step "Download datasets" "python src/data/download.py"

# ─────────────────────────────────────
# Step 2 — Clean datasets
# ─────────────────────────────────────
run_step "Clean datasets" "python src/data/clean.py"

# ─────────────────────────────────────
# Step 3 — Train tokenizer
# ─────────────────────────────────────
run_step "Train tokenizer" "python tokenizer/train_tokenizer.py"

# ─────────────────────────────────────
# Step 4 — Tokenize dataset
# ─────────────────────────────────────
run_step "Tokenize dataset" "python src/data/tokenizer.py"

# ─────────────────────────────────────
# Step 5 — Pack tokens
# ─────────────────────────────────────
run_step "Pack tokens" "python src/data/pack.py"

# ─────────────────────────────────────
# Step 6 — Start Training
# ─────────────────────────────────────
# (multi-GPU ready — adjust as needed)
run_step "Model training" "torchrun --nproc_per_node=8 train.py"

echo ""
echo "================================="
echo " Training Pipeline Complete"
echo "================================="

# ─────────────────────────────────────
# Step 7 — Evaluate best checkpoint
# ─────────────────────────────────────
echo ""
echo "Step 7: Evaluate best model"

BEST_CKPT=$(ls -t checkpoints/best/*.pt 2>/dev/null | head -n 1)

if [ -n "$BEST_CKPT" ]; then
    echo "Best checkpoint found: $BEST_CKPT"
    python src/evaluation/eval_runner.py --checkpoint "$BEST_CKPT"
else
    echo "No checkpoint found."
fi
