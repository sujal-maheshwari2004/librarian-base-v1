#!/usr/bin/env bash

set -e

echo "================================="
echo " Librarian Base v1 Training"
echo "================================="

echo ""
echo "Step 1: Download datasets"
python src/data/download.py

echo ""
echo "Step 2: Clean datasets"
python src/data/clean.py

echo ""
echo "Step 3: Train tokenizer"
python tokenizer/train_tokenizer.py

echo ""
echo "Step 4: Tokenize dataset"
python src/data/tokenizer.py

echo ""
echo "Step 5: Pack tokens"
python src/data/pack.py

echo ""
echo "Step 6: Start model training"
python train.py

echo ""
echo "================================="
echo " Training Pipeline Complete"
echo "================================="