Write-Host "=================================" -ForegroundColor Cyan
Write-Host " Librarian Base v1 Training"
Write-Host "=================================" -ForegroundColor Cyan

# ---------------------------------
# Step 1 — Download datasets
# ---------------------------------
Write-Host ""
Write-Host "Step 1: Download datasets" -ForegroundColor Yellow
python src/data/download.py


# ---------------------------------
# Step 2 — Clean datasets
# ---------------------------------
Write-Host ""
Write-Host "Step 2: Clean datasets" -ForegroundColor Yellow
python src/data/clean.py


# ---------------------------------
# Step 3 — Train tokenizer
# ---------------------------------
Write-Host ""
Write-Host "Step 3: Train tokenizer" -ForegroundColor Yellow
python tokenizer/train_tokenizer.py


# ---------------------------------
# Step 4 — Tokenize dataset
# ---------------------------------
Write-Host ""
Write-Host "Step 4: Tokenize dataset" -ForegroundColor Yellow
python src/data/tokenizer.py


# ---------------------------------
# Step 5 — Pack tokens
# ---------------------------------
Write-Host ""
Write-Host "Step 5: Pack tokens" -ForegroundColor Yellow
python src/data/pack.py


# ---------------------------------
# Step 6 — Start Training
# ---------------------------------
Write-Host ""
Write-Host "Step 6: Start model training" -ForegroundColor Yellow
python train.py


Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host " Training Pipeline Complete"
Write-Host "=================================" -ForegroundColor Cyan