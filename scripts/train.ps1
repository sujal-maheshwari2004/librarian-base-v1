Write-Host "=================================" -ForegroundColor Cyan
Write-Host " Librarian Base v1 Training"
Write-Host "=================================" -ForegroundColor Cyan

# ─────────────────────────────────────
# Shared RUN_ID so all stages appear
# under the same run in the dashboard
# ─────────────────────────────────────
$env:RUN_ID = [int][double]::Parse(
    (Get-Date -UFormat "%s")
)
Write-Host ""
Write-Host "RUN_ID: $($env:RUN_ID)" -ForegroundColor DarkCyan

# ─────────────────────────────────────
# Step 1 — Download datasets
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 1: Download datasets" -ForegroundColor Yellow
python src/data/download.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at download" -ForegroundColor Red; exit 1 }

# ─────────────────────────────────────
# Step 2 — Clean datasets
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 2: Clean datasets" -ForegroundColor Yellow
python src/data/clean.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at clean" -ForegroundColor Red; exit 1 }

# ─────────────────────────────────────
# Step 3 — Train tokenizer
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 3: Train tokenizer" -ForegroundColor Yellow
python tokenizer/train_tokenizer.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at train_tokenizer" -ForegroundColor Red; exit 1 }

# ─────────────────────────────────────
# Step 4 — Tokenize dataset
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 4: Tokenize dataset" -ForegroundColor Yellow
python src/data/tokenizer.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at tokenize" -ForegroundColor Red; exit 1 }

# ─────────────────────────────────────
# Step 5 — Pack tokens
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 5: Pack tokens" -ForegroundColor Yellow
python src/data/pack.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at pack" -ForegroundColor Red; exit 1 }

# ─────────────────────────────────────
# Step 6 — Start Training
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 6: Start model training" -ForegroundColor Yellow
python train.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at train" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host " Training Pipeline Complete"
Write-Host "=================================" -ForegroundColor Cyan

# ─────────────────────────────────────
# Step 7 — Evaluate best checkpoint
# ─────────────────────────────────────
Write-Host ""
Write-Host "Step 7: Evaluate best model" -ForegroundColor Yellow

$best_ckpt = Get-ChildItem checkpoints/best/*.pt |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($best_ckpt) {
    Write-Host "Best checkpoint found: $($best_ckpt.FullName)"
    python src/evaluation/eval_runner.py --checkpoint $best_ckpt.FullName
} else {
    Write-Host "No checkpoint found." -ForegroundColor Red
}