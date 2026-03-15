Write-Host "=================================" -ForegroundColor Cyan
Write-Host " Librarian Base v1 Inference"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Select mode:" -ForegroundColor Yellow
Write-Host "1) Chat mode (interactive)"
Write-Host "2) Single prompt inference"
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {

```
Write-Host ""
Write-Host "Starting chat mode..." -ForegroundColor Green
python scripts/run_chat.py
```

}
elseif ($choice -eq "2") {

```
Write-Host ""
$prompt = Read-Host "Enter prompt"

Write-Host ""
Write-Host "Running inference..." -ForegroundColor Green

python scripts/run_infer.py --prompt "$prompt"
```

}
else {

```
Write-Host "Invalid choice." -ForegroundColor Red
```

}
