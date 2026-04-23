# =============================================================================
# SemEval-2026 Task 13 - Perplexity Feature Extraction
# =============================================================================
# Cach chay:
#   .\run_perplexity.ps1                    # Chay full train set
#   .\run_perplexity.ps1 -Mode test         # Test 100 mau
#   .\run_perplexity.ps1 -Mode medium       # 5000 mau
#   .\run_perplexity.ps1 -Mode val          # Validation set
# =============================================================================

param(
    [ValidateSet("full", "test", "medium", "val")]
    [string]$Mode = "full"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SemEval-2026 Task 13 - Perplexity Extraction" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check GPU
Write-Host "`nChecking GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU - will use CPU (slow)')"

switch ($Mode) {
    "test" {
        Write-Host "`n[TEST MODE] Processing 100 samples only" -ForegroundColor Green
        python extract_perplexity.py `
            --input ./task_A/train.parquet `
            --output ./task_A/train_perplexity.parquet `
            --sample 100 `
            --save_every 50
    }
    "medium" {
        Write-Host "`n[MEDIUM MODE] Processing 5000 samples" -ForegroundColor Green
        python extract_perplexity.py `
            --input ./task_A/train.parquet `
            --output ./task_A/train_perplexity.parquet `
            --sample 5000 `
            --save_every 1000
    }
    "val" {
        Write-Host "`n[VAL MODE] Processing validation set" -ForegroundColor Green
        python extract_perplexity.py `
            --input ./task_A/validation.parquet `
            --output ./task_A/val_perplexity.parquet `
            --save_every 2000
    }
    "full" {
        Write-Host "`n[FULL MODE] Processing entire train set (~500K samples)" -ForegroundColor Red
        Write-Host "WARNING: This may take several hours!" -ForegroundColor Red
        Write-Host "Press Ctrl+C to cancel, or wait 5 seconds to continue..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        python extract_perplexity.py `
            --input ./task_A/train.parquet `
            --output ./task_A/train_perplexity.parquet `
            --save_every 5000
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Done! Check output in ./task_A/" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
