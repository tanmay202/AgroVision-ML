# AgroVision-ML — Launch Staging Server (PowerShell)
# ==================================================

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Starting AgroVision-ML Universal Staging Environment" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

$env:PYTHONPATH = "$PSScriptRoot/..;$PSScriptRoot/../member2-price-volatility/src"
$env:AGROVISION_ENV = "staging"
$env:AGROVISION_ENABLE_SHADOW_EVALUATION = "true"
$env:AGROVISION_ENABLE_RICE_SHADOW = "true"
$env:AGROVISION_ENABLE_WHEAT_SHADOW = "true"
$env:AGROVISION_ENABLE_POTATO_SHADOW = "true"
$env:AGROVISION_ENABLE_TEA_SHADOW = "true"

Write-Host "Loaded staging configuration." -ForegroundColor Cyan
Write-Host "Launching Uvicorn ASGI server on http://0.0.0.0:8000 ..." -ForegroundColor Cyan

python -m uvicorn api_v2:app --app-dir "$PSScriptRoot/../member2-price-volatility/src" --host 0.0.0.0 --port 8000 --reload
