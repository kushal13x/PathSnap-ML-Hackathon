<#
.SYNOPSIS
    Start the PathSnap FastAPI inference backend (Laptop 1).
.EXAMPLE
    .\scripts\start_backend.ps1
    .\scripts\start_backend.ps1 -DataDir "C:\evaluator_data\evaluator_test"
#>
param(
    [string]$DataDir = "",
    [string]$ModelPath = "",
    [string]$HostAddr = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Activate virtual environment if present
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "Activating virtual environment: $VenvActivate" -ForegroundColor Cyan
    & $VenvActivate
}

if ($DataDir -ne "") {
    $env:PATHSNAP_INFERENCE_DATA_DIR = (Resolve-Path $DataDir).Path
} elseif (-not $env:PATHSNAP_INFERENCE_DATA_DIR) {
    $env:PATHSNAP_INFERENCE_DATA_DIR = Join-Path $ProjectRoot "data\raw\student_test"
}

if ($ModelPath -ne "") {
    $env:PATHSNAP_MODEL_PATH = (Resolve-Path $ModelPath).Path
} elseif (-not $env:PATHSNAP_MODEL_PATH) {
    $env:PATHSNAP_MODEL_PATH = Join-Path $ProjectRoot "artifacts\hist_gradient_boosting_tuned.joblib"
}

if (-not $env:PATHSNAP_HMM_CONFIG_PATH) {
    $env:PATHSNAP_HMM_CONFIG_PATH = Join-Path $ProjectRoot "configs\hmm_config.json"
}
if (-not $env:PATHSNAP_PIPELINE_CONFIG_PATH) {
    $env:PATHSNAP_PIPELINE_CONFIG_PATH = Join-Path $ProjectRoot "configs\pipeline_config.json"
}

Write-Host "=== PathSnap Backend Startup (Laptop 1) ===" -ForegroundColor Green
Write-Host "Inference Data Dir : $env:PATHSNAP_INFERENCE_DATA_DIR"
Write-Host "Model Artifact     : $env:PATHSNAP_MODEL_PATH"
Write-Host "Bind Address       : http://${HostAddr}:${Port}"
Write-Host "Swagger Docs       : http://127.0.0.1:${Port}/docs"

# Display detected non-loopback IPv4 addresses dynamically for Laptop 2 pairing
$LanIps = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -ExpandProperty IPAddress
if ($LanIps) {
    Write-Host "Detected LAN IPv4  : $($LanIps -join ', ')" -ForegroundColor Yellow
    Write-Host "On Laptop 2, set PATHSNAP_API_URL to http://<THIS_LAPTOP_LAN_IP>:${Port}" -ForegroundColor Yellow
}

python -m uvicorn backend.app.api.main:app --host $HostAddr --port $Port
