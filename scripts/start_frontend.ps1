<#
.SYNOPSIS
    Start the PathSnap Streamlit Evaluator UI (Laptop 2).
.EXAMPLE
    .\scripts\start_frontend.ps1
    .\scripts\start_frontend.ps1 -ApiUrl "http://<LAPTOP_1_LAN_IP>:8000"
#>
param(
    [string]$ApiUrl = "",
    [string]$HostAddr = "0.0.0.0",
    [int]$Port = 8501
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

if ($ApiUrl -ne "") {
    $env:PATHSNAP_API_URL = $ApiUrl.TrimEnd("/")
} elseif (-not $env:PATHSNAP_API_URL) {
    $env:PATHSNAP_API_URL = "http://127.0.0.1:8000"
}

Write-Host "=== PathSnap Frontend Startup (Laptop 2) ===" -ForegroundColor Green
Write-Host "Backend API URL    : $env:PATHSNAP_API_URL"
Write-Host "Streamlit UI       : http://127.0.0.1:${Port} (bound on ${HostAddr}:${Port})"

python -m streamlit run frontend/app.py --server.address $HostAddr --server.port $Port --server.headless true
