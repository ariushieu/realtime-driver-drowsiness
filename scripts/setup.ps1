# Driver Drowsiness Detection - Setup Script for Windows PowerShell

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Driver Drowsiness Detection - Setup Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10 from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "   Found: $pythonVersion" -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "[2/5] Virtual environment already exists" -ForegroundColor Green
    $response = Read-Host "   Do you want to recreate it? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "   Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force .venv
        Write-Host "   Creating new virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
    }
}
else {
    Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "   Virtual environment created!" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}
Write-Host "   Virtual environment activated!" -ForegroundColor Green

# Upgrade pip
Write-Host "[4/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "   Pip upgraded!" -ForegroundColor Green

# Install dependencies
Write-Host "[5/5] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "   This may take several minutes..." -ForegroundColor Cyan

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install some dependencies" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Setup completed successfully!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the environment: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Run the application: python main.py" -ForegroundColor White
Write-Host "  3. Or open Jupyter notebook: jupyter notebook driver-drowsiness-detection.ipynb" -ForegroundColor White
Write-Host ""
Write-Host "To verify installation, run:" -ForegroundColor Cyan
Write-Host "  python verify_setup.py" -ForegroundColor White
Write-Host ""
