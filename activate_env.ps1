# Activation script for Neural Structural Optimization (PowerShell)
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Neural Structural Optimization Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if conda is available
try {
    $condaPath = Get-Command conda -ErrorAction Stop
    Write-Host "Found conda at: $($condaPath.Source)" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Conda is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda first" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if environment exists
$envExists = conda env list | Select-String "neural_structural_opt"
if (-not $envExists) {
    Write-Host "Environment 'neural_structural_opt' not found." -ForegroundColor Yellow
    Write-Host "Creating environment from environment.yml..." -ForegroundColor Yellow
    conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create conda environment" -ForegroundColor Red
        Write-Host "Please check environment.yml file" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate the conda environment
Write-Host "Activating conda environment..." -ForegroundColor Green
conda activate neural_structural_opt

# Set Python path for the project
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "Environment activated successfully!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "You can now run: python script/run.py" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate: conda deactivate" -ForegroundColor White
Write-Host ""
