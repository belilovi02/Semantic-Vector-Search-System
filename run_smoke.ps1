#!/usr/bin/env pwsh
Set-StrictMode -Version Latest

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Virtual environment not found at .venv. Create it first: python -m venv .venv"
    exit 1
}

# Prepare a small dataset (1k docs)
& $py -c "from data.dataset import prepare_dataset; prepare_dataset(1000, allow_synthetic=True)"
if ($LASTEXITCODE -ne 0) { Write-Error "prepare_dataset failed"; exit $LASTEXITCODE }

# Run smoke experiments (both models)
& $py main.py --action run_all --models sentence_transformer bert
if ($LASTEXITCODE -ne 0) { Write-Error "run_all experiments failed"; exit $LASTEXITCODE }

Write-Output "Smoke run completed. Results saved under experiments/results/"
