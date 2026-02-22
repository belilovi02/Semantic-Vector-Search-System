# Start local H1 run, stop after 3 hours, then produce a summary
$env:LOCAL_ONLY = '1'

# Start the H1 runner in a new process
$python = Join-Path $PSScriptRoot "..\ .venv\Scripts\python.exe" -Resolve -ErrorAction SilentlyContinue
if (-not (Test-Path $python)) {
    # fallback to repository-relative path
    $python = Join-Path (Get-Location) ".\.venv\Scripts\python.exe"
}
Write-Output "Using python: $python"

$proc = Start-Process -FilePath $python -ArgumentList '-u','-m','experiments.run_h1_local' -NoNewWindow -PassThru -WorkingDirectory (Get-Location)
Write-Output "Started H1 run (PID=$($proc.Id)). Will stop after 10800 seconds (3h)."

# Sleep for 3 hours
Start-Sleep -Seconds 10800

# If process still running, stop it
try {
    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($p) {
        Write-Output "Process still running after 3h; stopping PID=$($proc.Id)."
        Stop-Process -Id $proc.Id -Force
    } else {
        Write-Output "Process finished before 3h."
    }
} catch {
    Write-Output "Process check/stop failed: $_"
}

# Run summary collector (writes JSON under experiments/results)
$rc = & "$python" -u -m experiments.collect_h1_local_3h_summary
Write-Output "Summary collector output:\n$rc"
