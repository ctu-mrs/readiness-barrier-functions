# Run every experiment of the paper in order and write the results to
# results\generated\ (logs in results\generated\logs\). Takes a few hours on two cores.
# Set $env:PAPER_TEX to a local main.tex to also check the numbers quoted in the manuscript.
$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $PSScriptRoot "results\generated\logs"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Create it and install requirements.txt first."
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Invoke-Experiment {
    param(
        [Parameter(Mandatory = $true)][string]$EntryPoint,
        [string[]]$ScriptArgs = @()
    )

    $logName = ($EntryPoint -replace '[\\/]', '_') -replace '\.py$', ''
    $logPath = Join-Path $logDirectory "$logName.log"
    Write-Host "== $EntryPoint $($ScriptArgs -join ' ')"
    & $python (Join-Path $PSScriptRoot $EntryPoint) @ScriptArgs *> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "$EntryPoint failed with exit code $LASTEXITCODE. See $logPath"
    }
}

$experiments = @(
    "exp_timing.py",
    "exp_sweep.py",
    "exp_fig5.py",
    "exp_certmap.py",
    "exp_kappa.py",
    "exp_delay.py",
    "exp_ff.py"
)

foreach ($experiment in $experiments) {
    Invoke-Experiment $experiment
}
Invoke-Experiment "exp_robust.py" @("50")
Invoke-Experiment "exp_mc.py" @("100")
Invoke-Experiment "hifi\run_hifi.py"
Invoke-Experiment "export_results_mat.py"
# plotting tables for the figures, compared with the ones in data\ (see make_paper_data.log)
Invoke-Experiment "make_paper_data.py" @("--compare", "data")

if ($env:PAPER_TEX) {
    Invoke-Experiment "verify_numbers.py" @($env:PAPER_TEX)
} else {
    Write-Host "Skipping the manuscript check; set PAPER_TEX to a local main.tex to run it."
}

Write-Host "== done"
