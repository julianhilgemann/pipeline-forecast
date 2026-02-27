param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $RootDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function New-Venv {
    if ($env:PYTHON_BIN) {
        & $env:PYTHON_BIN -m venv $VenvDir
        return
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.11 -m venv $VenvDir
            return
        } catch {
            & py -3 -m venv $VenvDir
            return
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
        return
    }

    throw "No Python interpreter found. Install Python 3.11+ and retry."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in $VenvDir"
    New-Venv
}

if ($env:SKIP_PIP_INSTALL -ne "1") {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $RootDir "requirements.txt")
}

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    Write-Host "Virtual environment ready: $VenvDir"
    Write-Host "Activate it:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "Run a Python command inside the venv:"
    Write-Host "  .\scripts\bootstrap_venv.ps1 pipeline_forecast.py"
    Write-Host "  .\scripts\bootstrap_venv.ps1 -m unittest discover -s tests -v"
    exit 0
}

& $VenvPython @PythonArgs
