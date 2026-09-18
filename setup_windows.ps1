$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.13 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
$ProjectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $ProjectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $ProjectPython -m src.prepare
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed." }
& $ProjectPython -m src.evaluate
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed." }
& $ProjectPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
Write-Host "Setup and tests complete."
Write-Host "Run API: .\.venv\Scripts\python.exe -m uvicorn src.api:app --reload"
Write-Host "Run UI:  .\run_demo.ps1"
