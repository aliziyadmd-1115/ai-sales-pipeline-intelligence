$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py
if ($LASTEXITCODE -ne 0) { throw "Streamlit failed. Run setup_windows.ps1 first." }
