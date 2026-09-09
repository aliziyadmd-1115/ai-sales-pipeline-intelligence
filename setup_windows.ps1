$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_data --rows 3000
python -c "from pathlib import Path; from src.pipeline import run_pipeline; print(run_pipeline(Path('data/opportunities_raw.csv'), Path('data/opportunities_clean.csv')))"
python -m src.evaluate
pytest -q
Write-Host "Setup complete."
Write-Host "Run API: uvicorn src.api:app --reload"
Write-Host "Run UI:  streamlit run streamlit_app.py"
