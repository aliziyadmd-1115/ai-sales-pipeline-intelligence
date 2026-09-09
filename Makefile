.PHONY: setup data evaluate test api app

setup:
	python -m pip install -r requirements.txt

data:
	python -m src.generate_data --rows 3000
	python -c "from pathlib import Path; from src.pipeline import run_pipeline; print(run_pipeline(Path('data/opportunities_raw.csv'), Path('data/opportunities_clean.csv')))"

evaluate:
	python -m src.evaluate

test:
	pytest -q

api:
	uvicorn src.api:app --reload

app:
	streamlit run streamlit_app.py
