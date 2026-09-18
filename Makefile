.PHONY: setup data evaluate test api app

setup:
	python -m pip install -r requirements.txt

data:
	python -m src.prepare

evaluate:
	python -m src.evaluate

test:
	python -m pytest -q

api:
	uvicorn src.api:app --reload

app:
	streamlit run streamlit_app.py
