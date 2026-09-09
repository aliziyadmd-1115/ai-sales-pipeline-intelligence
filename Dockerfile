FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.generate_data --rows 3000 && \
    python -c "from pathlib import Path; from src.pipeline import run_pipeline; print(run_pipeline(Path('data/opportunities_raw.csv'), Path('data/opportunities_clean.csv')))" && \
    python -m src.evaluate
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
