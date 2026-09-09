# Deployment Guide

The project is packaged as a stateless Docker API. The default TF-IDF backend requires no external services, so the same image can run locally or on a managed container platform.

## Local Docker verification

```bash
docker build -t ai-sales-pipeline-intelligence .
docker run --rm -p 8000:8000 ai-sales-pipeline-intelligence
curl http://localhost:8000/health
```

The image regenerates the synthetic dataset, cleaned data, evaluation metrics, and trained model during the build. The container starts the FastAPI service on port `8000`.

## One-click Render deployment

The repository includes `render.yaml`. In Render, create a new Blueprint and select this repository. Render builds the Docker image, starts the API, and monitors `/health`.

The free plan may sleep after inactivity. Keep `RETRIEVAL_BACKEND=tfidf` unless the instance has enough memory and persistent storage for ChromaDB and sentence-transformer embeddings.

## AWS or Azure path

The same image can be pushed to Amazon ECR and run on ECS/Fargate, or pushed to Azure Container Registry and run with Azure Container Apps. A production deployment should add:

- HTTPS and authenticated API access;
- managed secrets instead of checked-in environment values;
- centralized logs, request metrics, and alerting;
- a model registry and controlled promotion process;
- persistent vector storage for the ChromaDB backend;
- scheduled drift, calibration, latency, and retrieval-quality monitoring.

No live cloud deployment is claimed in this portfolio repository until a public endpoint is added to the README.
