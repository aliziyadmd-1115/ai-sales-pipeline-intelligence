# Deployment guide

The default TF-IDF API requires no external services. Build from the repository root:

```bash
docker build -t ai-sales-pipeline-intelligence .
docker run --rm -p 8000:8000 ai-sales-pipeline-intelligence
```

In another terminal, run `python scripts/smoke_api.py`. It checks readiness, health, prediction, successful evidence retrieval, and unrelated-query abstention. Docker was unavailable for local verification of the current update; the committed CI job performs build and running-container checks after push.

The image regenerates synthetic data and evaluation artifacts at build time, uses an unprivileged account, and monitors `/ready`. The build context excludes local environments, Git metadata, .env files, and cached models. Direct dependencies are pinned; the base image tag and transitive dependencies are not a complete immutable environment lock.

## Hosting

The included Render blueprint uses the Docker image and `/ready`. Confirm the chosen hosting account's current plan and runtime limits before deploying. No public endpoint is claimed.

Other container platforms can run the same image. For actual users, add authentication, HTTPS, resource/rate limits, request monitoring, model/data promotion controls, trusted artifact provenance, and time-based validation. The demo API has no authentication and is not ready to accept confidential data.

## Optional services and lifecycle

Chroma needs its optional packages, embedding-model download, memory, and persistent storage. The default Docker image does not install requirements-vector.txt. To use Chroma, create a separately tested image that installs those dependencies and persists artifacts/chroma_db/. Optional client-contract tests do not verify live embedding downloads or semantic quality.

Ollama must already run and have the configured model. Inside a container, localhost refers to that container. Set OLLAMA_BASE_URL to a service address reachable from the API; do not assume the host's localhost is reachable. Keep the service private.

Configuration is read from process environment variables; .env.example is a reference and is not automatically loaded. After regenerating model/data or changing retrieval settings, restart the API and Streamlit to invalidate process caches. Do not load untrusted joblib files. Chroma collection snapshots are fingerprinted; old snapshots must be explicitly cleaned up if disk usage grows.
