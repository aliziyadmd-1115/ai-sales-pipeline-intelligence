"""Smoke-test a running API using only Python's standard library."""
import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(args.url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


for attempt in range(15):
    try:
        assert call("/ready")["status"] == "ready"
        break
    except (URLError, TimeoutError):
        if attempt == 14:
            raise
        time.sleep(2)

assert call("/health")["status"] == "ok"
prediction = call("/predict-win", {"notes": "Decision makers requested a final analytics proposal."})
assert 0 <= prediction["win_probability"] <= 1
assert prediction["predicted_outcome"] in {"won", "lost"}
evidence = call("/answer", {"query": "executive sponsor analytics proposal"})
assert evidence["citations"] and not evidence["llm_used"]
unrelated = call("/answer", {"query": "volcanology tectonic seismograph"})
assert unrelated["status"] == "insufficient_evidence"
print("PASS: readiness, health, prediction, retrieval, and evidence abstention")
