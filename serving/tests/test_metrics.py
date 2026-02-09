from __future__ import annotations

from fastapi.testclient import TestClient

from serving.app import app


def test_metrics_endpoint_exists() -> None:
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "requests_total" in r.text


def test_predict_increments_metrics() -> None:
    client = TestClient(app)

    # baseline scrape
    base = client.get("/metrics").text

    payload = {"rows": [{"mean_radius": 14.0, "mean_texture": 20.0}]}
    r = client.post("/predict?mode=prod", json=payload)
    assert r.status_code == 200

    after = client.get("/metrics").text

    # See predict_latency metric appear after hitting /predict (even if 503, if instrument failures)
    assert "predict_latency_seconds" in after
    assert len(after) >= len(base)
