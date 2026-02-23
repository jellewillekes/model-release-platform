from __future__ import annotations

import re

from fastapi.testclient import TestClient

from serving.app import app


def _get_metric_value(text: str, metric_name: str, *, labels: dict[str, str]) -> float:
    # Prometheus exposition: metric{a="b",c="d"} 123
    label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
    pattern = (
        rf"^{re.escape(metric_name)}\{{{re.escape(label_str)}\}}\s+([0-9eE\.\+\-]+)$"
    )

    for line in text.splitlines():
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    raise AssertionError(f"Metric not found: {metric_name} labels={labels}")


def test_metrics_endpoint_exists() -> None:
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "requests_total" in r.text


def test_predict_increments_metrics() -> None:
    client = TestClient(app)

    base = client.get("/metrics").text

    # This is the counter your middleware increments for /predict
    base_count = _get_metric_value(
        base,
        "requests_total",
        labels={"endpoint": "/predict", "mode": "prod", "status": "200"},
    )

    payload = {"rows": [{"mean_radius": 14.0, "mean_texture": 20.0}]}
    r = client.post("/predict?mode=prod", json=payload)
    assert r.status_code == 200

    after = client.get("/metrics").text

    # Histogram exists after /predict
    assert "predict_latency_seconds" in after

    after_count = _get_metric_value(
        after,
        "requests_total",
        labels={"endpoint": "/predict", "mode": "prod", "status": "200"},
    )
    assert after_count == base_count + 1
