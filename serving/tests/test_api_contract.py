from __future__ import annotations

from typing import Any, Protocol, Sequence

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from serving.constants import HEADER_REQUEST_ID


class _Predictor(Protocol):
    """Protocol for a minimal pyfunc-like model interface."""

    def predict(self, df: pd.DataFrame) -> Sequence[float]: ...


class _FakeModel:
    """A tiny fake model with an MLflow pyfunc-like interface."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def predict(self, df: pd.DataFrame) -> list[float]:
        # Return one value per row.
        return [self._value for _ in range(len(df))]


@pytest.fixture()
def client(monkeypatch: MonkeyPatch) -> TestClient:
    # Import lazily so monkeypatch applies after module import too if needed.
    import serving.app as app_module

    # Patch loaded models (these must exist as module globals in serving.app).
    monkeypatch.setattr(app_module, "model_prod", _FakeModel(0.2), raising=True)
    monkeypatch.setattr(app_module, "model_candidate", _FakeModel(0.8), raising=True)

    # Patch versions (optional but keeps output stable).
    monkeypatch.setattr(app_module, "prod_version", "1", raising=False)
    monkeypatch.setattr(app_module, "candidate_version", "2", raising=False)

    return TestClient(app_module.app)


def _payload() -> dict[str, Any]:
    return {
        "rows": [
            {
                "mean radius": 14.0,
                "mean texture": 20.0,
                "mean perimeter": 90.0,
                "mean area": 600.0,
            }
        ]
    }


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "prod_version" in body
    assert "candidate_version" in body


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("prod", 0.2),
        ("candidate", 0.8),
        ("shadow", 0.2),  # returns prod
    ],
)
def test_predict_modes(client: TestClient, mode: str, expected: float) -> None:
    r = client.post(f"/predict?mode={mode}", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == mode
    assert body["n"] == 1
    assert body["proba"] == [expected]


def test_predict_invalid_payload_422(client: TestClient) -> None:
    r = client.post("/predict", json={"not_rows": []})
    assert r.status_code == 422


def test_request_id_header_is_echoed_if_provided(client: TestClient) -> None:
    rid = "test-rid-123"
    r = client.post(
        "/predict?mode=prod",
        json=_payload(),
        headers={HEADER_REQUEST_ID: rid},
    )
    assert r.status_code == 200
    assert r.headers.get(HEADER_REQUEST_ID) == rid


def test_request_id_header_is_generated_if_missing(client: TestClient) -> None:
    r = client.post("/predict?mode=prod", json=_payload())
    assert r.status_code == 200
    gen = r.headers.get(HEADER_REQUEST_ID)
    assert gen is not None
    assert len(gen) >= 16


def test_canary_bucket_is_stable_for_same_request_id(client: TestClient) -> None:
    rid = "sticky-rid-999"
    r1 = client.post(
        "/predict?mode=canary",
        json=_payload(),
        headers={HEADER_REQUEST_ID: rid},
    )
    r2 = client.post(
        "/predict?mode=canary",
        json=_payload(),
        headers={HEADER_REQUEST_ID: rid},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["bucket"] == r2.json()["bucket"]
    assert r1.json()["bucket_seed_source"] == "request_id"
    assert r2.json()["bucket_seed_source"] == "request_id"
