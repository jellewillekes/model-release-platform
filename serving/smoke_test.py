from __future__ import annotations

import os
import time
from typing import Any

import requests

SERVE_URL = os.getenv("SERVE_URL", "http://localhost:8000")


def _wait_for_service() -> None:
    """Wait until the service becomes healthy or raise."""
    last_status: int | None = None
    last_body: str | None = None

    for _ in range(30):
        try:
            r = requests.get(f"{SERVE_URL}/health", timeout=2)
            last_status = r.status_code
            last_body = r.text
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    raise RuntimeError(
        f"Service did not become healthy in time. last_status={last_status} last_body={last_body!r}"
    )


def _payload() -> dict[str, Any]:
    """Return a minimal valid prediction payload.

    IMPORTANT: feature names must match the training dataset schema.
    sklearn breast_cancer uses spaces in column names (e.g. 'mean radius').
    """
    return {
        "rows": [
            {
                "mean radius": 14.0,
                "mean texture": 20.0,
                "mean perimeter": 90.0,
                "mean area": 600.0,
                "mean smoothness": 0.10,
                "mean compactness": 0.13,
                "mean concavity": 0.10,
                "mean concave points": 0.05,
                "mean symmetry": 0.18,
                "mean fractal dimension": 0.06,
                "radius error": 0.30,
                "texture error": 1.10,
                "perimeter error": 2.50,
                "area error": 30.0,
                "smoothness error": 0.006,
                "compactness error": 0.020,
                "concavity error": 0.030,
                "concave points error": 0.010,
                "symmetry error": 0.020,
                "fractal dimension error": 0.003,
                "worst radius": 16.0,
                "worst texture": 26.0,
                "worst perimeter": 105.0,
                "worst area": 800.0,
                "worst smoothness": 0.14,
                "worst compactness": 0.30,
                "worst concavity": 0.35,
                "worst concave points": 0.12,
                "worst symmetry": 0.28,
                "worst fractal dimension": 0.08,
            }
        ]
    }


def _assert_prediction_response(body: dict[str, Any]) -> None:
    proba = body.get("proba")
    assert isinstance(proba, list) and len(proba) == 1, f"bad proba={proba!r}"

    p = proba[0]
    assert isinstance(p, (float, int)), f"bad proba[0]={p!r}"
    p_float = float(p)
    assert 0.0 <= p_float <= 1.0, f"proba out of range: {p_float}"


def _call(mode: str, *, required: bool) -> None:
    r = requests.post(f"{SERVE_URL}/predict?mode={mode}", json=_payload(), timeout=15)

    # In many deployments, only prod is guaranteed.
    if r.status_code == 503 and not required:
        print(f"[smoke] mode={mode} SKIP (503): {r.text[:300]!r}")
        return

    if r.status_code >= 400:
        # Make failures actionable in CI logs.
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(
            f"[smoke] mode={mode} failed: {r.status_code} detail={detail!r}"
        )

    body = r.json()
    assert isinstance(body, dict)
    _assert_prediction_response(body)
    print(f"[smoke] mode={mode} OK:", body)


def main() -> None:
    _wait_for_service()

    # prod must always work
    _call("prod", required=True)

    # optional depending on deployment policy
    for mode in ["candidate", "shadow", "canary"]:
        _call(mode, required=False)

    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
