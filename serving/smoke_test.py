from __future__ import annotations

import os
import time
import requests

SERVE_URL = os.getenv("SERVE_URL", "http://localhost:8000")


def main() -> None:
    # Wait a bit for uvicorn
    for _ in range(30):
        try:
            r = requests.get(f"{SERVE_URL}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("Service did not become healthy in time.")

    # Minimal valid payload: the breast cancer dataset has these columns; we use a tiny subset with plausible values.
    payload = {
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

    r = requests.post(f"{SERVE_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    body = r.json()
    assert "proba" in body and len(body["proba"]) == 1
    p = body["proba"][0]
    assert 0.0 <= p <= 1.0
    print("[smoke] OK:", body)


if __name__ == "__main__":
    main()
