from __future__ import annotations

import os
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "breast_cancer_clf")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="MLflow Production Model Serving")

class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]

def _load_model():
    # Always serve Production
    uri = f"models:/{MODEL_NAME}/Production"
    return mlflow.pyfunc.load_model(uri)

_model = None

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "tracking_uri": MLFLOW_TRACKING_URI}

@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    global _model
    if _model is None:
        _model = _load_model()

    df = pd.DataFrame(req.rows)
    proba = _model.predict(df)
    proba_list = [float(x) for x in proba]

    # best-effort to include version info
    return {
        "model_name": MODEL_NAME,
        "n": len(proba_list),
        "proba": proba_list,
    }
