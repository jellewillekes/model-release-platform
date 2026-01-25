from __future__ import annotations

import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get("MODEL_NAME", "breast_cancer_clf")

# Load from alias 'prod' (not stages)
MODEL_URI = f"models:/{MODEL_NAME}@prod"

app = FastAPI()


class PredictRequest(BaseModel):
    # We accept named-feature rows so the DataFrame keeps the correct column names
    rows: list[dict[str, float]] = Field(..., min_length=1)


# Load model at startup
model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(req: PredictRequest):
    import pandas as pd

    # Build DataFrame with correct feature names
    X = pd.DataFrame(req.rows)

    # Run prediction
    proba = model.predict(X)

    # Make sure output is JSON-serializable
    try:
        # numpy array / pandas series case
        out0 = float(proba[0])
    except Exception:
        # already a Python scalar
        out0 = proba[0]

    return {
        "model_name": MODEL_NAME,
        "n": len(req.rows),
        "proba": [out0],
    }
