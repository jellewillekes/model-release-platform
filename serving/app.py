from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, cast, Literal

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from serving.constants import (
    ALIAS_CANDIDATE,
    ALIAS_PROD,
    DEFAULT_MODEL_NAME,
    ENV_CANARY_PCT,
    ENV_CANDIDATE_ALIAS,
    ENV_LOG_LEVEL,
    ENV_MODEL_CACHE_TTL_SEC,
    ENV_MODEL_NAME,
    ENV_PROD_ALIAS,
    ENV_UNIT_TESTING,
)
from serving.router import Mode, decide_routing, stable_bucket_from_rows

logger = logging.getLogger("serving")
logging.basicConfig(level=os.getenv(ENV_LOG_LEVEL, "INFO"))

MODEL_NAME = os.getenv(ENV_MODEL_NAME, DEFAULT_MODEL_NAME)
PROD_ALIAS = os.getenv(ENV_PROD_ALIAS, ALIAS_PROD)
CANDIDATE_ALIAS = os.getenv(ENV_CANDIDATE_ALIAS, ALIAS_CANDIDATE)
CANARY_PCT = int(os.getenv(ENV_CANARY_PCT, "10"))

# Cache (module-level so tests can monkeypatch easily).
model_prod: Any | None = None
model_candidate: Any | None = None
prod_version: str | None = None
candidate_version: str | None = None
_last_refresh_ts: float = 0.0
CACHE_TTL_SEC = float(os.getenv(ENV_MODEL_CACHE_TTL_SEC, "60"))

app = FastAPI()


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(
        ..., description="List of feature dicts (one per row)"
    )


class PredictResponse(BaseModel):
    mode: Mode
    n: int
    proba: list[float]
    chosen: Literal["prod", "candidate"]
    bucket: int | None = None
    canary_pct: int | None = None


def _models_uri(alias: str) -> str:
    return f"models:/{MODEL_NAME}@{alias}"


def _get_version(alias: str) -> str | None:
    try:
        if mlflow is None:
            return None
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        return str(mv.version)
    except Exception:
        return None


def _load_model(alias: str) -> Any:
    # Local unit tests may not have an MLflow server running.
    if os.getenv(ENV_UNIT_TESTING, "").lower() in {"1", "true", "yes"}:
        return None
    if mlflow is None:
        return None
    return mlflow.pyfunc.load_model(_models_uri(alias))


def _refresh_models_if_needed(force: bool = False) -> None:
    """Populate model_prod/model_candidate and versions.

    The globals are intentionally module-level so unit tests can monkeypatch them
    without depending on MLflow.
    """
    global \
        model_prod, \
        model_candidate, \
        prod_version, \
        candidate_version, \
        _last_refresh_ts

    now = time.time()
    if not force and (now - _last_refresh_ts) < CACHE_TTL_SEC:
        return

    # Only load what isn't already present (e.g., tests monkeypatch the models).
    if model_prod is None:
        model_prod = _load_model(PROD_ALIAS)
    if model_candidate is None:
        model_candidate = _load_model(CANDIDATE_ALIAS)

    prod_version = prod_version or _get_version(PROD_ALIAS)
    candidate_version = candidate_version or _get_version(CANDIDATE_ALIAS)
    _last_refresh_ts = now


def _get_model(alias: Literal["prod", "candidate"], required: bool) -> Any | None:
    _refresh_models_if_needed()
    model = model_prod if alias == ALIAS_PROD else model_candidate
    if required and model is None:
        raise RuntimeError(f"model for alias={alias} is not available")
    return model


@app.get("/health")
def health() -> dict[str, Any]:
    _refresh_models_if_needed()
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "prod_version": prod_version,
        "candidate_version": candidate_version,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    payload: PredictRequest,
    mode: Mode = Query(default="prod", description="prod|candidate|shadow|canary"),
) -> PredictResponse:
    bucket: int | None = None
    if mode == "canary":
        bucket = stable_bucket_from_rows(payload.rows)
        decision = decide_routing(mode=mode, canary_pct=CANARY_PCT, bucket=bucket)
    else:
        decision = decide_routing(mode=mode, canary_pct=CANARY_PCT, bucket=0)

    primary_alias: Literal["prod", "candidate"] = decision.chosen
    shadow_alias = cast(
        Literal["prod", "candidate"],
        ALIAS_CANDIDATE if primary_alias == ALIAS_PROD else ALIAS_PROD,
    )

    try:
        model_primary = _get_model(primary_alias, required=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    df = pd.DataFrame(payload.rows)
    y_primary = model_primary.predict(df)  # type: ignore[union-attr]
    y_primary_list = [float(x) for x in list(y_primary)]

    shadow_mae: float | None = None
    if decision.run_shadow:
        model_shadow = _get_model(shadow_alias, required=False)
        if model_shadow is not None:
            try:
                y_shadow = model_shadow.predict(df)
                y_shadow_list = [float(x) for x in list(y_shadow)]
                diffs = [abs(a - b) for a, b in zip(y_primary_list, y_shadow_list)]
                shadow_mae = sum(diffs) / max(len(diffs), 1)
            except Exception as e:
                logger.warning("shadow prediction failed: %s", e)

    log: dict[str, Any] = {
        "event": "predict",
        "client": request.client.host if request.client else None,
        "mode": mode,
        "chosen": primary_alias,
        "bucket": bucket,
        "canary_pct": CANARY_PCT if mode == "canary" else None,
        "shadow_mae": shadow_mae,
        "prod_version": prod_version,
        "candidate_version": candidate_version,
    }
    logger.info(json.dumps(log, separators=(",", ":")))

    return PredictResponse(
        mode=mode,
        n=len(payload.rows),
        proba=y_primary_list,
        chosen=primary_alias,
        bucket=bucket,
        canary_pct=CANARY_PCT if mode == "canary" else None,
    )
