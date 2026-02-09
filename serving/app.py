from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable, AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal, cast

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]

from serving.constants import ALIAS_CANDIDATE, ALIAS_PROD, HEADER_REQUEST_ID
from serving.metrics import PREDICT_LATENCY_SECONDS, REQUESTS_TOTAL, SHADOW_DIFF_MAE
from serving.router import (
    BucketContext,
    Mode,
    SeedSource,
    choose_canary_bucket,
    decide_routing,
)
from serving.settings import Settings, get_settings

logger = logging.getLogger("serving")


# Model cache (module-level so tests can monkeypatch)
model_prod: Any | None = None
model_candidate: Any | None = None
prod_version: str | None = None
candidate_version: str | None = None
_last_refresh_ts: float = 0.0


# App lifecycle
def _configure_logging(settings: Settings) -> None:
    # Being called repeatedly, basicConfig is a no-op after first call.
    logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    _configure_logging(settings)
    logger.info("serving started")
    yield
    logger.info("serving stopped")


app = FastAPI(lifespan=lifespan)


# Middleware
@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Ensure every request has a request-id.

    If client supplies X-Request-Id, keep it for deterministic bucketing.
    Otherwise generate one.
    """
    incoming = request.headers.get(HEADER_REQUEST_ID)
    if incoming and incoming.strip():
        request.state.request_id = incoming.strip()
        request.state.client_provided_request_id = True
    else:
        request.state.request_id = uuid.uuid4().hex
        request.state.client_provided_request_id = False

    response = await call_next(request)
    response.headers[HEADER_REQUEST_ID] = request.state.request_id
    return response


@app.middleware("http")
async def coarse_metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Count all requests (bounded labels)."""

    endpoint = request.url.path

    # Avoid self-scrape recursion / noise.
    if endpoint == "/metrics":
        return await call_next(request)

    mode_label = request.query_params.get("mode", "") if endpoint == "/predict" else ""

    try:
        response = await call_next(request)
    except HTTPException as e:
        REQUESTS_TOTAL.labels(
            endpoint=endpoint, mode=mode_label, status=str(e.status_code)
        ).inc()
        raise
    except Exception:
        REQUESTS_TOTAL.labels(endpoint=endpoint, mode=mode_label, status="500").inc()
        raise

    REQUESTS_TOTAL.labels(
        endpoint=endpoint, mode=mode_label, status=str(response.status_code)
    ).inc()
    return response


# Schemas
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
    bucket_seed_source: str | None = None


# Registry / model helpers
class _UnitTestModel:
    """Minimal model stub for unit tests.

    Keeps /predict behavior stable without requiring MLflow.
    """

    def predict(self, df: pd.DataFrame) -> list[float]:
        # Return deterministic "probabilities" in [0,1].
        n = len(df)
        return [1.0] * n


def _models_uri(settings: Settings, alias: str) -> str:
    return f"models:/{settings.model_name}@{alias}"


def _registry_resolves_prod_alias(settings: Settings) -> tuple[bool, str | None]:
    """Return (ok, detail). Simple MLflow call."""
    if settings.unit_testing:
        return True, None
    if mlflow is None:
        return False, "mlflow not available in serving image"

    try:
        client = mlflow.tracking.MlflowClient()
        _ = client.get_model_version_by_alias(settings.model_name, settings.prod_alias)
        return True, None
    except Exception as e:
        return False, f"registry check failed: {e}"


def _get_version(settings: Settings, alias: str) -> str | None:
    if settings.unit_testing or mlflow is None:
        return None
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(settings.model_name, alias)
        return str(mv.version)
    except Exception:
        return None


def _load_model(settings: Settings, alias: str) -> Any:
    # In unit tests, always return a deterministic stub model.
    if settings.unit_testing:
        return _UnitTestModel()

    if mlflow is None:
        raise RuntimeError("mlflow not available in serving image")

    # Defensive: sometimes people accidentally install a stub/namespace package named "mlflow".
    pyfunc = getattr(mlflow, "pyfunc", None)
    if pyfunc is None:
        raise RuntimeError("mlflow.pyfunc is missing (mlflow install is broken)")

    return pyfunc.load_model(_models_uri(settings, alias))


def _refresh_models_if_needed(
    settings: Settings,
    *,
    force: bool = False,
    load_candidate: bool = False,
) -> None:
    """Populate model_prod/model_candidate and versions.

    Globals are intentional: tests can monkeypatch without MLflow.
    """
    global \
        model_prod, \
        model_candidate, \
        prod_version, \
        candidate_version, \
        _last_refresh_ts

    now = time.time()
    if not force and (now - _last_refresh_ts) < settings.model_cache_ttl_sec:
        return

    if model_prod is None:
        model_prod = _load_model(settings, settings.prod_alias)

    if load_candidate and model_candidate is None:
        model_candidate = _load_model(settings, settings.candidate_alias)

    prod_version = prod_version or _get_version(settings, settings.prod_alias)
    if load_candidate:
        candidate_version = candidate_version or _get_version(
            settings, settings.candidate_alias
        )

    _last_refresh_ts = now


def _get_model(
    settings: Settings, alias: Literal["prod", "candidate"], required: bool
) -> Any | None:
    _refresh_models_if_needed(settings, load_candidate=(alias == ALIAS_CANDIDATE))
    model = model_prod if alias == ALIAS_PROD else model_candidate
    if required and model is None:
        raise RuntimeError(f"model for alias={alias} is not available")
    return model


def _prod_model_loadable(settings: Settings) -> tuple[bool, str | None]:
    """Return (ok, detail). Ensures prod model can be used for traffic."""
    try:
        _ = _get_model(settings, ALIAS_PROD, required=True)
        return True, None
    except Exception as e:
        return False, f"prod model not loadable: {e}"


# Health / metrics
@app.get("/livez")
def livez() -> dict[str, str]:
    # No dependencies, no MLflow calls.
    return {"status": "alive"}


@app.get("/readyz")
def readyz() -> Response:
    settings = get_settings()
    _configure_logging(settings)

    reg_ok, reg_detail = _registry_resolves_prod_alias(settings)
    if not reg_ok:
        return Response(
            content=reg_detail or "not ready", status_code=503, media_type="text/plain"
        )

    model_ok, model_detail = _prod_model_loadable(settings)
    if not model_ok:
        return Response(
            content=model_detail or "not ready",
            status_code=503,
            media_type="text/plain",
        )

    return Response(content="ready", status_code=200, media_type="text/plain")


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    _configure_logging(settings)

    reg_ok, reg_detail = _registry_resolves_prod_alias(settings)
    _refresh_models_if_needed(settings, load_candidate=False)

    model_loaded = model_prod is not None
    ready = bool(reg_ok and model_loaded)

    return {
        "status": "ok",
        "ready": ready,
        "model_name": settings.model_name,
        "prod_alias": settings.prod_alias,
        "candidate_alias": settings.candidate_alias,
        "prod_version": prod_version,
        "candidate_version": candidate_version,
        "registry_ok": reg_ok,
        "registry_detail": reg_detail,
        "prod_model_loaded": model_loaded,
        "cache_ttl_sec": settings.model_cache_ttl_sec,
    }


@app.get("/metrics")
def metrics() -> Response:
    payload = generate_latest()
    return Response(payload, media_type=CONTENT_TYPE_LATEST)


# Prediction
@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    payload: PredictRequest,
    mode: Mode = Query(default="prod", description="prod|candidate|shadow|canary"),
) -> PredictResponse:
    settings = get_settings()
    _configure_logging(settings)

    t0 = time.perf_counter()
    status_code: int = 200
    chosen_label: Literal["prod", "candidate", "unknown"] = "unknown"

    bucket: int | None = None
    bucket_seed_source: SeedSource | None = None
    shadow_mae: float | None = None

    try:
        # Routing decision (deterministic bucket only in canary mode)
        if mode == "canary":
            bd = choose_canary_bucket(
                BucketContext(
                    request_id=getattr(request.state, "request_id", None),
                    client_provided_request_id=bool(
                        getattr(request.state, "client_provided_request_id", False)
                    ),
                    rows=payload.rows,
                )
            )
            bucket = bd.bucket
            bucket_seed_source = bd.seed_source
            decision = decide_routing(
                mode=mode,
                canary_pct=settings.canary_pct,
                bucket=bucket,
            )
        else:
            decision = decide_routing(
                mode=mode,
                canary_pct=settings.canary_pct,
                bucket=0,
            )

        primary_alias: Literal["prod", "candidate"] = decision.chosen
        chosen_label = primary_alias

        shadow_alias = cast(
            Literal["prod", "candidate"],
            ALIAS_CANDIDATE if primary_alias == ALIAS_PROD else ALIAS_PROD,
        )

        # Important: ensure candidate is loaded only when needed (candidate traffic or shadow run).
        _refresh_models_if_needed(
            settings,
            load_candidate=(primary_alias == ALIAS_CANDIDATE or decision.run_shadow),
        )

        # Primary model must exist for prod/candidate routes.
        model_primary = _get_model(settings, primary_alias, required=True)
        if model_primary is None:
            status_code = 503
            raise HTTPException(
                status_code=503, detail=f"model not available: {primary_alias}"
            )

        df = pd.DataFrame(payload.rows)
        y_primary = model_primary.predict(df)  # type: ignore[union-attr]
        y_primary_list = [float(x) for x in list(y_primary)]

        # Optional shadow prediction (best-effort, never fails request)
        if decision.run_shadow:
            model_shadow = _get_model(settings, shadow_alias, required=False)
            if model_shadow is not None:
                try:
                    y_shadow = model_shadow.predict(df)  # type: ignore[union-attr]
                    y_shadow_list = [float(x) for x in list(y_shadow)]
                    diffs = [abs(a - b) for a, b in zip(y_primary_list, y_shadow_list)]
                    shadow_mae = sum(diffs) / max(len(diffs), 1)
                except Exception as e:
                    logger.warning("shadow prediction failed: %s", e)

        latency_s = time.perf_counter() - t0

        if shadow_mae is not None and math.isfinite(shadow_mae):
            SHADOW_DIFF_MAE.labels(mode=str(mode)).observe(shadow_mae)

        log: dict[str, Any] = {
            "event": "predict",
            "request_id": getattr(request.state, "request_id", None),
            "mode": mode,
            "chosen": primary_alias,
            "status": status_code,
            "latency_ms": int(latency_s * 1000),
            "bucket": bucket,
            "bucket_seed_source": str(bucket_seed_source)
            if bucket_seed_source
            else None,
            "canary_pct": settings.canary_pct if mode == "canary" else None,
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
            canary_pct=settings.canary_pct if mode == "canary" else None,
            bucket_seed_source=str(bucket_seed_source) if bucket_seed_source else None,
        )

    except HTTPException as e:
        status_code = e.status_code
        raise

    except RuntimeError as e:
        status_code = 503
        raise HTTPException(status_code=503, detail=str(e)) from e

    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail="internal error") from e

    finally:
        latency_s = time.perf_counter() - t0
        PREDICT_LATENCY_SECONDS.labels(
            mode=str(mode),
            status=str(status_code),
            chosen=chosen_label,
        ).observe(latency_s)
