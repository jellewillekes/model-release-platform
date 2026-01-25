#!/usr/bin/env sh
set -eu

: "${MLFLOW_HOST:=0.0.0.0}"
: "${MLFLOW_PORT:=5000}"
: "${BACKEND_STORE_URI:?BACKEND_STORE_URI must be set}"
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT must be set}"

exec mlflow server       --host "${MLFLOW_HOST}"       --port "${MLFLOW_PORT}"       --backend-store-uri "${BACKEND_STORE_URI}"       --default-artifact-root "${ARTIFACT_ROOT}"       --serve-artifacts
