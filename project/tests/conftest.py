from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pytest

# Allow running tests without requiring an editable install (useful for local dev).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def mlflow_sqlite(tmp_path: Path):
    """Provide a local MLflow tracking+registry backend for unit tests.

    We use sqlite as a backend store because the MLflow file store does not support
    the model registry.
    """
    db_path = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    os.environ["MLFLOW_TRACKING_URI"] = uri

    yield uri
