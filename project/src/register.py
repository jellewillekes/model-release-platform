from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import mlflow
from mlflow.tracking import MlflowClient

from src.common.config import get_experiment_name, get_model_name
from src.common.constants import (
    ALIAS_CANDIDATE,
    ART_GATE_OK,
    ART_REGISTERED_VERSION,
    ART_TRAIN_RUN_ID,
    GATE_PASSED,
    MLFLOW_ARTIFACT_PATH_MODEL,
    TAG_CONFIG_HASH,
    TAG_DATASET_CONTENT_HASH,
    TAG_DATASET_FINGERPRINT,
    TAG_DATASET_SCHEMA_HASH,
    TAG_DATA_SOURCE_URI,
    TAG_GATE,
    TAG_GIT_SHA,
    TAG_RELEASE_STATUS,
    TAG_ROW_COUNT,
    TAG_SOURCE_RUN_ID,
    TAG_TRAINING_RUN_ID,
)
from src.common.mlflow_utils import ensure_experiment

logger = logging.getLogger(__name__)

ART_DIR: Final[Path] = Path("/app/artifacts")

FINGERPRINT_TAG_KEYS: Final[tuple[str, ...]] = (
    TAG_GIT_SHA,
    TAG_DATASET_CONTENT_HASH,
    TAG_DATASET_SCHEMA_HASH,
    TAG_DATASET_FINGERPRINT,
    TAG_ROW_COUNT,
    TAG_DATA_SOURCE_URI,
    TAG_CONFIG_HASH,
    TAG_TRAINING_RUN_ID,
)


def _read_required_artifact_text(path: Path, artifact_name: str) -> str:
    if not path.exists():
        raise RuntimeError(f"{artifact_name} artifact not found at {path}.")
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    logging.basicConfig(level="INFO")

    experiment_name = get_experiment_name()
    ensure_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    model_name = get_model_name()
    client = MlflowClient()

    train_run_id = _read_required_artifact_text(
        ART_DIR / ART_TRAIN_RUN_ID, ART_TRAIN_RUN_ID
    )
    gate_ok_raw = _read_required_artifact_text(ART_DIR / ART_GATE_OK, ART_GATE_OK)
    gate_ok = gate_ok_raw.lower() == "true"

    model_uri = f"runs:/{train_run_id}/{MLFLOW_ARTIFACT_PATH_MODEL}"
    logger.info("model_uri=%s model_name=%s gate_ok=%s", model_uri, model_name, gate_ok)

    if not gate_ok:
        logger.info("Gate failed. Not registering model.")
        return

    run = client.get_run(train_run_id)
    run_tags = run.data.tags or {}

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Minimum traceability for the model version.
    client.set_model_version_tag(
        model_name, mv.version, TAG_SOURCE_RUN_ID, train_run_id
    )
    client.set_model_version_tag(model_name, mv.version, TAG_GATE, GATE_PASSED)
    client.set_model_version_tag(
        model_name, mv.version, TAG_RELEASE_STATUS, ALIAS_CANDIDATE
    )

    # Copy fingerprint/config tags from the training run onto the model version.
    for key in FINGERPRINT_TAG_KEYS:
        value = str(run_tags.get(key, "")).strip()
        if value:
            client.set_model_version_tag(model_name, mv.version, key, value)

    client.set_registered_model_alias(
        name=model_name, alias=ALIAS_CANDIDATE, version=mv.version
    )

    (ART_DIR / ART_REGISTERED_VERSION).write_text(str(mv.version), encoding="utf-8")
    logger.info(
        "Registered %s v%s -> alias %r", model_name, mv.version, ALIAS_CANDIDATE
    )


if __name__ == "__main__":
    main()
