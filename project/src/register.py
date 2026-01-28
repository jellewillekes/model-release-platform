from __future__ import annotations

from pathlib import Path

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
    TAG_DATASET_CONTENT_HASH,
    TAG_DATASET_SCHEMA_HASH,
    TAG_DATA_SOURCE_URI,
    TAG_GATE,
    TAG_GIT_SHA,
    TAG_RELEASE_STATUS,
    TAG_ROW_COUNT,
    TAG_SOURCE_RUN_ID,
)
from src.common.mlflow_utils import ensure_experiment

ART_DIR = Path("/app/artifacts")

FINGERPRINT_TAG_KEYS = [
    TAG_GIT_SHA,
    TAG_DATASET_CONTENT_HASH,
    TAG_DATASET_SCHEMA_HASH,
    TAG_ROW_COUNT,
    TAG_DATA_SOURCE_URI,
]


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    model_name = get_model_name()

    train_run_id_path = ART_DIR / ART_TRAIN_RUN_ID
    gate_ok_path = ART_DIR / ART_GATE_OK

    if not train_run_id_path.exists():
        raise RuntimeError(f"{ART_TRAIN_RUN_ID} artifact not found.")
    if not gate_ok_path.exists():
        raise RuntimeError(f"{ART_GATE_OK} artifact not found.")

    train_run_id = train_run_id_path.read_text(encoding="utf-8").strip()
    gate_ok = gate_ok_path.read_text(encoding="utf-8").strip().lower() == "true"

    model_uri = f"runs:/{train_run_id}/{MLFLOW_ARTIFACT_PATH_MODEL}"
    client = MlflowClient()

    print(f"[register] model_uri={model_uri} model_name={model_name} gate_ok={gate_ok}")

    if not gate_ok:
        print("[register] Gate failed. Not registering model.")
        return

    # Fetch run tags (to propagate fingerprint onto model version)
    run = client.get_run(train_run_id)
    run_tags = run.data.tags or {}

    # Register model version
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Tag the model version with metadata
    client.set_model_version_tag(
        model_name, mv.version, TAG_SOURCE_RUN_ID, train_run_id
    )
    client.set_model_version_tag(model_name, mv.version, TAG_GATE, GATE_PASSED)
    client.set_model_version_tag(
        model_name, mv.version, TAG_RELEASE_STATUS, ALIAS_CANDIDATE
    )

    # Propagate dataset fingerprint + git SHA tags onto the model version
    for k in FINGERPRINT_TAG_KEYS:
        v = run_tags.get(k)
        if v is not None and str(v).strip() != "":
            client.set_model_version_tag(model_name, mv.version, k, str(v))

    # Alias-based release: set candidate -> this version
    client.set_registered_model_alias(
        name=model_name, alias=ALIAS_CANDIDATE, version=mv.version
    )

    (ART_DIR / ART_REGISTERED_VERSION).write_text(str(mv.version), encoding="utf-8")
    print(
        f"[register] Registered {model_name} v{mv.version} -> alias {ALIAS_CANDIDATE!r}"
    )


if __name__ == "__main__":
    main()
