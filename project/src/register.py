from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.common.config import get_experiment_name, get_model_name
from src.common.mlflow_utils import ensure_experiment

ART_DIR = Path("/app/artifacts")

def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    model_name = get_model_name()

    train_run_id = (ART_DIR / "TRAIN_RUN_ID").read_text().strip()
    gate_ok = (ART_DIR / "gate_ok.txt").read_text().strip().lower() == "true"

    model_uri = f"runs:/{train_run_id}/model"
    client = MlflowClient()

    print(f"[register] model_uri={model_uri} model_name={model_name} gate_ok={gate_ok}")

    if not gate_ok:
        print("[register] Gate failed. Not registering model.")
        return

    # Register model
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    # Wait for registration
    client.set_model_version_tag(model_name, mv.version, "source_run_id", train_run_id)
    client.set_model_version_tag(model_name, mv.version, "gate", "passed")

    # Move to Staging; promotion to Production is separate (make promote)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    (ART_DIR / "REGISTERED_VERSION").write_text(str(mv.version))
    print(f"[register] Registered {model_name} v{mv.version} -> Staging")

if __name__ == "__main__":
    main()
