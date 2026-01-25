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

    train_run_id_path = ART_DIR / "TRAIN_RUN_ID"
    gate_ok_path = ART_DIR / "gate_ok.txt"

    if not train_run_id_path.exists():
        raise RuntimeError("TRAIN_RUN_ID artifact not found.")
    if not gate_ok_path.exists():
        raise RuntimeError("gate_ok.txt artifact not found.")

    train_run_id = train_run_id_path.read_text().strip()
    gate_ok = gate_ok_path.read_text().strip().lower() == "true"

    model_uri = f"runs:/{train_run_id}/model"
    client = MlflowClient()

    print(f"[register] model_uri={model_uri} model_name={model_name} gate_ok={gate_ok}")

    if not gate_ok:
        print("[register] Gate failed. Not registering model.")
        return

    # Register model version
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Tag the model version with metadata (useful for auditing/debugging)
    client.set_model_version_tag(model_name, mv.version, "source_run_id", train_run_id)
    client.set_model_version_tag(model_name, mv.version, "gate", "passed")
    client.set_model_version_tag(model_name, mv.version, "release_status", "candidate")

    # Alias-based release: set candidate -> this version
    client.set_registered_model_alias(name=model_name, alias="candidate", version=mv.version)

    (ART_DIR / "REGISTERED_VERSION").write_text(str(mv.version))
    print(f"[register] Registered {model_name} v{mv.version} -> alias 'candidate'")


if __name__ == "__main__":
    main()
