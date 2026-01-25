from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow

from src.common.config import get_experiment_name, get_tracking_uri
from src.common.mlflow_utils import ensure_experiment

ART_DIR = Path("/app/artifacts")


def _run_step(module: str) -> None:
    print(f"[orchestrate] Running step: {module}")
    subprocess.check_call(["python", "-m", f"src.{module}"])


def main() -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # Each step is its own MLflow run for clean lineage in the UI.
    # We capture the train run_id so evaluate/register can load the exact logged model.
    _run_step("ingest")
    _run_step("featurize")

    # Train step
    _run_step("train")

    # Find latest train run
    exp = mlflow.get_experiment_by_name(get_experiment_name())
    assert exp is not None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.step = 'train'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No train run found after training step.")

    train_run_id = runs.iloc[0]["run_id"]
    (ART_DIR / "TRAIN_RUN_ID").write_text(str(train_run_id))
    print(f"[orchestrate] Captured TRAIN_RUN_ID={train_run_id}")

    _run_step("evaluate")
    _run_step("register")

    print("[orchestrate] Pipeline complete.")


if __name__ == "__main__":
    main()
