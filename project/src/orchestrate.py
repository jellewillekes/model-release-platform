from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow

from src.common.config import get_experiment_name, get_tracking_uri
from src.common.constants import ART_TRAIN_RUN_ID, STEP_TRAIN, TAG_STEP
from src.common.mlflow_utils import ensure_experiment

ART_DIR = Path("/app/artifacts")


def _run_step(module: str) -> None:
    print(f"[orchestrate] Running step: {module}")
    subprocess.check_call(["python", "-m", f"src.{module}"])


def _latest_train_run_id(experiment_id: str) -> str:
    """Return the most recent train run id for an experiment.

    MLflow has changed the return type of `search_runs` across versions
    (DataFrame vs. list[Run]). Handle both to keep the orchestrator stable.
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{TAG_STEP} = '{STEP_TRAIN}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    # MLflow <=2.x returns a pandas DataFrame.
    if hasattr(runs, "empty") and hasattr(runs, "iloc"):
        if runs.empty:  # type: ignore[attr-defined]
            raise RuntimeError("No train run found after training step.")
        return str(runs.iloc[0]["run_id"])  # type: ignore[index]

    # MLflow 3.x can return list[Run].
    if isinstance(runs, list):
        if not runs:
            raise RuntimeError("No train run found after training step.")
        run0 = runs[0]
        run_id = getattr(getattr(run0, "info", None), "run_id", None)
        if not run_id:
            raise RuntimeError("Train run object missing run_id.")
        return str(run_id)

    raise TypeError(f"Unexpected mlflow.search_runs return type: {type(runs)}")


def main() -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    ART_DIR.mkdir(parents=True, exist_ok=True)

    _run_step("ingest")
    _run_step("featurize")
    _run_step("train")

    exp = mlflow.get_experiment_by_name(get_experiment_name())
    assert exp is not None

    train_run_id = _latest_train_run_id(exp.experiment_id)
    (ART_DIR / ART_TRAIN_RUN_ID).write_text(str(train_run_id), encoding="utf-8")
    print(f"[orchestrate] Captured {ART_TRAIN_RUN_ID}={train_run_id}")

    _run_step("evaluate")
    _run_step("register")

    print("[orchestrate] Pipeline complete.")


if __name__ == "__main__":
    main()
