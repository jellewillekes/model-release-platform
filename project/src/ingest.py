from __future__ import annotations

from pathlib import Path

import mlflow
from sklearn.datasets import load_breast_cancer

from src.common.config import get_experiment_name, get_model_name
from src.common.constants import RAW_CSV, STEP_INGEST, TAG_MODEL_NAME, TAG_STEP
from src.common.mlflow_utils import ensure_experiment

DATA_DIR = Path("/app/data")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    model_name = get_model_name()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="ingest") as run:
        mlflow.set_tag(TAG_STEP, STEP_INGEST)
        mlflow.set_tag(TAG_MODEL_NAME, model_name)

        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.copy()

        raw_path = DATA_DIR / RAW_CSV
        df.to_csv(raw_path, index=False)

        mlflow.log_params(
            {
                "dataset": "sklearn_breast_cancer",
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            }
        )
        mlflow.log_artifact(str(raw_path))

        print(f"[ingest] run_id={run.info.run_id} wrote={raw_path}")


if __name__ == "__main__":
    main()
