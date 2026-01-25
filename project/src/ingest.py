from __future__ import annotations

from pathlib import Path

import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.common.config import get_experiment_name
from src.common.mlflow_utils import ensure_experiment

DATA_DIR = Path("/app/data")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with mlflow.start_run(run_name="ingest") as run:
        mlflow.log_param("dataset", "sklearn_breast_cancer")
        mlflow.log_param("n_rows_train", len(train_df))
        mlflow.log_param("n_rows_test", len(test_df))
        mlflow.log_artifact(str(train_path), artifact_path="data")
        mlflow.log_artifact(str(test_path), artifact_path="data")
        mlflow.set_tag("step", "ingest")
        print(f"[ingest] run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
