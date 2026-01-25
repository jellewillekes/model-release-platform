from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.common.config import get_experiment_name
from src.common.mlflow_utils import ensure_experiment

DATA_DIR = Path("/app/data")
ART_DIR = Path("/app/artifacts")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    # test_df = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [c for c in train_df.columns if c != "target"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
        ],
        remainder="drop",
    )

    # Fit on train only (classic)
    preprocessor.fit(train_df[feature_cols])

    ART_DIR.mkdir(parents=True, exist_ok=True)
    preproc_path = ART_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preproc_path)

    with mlflow.start_run(run_name="featurize") as run:
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_artifact(str(preproc_path), artifact_path="features")
        mlflow.set_tag("step", "featurize")
        print(f"[featurize] run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
