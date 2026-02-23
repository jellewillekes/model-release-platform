from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import get_experiment_name, get_model_name
from src.common.constants import (
    ART_PREPROCESSOR,
    LABEL_COL,
    RAW_CSV,
    STEP_FEATURIZE,
    TAG_MODEL_NAME,
    TAG_STEP,
    TEST_CSV,
    TRAIN_CSV,
)
from src.common.mlflow_utils import ensure_experiment

DATA_DIR = Path("/app/data")
ART_DIR = Path("/app/artifacts")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    model_name = get_model_name()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / RAW_CSV
    if not raw_path.exists():
        raise RuntimeError(f"Missing raw dataset: {raw_path}. Run ingest first.")

    df = pd.read_csv(raw_path)
    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Expected column {LABEL_COL!r} in {RAW_CSV}")

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df[LABEL_COL] = y_train.values
    test_df[LABEL_COL] = y_test.values

    train_path = DATA_DIR / TRAIN_CSV
    test_path = DATA_DIR / TEST_CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Keep it simple + compatible with your train.py which loads this artifact.
    preprocessor = Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    preprocessor_path = ART_DIR / ART_PREPROCESSOR
    joblib.dump(preprocessor, preprocessor_path)

    with mlflow.start_run(run_name="featurize") as run:
        mlflow.set_tag(TAG_STEP, STEP_FEATURIZE)
        mlflow.set_tag(TAG_MODEL_NAME, model_name)

        mlflow.log_params(
            {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": True,
                "preprocessor": "StandardScaler",
            }
        )

        mlflow.log_artifact(str(train_path))
        mlflow.log_artifact(str(test_path))
        mlflow.log_artifact(str(preprocessor_path))

        print(
            f"[featurize] run_id={run.info.run_id} wrote={train_path} {test_path} {preprocessor_path}"
        )


if __name__ == "__main__":
    main()
