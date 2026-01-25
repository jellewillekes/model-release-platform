from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.common.config import get_experiment_name, get_model_name
from src.common.mlflow_utils import ensure_experiment

DATA_DIR = Path("/app/data")
ART_DIR = Path("/app/artifacts")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())
    model_name = get_model_name()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"].astype(int)

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].astype(int)

    preprocessor = joblib.load(ART_DIR / "preprocessor.joblib")

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", clf),
        ]
    )

    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tag("step", "train")
        mlflow.set_tag("model_name", model_name)

        mlflow.log_params(
            {
                "model_type": "logreg",
                "max_iter": 2000,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            }
        )

        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1": float(f1_score(y_test, pred)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
        }
        mlflow.log_metrics(metrics)

        # signature & input example
        input_example = X_test.head(5)
        signature = infer_signature(
            input_example, pipeline.predict_proba(input_example)[:, 1]
        )

        # Log model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=None,  # registration happens in register step
        )

        # Also log a small JSON summary
        summary_path = Path("/app/artifacts/train_summary.json")
        summary_path.write_text(str(metrics))
        mlflow.log_artifact(str(summary_path), artifact_path="reports")

        print(f"[train] run_id={run.info.run_id}")
        print(f"[train] metrics={metrics}")


if __name__ == "__main__":
    main()
