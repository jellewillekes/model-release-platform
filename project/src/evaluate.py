from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, roc_auc_score

from src.common.config import get_experiment_name
from src.common.mlflow_utils import ensure_experiment, write_json

DATA_DIR = Path("/app/data")
ART_DIR = Path("/app/artifacts")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].astype(int)

    # Load model from current run's artifacts via MLflow (best practice is to load from run URI,
    # but for this simple pipeline, we'll reload the same pipeline object by using mlflow artifact path.)
    # We'll just re-train-evaluate linkage via parent run in orchestrator.

    # Orchestrator will pass TRAIN_RUN_ID
    train_run_id = Path("/app/artifacts/TRAIN_RUN_ID").read_text().strip()

    model_uri = f"runs:/{train_run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    proba = model.predict(X_test)
    pred = (np.array(proba) >= 0.5).astype(int)

    metrics = {
        "eval_accuracy": float(accuracy_score(y_test, pred)),
        "eval_f1": float(f1_score(y_test, pred)),
        "eval_roc_auc": float(roc_auc_score(y_test, proba)),
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ART_DIR / "evaluation.json"
    write_json(report_path, metrics)

    # ROC curve
    fig_path = ART_DIR / "roc_curve.png"
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.set_tag("step", "evaluate")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(fig_path), artifact_path="reports")
        print(
            f"[evaluate] run_id={run.info.run_id} train_run_id={train_run_id} metrics={metrics}"
        )

        # Gate decision (simple): roc_auc >= 0.95
        gate_ok = metrics["eval_roc_auc"] >= 0.95
        gate_path = ART_DIR / "gate_ok.txt"
        gate_path.write_text("true" if gate_ok else "false")
        mlflow.log_artifact(str(gate_path), artifact_path="reports")
        print(f"[evaluate] gate_ok={gate_ok}")


if __name__ == "__main__":
    main()
