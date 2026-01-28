from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, roc_auc_score

from src.common.config import get_experiment_name
from src.common.constants import (
    ART_EVALUATION_JSON,
    ART_GATE_OK,
    ART_ROC_CURVE_PNG,
    ART_TRAIN_RUN_ID,
    LABEL_COL,
    MLFLOW_ARTIFACT_PATH_MODEL,
    MLFLOW_ARTIFACT_PATH_REPORTS,
    STEP_EVALUATE,
    TAG_STEP,
    TEST_CSV,
)
from src.common.mlflow_utils import ensure_experiment, write_json

DATA_DIR = Path("/app/data")
ART_DIR = Path("/app/artifacts")


def main() -> None:
    ensure_experiment(get_experiment_name())
    mlflow.set_experiment(get_experiment_name())

    test_df = pd.read_csv(DATA_DIR / TEST_CSV)
    X_test = test_df.drop(columns=[LABEL_COL])
    y_test = test_df[LABEL_COL].astype(int)

    # Orchestrator passes TRAIN_RUN_ID
    train_run_id = (ART_DIR / ART_TRAIN_RUN_ID).read_text(encoding="utf-8").strip()

    model_uri = f"runs:/{train_run_id}/{MLFLOW_ARTIFACT_PATH_MODEL}"
    model = mlflow.pyfunc.load_model(model_uri)

    proba = model.predict(X_test)
    pred = (np.array(proba) >= 0.5).astype(int)

    metrics = {
        "eval_accuracy": float(accuracy_score(y_test, pred)),
        "eval_f1": float(f1_score(y_test, pred)),
        "eval_roc_auc": float(roc_auc_score(y_test, proba)),
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ART_DIR / ART_EVALUATION_JSON
    write_json(report_path, metrics)

    # ROC curve
    fig_path = ART_DIR / ART_ROC_CURVE_PNG
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.set_tag(TAG_STEP, STEP_EVALUATE)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(report_path), artifact_path=MLFLOW_ARTIFACT_PATH_REPORTS
        )
        mlflow.log_artifact(str(fig_path), artifact_path=MLFLOW_ARTIFACT_PATH_REPORTS)

        # Gate decision (simple): roc_auc >= 0.95
        gate_ok = metrics["eval_roc_auc"] >= 0.95
        gate_path = ART_DIR / ART_GATE_OK
        gate_path.write_text("true" if gate_ok else "false", encoding="utf-8")
        mlflow.log_artifact(str(gate_path), artifact_path=MLFLOW_ARTIFACT_PATH_REPORTS)

        print(
            f"[evaluate] run_id={run.info.run_id} train_run_id={train_run_id} metrics={metrics}"
        )
        print(f"[evaluate] gate_ok={gate_ok}")


if __name__ == "__main__":
    main()
