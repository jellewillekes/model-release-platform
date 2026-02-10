from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Final

import joblib
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.common.config import get_experiment_name, get_model_name
from src.common.constants import (
    ART_DATASET_FINGERPRINT_JSON,
    ART_PREPROCESSOR,
    ART_TRAIN_SUMMARY_JSON,
    LABEL_COL,
    MLFLOW_ARTIFACT_PATH_MODEL,
    MLFLOW_ARTIFACT_PATH_REPORTS,
    STEP_TRAIN,
    TAG_CONFIG_HASH,
    TAG_DATASET_FINGERPRINT,
    TAG_MODEL_NAME,
    TAG_STEP,
    TAG_TRAINING_RUN_ID,
    TEST_CSV,
    TRAIN_CSV,
)
from src.common.mlflow_utils import ensure_experiment
from src.contracts.dataset_fingerprint import (
    compute_fingerprint,
    write_fingerprint_json,
)

logger = logging.getLogger(__name__)

DATA_DIR: Final[Path] = Path("/app/data")
ART_DIR: Final[Path] = Path("/app/artifacts")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    logging.basicConfig(level="INFO")

    experiment_name = get_experiment_name()
    ensure_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    model_name = get_model_name()

    train_df = pd.read_csv(DATA_DIR / TRAIN_CSV)
    test_df = pd.read_csv(DATA_DIR / TEST_CSV)

    X_train = train_df.drop(columns=[LABEL_COL])
    y_train = train_df[LABEL_COL].astype(int)

    X_test = test_df.drop(columns=[LABEL_COL])
    y_test = test_df[LABEL_COL].astype(int)

    preprocessor = joblib.load(ART_DIR / ART_PREPROCESSOR)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])

    # Used for traceability + reproducibility. In production, this becomes gs:// or bq://.
    data_source_uri = f"file://{DATA_DIR.as_posix()}"

    params = {
        "model_type": "logreg",
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    }

    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tag(TAG_STEP, STEP_TRAIN)
        mlflow.set_tag(TAG_MODEL_NAME, model_name)

        # Promotion guardrail: the candidate must always point back to the training run.
        mlflow.set_tag(TAG_TRAINING_RUN_ID, run.info.run_id)

        fp = compute_fingerprint(
            train_df=train_df,
            test_df=test_df,
            data_source_uri=data_source_uri,
            index_cols=None,
        )
        mlflow.set_tags(fp.as_tags())

        fp_path = ART_DIR / ART_DATASET_FINGERPRINT_JSON
        write_fingerprint_json(fp, fp_path)
        mlflow.log_artifact(str(fp_path), artifact_path=MLFLOW_ARTIFACT_PATH_REPORTS)

        mlflow.log_params(params)

        # Deterministic config hash is a promotion guardrail.
        config_hash = _sha256_text(
            json.dumps(params, sort_keys=True, separators=(",", ":"))
        )
        mlflow.set_tag(TAG_CONFIG_HASH, config_hash)

        # One stable dataset fingerprint hash (in addition to component tags).
        mlflow.set_tag(TAG_DATASET_FINGERPRINT, _sha256_text(fp.to_json()))

        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1": float(f1_score(y_test, pred)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
        }
        mlflow.log_metrics(metrics)

        input_example = X_test.head(5)
        signature = infer_signature(
            input_example, pipeline.predict_proba(input_example)[:, 1]
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=MLFLOW_ARTIFACT_PATH_MODEL,
            signature=signature,
            input_example=input_example,
            registered_model_name=None,
        )

        summary_path = ART_DIR / ART_TRAIN_SUMMARY_JSON
        summary_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        mlflow.log_artifact(
            str(summary_path), artifact_path=MLFLOW_ARTIFACT_PATH_REPORTS
        )

        logger.info("run_id=%s", run.info.run_id)
        logger.info("dataset_fingerprint=%s", fp_path)
        logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()
