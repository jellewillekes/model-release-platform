from __future__ import annotations

# Centralized constants across pipeline steps.

# Contracts schema versions
DATASET_FINGERPRINT_SCHEMA_VERSION = "dataset_fingerprint/v1"
MODEL_REF_SCHEMA_VERSION = "model_ref/v1"
FEATURE_STATS_SCHEMA_VERSION = "feature_stats/v1"

# MLflow tags (keys)
TAG_STEP = "step"
TAG_MODEL_NAME = "model_name"

TAG_GIT_SHA = "git_sha"
TAG_DATASET_CONTENT_HASH = "dataset_content_hash"
TAG_DATASET_SCHEMA_HASH = "dataset_schema_hash"
TAG_ROW_COUNT = "row_count"
TAG_DATA_SOURCE_URI = "data_source_uri"

TAG_SOURCE_RUN_ID = "source_run_id"
TAG_GATE = "gate"
TAG_RELEASE_STATUS = "release_status"
TAG_PROMOTED_FROM_ALIAS = "promoted_from_alias"
TAG_PREVIOUS_PROD_VERSION = "previous_prod_version"

# MLflow tag values
GATE_PASSED = "passed"
GATE_FAILED = "failed"  # for later - tests
RELEASE_STATUS_PREVIOUS_PROD = "previous_prod"

# Steps
STEP_INGEST = "ingest"
STEP_FEATURIZE = "featurize"
STEP_TRAIN = "train"
STEP_EVALUATE = "evaluate"
STEP_REGISTER = "register"
STEP_PROMOTE = "promote"

# Artifacts (filenames)
ART_TRAIN_RUN_ID = "TRAIN_RUN_ID"
ART_GATE_OK = "gate_ok.txt"
ART_REGISTERED_VERSION = "REGISTERED_VERSION"
ART_DATASET_FINGERPRINT_JSON = "dataset_fingerprint.json"
ART_EVALUATION_JSON = "evaluation.json"
ART_ROC_CURVE_PNG = "roc_curve.png"
ART_TRAIN_SUMMARY_JSON = "train_summary.json"

# Artifact paths (within MLflow)
MLFLOW_ARTIFACT_PATH_REPORTS = "reports"
MLFLOW_ARTIFACT_PATH_MODEL = "model"

# Aliases
ALIAS_CANDIDATE = "candidate"
ALIAS_PROD = "prod"
ALIAS_CHAMPION = "champion"

# Target file names
LABEL_COL = "target"
RAW_CSV = "raw.csv"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
ART_PREPROCESSOR = "preprocessor.joblib"
