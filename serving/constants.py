from __future__ import annotations

# Serving-side constants (serving is built as a separate image, we don't import from project/*)

DEFAULT_MODEL_NAME = "breast_cancer_clf"

ALIAS_PROD = "prod"
ALIAS_CANDIDATE = "candidate"

ENV_MODEL_NAME = "MODEL_NAME"
ENV_PROD_ALIAS = "PROD_ALIAS"
ENV_CANDIDATE_ALIAS = "CANDIDATE_ALIAS"
ENV_CANARY_PCT = "CANARY_PCT"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_UNIT_TESTING = "UNIT_TESTING"
ENV_MODEL_CACHE_TTL_SEC = "MODEL_CACHE_TTL_SEC"

# HTTP headers
HEADER_REQUEST_ID = "X-Request-Id"
