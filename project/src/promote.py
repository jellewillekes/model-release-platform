from __future__ import annotations

import logging
import os
from typing import Mapping

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.common.constants import (
    ALIAS_CANDIDATE,
    ALIAS_CHAMPION,
    ALIAS_PROD,
    TAG_CONFIG_HASH,
    TAG_DATASET_FINGERPRINT,
    TAG_GIT_SHA,
    TAG_PREVIOUS_PROD_VERSION,
    TAG_TRAINING_RUN_ID,
)

logger = logging.getLogger(__name__)

_REQUIRED_TAGS: tuple[str, ...] = (
    TAG_DATASET_FINGERPRINT,
    TAG_GIT_SHA,
    TAG_CONFIG_HASH,
    TAG_TRAINING_RUN_ID,
)


def _missing_required_tags(tags: Mapping[str, str] | None) -> list[str]:
    """Returns required tag keys that are missing or empty."""
    safe_tags = tags or {}
    return [k for k in _REQUIRED_TAGS if not str(safe_tags.get(k, "")).strip()]


def _try_get_prod_version(client: MlflowClient, model_name: str) -> str | None:
    """Returns current prod version as a string, or None if prod alias does not exist."""
    try:
        prod = client.get_model_version_by_alias(model_name, ALIAS_PROD)
    except (MlflowException, KeyError, ValueError):
        return None
    return str(prod.version)


def promote_candidate_to_prod(client: MlflowClient, model_name: str) -> None:
    """Promotes candidate -> prod (+ champion).

    Contract:
      - Candidate alias must exist.
      - Candidate version must have required metadata tags, otherwise promotion is blocked.
      - If prod exists, we record the previous prod version on the *new* prod version,
        enabling one-step rollback.

    Raises:
      RuntimeError: if the candidate is missing required metadata.
      MlflowException: for MLflow client errors (connectivity, permissions, etc.).
    """
    candidate = client.get_model_version_by_alias(model_name, ALIAS_CANDIDATE)

    missing = _missing_required_tags(candidate.tags)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Promotion blocked: candidate model version is missing required metadata: "
            f"{missing_str}"
        )

    prev_prod_version = _try_get_prod_version(client, model_name)

    client.set_registered_model_alias(model_name, ALIAS_PROD, candidate.version)
    client.set_registered_model_alias(model_name, ALIAS_CHAMPION, candidate.version)

    if prev_prod_version is not None:
        client.set_model_version_tag(
            name=model_name,
            version=candidate.version,
            key=TAG_PREVIOUS_PROD_VERSION,
            value=prev_prod_version,
        )

    logger.info(
        "Promoted %s v%s -> alias '%s' (and '%s')",
        model_name,
        candidate.version,
        ALIAS_PROD,
        ALIAS_CHAMPION,
    )
    if prev_prod_version is not None:
        logger.info(
            "Recorded %s=%s on v%s",
            TAG_PREVIOUS_PROD_VERSION,
            prev_prod_version,
            candidate.version,
        )
    else:
        logger.info(
            "No previous prod found (first promotion) -> rollback will be blocked until next promotion."
        )


def get_model_name() -> str:
    return os.environ.get("MODEL_NAME", "breast_cancer_clf")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    promote_candidate_to_prod(MlflowClient(), get_model_name())


if __name__ == "__main__":
    main()
