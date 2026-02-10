from __future__ import annotations

import logging
import os

from mlflow.tracking import MlflowClient

from src.common.constants import ALIAS_PROD, TAG_PREVIOUS_PROD_VERSION

logger = logging.getLogger(__name__)


def rollback_prod(client: MlflowClient, model_name: str) -> None:
    """Rolls back prod to the previously recorded prod version.

    Contract (strict):
      - prod alias must exist
      - current prod version must have TAG_PREVIOUS_PROD_VERSION
      - rollback flips prod alias to the previous version
      - after rollback, we set TAG_PREVIOUS_PROD_VERSION on the restored version so the
        user can "undo" the rollback once (swap back).

    Raises:
      RuntimeError: if rollback metadata is missing.
    """
    current_prod = client.get_model_version_by_alias(model_name, ALIAS_PROD)
    tags = current_prod.tags or {}
    prev = str(tags.get(TAG_PREVIOUS_PROD_VERSION, "")).strip()

    if not prev:
        raise RuntimeError(
            "Rollback blocked: current prod does not have a previous prod recorded. "
            f"Expected tag '{TAG_PREVIOUS_PROD_VERSION}' on current prod model version."
        )

    client.set_registered_model_alias(model_name, ALIAS_PROD, prev)

    # Allow a one-step "undo": after rollback to prev, set prev's pointer to the version we came from.
    client.set_model_version_tag(
        name=model_name,
        version=prev,
        key=TAG_PREVIOUS_PROD_VERSION,
        value=str(current_prod.version),
    )

    logger.info(
        "Rolled back %s prod -> v%s (from v%s)", model_name, prev, current_prod.version
    )


def get_model_name() -> str:
    return os.environ.get("MODEL_NAME", "breast_cancer_clf")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    rollback_prod(MlflowClient(), get_model_name())


if __name__ == "__main__":
    main()
