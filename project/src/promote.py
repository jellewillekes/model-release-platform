from __future__ import annotations


from mlflow.tracking import MlflowClient

from src.common.config import get_model_name
from src.common.constants import (
    ALIAS_CANDIDATE,
    ALIAS_CHAMPION,
    ALIAS_PROD,
    RELEASE_STATUS_PREVIOUS_PROD,
    TAG_PREVIOUS_PROD_VERSION,
    TAG_PROMOTED_FROM_ALIAS,
    TAG_RELEASE_STATUS,
)


def get_alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(name, alias)
        return str(mv.version)
    except Exception:
        return None


def main() -> None:
    client = MlflowClient()
    model_name = get_model_name()

    candidate_version = get_alias_version(client, model_name, ALIAS_CANDIDATE)
    if not candidate_version:
        raise RuntimeError(
            f"No alias {ALIAS_CANDIDATE!r} found for model: {model_name}"
        )

    current_prod = get_alias_version(client, model_name, ALIAS_PROD)

    # Promote candidate -> prod
    client.set_registered_model_alias(model_name, ALIAS_PROD, candidate_version)
    client.set_model_version_tag(
        model_name, candidate_version, TAG_RELEASE_STATUS, ALIAS_PROD
    )
    client.set_model_version_tag(
        model_name, candidate_version, TAG_PROMOTED_FROM_ALIAS, ALIAS_CANDIDATE
    )

    # Champion tracks current production (same as prod)
    client.set_registered_model_alias(model_name, ALIAS_CHAMPION, candidate_version)
    client.set_model_version_tag(
        model_name, candidate_version, TAG_RELEASE_STATUS, ALIAS_CHAMPION
    )

    # Rollback metadata – keep info about what prod used to be
    if current_prod:
        client.set_model_version_tag(
            model_name, candidate_version, TAG_PREVIOUS_PROD_VERSION, current_prod
        )
        client.set_model_version_tag(
            model_name, current_prod, TAG_RELEASE_STATUS, RELEASE_STATUS_PREVIOUS_PROD
        )
        print(f"[promote] previous prod was v{current_prod}")

    print(
        f"[promote] Promoted {model_name} v{candidate_version} -> alias {ALIAS_PROD!r} (and {ALIAS_CHAMPION!r})"
    )


if __name__ == "__main__":
    main()
