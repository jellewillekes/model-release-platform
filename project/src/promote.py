import os
from mlflow.tracking import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "breast_cancer_clf")


def get_alias_version(client: MlflowClient, name: str, alias: str):
    try:
        mv = client.get_model_version_by_alias(name, alias)
        return mv.version
    except Exception:
        return None


def main():
    client = MlflowClient()

    # Read candidate
    candidate_version = get_alias_version(client, MODEL_NAME, "candidate")
    if not candidate_version:
        raise RuntimeError(f"No alias 'candidate' found for model: {MODEL_NAME}")

    # Read current prod (if any)
    current_prod = get_alias_version(client, MODEL_NAME, "prod")

    # Move current prod to champion (for rollback)
    if current_prod:
        client.set_registered_model_alias(MODEL_NAME, "champion", current_prod)
        client.set_model_version_tag(MODEL_NAME, current_prod, "release_status", "champion")
        print(f"[promote] Set alias 'champion' -> v{current_prod}")

    # Promote candidate to prod
    client.set_registered_model_alias(MODEL_NAME, "prod", candidate_version)
    client.set_model_version_tag(MODEL_NAME, candidate_version, "release_status", "prod")
    client.set_model_version_tag(MODEL_NAME, candidate_version, "promoted_from_alias", "candidate")

    print(f"[promote] Promoted {MODEL_NAME} v{candidate_version} -> alias 'prod'")


if __name__ == "__main__":
    main()
