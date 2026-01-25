from __future__ import annotations

import os
from mlflow.tracking import MlflowClient
import mlflow

from src.common.config import get_model_name, get_tracking_uri

def main() -> None:
    mlflow.set_tracking_uri(get_tracking_uri())
    model_name = get_model_name()
    client = MlflowClient()

    # Find latest in Staging
    versions = client.search_model_versions(f"name='{model_name}'")
    staging = [v for v in versions if (v.current_stage or "").lower() == "staging"]
    if not staging:
        raise RuntimeError(f"No Staging versions found for model: {model_name}")

    # Choose newest by version int
    newest = max(staging, key=lambda v: int(v.version))
    v = newest.version

    # Promote: archive existing Production
    client.transition_model_version_stage(
        name=model_name,
        version=v,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[promote] Promoted {model_name} v{v} -> Production (archived previous Production).")

if __name__ == "__main__":
    main()
