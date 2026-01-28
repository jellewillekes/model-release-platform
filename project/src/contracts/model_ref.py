from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.common.constants import MODEL_REF_SCHEMA_VERSION


@dataclass(frozen=True)
class ModelRef:
    """Reference to a model in the registry or a specific run artifact."""

    model_name: str
    alias: Optional[str] = None
    version: Optional[str] = None
    source_run_id: Optional[str] = None
    schema_version: str = MODEL_REF_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "alias": self.alias,
            "version": self.version,
            "source_run_id": self.source_run_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ModelRef":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != MODEL_REF_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ModelRef schema_version={schema_version!r} "
                f"(expected {MODEL_REF_SCHEMA_VERSION!r})"
            )
        return ModelRef(
            model_name=str(payload["model_name"]),
            alias=(
                None
                if payload.get("alias") in (None, "")
                else str(payload.get("alias"))
            ),
            version=(
                None
                if payload.get("version") in (None, "")
                else str(payload.get("version"))
            ),
            source_run_id=(
                None
                if payload.get("source_run_id") in (None, "")
                else str(payload.get("source_run_id"))
            ),
        )

    @staticmethod
    def from_json(payload: str) -> "ModelRef":
        return ModelRef.from_dict(json.loads(payload))
