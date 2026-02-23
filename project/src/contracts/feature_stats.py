from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.common.constants import FEATURE_STATS_SCHEMA_VERSION


@dataclass(frozen=True)
class FeatureStats:
    """Skeleton contract for feature distribution stats."""

    stats: dict[str, dict[str, float]]
    schema_version: str = FEATURE_STATS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "stats": self.stats}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> FeatureStats:
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != FEATURE_STATS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported FeatureStats schema_version={schema_version!r} "
                f"(expected {FEATURE_STATS_SCHEMA_VERSION!r})"
            )
        raw = payload.get("stats")
        if not isinstance(raw, dict):
            raise TypeError("FeatureStats.stats must be a dict")
        return FeatureStats(stats={str(k): dict(v) for k, v in raw.items()})  # type: ignore[arg-type]

    @staticmethod
    def from_json(payload: str) -> FeatureStats:
        return FeatureStats.from_dict(json.loads(payload))
