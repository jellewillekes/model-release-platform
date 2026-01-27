from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.common.constants import (
    DATASET_FINGERPRINT_SCHEMA_VERSION,
    TAG_DATASET_CONTENT_HASH,
    TAG_DATASET_SCHEMA_HASH,
    TAG_DATA_SOURCE_URI,
    TAG_GIT_SHA,
    TAG_ROW_COUNT,
)


@dataclass(frozen=True)
class DatasetFingerprint:
    """Minimal dataset lineage contract for a model training run."""

    git_sha: str
    dataset_content_hash: str
    dataset_schema_hash: str
    row_count: int
    data_source_uri: str
    schema_version: str = DATASET_FINGERPRINT_SCHEMA_VERSION

    def as_tags(self) -> dict[str, str]:
        return {
            TAG_GIT_SHA: self.git_sha,
            TAG_DATASET_CONTENT_HASH: self.dataset_content_hash,
            TAG_DATASET_SCHEMA_HASH: self.dataset_schema_hash,
            TAG_ROW_COUNT: str(self.row_count),
            TAG_DATA_SOURCE_URI: self.data_source_uri,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "dataset_content_hash": self.dataset_content_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "row_count": self.row_count,
            "data_source_uri": self.data_source_uri,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "DatasetFingerprint":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != DATASET_FINGERPRINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported DatasetFingerprint schema_version={schema_version!r} "
                f"(expected {DATASET_FINGERPRINT_SCHEMA_VERSION!r})"
            )
        return DatasetFingerprint(
            git_sha=str(payload["git_sha"]),
            dataset_content_hash=str(payload["dataset_content_hash"]),
            dataset_schema_hash=str(payload["dataset_schema_hash"]),
            row_count=int(payload["row_count"]),
            data_source_uri=str(payload["data_source_uri"]),
        )

    @staticmethod
    def from_json(payload: str) -> "DatasetFingerprint":
        return DatasetFingerprint.from_dict(json.loads(payload))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get_git_sha() -> str:
    """Return current git SHA if available.

    Order of precedence:
      1) GIT_SHA env var (recommended in CI)
      2) `git rev-parse HEAD` if repo is present in container
      3) "unknown"
    """
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha.strip()

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def schema_hash(df: pd.DataFrame) -> str:
    """Hash only schema: column names + dtypes."""
    schema = [(str(c), str(df[c].dtype)) for c in df.columns]
    payload = json.dumps(schema, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return _sha256_bytes(payload)


def content_hash(
    df: pd.DataFrame, *, index_cols: Optional[Sequence[str]] = None
) -> str:
    """Hash dataset content in a deterministic way."""
    df2 = df.copy()
    df2 = df2.reindex(sorted(df2.columns), axis=1)

    if index_cols:
        missing = [c for c in index_cols if c not in df2.columns]
        if missing:
            raise ValueError(f"index_cols not in DataFrame: {missing}")
        df2 = df2.sort_values(list(index_cols), kind="mergesort").reset_index(drop=True)

    row_hashes = pd.util.hash_pandas_object(df2, index=True).to_numpy()
    return _sha256_bytes(row_hashes.tobytes())


def compute_fingerprint(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_source_uri: str,
    index_cols: Optional[Sequence[str]] = None,
) -> DatasetFingerprint:
    """Fingerprint over training+test membership."""
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    return DatasetFingerprint(
        git_sha=get_git_sha(),
        dataset_content_hash=content_hash(combined, index_cols=index_cols),
        dataset_schema_hash=schema_hash(combined),
        row_count=int(len(combined)),
        data_source_uri=data_source_uri,
    )


def write_fingerprint_json(fp: DatasetFingerprint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fp.to_json(), encoding="utf-8")


def read_fingerprint_json(path: Path) -> DatasetFingerprint:
    return DatasetFingerprint.from_json(path.read_text(encoding="utf-8"))
