from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetFingerprint:
    git_sha: str
    dataset_content_hash: str
    dataset_schema_hash: str
    row_count: int
    data_source_uri: str

    def as_tags(self) -> dict[str, str]:
        # MLflow tags are strings
        return {
            "git_sha": self.git_sha,
            "dataset_content_hash": self.dataset_content_hash,
            "dataset_schema_hash": self.dataset_schema_hash,
            "row_count": str(self.row_count),
            "data_source_uri": self.data_source_uri,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "git_sha": self.git_sha,
                "dataset_content_hash": self.dataset_content_hash,
                "dataset_schema_hash": self.dataset_schema_hash,
                "row_count": self.row_count,
                "data_source_uri": self.data_source_uri,
            },
            indent=2,
            sort_keys=True,
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get_git_sha() -> str:
    """
    Return current git SHA if available.
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
    """
    Hash only schema: column names + dtypes.
    Stable across row ordering / content changes.
    """
    schema = [(str(c), str(df[c].dtype)) for c in df.columns]
    payload = json.dumps(schema, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return _sha256_bytes(payload)


def content_hash(
    df: pd.DataFrame, *, index_cols: Optional[Sequence[str]] = None
) -> str:
    """
    Hash dataset content in a deterministic way.

    Approach:
      - sort columns alphabetically (stable)
      - optionally sort rows by `index_cols` (recommended for an ID / timestamp)
        otherwise keep current row order (good enough for CSVs that are deterministically written)
      - use pandas' stable row hashing, then SHA256 over the bytes

    Note: if the data source does not guarantee row order, pass index_cols.
    """
    df2 = df.copy()

    # Sort columns for stability
    df2 = df2.reindex(sorted(df2.columns), axis=1)

    if index_cols:
        missing = [c for c in index_cols if c not in df2.columns]
        if missing:
            raise ValueError(f"index_cols not in DataFrame: {missing}")
        df2 = df2.sort_values(list(index_cols), kind="mergesort").reset_index(drop=True)

    # Convert to stable per-row hashes; include index to protect against duplicate rows order when not sorting
    row_hashes = pd.util.hash_pandas_object(df2, index=True).to_numpy()
    return _sha256_bytes(row_hashes.tobytes())


def compute_fingerprint(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_source_uri: str,
    index_cols: Optional[Sequence[str]] = None,
) -> DatasetFingerprint:
    """
    Fingerprint is computed over the *training + test* membership.
    This matches what your model actually sees and prevents silent train/test drift.
    """
    # Keep schema/content hashes over the concatenated dataset
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


def read_fingerprint_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
