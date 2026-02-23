from __future__ import annotations

# Backwards-compatible import path.
# New canonical location: src.contracts.dataset_fingerprint
from src.contracts.dataset_fingerprint import (  # noqa: F401
    DatasetFingerprint,
    compute_fingerprint,
    content_hash,
    get_git_sha,
    read_fingerprint_json,
    schema_hash,
    write_fingerprint_json,
)
