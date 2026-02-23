from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal, cast

from serving.constants import ALIAS_CANDIDATE, ALIAS_PROD

MODE_PROD: Final[str] = "prod"
MODE_CANDIDATE: Final[str] = "candidate"
MODE_CANARY: Final[str] = "canary"
MODE_SHADOW: Final[str] = "shadow"

Mode = Literal["prod", "candidate", "canary", "shadow"]
Alias = Literal["prod", "candidate"]


class SeedSource(StrEnum):
    """Source of entropy used for deterministic bucketing."""

    REQUEST_ID = "request_id"
    PAYLOAD_HASH = "payload_hash"
    RANDOM = "random"


@dataclass(frozen=True)
class RoutingDecision:
    """Decision for which alias should be used for the response.

    - chosen: the model alias used for the returned prediction
    - run_shadow: whether we should ALSO run the other model for comparison/logging
    """

    chosen: Alias
    run_shadow: bool


@dataclass(frozen=True)
class BucketContext:
    """Inputs for stable bucketing.

    If the client provides a request id, we use it as the primary seed. This is
    the only way to guarantee a stable bucket across retries.

    If the client does NOT provide a request id, we fall back to the request
    payload (stable for identical payloads), and finally to a random fallback.
    """

    request_id: str | None
    client_provided_request_id: bool
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class BucketDecision:
    bucket: int
    seed_source: SeedSource


def stable_bucket_from_bytes(payload: bytes) -> int:
    """Returns a stable bucket in [0, 99] for arbitrary bytes."""
    h = hashlib.sha256(payload).hexdigest()
    return int(h, 16) % 100


def stable_bucket_from_str(seed: str) -> int:
    """Returns a stable bucket in [0, 99] for a text seed."""
    return stable_bucket_from_bytes(seed.encode("utf-8"))


def stable_bucket_from_rows(rows: list[dict[str, Any]]) -> int:
    """Returns a stable bucket in [0, 99] based on request content."""
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return stable_bucket_from_bytes(payload)


def choose_canary_bucket(ctx: BucketContext) -> BucketDecision:
    """Choose a bucket using a deterministic seed priority.

    Priority:
      1) client-provided request id (stable across retries)
      2) payload hash (stable for identical payload)
      3) random fallback (should be rare; kept for defensive robustness)
    """
    if ctx.client_provided_request_id and ctx.request_id:
        return BucketDecision(
            bucket=stable_bucket_from_str(ctx.request_id),
            seed_source=SeedSource.REQUEST_ID,
        )

    try:
        return BucketDecision(
            bucket=stable_bucket_from_rows(ctx.rows),
            seed_source=SeedSource.PAYLOAD_HASH,
        )
    except Exception:
        # Defensive fallback: if payload isn't JSON-serializable for some reason.
        seed = secrets.token_hex(16)
        return BucketDecision(
            bucket=stable_bucket_from_str(seed),
            seed_source=SeedSource.RANDOM,
        )


def decide_routing(mode: Mode, canary_pct: int, bucket: int) -> RoutingDecision:
    """Computes routing decision.

    Rules:
      - prod: return prod only
      - candidate: return candidate only
      - shadow: return prod, but also run candidate (compare/log)
      - canary: if bucket < canary_pct -> return candidate and also run prod
               else -> return prod and also run candidate

    Note: we clamp canary_pct to [0, 100] for safety.
    """
    if not (0 <= bucket <= 99):
        raise ValueError(f"bucket must be in [0, 99], got {bucket}")

    canary_pct = max(0, min(100, int(canary_pct)))

    # Use serving.constants for alias strings (single source of truth)
    prod: Alias = cast(Alias, ALIAS_PROD)
    candidate: Alias = cast(Alias, ALIAS_CANDIDATE)

    if mode == MODE_PROD:
        return RoutingDecision(chosen=prod, run_shadow=False)

    if mode == MODE_CANDIDATE:
        return RoutingDecision(chosen=candidate, run_shadow=False)

    if mode == MODE_SHADOW:
        return RoutingDecision(chosen=prod, run_shadow=True)

    if mode == MODE_CANARY:
        if bucket < canary_pct:
            return RoutingDecision(chosen=candidate, run_shadow=True)
        return RoutingDecision(chosen=prod, run_shadow=True)

    raise ValueError(f"Unknown mode: {mode}")
