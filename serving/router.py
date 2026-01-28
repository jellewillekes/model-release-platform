from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from serving.constants import ALIAS_CANDIDATE, ALIAS_PROD

MODE_PROD: Final[str] = "prod"
MODE_CANDIDATE: Final[str] = "candidate"
MODE_CANARY: Final[str] = "canary"
MODE_SHADOW: Final[str] = "shadow"

Mode = Literal["prod", "candidate", "canary", "shadow"]
Alias = Literal["prod", "candidate"]


@dataclass(frozen=True)
class RoutingDecision:
    """Decision for which alias should be used for the response.

    - chosen: the model alias used for the returned prediction
    - run_shadow: whether we should ALSO run the other model for comparison/logging
    """

    chosen: Alias
    run_shadow: bool


def stable_bucket_from_rows(rows: list[dict[str, Any]]) -> int:
    """Returns a stable bucket in [0, 99] based on request content.

    This enables deterministic canary routing (no per-request flapping).
    """
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h, 16) % 100


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
