from __future__ import annotations

import pytest

from serving.router import (
    BucketContext,
    SeedSource,
    choose_canary_bucket,
    decide_routing,
)


def test_prod_routes_to_prod_only() -> None:
    d = decide_routing(mode="prod", canary_pct=10, bucket=0)
    assert d.chosen == "prod"
    assert d.run_shadow is False


def test_candidate_routes_to_candidate_only() -> None:
    d = decide_routing(mode="candidate", canary_pct=10, bucket=0)
    assert d.chosen == "candidate"
    assert d.run_shadow is False


def test_shadow_returns_prod_and_runs_shadow() -> None:
    d = decide_routing(mode="shadow", canary_pct=10, bucket=0)
    assert d.chosen == "prod"
    assert d.run_shadow is True


@pytest.mark.parametrize(
    "canary_pct,bucket,expected",
    [
        (0, 0, "prod"),
        (0, 99, "prod"),
        (100, 0, "candidate"),
        (100, 99, "candidate"),
        (10, 0, "candidate"),
        (10, 9, "candidate"),
        (10, 10, "prod"),
        (10, 99, "prod"),
    ],
)
def test_canary_routing(canary_pct: int, bucket: int, expected: str) -> None:
    d = decide_routing(mode="canary", canary_pct=canary_pct, bucket=bucket)
    assert d.chosen == expected
    assert d.run_shadow is True


def test_bucket_bounds_are_enforced() -> None:
    with pytest.raises(ValueError):
        decide_routing(mode="canary", canary_pct=10, bucket=-1)
    with pytest.raises(ValueError):
        decide_routing(mode="canary", canary_pct=10, bucket=100)


def test_same_request_id_yields_same_bucket() -> None:
    ctx = BucketContext(
        request_id="abc-123",
        client_provided_request_id=True,
        rows=[{"x": 1, "y": 2}],
    )
    b1 = choose_canary_bucket(ctx)
    b2 = choose_canary_bucket(ctx)
    assert b1.bucket == b2.bucket
    assert b1.seed_source == SeedSource.REQUEST_ID


def test_payload_hash_fallback_is_stable() -> None:
    rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    ctx = BucketContext(
        request_id=None,
        client_provided_request_id=False,
        rows=rows,
    )
    b1 = choose_canary_bucket(ctx)
    b2 = choose_canary_bucket(ctx)
    assert b1.bucket == b2.bucket
    assert b1.seed_source == SeedSource.PAYLOAD_HASH
