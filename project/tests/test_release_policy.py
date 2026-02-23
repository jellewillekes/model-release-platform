from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from src.common.constants import (
    ALIAS_CANDIDATE,
    ALIAS_PROD,
    GATE_PASSED,
    TAG_CONFIG_HASH,
    TAG_DATASET_FINGERPRINT,
    TAG_GATE,
    TAG_GIT_SHA,
    TAG_RELEASE_STATUS,
    TAG_TRAINING_RUN_ID,
)
from src.policy.release_policy import evaluate_promotion_policy
from src.promote import main as promote_main


@dataclass
class _ModelVersion:
    version: str
    tags: dict[str, str] = field(default_factory=dict)


class MlflowClientStub:
    """Tiny stub for policy and promote dry-run tests."""

    def __init__(self) -> None:
        self._aliases: dict[tuple[str, str], str] = {}
        self._versions: dict[tuple[str, str], _ModelVersion] = {}

        # Mutation call tracking
        self.set_registered_model_alias_calls: list[tuple[Any, ...]] = []
        self.set_model_version_tag_calls: list[tuple[Any, ...]] = []

    def put_version(self, model_name: str, version: str, tags: dict[str, str]) -> None:
        self._versions[(model_name, version)] = _ModelVersion(
            version=version, tags=tags
        )

    def set_alias(self, model_name: str, alias: str, version: str) -> None:
        self._aliases[(model_name, alias)] = version

    # --- Read APIs used by policy ---
    def get_model_version_by_alias(self, model_name: str, alias: str) -> _ModelVersion:
        v = self._aliases[(model_name, alias)]
        return self._versions[(model_name, v)]

    def get_model_version(self, model_name: str, version: str) -> _ModelVersion:
        return self._versions[(model_name, version)]

    def get_run(self, run_id: str):
        # default: pretend it exists
        return {"run_id": run_id}

    # --- Write APIs used by apply_promotion (should not be called in dry-run) ---
    def set_registered_model_alias(self, *args, **kwargs) -> None:
        self.set_registered_model_alias_calls.append((args, kwargs))

    def set_model_version_tag(self, *args, **kwargs) -> None:
        self.set_model_version_tag_calls.append((args, kwargs))


def _valid_candidate_tags() -> dict[str, str]:
    return {
        TAG_DATASET_FINGERPRINT: "abc",
        TAG_GIT_SHA: "deadbeef",
        TAG_CONFIG_HASH: "cfg123",
        TAG_TRAINING_RUN_ID: "trainrun123",
        TAG_GATE: GATE_PASSED,
        TAG_RELEASE_STATUS: ALIAS_CANDIDATE,
    }


def test_policy_blocks_when_candidate_alias_missing():
    client = MlflowClientStub()
    decision = evaluate_promotion_policy(
        client, model_name="m", from_alias=ALIAS_CANDIDATE, to_alias=ALIAS_PROD
    )
    assert decision.allowed is False
    assert any(v.code == "MISSING_ALIAS" for v in decision.errors)


@pytest.mark.parametrize(
    "missing_key",
    [TAG_DATASET_FINGERPRINT, TAG_GIT_SHA, TAG_CONFIG_HASH, TAG_TRAINING_RUN_ID],
)
def test_policy_blocks_when_required_tag_missing(missing_key: str):
    client = MlflowClientStub()
    tags = _valid_candidate_tags()
    tags[missing_key] = ""  # missing/empty
    client.put_version("m", "1", tags)
    client.set_alias("m", ALIAS_CANDIDATE, "1")

    decision = evaluate_promotion_policy(client, model_name="m")
    assert decision.allowed is False
    assert any(v.code == "MISSING_REQUIRED_TAGS" for v in decision.errors)


def test_policy_blocks_when_gate_not_passed():
    client = MlflowClientStub()
    tags = _valid_candidate_tags()
    tags[TAG_GATE] = "failed"
    client.put_version("m", "1", tags)
    client.set_alias("m", ALIAS_CANDIDATE, "1")

    decision = evaluate_promotion_policy(client, model_name="m")
    assert decision.allowed is False
    assert any(v.code == "GATE_NOT_PASSED" for v in decision.errors)


def test_policy_blocks_when_release_status_not_candidate():
    client = MlflowClientStub()
    tags = _valid_candidate_tags()
    tags[TAG_RELEASE_STATUS] = "something_else"
    client.put_version("m", "1", tags)
    client.set_alias("m", ALIAS_CANDIDATE, "1")

    decision = evaluate_promotion_policy(client, model_name="m")
    assert decision.allowed is False
    assert any(v.code == "INVALID_RELEASE_STATUS" for v in decision.errors)


def test_policy_blocks_noop_promotion_when_candidate_is_current_prod():
    client = MlflowClientStub()
    tags = _valid_candidate_tags()
    client.put_version("m", "1", tags)
    client.set_alias("m", ALIAS_CANDIDATE, "1")
    client.set_alias("m", ALIAS_PROD, "1")

    decision = evaluate_promotion_policy(client, model_name="m")
    assert decision.allowed is False
    assert any(v.code == "NOOP_PROMOTION" for v in decision.errors)


def test_dry_run_mode_has_zero_mutations(monkeypatch):
    client = MlflowClientStub()
    tags = _valid_candidate_tags()
    client.put_version("m", "1", tags)
    client.set_alias("m", ALIAS_CANDIDATE, "1")

    # Patch MlflowClient used in src.promote to our stub instance
    import src.promote as promote_mod

    monkeypatch.setattr(promote_mod, "MlflowClient", lambda: client)

    # Run promote in dry-run mode: must exit 0 and must not mutate
    with pytest.raises(SystemExit) as e:
        promote_main(["--model-name", "m", "--dry-run", "--format", "json"])
    assert e.value.code == 0

    assert client.set_registered_model_alias_calls == []
    assert client.set_model_version_tag_calls == []
