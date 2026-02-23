from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

import serving.app as appmod


def _client() -> TestClient:
    return TestClient(appmod.app)


def test_livez_is_always_200() -> None:
    c = _client()
    r = c.get("/livez")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_readyz_returns_503_when_registry_unreachable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        appmod,
        "_registry_resolves_prod_alias",
        lambda _settings: (False, "registry down"),
    )
    monkeypatch.setattr(appmod, "_prod_model_loadable", lambda _settings: (True, None))

    c = _client()
    r = c.get("/readyz")
    assert r.status_code == 503
    assert "registry down" in r.text


def test_readyz_returns_503_when_prod_model_not_loadable(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        appmod, "_registry_resolves_prod_alias", lambda _settings: (True, None)
    )
    monkeypatch.setattr(
        appmod,
        "_prod_model_loadable",
        lambda _settings: (False, "prod model not loadable"),
    )

    c = _client()
    r = c.get("/readyz")
    assert r.status_code == 503
    assert "prod model not loadable" in r.text


def test_readyz_returns_200_when_registry_ok_and_prod_model_loadable(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        appmod, "_registry_resolves_prod_alias", lambda _settings: (True, None)
    )
    monkeypatch.setattr(appmod, "_prod_model_loadable", lambda _settings: (True, None))

    c = _client()
    r = c.get("/readyz")
    assert r.status_code == 200
    assert r.text == "ready"


def test_health_exposes_readiness_flags(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        appmod, "_registry_resolves_prod_alias", lambda _settings: (True, None)
    )

    # Ensure globals are in a known state for this test
    appmod.model_prod = object()
    appmod.prod_version = "123"

    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()

    assert body["registry_ok"] is True
    assert body["prod_model_loaded"] is True
    assert body["ready"] is True
    assert body["prod_version"] == "123"
