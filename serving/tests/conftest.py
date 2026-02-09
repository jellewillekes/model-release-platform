from __future__ import annotations

import pytest

import serving.app as appmod
import serving.settings as settings_mod
from serving.constants import ENV_UNIT_TESTING


@pytest.fixture(autouse=True)
def _force_unit_testing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force settings to ensure unit_testing=True in every test.
    monkeypatch.setenv(ENV_UNIT_TESTING, "true")
    settings_mod.get_settings.cache_clear()

    appmod.model_prod = None
    appmod.model_candidate = None
    appmod.prod_version = None
    appmod.candidate_version = None
    appmod._last_refresh_ts = 0.0
