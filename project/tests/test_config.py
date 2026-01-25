from __future__ import annotations

from src.common.config import get_experiment_name, get_model_name


def test_get_experiment_name_default() -> None:
    assert isinstance(get_experiment_name(), str)
    assert get_experiment_name() != ""


def test_get_model_name_default() -> None:
    assert isinstance(get_model_name(), str)
    assert get_model_name() != ""
