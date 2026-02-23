from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from serving.constants import (
    ALIAS_CANDIDATE,
    ALIAS_PROD,
    DEFAULT_MODEL_NAME,
    ENV_CANARY_PCT,
    ENV_CANDIDATE_ALIAS,
    ENV_LOG_LEVEL,
    ENV_MODEL_CACHE_TTL_SEC,
    ENV_MODEL_NAME,
    ENV_PROD_ALIAS,
    ENV_UNIT_TESTING,
)


class Settings(BaseSettings):
    """Serving configuration.

    Small and operationally relevant.
    """

    model_config = SettingsConfigDict(extra="ignore")

    model_name: str = Field(default=DEFAULT_MODEL_NAME, validation_alias=ENV_MODEL_NAME)
    prod_alias: str = Field(default=ALIAS_PROD, validation_alias=ENV_PROD_ALIAS)
    candidate_alias: str = Field(
        default=ALIAS_CANDIDATE, validation_alias=ENV_CANDIDATE_ALIAS
    )

    canary_pct: int = Field(default=10, validation_alias=ENV_CANARY_PCT)
    model_cache_ttl_sec: float = Field(
        default=60.0, validation_alias=ENV_MODEL_CACHE_TTL_SEC
    )

    log_level: str = Field(default="INFO", validation_alias=ENV_LOG_LEVEL)

    # Used to disable real MLflow loads during unit tests.
    unit_testing: bool = Field(default=False, validation_alias=ENV_UNIT_TESTING)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Cached for stability + to avoid reparsing env vars on each request.
    return Settings()
