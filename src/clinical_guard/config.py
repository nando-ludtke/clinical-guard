"""Application configuration powered by pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the clinical-guard service."""

    model_config = SettingsConfigDict(
        env_prefix="CLINICAL_GUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "clinical-guard"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_redact_queries: bool = True

    e5_guard_url: str | None = None
    prompt_guard_url: str | None = None
    ner_guard_url: str | None = None
    worker_timeout_seconds: float = 10.0
    enable_local_ner: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
