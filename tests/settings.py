"""Test-only settings loaded from .env.test, separate from development config."""

from pydantic_settings import SettingsConfigDict

from app.config import Settings


class TestSettings(Settings):
    model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_env: str = "test"
    app_debug: bool = False
    api_key_hash_secret: str = "test-secret"
    internal_bearer_token: str = "test-bearer-token"
    openrouter_api_key: str = ""
    e2e_report_dir: str = "/tmp"