"""Environment configuration loaded from .env via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "development"
    app_debug: bool = True
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_crawler"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    api_key_hash_secret: str

    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 500
    chunk_overlap_percent: int = 10

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    return Settings()
