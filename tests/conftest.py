"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from functools import lru_cache

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from app.config import get_settings
from app.database import Base, get_db
from app.main import app
from app.services.collections import create_collection
from app.services.system_user import create_system_user
from tests.settings import TestSettings

os.environ.setdefault("API_KEY_HASH_SECRET", "test-secret")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    return postgres_container.get_connection_url()


@pytest.fixture(scope="session")
def test_settings(postgres_url: str) -> TestSettings:
    return TestSettings(database_url=postgres_url)


@pytest.fixture(scope="session", autouse=True)
def configure_test_settings(test_settings: TestSettings) -> Generator[TestSettings, None, None]:
    """Patch get_settings for the entire test session."""
    import app.config

    get_settings.cache_clear()
    original_get_settings = app.config.get_settings

    @lru_cache
    def _get_test_settings() -> TestSettings:
        return test_settings

    app.config.get_settings = _get_test_settings  # type: ignore[assignment]

    patched_modules = []
    for module in sys.modules.values():
        if module is not None and getattr(module, "get_settings", None) is original_get_settings:
            setattr(module, "get_settings", _get_test_settings)
            patched_modules.append(module)

    yield test_settings

    app.config.get_settings = original_get_settings
    for module in patched_modules:
        setattr(module, "get_settings", original_get_settings)
    _get_test_settings.cache_clear()
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def db_engine(postgres_url: str):
    engine = create_engine(postgres_url, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def api_key_secret(test_settings: TestSettings) -> str:
    return test_settings.api_key_hash_secret


@pytest.fixture
def test_user(db_session, api_key_secret: str):
    user, api_key = create_system_user(db_session, name="Test Tenant")
    return user, api_key


@pytest.fixture
def test_collection(db_session, test_user):
    user, _ = test_user
    return create_collection(db_session, user, name="Test Collection", slug="test-collection")


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_redis_job_status(monkeypatch):
    monkeypatch.setattr("app.services.job_status.get_job_status", lambda document_id: None)
    monkeypatch.setattr("app.services.job_status.set_job_step", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.services.job_status.delete_job_status", lambda document_id: None)


@pytest.fixture
def auth_headers(test_user):
    _, api_key = test_user
    return {"X-API-Key": api_key}
