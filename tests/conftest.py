"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer

from app.config import get_settings
from app.database import Base, get_db
from app.main import app
from app.services.collections import create_collection
from app.services.system_user import create_system_user


def _configure_podman_for_testcontainers() -> None:
    """Point testcontainers at rootless Podman when its socket is available.

    Leaves an explicit DOCKER_HOST alone. Disables Ryuk under rootless Podman
    (required); the PostgresContainer context manager still cleans up the session.
    """
    if os.environ.get("DOCKER_HOST"):
        return

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    podman_sock = Path(runtime_dir) / "podman" / "podman.sock"
    if not podman_sock.exists():
        return

    os.environ["DOCKER_HOST"] = f"unix://{podman_sock}"
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


_configure_podman_for_testcontainers()


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    return postgres_container.get_connection_url()


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
def api_key_secret() -> str:
    return "test-secret"


@pytest.fixture
def test_user(db_session, api_key_secret: str, monkeypatch):
    monkeypatch.setenv("API_KEY_HASH_SECRET", api_key_secret)
    get_settings.cache_clear()
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
