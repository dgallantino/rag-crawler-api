"""Shared pytest fixtures."""

import os

os.environ.setdefault("API_KEY_HASH_SECRET", "test-secret")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")

import pytest
from fastapi.testclient import TestClient
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, event
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.services.system_user import create_system_user


@compiles(Vector, "sqlite")
def _compile_vector_sqlite(type_, compiler, **kw):
    return "BLOB"


@pytest.fixture
def api_key_secret() -> str:
    return "test-secret"


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine):
    session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session, api_key_secret: str):
    user, api_key = create_system_user(db_session, name="Test Tenant")
    return user, api_key


@pytest.fixture
def client(db_session, test_user):
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
