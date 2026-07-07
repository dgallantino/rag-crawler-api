"""Planned: FastAPI application entry point and router registration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import get_settings
from app.database import Base, engine
import app.models  # noqa: F401 — register ORM models with Base.metadata
from app.schemas.common import MessageResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev convenience: auto-create tables. Replace with Alembic migrations later.
    if get_settings().app_env == "development":
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(bind=engine)
        except OperationalError:
            logger.warning("Database unavailable; skipping table creation")
    yield


app = FastAPI(
    title="RAG Crawler API",
    description="Web crawling and RAG API scaffold",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(documents_router)

@app.get("/", response_model=MessageResponse)
def root() -> MessageResponse:
    return MessageResponse(message="RAG Crawler API")
