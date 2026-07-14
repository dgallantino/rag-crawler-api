"""FastAPI application entry point and router registration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.api.stubs import raise_not_implemented
from app.api.agent import router as agent_router
from app.api.collections import router as collections_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.config import get_settings
from app.database import Base, engine
from app.exceptions import register_exception_handlers
import app.models  # noqa: F401 — register ORM models with Base.metadata
from app.middleware.request_id import RequestIDMiddleware
from app.schemas.common import MessageResponse

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/v1"


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

app.add_middleware(RequestIDMiddleware)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(collections_router, prefix=API_V1_PREFIX)
app.include_router(documents_router, prefix=API_V1_PREFIX)
app.include_router(query_router, prefix=API_V1_PREFIX)
app.include_router(agent_router, prefix=API_V1_PREFIX)


@app.get("/", response_model=MessageResponse)
def root() -> MessageResponse:
    raise_not_implemented()
