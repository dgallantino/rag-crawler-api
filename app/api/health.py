"""Planned: Health check endpoint for liveness/readiness probes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/db", response_model=HealthResponse)
def db_health_check(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute("SELECT 1")
        return HealthResponse(status="ok")
    except Exception as e:
        return HealthResponse(status="error")   