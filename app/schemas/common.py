"""Shared Pydantic schemas used across API endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    """Shared error envelope across all endpoints."""

    error: str
    message: str
    request_id: str | None = None
