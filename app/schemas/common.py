"""Shared Pydantic schemas used across API endpoints."""

from pydantic import BaseModel, ConfigDict

from app.schemas.examples import EXAMPLES


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [EXAMPLES["health_ok"], EXAMPLES["health_error"]]},
    )

    status: str


class ErrorResponse(BaseModel):
    """Shared error envelope across all endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                EXAMPLES["unauthorized"],
                EXAMPLES["not_found"],
                EXAMPLES["conflict"],
                EXAMPLES["validation_error"],
            ]
        },
    )

    error: str
    message: str
    request_id: str | None = None
