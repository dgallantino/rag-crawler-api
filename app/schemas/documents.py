"""Pydantic schemas for document endpoints."""

from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    accepted: bool = True


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    status: str
    step: str | None = None
    steps: dict[str, str] | None = None
    error_message: str | None = None


class DocumentValidationErrorResponse(BaseModel):
    accepted: bool = False
    reason: str = Field(description="Why the upload was rejected")
