"""Pydantic schemas for document endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


def validate_collection_identifier(
    collection_id: UUID | None,
    collection_slug: str | None,
) -> None:
    has_id = collection_id is not None
    has_slug = collection_slug is not None
    if has_id == has_slug:
        raise ValueError("Exactly one of collection_id or collection_slug is required")


class DocumentUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Markdown file content (UTF-8)")
    filename: str = Field(min_length=1, max_length=255, description="Original .md filename")
    collection_id: UUID | None = Field(default=None, description="Target collection UUID")
    collection_slug: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Target collection slug",
    )

    @model_validator(mode="after")
    def require_one_collection_identifier(self) -> "DocumentUploadRequest":
        validate_collection_identifier(self.collection_id, self.collection_slug)
        return self


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


class DocumentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=512)
    content: str | None = Field(
        default=None, min_length=1, description="Markdown file content (UTF-8)"
    )

    @model_validator(mode="after")
    def require_at_least_one_field(self) -> "DocumentUpdateRequest":
        if self.title is None and self.content is None:
            raise ValueError("At least one of title or content is required")
        return self


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    title: str | None
    url: str
    status: str | None
    error_message: str | None
    created_at: datetime


class DocumentResponse(DocumentListItem):
    content: str | None
