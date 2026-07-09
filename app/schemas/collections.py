"""Pydantic schemas for collection endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    slug: str | None = Field(default=None, min_length=1, max_length=255)


class CollectionCreateResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    created_at: datetime
