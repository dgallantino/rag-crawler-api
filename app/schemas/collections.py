"""Pydantic schemas for collection endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.examples import EXAMPLES


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLES["create_collection_with_slug"]]},
    )

    name: str = Field(min_length=1, max_length=255)
    slug: str | None = Field(default=None, min_length=1, max_length=255)


class CollectionUpdateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLES["update_collection_rename"]]},
    )

    name: str | None = Field(default=None, min_length=1, max_length=255)
    slug: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def require_at_least_one_field(self) -> "CollectionUpdateRequest":
        if self.name is None and self.slug is None:
            raise ValueError("At least one of name or slug is required")
        return self


class CollectionResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"examples": [EXAMPLES["collection"]]},
    )

    id: UUID
    name: str
    slug: str
    created_at: datetime
