"""SQLAlchemy ORM models for system users, crawled documents, and related entities."""

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.utils import utc_now


class SystemUser(Base):
    __tablename__ = "system_users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    ratelimit: Mapped[int] = mapped_column(Integer, nullable=False, default=100)

    documents: Mapped[list["Document"]] = relationship(back_populates="system_user")
    document_chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="system_user")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("system_user_id", "url"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    system_user_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("system_users.id"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    system_user: Mapped["SystemUser"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.id"), nullable=False, index=True
    )
    system_user_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("system_users.id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    chunk_vector: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    system_user: Mapped["SystemUser"] = relationship(back_populates="document_chunks")
