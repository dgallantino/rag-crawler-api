"""Collection provisioning and lookup service."""

import re
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Collection, Document, DocumentChunk, SystemUser
from app.services import job_status


class CollectionNotFoundError(Exception):
    pass


class CollectionConflictError(Exception):
    pass


def _derive_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "collection"


def create_collection(
    db: Session,
    user: SystemUser,
    name: str,
    slug: str | None = None,
) -> Collection:
    """Create a collection for a system user.

    Raises:
        CollectionConflictError: If a collection with the same slug already exists for the user.
    """
    resolved_slug = slug if slug else _derive_slug(name)
    collection = Collection(
        system_user_id=user.id,
        name=name,
        slug=resolved_slug,
    )
    db.add(collection)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise CollectionConflictError(
            f"Collection with slug '{resolved_slug}' already exists"
        ) from exc
    db.refresh(collection)
    return collection


def list_collections(db: Session, user: SystemUser) -> list[Collection]:
    """Return all collections owned by ``user``. Empty list is OK."""
    return db.query(Collection).filter(Collection.system_user_id == user.id).all()


def get_collection(db: Session, user: SystemUser, collection_id: UUID) -> Collection:
    """Fetch a collection by ID, scoped to the given user.

    Raises:
        CollectionNotFoundError: If no matching collection is found.
    """
    collection = (
        db.query(Collection)
        .filter(Collection.id == collection_id, Collection.system_user_id == user.id)
        .one_or_none()
    )
    if collection is None:
        raise CollectionNotFoundError(str(collection_id))
    return collection


def update_collection(
    db: Session,
    user: SystemUser,
    collection_id: UUID,
    *,
    name: str | None = None,
    slug: str | None = None,
) -> Collection:
    """Update name and/or slug on a collection owned by ``user``.

    Omitted fields are left unchanged. Does not re-derive slug from name.

    Raises:
        CollectionNotFoundError: If no matching collection is found.
        CollectionConflictError: If the new slug already exists for the user.
    """
    collection = get_collection(db, user, collection_id)
    if name is not None:
        collection.name = name
    if slug is not None:
        collection.slug = slug
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise CollectionConflictError(
            f"Collection with slug '{slug}' already exists"
        ) from exc
    db.refresh(collection)
    return collection


def delete_collection(db: Session, user: SystemUser, collection_id: UUID) -> None:
    """Delete a collection owned by ``user``, including its documents and chunks.

    Raises:
        CollectionNotFoundError: If no matching collection is found.
    """
    collection = get_collection(db, user, collection_id)
    document_ids = [
        str(row[0])
        for row in db.query(Document.id)
        .filter(Document.collection_id == collection.id)
        .all()
    ]
    db.query(DocumentChunk).filter(
        DocumentChunk.collection_id == collection.id
    ).delete()
    db.query(Document).filter(Document.collection_id == collection.id).delete()
    db.delete(collection)
    db.commit()
    for document_id in document_ids:
        job_status.delete_job_status(document_id)


def get_collection_by_slug(
    db: Session, user: SystemUser, slug: str | None = None
) -> list[Collection]:
    """Fetch collections for a user, optionally filtered by slug.

    When ``slug`` is set, returns that single collection (scoped to ``user``).
    When ``slug`` is ``None``, returns all collections owned by ``user``.

    Raises:
        CollectionNotFoundError: If no matching collection is found.
    """
    if slug is not None:
        collection = (
            db.query(Collection)
            .filter(Collection.slug == slug, Collection.system_user_id == user.id)
            .one_or_none()
        )
        if collection is None:
            raise CollectionNotFoundError(f"slug='{slug}'")
        return [collection]

    collections = (
        db.query(Collection).filter(Collection.system_user_id == user.id).all()
    )
    if not collections:
        raise CollectionNotFoundError(f"no collections for user id={user.id}")
    return collections
