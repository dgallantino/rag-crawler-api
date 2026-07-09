"""Collection provisioning and lookup service."""

import re
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Collection, SystemUser


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


def get_collection_by_slug(db: Session, user: SystemUser, slug: str) -> Collection:
    """Fetch a collection by slug, scoped to the given user.

    Raises:
        CollectionNotFoundError: If no matching collection is found.
    """
    collection = (
        db.query(Collection)
        .filter(Collection.slug == slug, Collection.system_user_id == user.id)
        .one_or_none()
    )
    if collection is None:
        raise CollectionNotFoundError(f"slug='{slug}'")
    return collection
