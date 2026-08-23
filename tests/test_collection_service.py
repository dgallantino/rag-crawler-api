from uuid import uuid4

import pytest

from app.models import Collection, Document, DocumentChunk
from app.services.collections import (
    CollectionConflictError,
    CollectionNotFoundError,
    create_collection,
    delete_collection,
    get_collection,
    list_collections,
    update_collection,
)
from app.services.system_user import create_system_user


def test_list_collections_empty(db_session) -> None:
    user, _ = create_system_user(db_session, name="Empty Tenant")
    assert list_collections(db_session, user) == []


def test_list_collections_populated(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    other = create_collection(db_session, user, name="Second", slug="second")

    result = list_collections(db_session, user)
    slugs = {c.slug for c in result}
    assert slugs == {test_collection.slug, other.slug}


def test_list_collections_scoped_to_user(db_session, test_user, test_collection) -> None:
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    create_collection(db_session, other_user, name="Other Coll", slug="other")

    user, _ = test_user
    result = list_collections(db_session, user)
    assert [c.id for c in result] == [test_collection.id]


def test_get_collection_not_found(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(CollectionNotFoundError):
        get_collection(db_session, user, uuid4())


def test_get_collection_rejects_other_users_id(
    db_session, test_user, test_collection
) -> None:
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    with pytest.raises(CollectionNotFoundError):
        get_collection(db_session, other_user, test_collection.id)


def test_update_collection_name(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    original_slug = test_collection.slug

    updated = update_collection(
        db_session, user, test_collection.id, name="Renamed Collection"
    )

    assert updated.name == "Renamed Collection"
    assert updated.slug == original_slug


def test_update_collection_slug(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    original_name = test_collection.name

    updated = update_collection(db_session, user, test_collection.id, slug="new-slug")

    assert updated.slug == "new-slug"
    assert updated.name == original_name


def test_update_collection_name_and_slug(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    updated = update_collection(
        db_session, user, test_collection.id, name="Both", slug="both"
    )
    assert updated.name == "Both"
    assert updated.slug == "both"


def test_update_collection_slug_conflict(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    create_collection(db_session, user, name="Taken", slug="taken")

    with pytest.raises(CollectionConflictError, match="taken"):
        update_collection(db_session, user, test_collection.id, slug="taken")


def test_update_collection_not_found(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(CollectionNotFoundError):
        update_collection(db_session, user, uuid4(), name="Nope")


def test_delete_collection(db_session, test_user, test_collection) -> None:
    user, _ = test_user
    collection_id = test_collection.id

    delete_collection(db_session, user, collection_id)

    assert db_session.get(Collection, collection_id) is None
    with pytest.raises(CollectionNotFoundError):
        get_collection(db_session, user, collection_id)


def test_delete_collection_removes_documents_and_chunks(
    db_session, test_user, test_collection
) -> None:
    from tests.rag_helpers import _make_chunk

    user, _ = test_user
    chunk = _make_chunk(db_session, test_collection, content="to-delete")
    document_id = chunk.document_id
    chunk_id = chunk.id

    delete_collection(db_session, user, test_collection.id)

    assert db_session.get(Collection, test_collection.id) is None
    assert db_session.get(Document, document_id) is None
    assert db_session.get(DocumentChunk, chunk_id) is None


def test_delete_collection_not_found(db_session, test_user) -> None:
    user, _ = test_user
    with pytest.raises(CollectionNotFoundError):
        delete_collection(db_session, user, uuid4())


def test_delete_collection_rejects_other_users_id(
    db_session, test_user, test_collection
) -> None:
    other_user, _ = create_system_user(db_session, name="Other Tenant")
    with pytest.raises(CollectionNotFoundError):
        delete_collection(db_session, other_user, test_collection.id)
    assert db_session.get(Collection, test_collection.id) is not None
