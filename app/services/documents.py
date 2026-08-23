"""Document upload and status services."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Collection, Document, DocumentChunk, SystemUser
from app.schemas.documents import DocumentStatusResponse
from app.services import job_status
from app.services.collections import get_collection, get_collection_by_slug
from app.services.triggers import trigger_process_document


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate_markdown_upload(filename: str, content: bytes) -> ValidationResult:
    if not filename.lower().endswith(".md"):
        return ValidationResult(valid=False, reason="Only .md files are accepted")

    if b"\x00" in content:
        return ValidationResult(valid=False, reason="File contains binary content")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ValidationResult(valid=False, reason="File must be valid UTF-8 text")

    if not text.strip():
        return ValidationResult(valid=False, reason="File content is empty")

    return ValidationResult(valid=True)


class DocumentConflictError(Exception):
    pass


class DocumentNotFoundError(Exception):
    pass


class DocumentValidationError(Exception):
    pass


def create_document_upload(
    db: Session,
    collection: Collection,
    filename: str,
    content: str,
) -> Document:
    """
    Create a new document entry in the database based on an uploaded file.

    This function will:
    - Attempt to create a new Document with the given filename and content,
      associated with the provided collection.
    - If a document with the same filename for the collection already exists, it will raise
      DocumentConflictError.
    - On success, commits the new document, triggers the background processing job,
      and returns the created Document object.

    Args:
        db (Session): SQLAlchemy database session.
        collection (Collection): The collection the document belongs to.
        filename (str): Name of the uploaded file.
        content (str): Raw content of the file.

    Returns:
        Document: The SQLAlchemy Document instance just created.

    Raises:
        DocumentConflictError: If a document with the same filename already exists in the collection.
    """
    document = Document(
        collection_id=collection.id,
        title=filename,
        url=f"file://{filename}",
        content=content,
        status=None,
    )
    db.add(document)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise DocumentConflictError(f"Document with filename '{filename}' already exists") from exc

    db.refresh(document)
    trigger_process_document(str(document.id))
    return document


def get_document(db: Session, user: SystemUser, document_id: UUID) -> Document:
    """Fetch a document by ID, scoped to the given user.

    Raises:
        DocumentNotFoundError: If the document does not exist for the given user.
    """
    document = (
        db.query(Document)
        .join(Collection, Document.collection_id == Collection.id)
        .filter(Document.id == document_id, Collection.system_user_id == user.id)
        .one_or_none()
    )
    if document is None:
        raise DocumentNotFoundError(str(document_id))
    return document


def list_documents(
    db: Session,
    user: SystemUser,
    collection_id: UUID | None = None,
    collection_slug: str | None = None,
) -> list[Document]:
    """Return documents owned by ``user``, optionally filtered by collection.

    Empty list is OK. Both ``collection_id`` and ``collection_slug`` is invalid.

    Raises:
        ValueError: If both collection identifiers are set.
        CollectionNotFoundError: If the requested collection does not exist for the user.
    """
    if collection_id is not None and collection_slug is not None:
        raise ValueError("Provide at most one of collection_id or collection_slug")

    query = (
        db.query(Document)
        .join(Collection, Document.collection_id == Collection.id)
        .filter(Collection.system_user_id == user.id)
    )
    if collection_id is not None:
        collection = get_collection(db, user, collection_id)
        query = query.filter(Document.collection_id == collection.id)
    elif collection_slug is not None:
        collection = get_collection_by_slug(db, user, collection_slug)[0]
        query = query.filter(Document.collection_id == collection.id)
    return query.all()


def update_document(
    db: Session,
    user: SystemUser,
    document_id: UUID,
    *,
    title: str | None = None,
    content: str | None = None,
) -> Document:
    """Update title and/or content on a document owned by ``user``.

    Content changes reset ``status`` and ``error_message`` like a new upload.
    Title or content changes re-queue ``process_document``.

    Raises:
        DocumentNotFoundError: If the document does not exist for the given user.
        DocumentValidationError: If content fails markdown upload validation.
    """
    document = get_document(db, user, document_id)
    if content is not None:
        validation = validate_markdown_upload("update.md", content.encode("utf-8"))
        if not validation.valid:
            raise DocumentValidationError(validation.reason or "Invalid content")
        document.content = content
        document.status = None
        document.error_message = None
    if title is not None:
        document.title = title
    db.commit()
    db.refresh(document)
    if title is not None or content is not None:
        trigger_process_document(str(document.id))
    return document


def delete_document(db: Session, user: SystemUser, document_id: UUID) -> None:
    """Delete a document owned by ``user``, including its chunks and job status.

    Raises:
        DocumentNotFoundError: If the document does not exist for the given user.
    """
    document = get_document(db, user, document_id)
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
    db.delete(document)
    db.commit()
    job_status.delete_job_status(str(document_id))


def get_document_status(
    db: Session,
    user: SystemUser,
    document_id: UUID,
) -> DocumentStatusResponse:
    """
    Retrieve the processing status of a specific document for a user.

    Checks both the database and any available job status information in Redis
    to provide the most up-to-date status for the requested document.

    Args:
        db (Session): SQLAlchemy database session.
        user (SystemUser): The user requesting the status.
        document_id (UUID): The UUID of the document.

    Returns:
        DocumentStatusResponse: Structured response containing the document's
            status, processing step (if available), step details, and error message.

    Raises:
        DocumentNotFoundError: If the document does not exist for the given user.
    """
    document = get_document(db, user, document_id)

    redis_status = job_status.get_job_status(str(document_id))
    if redis_status:
        return DocumentStatusResponse(
            document_id=document.id,
            status=document.status or "processing",
            step=redis_status.get("step"),
            steps=redis_status.get("steps"),
            error_message=document.error_message,
        )

    status = document.status or "queued"
    return DocumentStatusResponse(
        document_id=document.id,
        status=status,
        error_message=document.error_message,
    )
