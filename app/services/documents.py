"""Document upload and status services."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Document, SystemUser
from app.schemas.documents import DocumentStatusResponse
from app.services import job_status
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


def create_document_upload(
    db: Session,
    user: SystemUser,
    filename: str,
    content: str,
) -> Document:
    """
    Create a new document entry in the database based on an uploaded file.

    This function will:
    - Attempt to create a new Document with the given filename and content,
      associated with the provided system user.
    - If a document with the same filename for the user already exists, it will raise
      DocumentConflictError.
    - On success, commits the new document, triggers the background processing job,
      and returns the created Document object.

    Args:
        db (Session): SQLAlchemy database session.
        user (SystemUser): The user uploading the document.
        filename (str): Name of the uploaded file.
        content (str): Raw content of the file.

    Returns:
        Document: The SQLAlchemy Document instance just created.

    Raises:
        DocumentConflictError: If a document with the same filename already exists.
    """
    document = Document(
        system_user_id=user.id,
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
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.system_user_id == user.id)
        .one_or_none()
    )
    if document is None:
        raise DocumentNotFoundError(str(document_id))

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
