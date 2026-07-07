"""Document upload and status services."""

from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Document, SystemUser
from app.schemas.documents import DocumentStatusResponse
from app.services import job_status
from app.services.triggers import trigger_process_document


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
