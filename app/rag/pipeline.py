"""RAG document processing pipeline."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.rag import chunking, embeddings
import app.services.job_status as job_status


def process_document(db: Session, document_id: str) -> None:
    document = db.query(Document).filter(Document.id == UUID(document_id)).one_or_none()
    if document is None:
        return

    try:
        document.status = "processing"
        document.error_message = None
        db.commit()

        job_status.set_job_step(document_id, "chunking", "in_progress")
        chunks = chunking.chunk_markdown(document.content or "")
        job_status.set_job_step(document_id, "chunking", "completed")

        job_status.set_job_step(document_id, "embedding", "in_progress")
        vectors = embeddings.embed_texts([chunk.content for chunk in chunks])
        job_status.set_job_step(document_id, "embedding", "completed")

        job_status.set_job_step(document_id, "storing", "in_progress")
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()

        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    system_user_id=document.system_user_id,
                    chunk_index=index,
                    content=chunk.content,
                    chunk_metadata=chunk.metadata or None,
                    chunk_vector=vector,
                )
            )

        document.status = "success"
        document.error_message = None
        db.commit()
        job_status.set_job_step(document_id, "storing", "completed")
        job_status.delete_job_status(document_id)
    except Exception as exc:
        db.rollback()
        document = db.query(Document).filter(Document.id == UUID(document_id)).one_or_none()
        if document is not None:
            document.status = "failed"
            document.error_message = str(exc)
            db.commit()
        job_status.delete_job_status(document_id)
        raise
