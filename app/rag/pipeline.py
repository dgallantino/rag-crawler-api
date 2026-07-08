"""RAG document processing pipeline."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.rag import chunking, embeddings
from app.rag.events import DocumentStatusEvent, DocumentStatusHandler


def process_document(
        db: Session, document_id: str,
        *,
        chunk_max_tokens: int = 500,
        chunk_overlap_percent: float = 10,
        on_status: DocumentStatusHandler
    ) -> None:
    """Process a document by chunking, embedding, and storing the chunks.

    Args:
        db: The database session.
        document_id: The ID of the document to process.
        on_status: A callback invoked when a processing step changes status.
    """
    document = db.query(Document).filter(Document.id == UUID(document_id)).one_or_none()
    if document is None:
        return

    try:
        document.status = "processing"
        document.error_message = None
        db.commit()

        on_status(DocumentStatusEvent(document_id, "chunking", "in_progress"))
        chunks = chunking.chunk_markdown(document.content or "", chunk_max_tokens, chunk_overlap_percent)
        on_status(DocumentStatusEvent(document_id, "chunking", "completed"))

        on_status(DocumentStatusEvent(document_id, "embedding", "in_progress"))
        vectors = embeddings.embed_texts([chunk.content for chunk in chunks])
        on_status(DocumentStatusEvent(document_id, "embedding", "completed"))

        on_status(DocumentStatusEvent(document_id, "storing", "in_progress"))
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
        on_status(DocumentStatusEvent(document_id, "storing", "completed"))
    except Exception as exc:
        db.rollback()
        document = db.query(Document).filter(Document.id == UUID(document_id)).one_or_none()
        if document is not None:
            document.status = "failed"
            document.error_message = str(exc)
            db.commit()
        on_status(DocumentStatusEvent(document_id, "failed", "completed"))
        raise
