"""Celery application and background task definitions."""


from celery import Celery

from app.config import get_settings
from app.database import SessionLocal
from app.rag.pipeline import process_document as run_process_document

settings = get_settings()

celery_app = Celery(
    "rag_crawler",
    broker=settings.celery_broker_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="process_document", bind=True, max_retries=0)
def process_document(self, document_id: str) -> None:
    """Chunk, embed, and store a document."""
    db = SessionLocal()
    try:
        run_process_document(db, document_id)
    finally:
        db.close()
