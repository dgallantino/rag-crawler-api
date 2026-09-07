"""Business logic bridging API routes, DB, and background jobs."""


class QueueEnqueueError(Exception):
    """Failed to enqueue a Celery task for document processing."""

    def __init__(self, message: str = "Failed to queue document processing") -> None:
        super().__init__(message)
        self.message = message


def trigger_process_document(document_id: str) -> str:
    """Queue a document processing job."""
    from app.services.jobs import run_process_document

    try:
        result = run_process_document.delay(document_id)
    except Exception as exc:
        raise QueueEnqueueError() from exc
    return result.id
