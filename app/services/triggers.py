"""Business logic bridging API routes, DB, and background jobs."""



def trigger_process_document(document_id: str) -> str:
    """Queue a document processing job."""
    from app.services.jobs import process_document

    result = process_document.delay(document_id)
    return result.id
