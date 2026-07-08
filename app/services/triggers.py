"""Business logic bridging API routes, DB, and background jobs."""

from app.jobs import crawl_url, process_document


def trigger_crawl(url: str, system_user_id: str | None = None) -> str:
    """Queue a crawl job for the given URL."""
    result = crawl_url.delay(url, system_user_id=system_user_id)
    return result.id


def trigger_process_document(document_id: str) -> str:
    """Queue a document processing job."""
    result = process_document.delay(document_id)
    return result.id
