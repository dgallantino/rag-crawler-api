"""
Production crawl orchestration.

Pipeline composition for crawl jobs lives here (or in services/jobs.py), not in runner.py.
runner.py is CLI-only for local debugging.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.crawler.extractors import DOMChunker
from app.crawler.formatters import EmbeddingInputBuilder
from app.crawler.pipeline import Pipeline, StageContext
from app.crawler.settings import DEFAULT_HEADLESS, DEFAULT_MAX_PAGES
from app.crawler.sources import JSCrawler
from app.crawler.storage import DBStorage


def run_crawl_for_url(
    db: Session,
    url: str,
    system_user_id: UUID,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    headless: bool = DEFAULT_HEADLESS,
    job_id: str | None = None,
) -> dict:
    """
    Run a production crawl for a single URL.

    Composes Pipeline([...]) directly — do not delegate to runner.run_crawl_debug().
    """
    context = StageContext()
    context.set("seed_urls", [url])
    if job_id:
        context.set("job_id", job_id)

    pipeline = Pipeline([
        JSCrawler(max_pages=max_pages, headless=headless),
        DOMChunker(),
        EmbeddingInputBuilder(),
        DBStorage(db=db, system_user_id=system_user_id),
    ])

    # TODO: persist crawled pages to Document and trigger process_document
    results = pipeline.run(context=context)
    return {
        "url": url,
        "job_id": context.get("job_id"),
        "items": len(results),
        "status": "stub",
    }
