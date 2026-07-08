"""
CLI-only developer helper for manually running the crawler pipeline.

Do NOT import this module from jobs, services, or API code.
Production paths should compose Pipeline([...]) directly at the call site.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from .extractors import DOMChunker
from .formatters import ChunkPrinter
from .pipeline import DataRetrieverError, Item, Pipeline, StageContext
from .schemas import PageChunk
from .settings import DEFAULT_HEADLESS, DEFAULT_MAX_PAGES
from .sources import JSCrawler

logger = logging.getLogger(__name__)


def run_crawl_debug(
    seed_urls: list[str],
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    headless: bool = DEFAULT_HEADLESS,
    job_id: str | None = None,
) -> list[Any]:
    """
    Run the default debug pipeline: JSCrawler → DOMChunker → ChunkPrinter.

    CLI-only — do NOT use as the main entry point in jobs, services, or API code.
    Production callers should compose Pipeline([...]) directly at the call site.
    """
    if not seed_urls:
        raise ValueError("seed_urls must be a non-empty list")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    context = StageContext()
    context.set("seed_urls", seed_urls)
    context.set("job_id", job_id or str(uuid4()))

    pipeline = Pipeline([
        JSCrawler(max_pages=max_pages, headless=headless),
        DOMChunker(),
        ChunkPrinter(),
    ])

    logger.info(
        "starting debug crawl job %s with %s seed URL(s)",
        context.get("job_id"),
        len(seed_urls),
    )

    try:
        return pipeline.run(context=context)
    except DataRetrieverError:
        raise


def count_crawl_results(results: list[Any]) -> tuple[int, int]:
    """Return (page_count, chunk_count) from pipeline results."""
    pages = 0
    chunks = 0
    for item in results:
        if isinstance(item, str):
            chunks += 1
        elif isinstance(item, Item) and isinstance(item.data, PageChunk):
            chunks += 1
        elif isinstance(item, Item):
            pages += 1
    return pages, chunks
