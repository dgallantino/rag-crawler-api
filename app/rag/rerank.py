"""Optional reranking of retrieved chunks.

Only invoked when the service layer passes ``use_rerank=True``. The actual
rerank provider is injected as ``RerankServiceFn`` (built in the service
layer); this module only invokes it and falls back on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from app.models import DocumentChunk
from app.rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    """A DocumentChunk plus its rerank-stage score."""

    chunk: DocumentChunk
    rerank_score: float
    similarity_score: float | None = None  # assigned later after rerank result remapped


# Contract for the service-layer rerank provider.
RerankServiceFn = Callable[[str, int, list[RetrievedChunk]], list[RerankedChunk]]


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    *,
    rerank_service_fn: RerankServiceFn,
) -> list[RerankedChunk] | list[RetrievedChunk]:
    """Re-score ``candidates`` against ``query`` and return the top_k.

    On any failure, falls back to the original retrieval order.
    """
    if not rerank_service_fn:
        return []

    try:
        reranked_chunks = rerank_service_fn(query, top_k, candidates)
    except Exception:
        logger.warning("Rerank failed; falling back to vector-search order", exc_info=True)
        return candidates[:top_k]

    return reranked_chunks[:top_k]
