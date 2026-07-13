"""Optional reranking of retrieved chunks.

Only invoked when the service layer passes rerank=True. No dedicated
cross-encoder/reranker model is available in this system, so this uses
the completion client (LLM-as-reranker) supplied by the service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Callable

from app.models import DocumentChunk
from app.rag.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

_CONTENT_PREVIEW_LEN = 500


@dataclass
class RerankedChunk:
    """A RetrievedChunk plus its rerank-stage similarity score."""

    chunk: DocumentChunk
    rerank_score: float
    similarity_score: float | None = None  # assigned later after rerank result remapped


# contract for how service layer should supply rerank client
# this module would parse the rerank response 
RerankServiceFn = Callable[[str,int, list[RetrievedChunk]], list[RerankedChunk]]

def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    *,
    rerank_service_fn: RerankServiceFn,
) -> list[RetrievedChunk]:
    """Re-score `candidates` against `query` and return the top_k.

    Uses a listwise LLM call returning a JSON array of chunk IDs.
    On any failure, falls back to the original vector-search order.
    """

    # if this is just a runner now maybe better to have a service layer that does this?
    if not rerank_service_fn:
        return []

    try:
        reranked_chunks = rerank_service_fn(query, top_k, candidates)
    except Exception:
        logger.warning("Rerank failed; falling back to vector-search order", exc_info=True)
        return candidates[:top_k]

    return reranked_chunks[:top_k]
