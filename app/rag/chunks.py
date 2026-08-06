"""Unified scored-chunk type shared across the RAG pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from app.models import DocumentChunk


@dataclass
class ScoredChunk:
    """A DocumentChunk plus its current pipeline-stage score.

    ``score`` always holds the most recent stage's score and is never None:
    cosine similarity (vector-only), RRF fusion score (hybrid), or the
    rerank relevance score once reranking has run. The stage-specific
    fields keep earlier scores around for response shaping/debugging.
    """

    chunk: DocumentChunk
    score: float
    retrieval_score: float | None = None
    rerank_score: float | None = None
