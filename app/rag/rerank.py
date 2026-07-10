"""Optional reranking of retrieved chunks.

Only invoked when the service layer passes rerank=True. No dedicated
cross-encoder/reranker model is available in this system, so this uses
the completion client (LLM-as-reranker) supplied by the service layer.
"""

from __future__ import annotations

import json
import logging
import re

from app.rag.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

_CONTENT_PREVIEW_LEN = 500


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    completion_client,
    *,
    top_k: int,
    completion_model: str,
) -> list[ScoredChunk]:
    """Re-score `candidates` against `query` and return the top_k.

    Uses a listwise LLM call returning a JSON array of chunk IDs.
    On any failure, falls back to the original vector-search order.
    """
    if not candidates:
        return candidates

    try:
        prompt = _build_rerank_prompt(query, candidates)
        response = completion_client.chat.completions.create(
            model=completion_model,
            messages=[{"role": "user", "content": prompt}],
        )
        ranked_ids = _parse_rerank_response(response, candidates)
        reordered = _reorder_with_fallback(candidates, ranked_ids)
    except Exception:
        logger.warning("Rerank failed; falling back to vector-search order", exc_info=True)
        return candidates[:top_k]

    return _assign_rank_scores(reordered)[:top_k]


def _build_rerank_prompt(query: str, candidates: list[ScoredChunk]) -> str:
    lines = [
        "Rank the following document chunks by relevance to the query.",
        "Return ONLY a JSON array of chunk IDs, most relevant first.",
        'Example: ["uuid-1", "uuid-3", "uuid-2"]',
        "",
        f"Query: {query}",
        "",
        "Chunks:",
    ]
    for candidate in candidates:
        chunk_id = str(candidate.chunk.id)
        preview = candidate.chunk.content[:_CONTENT_PREVIEW_LEN]
        if len(candidate.chunk.content) > _CONTENT_PREVIEW_LEN:
            preview += "..."
        lines.append(f"[id: {chunk_id}] {preview}")
    return "\n".join(lines)


def _parse_rerank_response(response, candidates: list[ScoredChunk]) -> list[str]:
    content = response.choices[0].message.content or ""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    return [str(c.chunk.id) for c in candidates]


def _reorder_with_fallback(
    candidates: list[ScoredChunk], ranked_ids: list[str]
) -> list[ScoredChunk]:
    by_id = {str(c.chunk.id): c for c in candidates}
    seen: set[str] = set()
    reordered: list[ScoredChunk] = []

    for chunk_id in ranked_ids:
        if chunk_id in by_id and chunk_id not in seen:
            reordered.append(by_id[chunk_id])
            seen.add(chunk_id)

    for candidate in candidates:
        chunk_id = str(candidate.chunk.id)
        if chunk_id not in seen:
            reordered.append(candidate)

    return reordered


def _assign_rank_scores(candidates: list[ScoredChunk]) -> list[ScoredChunk]:
    total = len(candidates)
    if total == 0:
        return candidates
    if total == 1:
        return [ScoredChunk(chunk=candidates[0].chunk, score=1.0)]

    return [
        ScoredChunk(
            chunk=candidate.chunk,
            score=1.0 - (index / (total - 1)),
        )
        for index, candidate in enumerate(candidates)
    ]


def _reorder(candidates: list[ScoredChunk], ranked_ids: list[str]) -> list[ScoredChunk]:
    by_id = {str(c.chunk.id): c for c in candidates}
    return [by_id[i] for i in ranked_ids if i in by_id]
