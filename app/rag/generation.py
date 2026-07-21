"""Answer generation from retrieved chunks.

Combines chunk contents into a context block, calls the completion
client to generate an answer grounded in that context, and shapes the
final RagResponse returned up through app.rag.retrieval.retrieve().
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel

from app.models import DocumentChunk
from app.rag.processor import _token_length
from app.rag.rerank import RerankedChunk
from app.rag.retrieval import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    "If the context does not contain enough information, say so clearly. Do not invent facts."
)


class Source(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float


class RagResponse(BaseModel):
    answer: str
    sources: list[Source]


@dataclass
class ScoredChunk:
    """A RetrievedChunk or RerankedChunk that has been normalized."""

    chunk: DocumentChunk
    score: float


@dataclass
class MergedChunk:
    """One or more adjacent ScoredChunks from the same document."""

    members: list[ScoredChunk]
    score: float

    @property
    def document_id(self) -> UUID:
        return self.members[0].chunk.document_id

    @property
    def collection_id(self) -> UUID:
        return self.members[0].chunk.collection_id

    @property
    def chunk_indices(self) -> list[int]:
        return [m.chunk.chunk_index for m in self.members]

    @property
    def content(self) -> str:
        return "\n\n".join(m.chunk.content for m in self.members)

    def source_label(self) -> str:
        indices = self.chunk_indices
        if len(indices) == 1:
            return f"{self.document_id}#{indices[0]}"
        joined = ",".join(str(i) for i in indices)
        return f"{self.document_id}#[{joined}]"


def normalize_chunks(chunks: list[RetrievedChunk] | list[RerankedChunk]) -> list[ScoredChunk]:
    """Normalize chunks by extracting the chunk and score."""
    return [
        ScoredChunk(
            chunk=chunk.chunk,
            score=getattr(chunk, "rerank_score", getattr(chunk, "similarity_score", None)),
        )
        for chunk in chunks
        if chunk.chunk is not None
    ]


def merge_adjacent_chunks(chunks: list[ScoredChunk]) -> list[MergedChunk]:
    """Merge consecutive chunks that share collection_id and document_id.

    Adjacent means chunk_index differs by exactly 1. Each merged group is
    scored as the max of its members; groups are returned highest-score first.
    """
    if not chunks:
        return []

    by_doc: dict[tuple[UUID, UUID], list[ScoredChunk]] = defaultdict(list)
    for scored in chunks:
        key = (scored.chunk.collection_id, scored.chunk.document_id)
        by_doc[key].append(scored)

    merged: list[MergedChunk] = []
    for group in by_doc.values():
        group.sort(key=lambda sc: sc.chunk.chunk_index)
        run = [group[0]]
        for scored in group[1:]:
            if scored.chunk.chunk_index == run[-1].chunk.chunk_index + 1:
                run.append(scored)
            else:
                merged.append(
                    MergedChunk(members=run, score=max(m.score for m in run))
                )
                run = [scored]
        merged.append(MergedChunk(members=run, score=max(m.score for m in run)))

    merged.sort(key=lambda m: m.score, reverse=True)
    return merged


def build_context(
    chunks: list[ScoredChunk],
    *,
    max_tokens: int | None = None,
) -> str:
    """Concatenate chunk contents into a single context string.

    Adjacent same-document chunks are merged first, then groups are ordered
    by max score (descending), deduplicated by content, and included whole
    until max_tokens would be exceeded.
    """
    merged_chunks = merge_adjacent_chunks(chunks)
    seen_content: set[str] = set()
    parts: list[str] = []
    total_tokens = 0

    for merged in merged_chunks:
        content = merged.content
        if content in seen_content:
            continue
        seen_content.add(content)

        block = f"[source: {merged.source_label()}]\n{content}"
        separator = "\n\n" if parts else ""
        block_tokens = _token_length(separator + block) if max_tokens is not None else 0

        if max_tokens is not None and total_tokens + block_tokens > max_tokens:
            break

        parts.append(block)
        total_tokens += block_tokens

    return "\n\n".join(parts)


def generate_answer(
    query: str,
    context: str,
    completion_client,
    *,
    completion_model: str,
) -> str:
    """Call the completion client to produce a grounded answer."""
    response = completion_client.chat.completions.create(
        model=completion_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content or ""


def answer_with_retrieval(
    query: str,
    chunks: list[ScoredChunk],
    completion_client,
    *,
    completion_model: str,
    max_tokens_context: int | None = None,
) -> RagResponse:
    """Build the final response returned by app.rag.retrieval.retrieve()."""
    # TODO: only return answer do not return chunks
    context = build_context(chunks, max_tokens=max_tokens_context)
    answer = generate_answer(
        query,
        context,
        completion_client,
        completion_model=completion_model,
    )

    return RagResponse(
        answer=answer,
        sources=[
            Source(
                chunk_id=str(sc.chunk.id),
                document_id=str(sc.chunk.document_id),
                chunk_index=sc.chunk.chunk_index,
                score=sc.score,
            )
            for sc in chunks
        ],
    )
