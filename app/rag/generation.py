"""Answer generation from retrieved chunks.

Combines chunk contents into a context block, calls the completion
client to generate an answer grounded in that context, and shapes the
final RagResponse returned up through app.rag.retrieval.retrieve().
"""

from __future__ import annotations

from pydantic import BaseModel

from app.rag.retrieval import ScoredChunk

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


def build_context(chunks: list[ScoredChunk], *, max_chars: int | None = None) -> str:
    """Concatenate chunk contents into a single context string.

    Chunks are ordered by score (descending), deduplicated by content,
    and included whole until max_chars would be exceeded.
    """
    sorted_chunks = sorted(chunks, key=lambda sc: sc.score, reverse=True)
    seen_content: set[str] = set()
    parts: list[str] = []
    total_chars = 0

    for scored in sorted_chunks:
        content = scored.chunk.content
        if content in seen_content:
            continue
        seen_content.add(content)

        block = (
            f"[source: {scored.chunk.document_id}#{scored.chunk.chunk_index}]\n"
            f"{content}"
        )
        separator_len = 2 if parts else 0  # "\n\n" between blocks
        if max_chars is not None and total_chars + separator_len + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += separator_len + len(block)

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
) -> RagResponse:
    """Build the final response returned by app.rag.retrieval.retrieve()."""
    context = build_context(chunks)
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
