"""RAG document processors: chunking, embedding, and persistence."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.rag.events import DocumentStatusEvent, DocumentStatusHandler

BATCH_SIZE = 100

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TAGS_PATTERN = re.compile(r"^tags:\s*\[(.*?)\]", re.MULTILINE)

_encoding: tiktoken.Encoding | None = None


@dataclass(frozen=True)
class ChunkResult:
    content: str
    metadata: dict


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _token_length(text: str) -> int:
    return len(_get_encoding().encode(text))


def _extract_frontmatter_tags(text: str) -> list[str]:
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return []

    frontmatter = match.group(1)
    tags_match = TAGS_PATTERN.search(frontmatter)
    if not tags_match:
        return []

    return [tag.strip().strip("'\"") for tag in tags_match.group(1).split(",") if tag.strip()]


def _build_heading_index(text: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for match in HEADING_PATTERN.finditer(text):
        headings.append((match.start(), match.group(2).strip()))
    return headings


def _nearest_heading(position: int, headings: list[tuple[int, str]]) -> str | None:
    current: str | None = None
    for heading_pos, heading_text in headings:
        if heading_pos <= position:
            current = heading_text
        else:
            break
    return current


class DocumentProcessor(ABC):
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @abstractmethod
    def chunk(self, content: str) -> list[ChunkResult]:
        """Split document content into chunks with metadata."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in sorted_data)

        return embeddings

    def process_document(
        self,
        db: Session,
        document_id: str,
        *,
        on_status: DocumentStatusHandler,
    ) -> None:
        """Process a document by chunking, embedding, and storing the chunks."""
        document = db.query(Document).filter(Document.id == UUID(document_id)).one_or_none()
        if document is None:
            return

        try:
            document.status = "processing"
            document.error_message = None
            db.commit()

            on_status(DocumentStatusEvent(document_id, "chunking", "in_progress"))
            chunks = self.chunk(document.content or "")
            on_status(DocumentStatusEvent(document_id, "chunking", "completed"))

            on_status(DocumentStatusEvent(document_id, "embedding", "in_progress"))
            vectors = self.embed_texts([chunk.content for chunk in chunks])
            on_status(DocumentStatusEvent(document_id, "embedding", "completed"))

            on_status(DocumentStatusEvent(document_id, "storing", "in_progress"))
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()

            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                db.add(
                    DocumentChunk(
                        document_id=document.id,
                        collection_id=document.collection_id,
                        chunk_index=index,
                        content=chunk.content,
                        chunk_metadata=chunk.metadata or None,
                        chunk_vector=vector,
                    )
                )

            document.status = "success"
            document.error_message = None
            db.commit()
            on_status(DocumentStatusEvent(document_id, "storing", "completed"))
        except Exception as exc:
            db.rollback()
            document = db.query(Document).filter(Document.id == UUID(document_id)).one_or_none()
            if document is not None:
                document.status = "failed"
                document.error_message = str(exc)
                db.commit()
            on_status(DocumentStatusEvent(document_id, "failed", "completed"))
            raise


class MarkdownProcessor(DocumentProcessor):
    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        chunk_max_tokens: int,
        chunk_overlap_percent: float,
    ) -> None:
        super().__init__(client, model)
        self._chunk_max_tokens = chunk_max_tokens
        self._chunk_overlap_percent = chunk_overlap_percent

    def chunk(self, content: str) -> list[ChunkResult]:
        overlap = int(self._chunk_max_tokens * self._chunk_overlap_percent / 100)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_max_tokens,
            chunk_overlap=overlap,
            length_function=_token_length,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        )

        tags = _extract_frontmatter_tags(content)
        headings = _build_heading_index(content)
        chunks = splitter.split_text(content)

        results: list[ChunkResult] = []
        search_from = 0
        for chunk in chunks:
            position = content.find(chunk, search_from)
            if position == -1:
                position = search_from
            search_from = position + len(chunk)

            metadata: dict = {}
            section_header = _nearest_heading(position, headings)
            if section_header:
                metadata["section_header"] = section_header
            if headings:
                metadata["all_headings"] = [heading_text for _, heading_text in headings]
            if tags:
                metadata["tags"] = tags

            results.append(ChunkResult(content=chunk, metadata=metadata))

        return results
