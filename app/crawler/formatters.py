from __future__ import annotations

from typing import Any, Iterable, Iterator

from .pipeline import BaseStage, Item, StageContext
from .schemas import CrawlerData, EmbeddingInput, PageChunk


class DocFromCrawler(BaseStage):
    def run(
        self,
        items: Iterable[Item[CrawlerData]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        for item in items:
            yield f"{'-' * 10}\n{item.data.title}\n{item.data.text}\n"


class ChunkPrinter(BaseStage):
    """
    Development-friendly text formatter for DOMChunker output.
    """

    def run(
        self,
        items: Iterable[Item[PageChunk]],
        context: StageContext,
    ) -> Iterator[str]:
        for item in items:
            chunk = item.data
            headings = " > ".join(chunk.headings) if chunk.headings else ""
            yield (
                f"{'-' * 10}\n"
                f"chunk_index: {chunk.chunk_index}\n"
                f"url: {chunk.url or ''}\n"
                f"title: {chunk.title or ''}\n"
                f"headings: {headings}\n"
                f"selector: {chunk.selector}\n"
                f"chars: {chunk.char_count}\n"
                f"words: {chunk.word_count}\n\n"
                f"{chunk.text}\n"
            )


class EmbeddingInputBuilder(BaseStage):
    """
    Prepare PageChunk content for an embedding model without calling one.
    """

    def run(
        self,
        items: Iterable[Item[PageChunk]],
        context: StageContext,
    ) -> Iterator[Item[EmbeddingInput]]:
        for item in items:
            chunk = item.data
            heading = " > ".join(chunk.headings) if chunk.headings else None
            text = self._embedding_text(chunk, heading)
            metadata = {
                "url": chunk.url,
                "chunk_index": chunk.chunk_index,
                "title": chunk.title,
                "heading": heading,
                "headings": chunk.headings,
                "selector": chunk.selector,
                "char_count": chunk.char_count,
                "word_count": chunk.word_count,
                "node_names": chunk.node_names,
            }

            yield Item(
                data=EmbeddingInput(
                    text=text,
                    source_text=chunk.text,
                    title=chunk.title,
                    heading=heading,
                    chunk_index=chunk.chunk_index,
                ),
                meta={**item.meta, **metadata},
            )

    @staticmethod
    def _embedding_text(chunk: PageChunk, heading: str | None) -> str:
        context_lines = []
        if chunk.title:
            context_lines.append(f"Title: {chunk.title}")
        if heading:
            context_lines.append(f"Heading: {heading}")

        if not context_lines:
            return chunk.text

        return "\n".join(context_lines) + f"\n\n{chunk.text}"
