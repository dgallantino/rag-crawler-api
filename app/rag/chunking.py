"""Markdown chunking with token-aware splitting and metadata extraction."""

import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def chunk_markdown(text: str, chunk_max_tokens: int, chunk_overlap_percent: float) -> list[ChunkResult]:
    overlap = int(chunk_max_tokens * chunk_overlap_percent / 100)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_max_tokens,
        chunk_overlap=overlap,
        length_function=_token_length,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
    )

    tags = _extract_frontmatter_tags(text)
    headings = _build_heading_index(text)
    chunks = splitter.split_text(text)

    results: list[ChunkResult] = []
    search_from = 0
    for chunk in chunks:
        position = text.find(chunk, search_from)
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
