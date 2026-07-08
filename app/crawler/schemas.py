from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CrawlerData:
    url: str
    title: str
    text: str


@dataclass
class PageChunk:
    chunk_index: int
    url: Optional[str]
    text: str
    headings: List[str]
    title: Optional[str]
    selector: str
    char_count: int
    word_count: int
    node_names: List[str]


@dataclass
class EmbeddingInput:
    text: str
    source_text: str
    title: Optional[str]
    heading: Optional[str]
    chunk_index: int
