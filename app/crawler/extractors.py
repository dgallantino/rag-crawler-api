from __future__ import annotations

import logging
import math
import re
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from .pipeline import BaseStage, Item, StageContext
from .schemas import CrawlerData, PageChunk


logger = logging.getLogger(__name__)


class JSTextExtractor(BaseStage):
    """
    """

    def _extract_data(self, html: str, ) -> CrawlerData:
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        text = soup.get_text(" ", strip=True)
        return  title, text

    def run(
        self,
        items: Iterable[Item[CrawlerData]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        for item in items:
            title, extracted_text = self._extract_data(item.data.text)
            item.data.text = extracted_text
            item.data.title = title
            yield Item(
                data=item.data,
                meta={
                    **item.meta,
                    "html_parsed": True,
                },
            )

class DOMChunker(BaseStage):
    """
    Pipeline stage that chunks rendered HTML into PageChunk items.
    """

    DEFAULT_BLOCK_TAGS = {
        "article", "section", "div", "main", "meta",
        "p", "ul", "ol", "li",
        "table", "thead", "tbody", "tr", "td", "th",
        "blockquote", "pre",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }

    PRESERVED_ROOT_TAGS = {"html", "body", "main", "article"}

    NOISE_TAGS = {
        "script", "style", "noscript", "svg", "canvas", "iframe",
        "footer", "nav", "aside", "form",
    }

    NOISE_CLASS_ID_PATTERNS = [
        r"nav",
        r"menu",
        r"breadcrumb",
        r"footer",
        r"header",
        r"sidebar",
        r"share",
        r"social",
        r"related",
        r"recommend",
        r"promo",
        r"banner",
        r"advert",
        r"ads",
        r"cookie",
        r"consent",
        r"modal",
        r"popup",
        r"comment",
        r"newsletter",
        r"subscribe",
        r"pagination",
        r"toolbar",
        r"masthead",
    ]

    CONTENT_ROOT_SELECTORS = [
        "[role='main']",
        "main",
        "article",
        ".entry-content",
        ".post-content",
        ".page-content",
        ".article-content",
        ".content-area",
        "#main",
        "#content",
    ]

    def __init__(
        self,
        min_words: int = 8,
        min_chars: int = 40,
        max_chars: int = 3500,
        target_chars: int = 1800,
        max_link_density: float = 0.45,
        merge_small_chunks: bool = True,
    ) -> None:
        self.min_words = min_words
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.target_chars = target_chars
        self.max_link_density = max_link_density
        self.merge_small_chunks = merge_small_chunks

        self._noise_class_id_regexes = [
            re.compile(rf"(?:^|[\s_\-]){p}(?:$|[\s_\-])", re.IGNORECASE)
            for p in self.NOISE_CLASS_ID_PATTERNS
        ]

    def run(
        self,
        items: Iterable[Item[CrawlerData]],
        context: StageContext,
    ) -> Iterator[Item[PageChunk]]:
        for item in items:
            yield from self._chunk_page(item)

    def _chunk_page(self, item: Item[CrawlerData]) -> List[Item[PageChunk]]:
        soup = BeautifulSoup(item.data.text, "html.parser")
        self._remove_noise(soup)

        title = self._extract_title(soup)
        root = self._find_main_content_root(soup)

        chunks = self._build_chunks(root=root, title=title, url=item.data.url)
        chunks = self._merge_small_chunks(chunks) if self.merge_small_chunks else chunks
        chunks = self._split_oversized_chunks(chunks)
        chunks = self._filter_final_chunks(chunks)

        return [
            Item(data=chunk, meta={**item.meta, "chunk": True})
            for chunk in chunks
        ]

    # ----------------------------
    # Cleaning / root detection
    # ----------------------------

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        for tag_name in self.NOISE_TAGS:
            for node in soup.find_all(tag_name):
                node.decompose()

        for node in list(soup.find_all(True)):
            if self._looks_like_noise_node(node):
                node.decompose()

    def _looks_like_noise_node(self, node: Tag) -> bool:
        if node.name in self.PRESERVED_ROOT_TAGS:
            return False

        if not getattr(node, "attrs", None):
            return False

        attrs = self._node_attr_text(node)

        if attrs:
            for rx in self._noise_class_id_regexes:
                if rx.search(attrs):
                    return True

        # Hidden-ish nodes
        style = (node.get("style") or "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True

        return False

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        if soup.title and soup.title.get_text(strip=True):
            return self._normalize_text(soup.title.get_text(" ", strip=True))
        h1 = soup.find("h1")
        if h1:
            return self._normalize_text(h1.get_text(" ", strip=True))
        return None

    def _find_main_content_root(self, soup: BeautifulSoup) -> Tag:
        # Prefer conventional content containers before falling back to scoring.
        for selector in self.CONTENT_ROOT_SELECTORS:
            node = soup.select_one(selector)
            if (
                node
                and isinstance(node, Tag)
                and not self._looks_like_noise_node(node)
                and self._node_text_len(node) >= self.min_chars
            ):
                return node

        candidates = soup.find_all(["div", "section", "article", "main"])
        best = soup.body or soup

        best_score = float("-inf")
        for node in candidates:
            if self._looks_like_noise_node(node):
                continue
            score = self._content_score(node)
            if score > best_score:
                best_score = score
                best = node

        return best

    def _content_score(self, node: Tag) -> float:
        text = self._extract_text(node)
        text_chars = len(text)
        if text_chars == 0:
            return -999.0

        descendants = max(1, len(node.find_all(True)))
        paragraphs = len(node.find_all("p"))
        headings = len(node.find_all(re.compile(r"^h[1-6]$")))
        link_density = self._link_density(node)

        # penalize nodes with too many short links / low text density
        text_density = text_chars / descendants

        score = (
            2.0 * math.log1p(text_chars)
            + 1.2 * paragraphs
            + 1.0 * headings
            + 0.02 * text_density
            - 3.0 * link_density
        )

        return score

    # ----------------------------
    # Core chunk building
    # ----------------------------

    def _build_chunks(
        self,
        root: Tag,
        title: Optional[str],
        url: Optional[str],
    ) -> List[PageChunk]:
        chunks: List[PageChunk] = []
        heading_stack: List[Tuple[int, str]] = []
        current_parts: List[str] = []
        current_nodes: List[Tag] = []
        chunk_index = 0

        for node in self._iter_relevant_nodes(root):
            if not isinstance(node, Tag):
                continue

            if self._is_heading(node):
                heading_text = self._normalize_text(node.get_text(" ", strip=True))
                if not heading_text:
                    continue

                # flush existing content before starting new section
                if self._joined_length(current_parts) >= self.min_chars:
                    chunks.append(
                        self._make_chunk(
                            chunk_index=chunk_index,
                            parts=current_parts,
                            nodes=current_nodes,
                            headings=[h[1] for h in heading_stack],
                            title=title,
                            url=url,
                        )
                    )
                    chunk_index += 1
                    current_parts = []
                    current_nodes = []

                level = int(node.name[1])
                heading_stack = [h for h in heading_stack if h[0] < level]
                heading_stack.append((level, heading_text))
                continue

            text = self._extract_meaningful_text_from_node(node)
            if not text:
                continue

            if not self._is_meaningful_node(node, text):
                continue

            projected_len = self._joined_length(current_parts) + len(text) + 2
            if projected_len > self.max_chars and current_parts:
                chunks.append(
                    self._make_chunk(
                        chunk_index=chunk_index,
                        parts=current_parts,
                        nodes=current_nodes,
                        headings=[h[1] for h in heading_stack],
                        title=title,
                        url=url,
                    )
                )
                chunk_index += 1
                current_parts = []
                current_nodes = []

            logger.debug("adding chunk %s"%text)
            current_parts.append(text)
            current_nodes.append(node)

        if current_parts:
            chunks.append(
                self._make_chunk(
                    chunk_index=chunk_index,
                    parts=current_parts,
                    nodes=current_nodes,
                    headings=[h[1] for h in heading_stack],
                    title=title,
                    url=url,
                )
            )

        return chunks

    def _iter_relevant_nodes(self, root: Tag):
        """
        Walk block-like nodes in reading order while avoiding nested duplication.
        We prefer leaf-ish content nodes plus headings.
        """
        for node in root.descendants:
            if not isinstance(node, Tag):
                continue

            if node.name not in self.DEFAULT_BLOCK_TAGS:
                continue

            if self._looks_like_noise_node(node):
                continue

            if self._is_heading(node):
                yield node
                continue

            # Yield paragraph/list/table/pre/blockquote directly
            if node.name in {"p", "ul", "ol", "table", "blockquote", "pre"}:
                yield node
                continue

            # Yield div/section/article only if they are leaf-ish content containers
            if node.name in {"div", "section", "article", "main"}:
                child_blocks = [
                    c for c in node.find_all(recursive=False)
                    if isinstance(c, Tag) and c.name in self.DEFAULT_BLOCK_TAGS
                ]
                if not child_blocks:
                    yield node

    # ----------------------------
    # Meaningfulness
    # ----------------------------

    def _is_heading(self, node: Tag) -> bool:
        return bool(re.match(r"^h[1-6]$", node.name or ""))

    def _is_meaningful_node(self, node: Tag, text: str) -> bool:
        if self._looks_like_boilerplate(text):
            return False

        link_density = self._link_density(node)
        if link_density > self.max_link_density and not self._has_structured_context(node):
            return False

        if node.name in {"ul", "ol", "table"} and len(text) >= self.min_chars:
            return True

        word_count = self._word_count(text)
        if self._looks_like_structured_fact(node, text):
            return True

        if word_count < self.min_words:
            return False

        if len(text) < self.min_chars:
            return False

        if self._sentence_count(text) < 1:
            return False

        return True

    def _has_structured_context(self, node: Tag) -> bool:
        current = node.parent
        while current and isinstance(current, Tag) and current.name not in {"html", "body"}:
            if current.find(re.compile(r"^h[1-6]$")):
                return True
            current = current.parent
        return False

    def _looks_like_structured_fact(self, node: Tag, text: str) -> bool:
        if len(text) < 3:
            return False

        lowered = text.lower()
        word_count = self._word_count(text)

        if node.name in {"li", "td", "th", "tr"} and word_count >= 2:
            return True

        fact_patterns = [
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(?:mon|tue|wed|thu|fri|sat|sun)\b",
            r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b",
            r"(?:\+?\d[\d\s().-]{6,}\d)",
            r"\b(?:phone|whatsapp|email|address|location|hours?|open|closed)\b",
            r"\b(?:jl\.|jalan|street|st\.|road|rd\.|avenue|ave\.|suite|floor|building|blok|block|rt\.|rw\.)\b",
            r"(?:rp|idr|usd|\$|€|£)\s*\d",
            r"\b\d+\s*(?:km|m|minutes?|mins?|hours?|hrs?)\b",
        ]

        if any(re.search(pattern, lowered) for pattern in fact_patterns):
            return True

        if ":" in text and word_count >= 2:
            return True

        if len(text) >= self.min_chars and word_count >= max(3, self.min_words // 2):
            return True

        return False

    def _looks_like_boilerplate(self, text: str) -> bool:
        lowered = text.lower()

        boilerplate_patterns = [
            r"all rights reserved",
            r"privacy policy",
            r"terms of service",
            r"cookie policy",
            r"sign up",
            r"subscribe",
            r"follow us",
            r"share this",
            r"related articles",
            r"table of contents",
            r"skip to content",
        ]

        for pattern in boilerplate_patterns:
            if re.search(pattern, lowered):
                return True

        # Too repetitive can indicate nav/menu junk
        tokens = re.findall(r"\w+", lowered)
        if len(tokens) >= 12:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.35:
                return True

        return False

    # ----------------------------
    # Splitting / merging
    # ----------------------------

    def _merge_small_chunks(self, chunks: List[PageChunk]) -> List[PageChunk]:
        if not chunks:
            return []

        merged: List[PageChunk] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = merged[-1]
            should_merge = (
                chunk.char_count < self.min_chars
                or chunk.word_count < self.min_words
            )

            same_heading_context = chunk.headings == prev.headings

            if should_merge or same_heading_context and prev.char_count < self.target_chars:
                combined_text = prev.text + "\n\n" + chunk.text
                headings = prev.headings or chunk.headings
                node_names = prev.node_names + chunk.node_names
                char_count = len(combined_text)
                word_count = self._word_count(combined_text)
                merged[-1] = PageChunk(
                    chunk_index=prev.chunk_index,
                    url=prev.url,
                    text=combined_text,
                    headings=headings,
                    title=prev.title,
                    selector=prev.selector,
                    char_count=char_count,
                    word_count=word_count,
                    node_names=node_names,
                )
            else:
                merged.append(chunk)

        # re-number
        for i, c in enumerate(merged):
            c.chunk_index = i

        return merged

    def _split_oversized_chunks(self, chunks: List[PageChunk]) -> List[PageChunk]:
        out: List[PageChunk] = []
        next_id = 0

        for chunk in chunks:
            if chunk.char_count <= self.max_chars:
                chunk.chunk_index = next_id
                out.append(chunk)
                next_id += 1
                continue

            pieces = self._split_text_by_paragraphs(chunk.text, self.max_chars)
            for piece in pieces:
                char_count = len(piece)
                word_count = self._word_count(piece)
                out.append(
                    PageChunk(
                        chunk_index=next_id,
                        url=chunk.url,
                        text=piece,
                        headings=chunk.headings,
                        title=chunk.title,
                        selector=chunk.selector,
                        char_count=char_count,
                        word_count=word_count,
                        node_names=chunk.node_names,
                    )
                )
                next_id += 1

        return out

    def _split_text_by_paragraphs(self, text: str, max_chars: int) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paras:
            return [text]

        pieces: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paras:
            p_len = len(para)

            if current and current_len + p_len + 2 > max_chars:
                pieces.append("\n\n".join(current))
                current = [para]
                current_len = p_len
                continue

            if p_len > max_chars:
                subparts = self._split_long_paragraph(para, max_chars)
                for sp in subparts:
                    if current and current_len + len(sp) + 2 > max_chars:
                        pieces.append("\n\n".join(current))
                        current = [sp]
                        current_len = len(sp)
                    else:
                        current.append(sp)
                        current_len += len(sp) + (2 if len(current) > 1 else 0)
                continue

            current.append(para)
            current_len += p_len + (2 if len(current) > 1 else 0)

        if current:
            pieces.append("\n\n".join(current))

        return pieces

    def _split_long_paragraph(self, paragraph: str, max_chars: int) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        if len(sentences) <= 1:
            return [paragraph[i:i + max_chars] for i in range(0, len(paragraph), max_chars)]

        parts: List[str] = []
        current: List[str] = []
        current_len = 0

        for sent in sentences:
            s_len = len(sent)
            if current and current_len + s_len + 1 > max_chars:
                parts.append(" ".join(current))
                current = [sent]
                current_len = s_len
            else:
                current.append(sent)
                current_len += s_len + (1 if len(current) > 1 else 0)

        if current:
            parts.append(" ".join(current))

        return parts

    def _filter_final_chunks(self, chunks: List[PageChunk]) -> List[PageChunk]:
        kept: List[PageChunk] = []

        for c in chunks:
            if c.char_count < self.min_chars:
                continue
            if c.word_count < max(3, self.min_words // 2):
                continue
            kept.append(c)

        for i, c in enumerate(kept):
            c.chunk_index = i

        return kept

    # ----------------------------
    # Chunk creation helpers
    # ----------------------------

    def _make_chunk(
        self,
        chunk_index: int,
        parts: List[str],
        nodes: List[Tag],
        headings: List[str],
        title: Optional[str],
        url: Optional[str],
    ) -> PageChunk:
        text = "\n\n".join(p.strip() for p in parts if p.strip())
        selector = self._css_path(nodes[0]) if nodes else ""
        node_names = [n.name for n in nodes if isinstance(n, Tag)]
        char_count = len(text)
        word_count = self._word_count(text)

        return PageChunk(
            chunk_index=chunk_index,
            url=url,
            text=text,
            headings=headings,
            title=title,
            selector=selector,
            char_count=char_count,
            word_count=word_count,
            node_names=node_names,
        )

    def _css_path(self, node: Tag) -> str:
        parts: List[str] = []
        current: Optional[Tag] = node

        while current and current.name not in {"[document]", "html"}:
            if not current.parent or not isinstance(current.parent, Tag):
                break

            siblings = [
                s for s in current.parent.find_all(current.name, recursive=False)
                if isinstance(s, Tag)
            ]

            if len(siblings) == 1:
                parts.append(current.name)
            else:
                idx = siblings.index(current) + 1
                parts.append(f"{current.name}:nth-of-type({idx})")

            current = current.parent

        return " > ".join(reversed(parts))

    # ----------------------------
    # Text extraction helpers
    # ----------------------------

    def _extract_meaningful_text_from_node(self, node: Tag) -> str:
        if node.name in {"ul", "ol"}:
            items = []
            for li in node.find_all("li", recursive=False):
                t = self._normalize_text(li.get_text(" ", strip=True))
                if t:
                    items.append(f"- {t}")
            return "\n".join(items)

        if node.name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [
                    self._normalize_text(cell.get_text(" ", strip=True))
                    for cell in tr.find_all(["th", "td"])
                ]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(" | ".join(cells))
            return "\n".join(rows)

        return self._extract_text(node)

    def _extract_text(self, node: Tag) -> str:
        text = node.get_text(" ", strip=True)
        return self._normalize_text(text)

    def _node_attr_text(self, node: Tag) -> str:
        classes = node.get("class", [])
        if isinstance(classes, str):
            classes = [classes]

        attrs = [
            node.get("id", ""),
            " ".join(classes),
            node.get("role", ""),
            node.get("aria-label", ""),
        ]
        return " ".join(str(a) for a in attrs if a)

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def _node_text_len(self, node: Tag) -> int:
        return len(self._extract_text(node))

    def _link_density(self, node: Tag) -> float:
        total_text = self._extract_text(node)
        total_len = max(1, len(total_text))

        link_text = " ".join(
            self._normalize_text(a.get_text(" ", strip=True))
            for a in node.find_all("a")
        ).strip()

        return len(link_text) / total_len

    def _joined_length(self, parts: List[str]) -> int:
        if not parts:
            return 0
        return sum(len(p) for p in parts) + (2 * (len(parts) - 1))

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def _sentence_count(self, text: str) -> int:
        parts = re.split(r"[.!?]+(?:\s|$)", text.strip())
        return len([p for p in parts if p.strip()])
