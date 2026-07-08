from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin, urlparse
from uuid import uuid4

from bs4 import BeautifulSoup
from playwright.sync_api import Page, sync_playwright

from .pipeline import BaseStage, Item, StageContext
from .schemas import CrawlerData


logger = logging.getLogger(__name__)


class JSCrawler(BaseStage):
    """
    Source-like stage.

    In the future this can:
    - read seed URLs from context
    - fetch pages
    - yield one Item per crawled page
    """

    def __init__(self, max_pages: int = 20, headless: bool = True):
        super().__init__()
        self.max_pages = max_pages
        self.headless = headless

    def is_valid_url(self, url: str, allowed_domains: set[str]) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and parsed.netloc in allowed_domains

    def normalize_url(self, url: str) -> str:
        return url.split("#")[0].rstrip("/")

    def extract_links(
        self,
        html: str,
        current_url: str,
        allowed_domains: set[str],
    ) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            absolute_url = urljoin(current_url, href)
            absolute_url = self.normalize_url(absolute_url)

            if self.is_valid_url(absolute_url, allowed_domains):
                links.append(absolute_url)

        return list(dict.fromkeys(links))

    def render_page(self, page: Page, url: str) -> str | None:
        try:
            logger.info("visiting url: %s", url)

            page.goto(url, wait_until="domcontentloaded", timeout=30000)

            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                logger.warning("networkidle timeout: %s", url)

            # Prefer meaningful selector when possible
            try:
                page.wait_for_selector("body", timeout=10000)
            except Exception:
                pass

            previous_height = 0
            stable_rounds = 0

            for _ in range(12):
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(750)

                current_height = page.evaluate("document.body.scrollHeight")

                if current_height == previous_height:
                    stable_rounds += 1
                    if stable_rounds >= 2:
                        break
                else:
                    stable_rounds = 0

                previous_height = current_height

            # Return to top if later extraction assumes natural page order
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)

            return page.content()

        except Exception:
            logger.exception("failed to render page: %s", url)
            return None

    def _initial_stats(self) -> dict[str, int]:
        return {
            "pages_seen": 0,
            "pages_crawled": 0,
            "pages_failed": 0,
            "links_discovered": 0,
            "pages_skipped_duplicate": 0,
        }

    def _prepare_job(self, context: StageContext) -> tuple[str, list[str], set[str], dict[str, int]]:
        seed_urls = context.get("seed_urls")
        if not seed_urls or isinstance(seed_urls, str):
            raise ValueError("JSCrawler requires context seed_urls as a non-empty list.")

        normalized_seed_urls = list(dict.fromkeys(self.normalize_url(url) for url in seed_urls))
        invalid_seed_urls = []
        allowed_domains = set()
        for url in normalized_seed_urls:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                invalid_seed_urls.append(url)
                continue
            allowed_domains.add(parsed.netloc)

        if invalid_seed_urls:
            raise ValueError("JSCrawler seed_urls must contain only http(s) URLs.")
        if not allowed_domains:
            raise ValueError("JSCrawler requires at least one http(s) seed URL.")

        job_id = context.get("job_id") or str(uuid4())
        context.set("job_id", job_id)

        jobs = context.get("jobs", {})
        stats = self._initial_stats()
        jobs[job_id] = {
            "seed_urls": normalized_seed_urls,
            "stats": stats,
        }
        context.set("jobs", jobs)

        return job_id, normalized_seed_urls, allowed_domains, stats

    def _crawl_pages(
        self,
        page: Page,
        seed_urls: list[str],
        allowed_domains: set[str],
        stats: dict[str, int],
    ) -> Iterator[Item[CrawlerData]]:
        queue = deque(seed_urls)
        visited: set[str] = set()

        while queue and len(visited) < self.max_pages:
            url = queue.popleft()
            url = self.normalize_url(url)
            stats["pages_seen"] += 1

            if url in visited:
                stats["pages_skipped_duplicate"] += 1
                continue

            visited.add(url)

            html = self.render_page(page, url)
            if not html:
                stats["pages_failed"] += 1
                continue

            links = self.extract_links(html, url, allowed_domains)
            stats["links_discovered"] += len(links)
            for link in links:
                if link not in visited:
                    queue.append(link)

            stats["pages_crawled"] += 1
            yield Item(
                data=CrawlerData(url=url, text=html, title=None),
                meta={
                    "visit_time": datetime.now(),
                    "stage": self.stage_name,
                },
            )

    def crawl(
        self,
        seed_urls: list[str],
        allowed_domains: set[str],
        stats: dict[str, int],
    ) -> Iterator[Item[CrawlerData]]:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()
            try:
                yield from self._crawl_pages(page, seed_urls, allowed_domains, stats)
            finally:
                logger.info("closing playwright")
                context.close()
                browser.close()
    
    def run(
        self,
        items: Iterable[Item[Any]],
        context: StageContext,
    ) -> Iterator[Item[Any]]:
        # Consume upstream trigger items, but ignore their content.
        job_id, seed_urls, allowed_domains, stats = self._prepare_job(context)
        logger.info("starting crawl job %s with %s seed URLs", job_id, len(seed_urls))
        yield from self.crawl(seed_urls, allowed_domains, stats)
