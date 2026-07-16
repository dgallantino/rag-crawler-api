# Roadmap — Crawler

**Date:** 2026-07-16  
**Parent index:** [ROADMAP.md](ROADMAP.md)  
**Sibling:** [ROADMAP_RAG.md](ROADMAP_RAG.md)

## 0. Snapshot

**Verdict:** The crawler library can fetch and chunk pages (Playwright BFS + DOM extraction). Product ingest is unfinished: `DBStorage` does not write Documents, and `run_crawl_for_url` returns `"status": "stub"`. There is no crawl HTTP route yet. Debug CLI (`run-crawler`) works without DB.

**Scope:** `app/crawler/*`, `app/services/crawl.py`, future crawl API route/schemas, CLI `run-crawler`.

**Present today:**

| Capability | Where |
|------------|--------|
| Playwright domain-scoped BFS crawl | `JSCrawler` (`app/crawler/sources.py`) |
| HTML → text / char chunks | `DOMChunker`, extractors (`app/crawler/extractors.py`) |
| Filters (empty pages, image-name noise) | `app/crawler/filters.py` |
| Title/heading-prefixed embed prep | `EmbeddingInputBuilder` (`app/crawler/formatters.py`) |
| Production pipeline skeleton | `app/services/crawl.py` |
| Debug CLI path (no persist) | `app/crawler/runner.py`, `app.cli run-crawler` |
| Storage stages | `DBStorage` / `FileStorage` — **placeholders** (`app/crawler/storage.py`) |

---

## A — Main functionality (services)

Highest-priority gaps that unblock crawl → index:

1. **Implement `DBStorage`** — upsert `Document` (url, title, content) for `collection_id`, set status `pending`, commit (`app/crawler/storage.py` TODO).
2. **Finish `run_crawl_for_url`** — after pipeline: persist results, call `trigger_process_document` for created/updated docs, clear `"status": "stub"` (`app/services/crawl.py`).
3. **Unify ingest handoff** — define a single contract from crawler output → `Document.content` shape that `MarkdownProcessor` (or a dedicated HTML processor) can index. Prefer storing full page text and letting RAG chunk once, or explicitly align DOM char chunking with token chunking.
4. **Celery crawl job + job_status** — if crawls are async, add a worker task and Redis step tracking for the crawl pipeline (mirror `process_document`).
5. **Production composition only via `services/crawl.py`** — do not import `runner.py` from product paths (`runner.py` is temporary / CLI-only).

---

## B — Gold-standard improvements (accuracy / efficiency)

1. **URL normalize + dedup** for upserts — `normalize_url` in `app/utils.py` is a documented placeholder (strip only).
2. **Robots / politeness / concurrency limits** — rate, delay, and max concurrency per host.
3. **Align chunking strategies** — DOM char targets (`target_chars` / `max_chars`) vs RAG token splitter; avoid double-chunking or mismatched units (ties to A.3).
4. **Incremental recrawl / change detection** — skip re-index when content hash unchanged.
5. **Playwright efficiency** — browser/context reuse, headless defaults, retries on transient failures.
6. **Extraction quality** — tables, nav/footer noise beyond current `DOMChunker` / filter stages.

---

## C — REST API (contract first, then implementation)

### C1 — Contract freeze

Define crawl API **before** coding handlers:

- Method/path (e.g. `POST /v1/crawl`)
- Request: url, collection id/slug, `max_pages`, headless/options
- Response: job_id, status, accepted counts; async status shape if queued
- Shared `ErrorResponse` envelope (align with RAG routes)

No crawl route exists today; README mentions `POST /crawl` as planned.

### C2 — Implementation

- Thin route → `run_crawl_for_url` and/or Celery crawl task
- Status endpoint if async
- API tests for happy path, validation, and auth (once D lands)

---

## D — Security (crawler-facing)

1. **SSRF controls** — block private, link-local, and cloud metadata IPs (`127.0.0.1`, RFC1918, `169.254.169.254`, etc.); optional domain allowlist. Today only same-`netloc` link following restricts crawl breadth, not the seed URL itself.
2. **Auth on crawl endpoint** — same API-key / `SystemUser` model as other `/v1` routes.
3. **Rate limit and `max_pages` caps** per tenant.
4. **URL / metadata sanitization** when persisting Documents (no unsafe or misleading stored URLs).

---

## E — Bugs / chores (crawler)

- Remove or quarantine temporary `app/crawler/runner.py` once production path is complete.
- Bare `pass` on selector wait timeout in `JSCrawler` (swallows failures silently).
- `FileStorage` placeholder; address GUIDELINES.md debt (`JSTextExtractor` mutates in place).
- Install Playwright browsers in the Docker image (image currently lacks browser install).
- Crawler integration tests beyond runner mocks (persist + trigger index).

---

## Suggested sequencing (crawler)

| Step | Focus |
|------|--------|
| **M1** | `DBStorage` + crawl → Document → `process_document` E2E |
| **M2** | URL normalize/dedup, chunk alignment, politeness basics |
| **M3** | Freeze crawl request/response contract |
| **M4** | Wire crawl API (+ status if async) + tests |
| **M5** | SSRF, auth, per-tenant caps |
| **M6** | Runner cleanup, Docker Playwright, GUIDELINES debt, richer tests |
