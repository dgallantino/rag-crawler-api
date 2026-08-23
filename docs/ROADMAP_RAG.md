# Roadmap — RAG

**Date:** 2026-07-16  
**Parent index:** [ROADMAP.md](ROADMAP.md)  
**Sibling:** [ROADMAP_CRAWLER.md](ROADMAP_CRAWLER.md)

## 0. Snapshot

**Verdict:** Library-level RAG works end-to-end via CLI and Celery (chunk → embed → pgvector → optional Cohere rerank → LLM answer). Public HTTP routes for retrieve, query, documents, and collections are stubbed (501). README Planned Work item 3 (“implement RAG pipeline”) is obsolete.

**Scope:** `app/rag/*`, `app/services/rag.py`, `app/services/documents.py`, `app/services/collections.py`, query/document schemas and routes, related tests.

**Present today:**

| Capability | Where |
|------------|--------|
| Markdown chunk / embed / store | `MarkdownProcessor` (`app/rag/processor.py`) |
| Dense cosine retrieval + metadata/date filters | `app/rag/retrieval.py` |
| Optional Cohere rerank (`top_k * 4` over-fetch) | `app/rag/rerank.py`, `app/services/rag.py` |
| Context build + grounded LLM answer + `Source` list | `app/rag/generation.py` |
| Upload → Celery `process_document` | `app/services/documents.py`, `app/services/jobs.py` |
| CLI `retrieve` / `query` | `app/cli.py` |

---

## A — Main functionality (services)

Highest-priority gaps that unblock correct, tenant-safe RAG behavior:

1. **Tenant-safe retrieval** — ~~`owner_ref` in `Filters` is silently skipped~~ ~~Done: `retrieval_service` resolves collections via `get_collection_by_slug` (all of the user's collections when slug is `None`; raises if none). `retrieve` / `vector_search` only filter by collection UUID(s). `owner_ref` remains unused as a SQL filter.~~
2. **Apply `max_tokens_context` end-to-end** — ~~callers do not pass caps~~ ~~Done: `build_context(max_tokens=...)` via tiktoken; plumbed through `answer_with_retrieval` / `answer_service` / CLI `--max-tokens-context`.~~
3. **Honor `chunk_min_tokens`** — ~~unused in `MarkdownProcessor`~~ ~~Done: consecutive undersized chunks are merged (capped by `chunk_max_tokens`).~~
4. **Document / collection services vs HTTP** — ~~service layer for upload/status/collections is ready for CLI; wire to HTTP only after Milestone C contracts. No blocking service gaps for the markdown CLI path beyond the items above.~~
5. **Adjacent-chunk merge before context** — ~~TODO in `app/rag/generation.py`; improves answer coherence (also reinforced under B).~~

---

## B — Gold-standard improvements (accuracy / efficiency)

Already present: dense recall, Cohere rerank, partial metadata filters, structured `Source` sidecar.

Prioritized additions for this stack:

1. ~~**HNSW / IVFFlat** on `DocumentChunk.chunk_vector` — cosine search is currently an unindexed sequential sort. Implement and benchmark a vector index (assess current state, recommend HNSW, define scope and required tests, and perform a smoke test).~~
2. ~~**Hybrid search** — Combine Postgres FTS, dense vector retrieval, and RRF fusion (not BM25) into a unified hybrid search pipeline.~~
3. ~~**Contextualized embeddings on the markdown path** — prefix title/heading into embed text (crawler `EmbeddingInputBuilder` already does this; RAG markdown path stores raw chunk text).~~
4. **Adjacent-chunk merge** before building LLM context; then **query rewrite / multi-query** if recall still lags.
5. **Embedding / answer caching** via Redis (today Redis is used for job status only).
6. **Offline eval harness** — golden Q&A sets and metrics; do not rely only on optional live OpenRouter e2e smoke.

---

## C — REST API (contract first, then implementation)

### C1 — Contract freeze

Done. Live OpenAPI is FastAPI `/docs` (no checked-in yaml). Tenant is the authenticated `SystemUser` (API key; D1 wires `Depends`) — request bodies have no `user_id`. Retrieve-only is `POST /v1/retrieve` (`RetrievalResult`); `POST /v1/query` returns `RagResponse`. `/v1/agent/search` removed. Errors use `ErrorResponse` (`error`, `message`, `request_id`) including Pydantic 422, 409 conflict, and 501 stubs. Documents: upload, list, detail, PATCH (`title` and `content`; reindexes), status, and delete (chunks + job status).

### C2 — Implementation

Wire thin routes → existing services, in order:

1. `POST /v1/collections`
2. `POST /v1/documents`, `POST /v1/documents/json`, `GET /v1/documents`, `GET /v1/documents/{id}`, `PATCH /v1/documents/{id}`, `DELETE /v1/documents/{id}`, `GET /v1/documents/{id}/status`
3. `POST /v1/retrieve`
4. `POST /v1/query` (`retrieval_service` then `answer_service`)

Add HTTP API tests (suite today is ~70 tests, almost none on routes).

---

## D — Security (RAG-facing)

1. Wire API-key auth (`app/auth.py` + `SystemUser`) via FastAPI `Depends` **before** enabling query/upload routes.
2. Do not add body `user_id`; tenant comes from the authenticated system user.
3. Rate limiting on `/v1/retrieve` and `/v1/query` using `SystemUser.ratelimit`.
4. Bound caller-controlled metadata filter keys/values; treat prompt injection into retrieved context as an inherent RAG risk (grounded prompts help but do not eliminate it).

---

## E — Bugs / chores (RAG)

- Update stale README Planned Work / structure that still describe RAG as unimplemented.
- Fix docstring drift: `rerank.py` (still mentions LLM-as-reranker), `retrieve()` (claims orchestration it does not do).
- Fix `httpx2` entry in `requirements.txt` if incorrect (code imports `httpx`).
- Alembic migrations for vector/chunk schema (replace dev `create_all`).
- Close RAG API test gap; reduce reliance on live e2e-only evaluation.

---

## Suggested sequencing (RAG)

| Step | Focus |
|------|--------|
| **M1** | Tenant/collection scoping + `max_tokens_context` + `chunk_min_tokens` |
| **M2** | HNSW, hybrid search, contextual embeddings, adjacent-chunk merge |
| **M3** | Freeze retrieve/query/document contracts (OpenAPI from FastAPI `/docs`) |
| **M4** | Wire collections → documents → retrieve → query + API tests |
| **M5** | Auth, rate limits, filter bounds |
| **M6** | Docs, deps, Alembic, eval harness chores |
