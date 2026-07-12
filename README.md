# RAG Crawler API

A scaffold for a web-crawling and RAG (Retrieval-Augmented Generation) API. This repo provides the initial project structure and stub code — not a complete implementation.

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| Validation | Pydantic |
| Database | SQLAlchemy + PostgreSQL |
| Background jobs | Celery + Redis |
| Crawler | Custom pipeline (Playwright + BeautifulSoup) |
| RAG | Custom pipeline |
| Tests | pytest |

## Project Structure

```
app/
├── main.py          # FastAPI entry point
├── config.py        # Environment settings
├── database.py      # SQLAlchemy engine and session
├── models.py        # ORM models
├── jobs.py          # Celery tasks
├── services.py      # Business logic layer
├── utils.py         # Shared helpers
├── api/             # HTTP route handlers
├── schemas/         # Pydantic request/response models
├── crawler/         # Composable crawl pipeline library
└── rag/             # RAG pipeline
tests/               # pytest suite
```

## Quick Start (Local)

```bash
cd rag-crawler-api
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Quick Start (Docker)

```bash
cp .env.example .env
docker compose up --build
```

Services: `api` (port 8000), `postgres` (5432), `redis` (6379), `celery-worker`.

## Running Tests

Tests use an ephemeral PostgreSQL instance (pgvector) via [testcontainers-python](https://testcontainers-python.readthedocs.io/). **Docker must be running.**

```bash
source venv/bin/activate
pip install -r requirements-test.txt
pytest
```

By default, live e2e tests (real OpenRouter calls) are **excluded**. To run them:

```bash
cp .env.test.example .env.test
# Set OPENROUTER_API_KEY in .env.test

RUN_E2E=1 pytest -m e2e
```

E2e tests write a YAML test report (request, response, database dumps) to `E2E_REPORT_DIR` from `.env.test`, or `/tmp` if unset. Example: `/tmp/e2e_query_20260712T135600Z_test_query_backend_live_openrouter.yaml`.

## What's Implemented vs Stubbed

| Component | Status |
|-----------|--------|
| FastAPI app + health endpoint | Working |
| Settings from `.env` | Working |
| SQLAlchemy models + session | Stub (tables auto-created in dev) |
| Celery tasks | Partial (process_document working; crawl_url scaffold) |
| Custom crawler | Pipeline library (CLI debug available) |
| RAG pipeline | Working (chunk, embed, store) |

## Crawler CLI (local debugging)

Run the default pipeline (`JSCrawler → DOMChunker → ChunkPrinter`) without DB or API keys:

```bash
python -m app.cli run-crawler --url https://example.com
python -m app.cli run-crawler --url https://example.com --max-pages 5 --no-headless
```

Requires `playwright install chromium` after `pip install`.

Production code should compose `Pipeline([...])` directly (see `app/services/crawl.py`), not import `app/crawler/runner.py`.

## Developer Notes — Planned Work (by priority)

### 1. Database migrations (Alembic)

Replace `Base.metadata.create_all()` with versioned migrations. Set up Alembic, generate an initial migration for the `Document` model, and wire `alembic upgrade head` into Docker startup. This is the foundation for any schema changes.

### 2. Custom crawler integration

Wire `DBStorage` to persist crawled pages to the `Document` model and chain `trigger_process_document`. Add `POST /crawl` API route. See `app/services/crawl.py` for the production pipeline skeleton.

### 3. RAG pipeline

Implement `RAGPipeline.index()` to chunk documents, generate embeddings, and store vectors. Implement `query()` for semantic search and optional LLM-augmented answers. Choose and integrate a vector store (e.g. pgvector, Chroma, Pinecone).

### 4. API endpoints for crawl and query

Add routes to trigger crawls (`POST /crawl`), list documents, and run RAG queries (`POST /query`). Wire routes through `services.py` with proper Pydantic schemas and error handling.

### 5. Auth, rate limiting, and production hardening

Add API key or OAuth authentication, request rate limits, structured logging, health checks that verify DB/Redis connectivity, and environment-specific config (no `create_all` in production).

### 6. CI/CD

GitHub Actions (or similar) for lint, test, and Docker image build/push on every PR and merge to main.
