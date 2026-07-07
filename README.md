# RAG Crawler API

A scaffold for a web-crawling and RAG (Retrieval-Augmented Generation) API. This repo provides the initial project structure and stub code — not a complete implementation.

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| Validation | Pydantic |
| Database | SQLAlchemy + PostgreSQL |
| Background jobs | Celery + Redis |
| Crawler | Custom (BeautifulSoup + httpx, stub) |
| RAG | Custom pipeline (stub) |
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
├── crawler/         # Custom crawler config (stub)
└── rag/             # RAG pipeline (stub)
tests/               # pytest suite
```

## Quick Start (Local)

```bash
cd rag-crawler-api
source venv/bin/activate
pip install -r requirements.txt
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

```bash
source venv/bin/activate
pytest
```

## What's Implemented vs Stubbed

| Component | Status |
|-----------|--------|
| FastAPI app + health endpoint | Working |
| Settings from `.env` | Working |
| SQLAlchemy models + session | Stub (tables auto-created in dev) |
| Celery tasks | Stub (placeholder tasks) |
| Custom crawler | Stub (settings only) |
| RAG pipeline | Stub (`NotImplementedError`) |

## Developer Notes — Planned Work (by priority)

### 1. Database migrations (Alembic)

Replace `Base.metadata.create_all()` with versioned migrations. Set up Alembic, generate an initial migration for the `Document` model, and wire `alembic upgrade head` into Docker startup. This is the foundation for any schema changes.

### 2. Custom crawler implementation

Add fetch-and-parse logic under `app/crawler/` using httpx (or requests) and BeautifulSoup. Connect the `crawl_url` Celery task to fetch pages, extract content, and persist results to the `Document` model. Add rate limiting, robots.txt checks, and HTML cleaning as needed.

### 3. RAG pipeline

Implement `RAGPipeline.index()` to chunk documents, generate embeddings, and store vectors. Implement `query()` for semantic search and optional LLM-augmented answers. Choose and integrate a vector store (e.g. pgvector, Chroma, Pinecone).

### 4. API endpoints for crawl and query

Add routes to trigger crawls (`POST /crawl`), list documents, and run RAG queries (`POST /query`). Wire routes through `services.py` with proper Pydantic schemas and error handling.

### 5. Auth, rate limiting, and production hardening

Add API key or OAuth authentication, request rate limits, structured logging, health checks that verify DB/Redis connectivity, and environment-specific config (no `create_all` in production).

### 6. CI/CD

GitHub Actions (or similar) for lint, test, and Docker image build/push on every PR and merge to main.
