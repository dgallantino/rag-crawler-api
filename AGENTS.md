# AGENTS.md

## Cursor Cloud specific instructions

This is a Python 3 (FastAPI) RAG system. The dev environment (Docker engine, a
`venv/` with `requirements-test.txt` installed, and a `.env`) is provisioned by
the startup update script; the notes below are the non-obvious runtime caveats.
Standard commands live in `README.md`; prefer those and only the deltas here.

### Services and how to run them

Run everything from the repo root using the project virtualenv at `venv/`
(e.g. `./venv/bin/python`, `./venv/bin/uvicorn`, `./venv/bin/celery`,
`./venv/bin/pytest`).

- Postgres (pgvector) + Redis: `docker compose up -d postgres redis`. Do NOT run
  the full `docker compose up` for local dev — the `api`/`celery-worker`
  compose services rebuild the image and don't use `venv/`. Just run those two
  processes locally instead.
- API: `./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000`. On startup
  in `APP_ENV=development` it auto-creates tables and the pgvector extension, so
  no migrations are needed. Docs at `/docs`, health at `/health` and
  `/health/db`.
- Celery worker (required for document ingestion): the correct app path is
  `./venv/bin/celery -A app.services.jobs:celery_app worker --loglevel=info`
  (the `README` mentions `app.celery_app`, which does not exist — use the
  `docker-compose.yml` path).

### Docker caveats (tests + infra)

- The test suite spins up an ephemeral `pgvector/pgvector:pg16` container via
  testcontainers, so the Docker daemon must be running. If `docker ps` fails,
  start it with `sudo service docker start`.
- The Docker daemon uses the `fuse-overlayfs` storage driver and
  `iptables-legacy` (configured during environment setup) to work inside the
  Cloud VM.
- The `ubuntu` user is in the `docker` group. In a brand-new shell that already
  works; if you hit `permission denied` on `/var/run/docker.sock` in an existing
  session, run `sudo chmod 666 /var/run/docker.sock` (resets on daemon restart).

### OpenRouter API key (external, gated)

- Embeddings, reranking, and answer generation call OpenRouter and require
  `OPENROUTER_API_KEY` in `.env`. Without it, `create_openai_client` raises
  `RuntimeError("OPENROUTER_API_KEY is not configured")`, so document ingestion
  (Celery `run_process_document`), CLI `retrieve`/`query`, and live `e2e` tests
  cannot complete. Everything else (services, health, tenant/collection/document
  provisioning, the default test suite) works without it because unit tests mock
  OpenRouter.

### Tests

- `./venv/bin/pytest` runs the default suite (107 tests). `e2e` (live OpenRouter)
  and `slow` markers are excluded via `pytest.ini`. Run live e2e with
  `RUN_E2E=1 ./venv/bin/pytest -m e2e` only when `OPENROUTER_API_KEY` is set.

### Notes

- Playwright/Chromium is a declared dependency for the crawler, but the crawler
  is skeletal and unwired (`app/cli.py run-crawler` raises `NotImplementedError`),
  and its tests mock the pipeline — so a Chromium browser download is not needed
  for tests or the working RAG flow.
- HTTP `/v1/*` business routes are wired to services (API-key auth via
  `Authorization: Bearer`). The CLI (`python -m app.cli ...`) plus Celery remain
  valid for local debugging.
