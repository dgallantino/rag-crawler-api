# Crawler Pipeline — Developer Guidelines

This package uses a **lazy iterator pipeline**: each stage receives a stream of `Item`s and yields a stream of `Item`s. Stages are composed linearly via `Pipeline`.

```
Item[T] stream → Stage → Item[U] stream → Stage → …
                      ↘ StageContext (read-only config) ↗
```

The goal is a **composable library** with a clear protocol. Follow these rules when adding or changing stages.

---

## Core rules

### 1. Data flows through `Item`, not `StageContext`

- **`Item.data`** — the record payload (`CrawlerData`, `PageChunk`, `EmbeddingInput`, …). This is the primary input/output between stages.
- **`Item.meta`** — per-record metadata (provenance, flags, timings). Copy forward with `{**item.meta, ...}` when enriching.
- **`StageContext`** — **pipeline-wide configuration only** (seed URLs, job ID, limits, feature flags). Set once before `Pipeline.run()` / `Pipeline.iter()`.

**Do**

```python
for item in items:
    yield Item(data=new_payload, meta={**item.meta, "stage": self.stage_name})
```

**Don't**

```python
# Passing business data between stages via context
context.set("last_url", item.data.url)
url = context.get("last_url")  # next stage reads it — wrong channel
```

### 2. Keep stages small and single-purpose

One stage = one responsibility. If a class needs a comment like "and also …", split it.

| Kind | Role | Examples |
|------|------|----------|
| **Source** | Produce the first real items (may ignore upstream trigger) | `JSCrawler` |
| **Extract** | Parse or transform `Item.data` | `JSTextExtractor` |
| **Filter** | Drop items that fail a predicate | `RemoveEmptyPages` |
| **Transform** | Map one item type to another | `EmbeddingInputBuilder` |
| **Sink** | Persist or hand off; may still yield items downstream | `FileStorage` |

Aim for stages that fit in one screen (~50–100 lines of `run()` logic). Extract helpers privately or into separate stage classes.

### 3. Prefer new objects over in-place mutation

When changing `Item.data`, create a new dataclass instance instead of mutating fields on the incoming object. Downstream stages and tests can reason about immutability.

```python
# Good
yield Item(
    data=CrawlerData(url=item.data.url, title=title, text=text),
    meta={**item.meta, "html_parsed": True},
)

# Avoid
item.data.text = text
item.data.title = title
yield Item(data=item.data, meta=item.meta)
```

### 4. Always yield `Item`, never raw values

`BaseStage.run()` must return `Iterator[Item[Any]]`. Terminal formatters should wrap output in a schema (e.g. `FormattedOutput`) or write to stdout/files inside the stage and yield the original `Item` with updated `meta`.

Dropping the `Item` wrapper breaks composability — nothing can be chained after that stage.

### 5. Filters drop silently; log when you do

Filter stages `continue` without yielding. Always log at warning level so operators can diagnose empty pipelines.

### 6. Document the type contract in the class docstring

State expected input and output types explicitly:

```python
class MyStage(BaseStage):
    """Filter: Item[CrawlerData] → Item[CrawlerData] (drops empty pages)."""
```

The framework does not enforce types yet; the docstring is the contract.

### 7. `StageContext` keys are config, not a data store

Allowed context keys (examples):

| Key | Set by | Purpose |
|-----|--------|---------|
| `seed_urls` | Caller | Crawl entry points |
| `job_id` | Caller or source stage | Correlation ID |
| `max_pages` | Caller | Global limit (if not on stage ctor) |

Avoid writing per-page results, stats blobs, or intermediate artifacts to context. Put those on `Item.meta` or a dedicated sink stage.

---

## Recommended pipeline shapes

**RAG ingestion (target)**

```
JSCrawler → JSTextExtractor → RemoveEmptyPages → RemoveImageNames
         → DOMChunker* → EmbeddingInputBuilder → FileStorage
```

**Debug / inspect**

```
JSCrawler → JSTextExtractor → DOMChunker* → ChunkPrinter*
```

\* See known deviations below.

---

## Warnings

1. **Order matters.** There is no runtime type check. `ChunkPrinter` after `JSCrawler` (skipping chunking) will fail at runtime. Add integration tests for each supported pipeline recipe.

2. **Do not grow `StageContext`.** It becomes a hidden global. If two stages need the same derived data, pass it on `Item.meta` or add an explicit stage between them.

3. **Do not add fat stages.** Large monolithic stages (parsing + heuristics + merging + scoring in one class) are hard to test, reorder, and disable. Split instead.

4. **Source stages are special but still bounded.** `JSCrawler` may ignore upstream `Item.data`, but it must not read business payloads from context — only config (`seed_urls`, `job_id`).

5. **1→N expansion is fine.** A stage may yield multiple items per input (e.g. one page → many chunks). Keep each output `Item` self-contained.

6. **Side effects belong in Source or Sink stages.** Network I/O, DB writes, and file persistence should not hide inside transform/filter stages.

---

## Known deviations (fix when touching these files)

These classes work today but do not fully match the intended design. Refactor toward the fix when you change them.

| Class | File | Issue | Fix |
|-------|------|-------|-----|
| **`DOMChunker`** | `extractors.py` | Fat stage (~700 lines): noise removal, content-root detection, block extraction, scoring, merging, and chunk assembly in one class. | Split into focused stages, e.g. `NoiseStripper` → `ContentRootResolver` → `BlockExtractor` → `ChunkMerger` → `PageChunkEmitter`. Share HTML parsing helpers as private functions or a small non-stage utility module — not as one mega-stage. |
| **`JSCrawler`** | `sources.py` | Reads `seed_urls` from context (OK for config) but also writes `jobs` / stats dict back to context — using context as output. Ignores upstream items (acceptable for sources). | Keep only config reads (`seed_urls`, `job_id`). Emit stats on each `Item.meta` (e.g. `crawl_stats`) or yield a final summary `Item`. Do not store per-job state in `context.data["jobs"]`. |
| **`JSTextExtractor`** | `extractors.py` | Mutates `item.data` in place instead of producing a new `CrawlerData`. | Build and yield a fresh `CrawlerData(url=..., title=..., text=...)`. |
| **`DocFromCrawler`** | `formatters.py` | Yields raw `str`, not `Item`. Return type annotation does not match behaviour. | Yield `Item(data=FormattedDocument(text=...), meta=item.meta)` or write to stdout inside the stage and re-yield the input `Item` with `meta["printed"] = True`. |
| **`ChunkPrinter`** | `formatters.py` | Same as above — yields `str`, breaking the pipeline contract. | Same pattern: wrap in a schema or side-effect + re-yield `Item`. |
| **`FileStorage`** | `storage.py` | Placeholder only; sets `meta["stored"] = True` without persisting. | Implement real I/O in `run()`, then yield the `Item` (or a copy) with storage metadata (path, record ID). |

### Stages that follow the spirit (use as templates)

- **`RemoveEmptyPages`** / **`RemoveImageNames`** — small, single predicate, clear input/output type.
- **`EmbeddingInputBuilder`** — maps `PageChunk` → `EmbeddingInput`, preserves `meta`, no context reads.

---

## Checklist for new stages

- [ ] One clear responsibility; `run()` fits on one screen
- [ ] Input/output types documented in the class docstring
- [ ] Yields `Item` only — never raw strings or dicts
- [ ] Does not read business data from `StageContext`
- [ ] Does not write per-item results to `StageContext`
- [ ] Produces new dataclass instances instead of mutating incoming `data`
- [ ] Copies `item.meta` forward when enriching
- [ ] Has a unit test with a minimal fake `Item` (no Playwright required unless it is a source stage)

---

## File layout convention

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Framework (`Item`, `StageContext`, `BaseStage`, `Pipeline`) — avoid domain logic here |
| `schemas.py` | Dataclass payloads exchanged between stages |
| `sources.py` | Stages that fetch or ingest external data |
| `extractors.py` | Parse / transform raw content |
| `filters.py` | Drop items |
| `formatters.py` | Reshape for output or downstream consumers |
| `storage.py` | Persistence sinks |

When a module grows many stages, prefer splitting by responsibility (e.g. `chunking.py`, `html_cleanup.py`) over adding more classes to an already large file.
