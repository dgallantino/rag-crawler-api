"""CLI for administrative tasks such as system user provisioning."""

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import UUID

from app.config import get_settings
from app.database import SessionLocal, create_all_tables, drop_all_tables
import app.models  # noqa: F401 — register ORM models with Base.metadata
from app.services.collections import (
    CollectionConflictError,
    CollectionNotFoundError,
    create_collection,
    get_collection_by_slug,
)
from app.services.documents import validate_markdown_upload
from app.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    create_document_upload,
    get_document_status,
)
from app.services.rag import answer_service, chunks_to_retrieval_result, retrieval_service
from app.services.system_user import (
    SystemUserLookupError,
    create_system_user,
    get_system_user_by_name,
)


def cmd_run_crawler(args: argparse.Namespace) -> int:
    """
    Run the crawler pipeline locally for debugging (no DB required).
    This is a temporary command to help with debugging the crawler pipeline.
    It will be removed once the pipeline is fully implemented.
    or replaced with a production-grade command
    """
    raise NotImplementedError("Crawler pipeline is not implemented yet")
    from app.crawler.pipeline import DataRetrieverError
    from app.crawler.runner import count_crawl_results, run_crawl_debug
    from app.crawler.settings import DEFAULT_MAX_PAGES

    try:
        results = run_crawl_debug(
            args.url,
            max_pages=args.max_pages,
            headless=not args.no_headless,
            job_id=args.job_id,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except DataRetrieverError as exc:
        print(f"error: crawl pipeline failed: {exc}", file=sys.stderr)
        return 1

    for item in results:
        if isinstance(item, str):
            print(item)

    pages, chunks = count_crawl_results(results)
    print(f"\ncrawl complete: {pages} page(s), {chunks} chunk(s) printed", file=sys.stderr)
    return 0


def _parse_collection_arg(value: str) -> tuple[str, str | None]:
    """Parse a 'name' or 'name:slug' collection argument into (name, slug)."""
    if ":" in value:
        name, slug = value.split(":", 1)
        return name.strip(), slug.strip() or None
    return value.strip(), None


def cmd_create_collection(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        user = get_system_user_by_name(db, args.name)
        name, slug = _parse_collection_arg(args.collection)
        try:
            col = create_collection(db, user, name=name, slug=slug)
        except CollectionConflictError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    except SystemUserLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(
        json.dumps(
            {
                "id": str(col.id),
                "name": col.name,
                "slug": col.slug,
                "system_user_id": str(col.system_user_id),
            }
        )
    )
    return 0


def cmd_create_system_user(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        user, api_key = create_system_user(db, name=args.name, ratelimit=args.ratelimit)

        collections_output = []
        for raw in args.collection or []:
            name, slug = _parse_collection_arg(raw)
            try:
                col = create_collection(db, user, name=name, slug=slug)
            except CollectionConflictError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            collections_output.append(
                {"id": str(col.id), "name": col.name, "slug": col.slug}
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(
        json.dumps(
            {
                "id": str(user.id),
                "name": user.name,
                "api_key": api_key,
                "ratelimit": user.ratelimit,
                "created_at": user.created_at.isoformat(),
                "collections": collections_output,
            }
        )
    )
    return 0


def cmd_upload_document(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    content = path.read_bytes()
    validation = validate_markdown_upload(path.name, content)
    if not validation.valid:
        print(f"error: {validation.reason}", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        user = get_system_user_by_name(db, args.name)
        collection = get_collection_by_slug(db, user, args.collection_slug)
        document = create_document_upload(
            db,
            collection,
            path.name,
            content.decode("utf-8"),
        )
    except SystemUserLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except CollectionNotFoundError as exc:
        print(f"error: collection not found: {exc}", file=sys.stderr)
        return 1
    except DocumentConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(
        json.dumps(
            {
                "document_id": str(document.id),
                "filename": path.name,
                "accepted": True,
            }
        )
    )
    return 0


def _parse_filters(value: str | None) -> dict | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for filters: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("filters must be a JSON object")
    return parsed


def _add_rag_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--name", required=True, help="System user name")
    parser.add_argument(
        "--collection-slug",
        default=None,
        help="Optional collection slug to restrict search to",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply a reranking pass after vector retrieval",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help='Optional JSON object of retrieval filters, e.g. \'{"metadata": {"doc_type": "contract"}}\'',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )


def cmd_retrieve(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        user = get_system_user_by_name(db, args.name)
        try:
            filters = _parse_filters(args.filters)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        collection_id: str | None = None
        if args.collection_slug is not None:
            try:
                collection = get_collection_by_slug(db, user, args.collection_slug)
            except CollectionNotFoundError as exc:
                print(f"error: collection not found: {exc}", file=sys.stderr)
                return 1
            collection_id = str(collection.id)

        start = time.monotonic()
        candidates = retrieval_service(
            query=args.query,
            top_k=args.top_k,
            filters=filters,
            collection=collection_id,
            use_rerank=args.rerank,
            session=db,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        result = chunks_to_retrieval_result(
            args.query,
            candidates,
            top_k=args.top_k,
            use_rerank=args.rerank,
            latency_ms=elapsed_ms,
        )
    except SystemUserLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    if args.json:
        print(result.model_dump_json())
        return 0

    result_dict = result.model_dump()
    for i, res in enumerate(result_dict.get("results", []), 1):
        print(f"\nResult {i}:")
        print(f"score: {res['score']}")
        print("=" * 80)
        print(res["text"])
        print("=" * 80)
    print(f"latency_ms: {result_dict['latency_ms']}")
    print(f"top_k: {result_dict['top_k']}")
    print(f"reranked: {result_dict['reranked']}")
    print(f"query_used: {result_dict['query_used']}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        user = get_system_user_by_name(db, args.name)
        try:
            filters = _parse_filters(args.filters)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        collection_id: str | None = None
        if args.collection_slug is not None:
            try:
                collection = get_collection_by_slug(db, user, args.collection_slug)
            except CollectionNotFoundError as exc:
                print(f"error: collection not found: {exc}", file=sys.stderr)
                return 1
            collection_id = str(collection.id)

        candidates = retrieval_service(
            query=args.query,
            top_k=args.top_k,
            filters=filters,
            collection=collection_id,
            use_rerank=args.rerank,
            session=db,
        )
        response = answer_service(args.query, candidates)
    except SystemUserLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(response.model_dump_json())
    return 0


def cmd_document_status(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        user = get_system_user_by_name(db, args.name)
        status = get_document_status(db, user, UUID(args.document_id))
    except SystemUserLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except DocumentNotFoundError as exc:
        print(f"error: document not found: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(status.model_dump_json())
    return 0


def _require_development() -> int | None:
    if get_settings().app_env != "development":
        print(
            "error: db schema commands only allowed when APP_ENV=development",
            file=sys.stderr,
        )
        return 1
    return None


def cmd_db_create_all(args: argparse.Namespace) -> int:
    if (err := _require_development()) is not None:
        return err
    try:
        create_all_tables()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("tables created")
    return 0


def cmd_db_delete_all(args: argparse.Namespace) -> int:
    if (err := _require_development()) is not None:
        return err
    try:
        drop_all_tables()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("tables dropped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG Crawler API administrative CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-system-user", help="Provision a new system user")
    create_parser.add_argument("--name", required=True, help="Display name for the tenant")
    create_parser.add_argument(
        "--ratelimit",
        type=int,
        default=100,
        help="Requests-per-minute limit (default: 100)",
    )
    create_parser.add_argument(
        "--collection",
        action="append",
        metavar="NAME[:SLUG]",
        help="Create a collection (repeatable); slug auto-derived from name when omitted",
    )
    create_parser.set_defaults(func=cmd_create_system_user)

    collection_parser = subparsers.add_parser(
        "create-collection", help="Create a collection for an existing system user"
    )
    collection_parser.add_argument("--name", required=True, help="System user name")
    collection_parser.add_argument(
        "--collection",
        required=True,
        metavar="NAME[:SLUG]",
        help="Collection name; slug auto-derived from name when omitted",
    )
    collection_parser.set_defaults(func=cmd_create_collection)

    upload_parser = subparsers.add_parser("upload-document", help="Upload a markdown document")
    upload_parser.add_argument("--path", required=True, help="Path to local .md file")
    upload_parser.add_argument("--name", required=True, help="System user name")
    upload_parser.add_argument(
        "--collection-slug",
        required=True,
        help="Slug of the collection to upload into",
    )
    upload_parser.set_defaults(func=cmd_upload_document)

    status_parser = subparsers.add_parser("document-status", help="Check document processing status")
    status_parser.add_argument("--document-id", required=True, help="Document UUID")
    status_parser.add_argument("--name", required=True, help="System user name")
    status_parser.set_defaults(func=cmd_document_status)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve relevant chunks for a query")
    _add_rag_args(retrieve_parser)
    retrieve_parser.set_defaults(func=cmd_retrieve)

    query_parser = subparsers.add_parser(
        "query", help="Retrieve relevant chunks and generate a grounded answer"
    )
    _add_rag_args(query_parser)
    query_parser.set_defaults(func=cmd_query)

    db_parser = subparsers.add_parser(
        "db", help="Development database schema helpers (APP_ENV=development only)"
    )
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)

    db_create_parser = db_subparsers.add_parser(
        "create-all", help="Create the vector extension and all ORM tables"
    )
    db_create_parser.set_defaults(func=cmd_db_create_all)

    db_delete_parser = db_subparsers.add_parser(
        "delete-all", help="Drop all ORM tables"
    )
    db_delete_parser.set_defaults(func=cmd_db_delete_all)

    from app.crawler.settings import DEFAULT_MAX_PAGES

    crawler_parser = subparsers.add_parser(
        "run-crawler",
        help="Run the crawler pipeline locally for debugging (no DB required)",
    )
    crawler_parser.add_argument(
        "--url",
        action="append",
        required=True,
        dest="url",
        help="Seed URL to crawl (repeatable)",
    )
    crawler_parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Maximum pages to crawl (default: {DEFAULT_MAX_PAGES})",
    )
    crawler_parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser with a visible window",
    )
    crawler_parser.add_argument(
        "--job-id",
        default=None,
        help="Optional correlation ID for logs",
    )
    crawler_parser.set_defaults(func=cmd_run_crawler)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
