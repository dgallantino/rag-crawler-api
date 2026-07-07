"""CLI for administrative tasks such as system user provisioning."""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from app.database import SessionLocal
from app.services.document_validation import validate_markdown_upload
from app.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    create_document_upload,
    get_document_status,
)
from app.services.system_user import (
    SystemUserLookupError,
    create_system_user,
    get_system_user_by_name,
)


def cmd_create_system_user(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        user, api_key = create_system_user(db, name=args.name, ratelimit=args.ratelimit)
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
        document = create_document_upload(
            db,
            user,
            path.name,
            content.decode("utf-8"),
        )
    except SystemUserLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
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
    create_parser.set_defaults(func=cmd_create_system_user)

    upload_parser = subparsers.add_parser("upload-document", help="Upload a markdown document")
    upload_parser.add_argument("--path", required=True, help="Path to local .md file")
    upload_parser.add_argument("--name", required=True, help="System user name")
    upload_parser.set_defaults(func=cmd_upload_document)

    status_parser = subparsers.add_parser("document-status", help="Check document processing status")
    status_parser.add_argument("--document-id", required=True, help="Document UUID")
    status_parser.add_argument("--name", required=True, help="System user name")
    status_parser.set_defaults(func=cmd_document_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
