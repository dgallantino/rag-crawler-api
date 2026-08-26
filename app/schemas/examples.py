"""Canonical OpenAPI example payloads (RS Sehat Sentosa).

Placeholder IDs are documentation-only; they are not real database rows.
"""

from __future__ import annotations

from typing import Any

COLLECTION_ID = "11111111-1111-4111-8111-111111111111"
DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"
CHUNK_ID = "33333333-3333-4333-8333-333333333333"
CREATED_AT = "2024-01-15T08:00:00Z"
REQUEST_ID = "req-a1b2c3d4"
COLLECTION_NAME = "RS Sehat Sentosa"
COLLECTION_SLUG = "rs-sehat-sentosa"
DOCUMENT_FILENAME = "rs_sehat_sentosa_jadwal_operasional.md"
DOCUMENT_TITLE = "Jadwal Operasional RS Sehat Sentosa"
QUERY_VISIT = "Jam kunjungan rawat inap RS Sehat Sentosa?"
QUERY_IGD = "Apakah IGD RS Sehat Sentosa buka 24 jam?"

DOCUMENT_MARKDOWN = """# Jadwal Operasional RS Sehat Sentosa

## Instalasi Gawat Darurat

IGD beroperasi 24 jam dengan dokter jaga yang selalu tersedia.

## Rawat Inap dan Jam Kunjungan

Jam kunjungan pengunjung adalah pukul 11.00–13.00 dan 17.00–20.00.
"""

EXAMPLES: dict[str, Any] = {
    "create_collection_with_slug": {
        "name": COLLECTION_NAME,
        "slug": COLLECTION_SLUG,
    },
    "create_collection_name_only": {
        "name": COLLECTION_NAME,
    },
    "update_collection_rename": {
        "name": "RS Sehat Sentosa Bandung",
        "slug": "rs-sehat-sentosa-bandung",
    },
    "upload_document_json_by_slug": {
        "content": DOCUMENT_MARKDOWN,
        "filename": DOCUMENT_FILENAME,
        "collection_slug": COLLECTION_SLUG,
    },
    "upload_document_json_by_id": {
        "content": DOCUMENT_MARKDOWN,
        "filename": DOCUMENT_FILENAME,
        "collection_id": COLLECTION_ID,
    },
    "update_document_title": {
        "title": DOCUMENT_TITLE,
    },
    "update_document_content": {
        "content": DOCUMENT_MARKDOWN,
    },
    "retrieve_simple": {
        "query": QUERY_VISIT,
        "top_k": 5,
    },
    "retrieve_filtered": {
        "query": QUERY_IGD,
        "top_k": 5,
        "use_rerank": True,
        "collection_slug": COLLECTION_SLUG,
        "filters": {
            "metadata": {"section_header": "Instalasi Gawat Darurat"},
        },
    },
    "query_simple": {
        "query": QUERY_VISIT,
        "top_k": 5,
        "collection_slug": COLLECTION_SLUG,
    },
    "query_grounded": {
        "query": QUERY_VISIT,
        "top_k": 5,
        "collection_slug": COLLECTION_SLUG,
        "max_tokens_context": 2000,
    },
    "filters": {
        "metadata": {"section_header": "Instalasi Gawat Darurat"},
    },
    "date_range": {
        "after": "2024-01-01",
        "before": "2024-12-31",
    },
    "collection": {
        "id": COLLECTION_ID,
        "name": COLLECTION_NAME,
        "slug": COLLECTION_SLUG,
        "created_at": CREATED_AT,
    },
    "document_list_item": {
        "id": DOCUMENT_ID,
        "collection_id": COLLECTION_ID,
        "title": DOCUMENT_TITLE,
        "url": f"file://{DOCUMENT_FILENAME}",
        "status": "success",
        "error_message": None,
        "created_at": CREATED_AT,
    },
    "document": {
        "id": DOCUMENT_ID,
        "collection_id": COLLECTION_ID,
        "title": DOCUMENT_TITLE,
        "url": f"file://{DOCUMENT_FILENAME}",
        "status": "success",
        "error_message": None,
        "created_at": CREATED_AT,
        "content": DOCUMENT_MARKDOWN,
    },
    "document_upload_accepted": {
        "document_id": DOCUMENT_ID,
        "filename": DOCUMENT_FILENAME,
        "accepted": True,
    },
    "document_status_queued": {
        "document_id": DOCUMENT_ID,
        "status": "queued",
        "step": None,
        "steps": None,
        "error_message": None,
    },
    "document_status_processing": {
        "document_id": DOCUMENT_ID,
        "status": "processing",
        "step": "embedding",
        "steps": {
            "chunking": "completed",
            "embedding": "in_progress",
            "storing": "pending",
        },
        "error_message": None,
    },
    "document_status_success": {
        "document_id": DOCUMENT_ID,
        "status": "success",
        "step": "storing",
        "steps": {
            "chunking": "completed",
            "embedding": "completed",
            "storing": "completed",
        },
        "error_message": None,
    },
    "document_status_failed": {
        "document_id": DOCUMENT_ID,
        "status": "failed",
        "step": "embedding",
        "steps": {
            "chunking": "completed",
            "embedding": "failed",
            "storing": "pending",
        },
        "error_message": "OpenRouter embedding request failed",
    },
    "chunk_source": {
        "document": DOCUMENT_FILENAME,
        "page": None,
        "url": f"file://{DOCUMENT_FILENAME}",
    },
    "retrieval_chunk": {
        "chunk_id": CHUNK_ID,
        "text": (
            "Jam kunjungan pengunjung adalah pukul 11.00–13.00 dan 17.00–20.00."
        ),
        "score": 0.91,
        "source": {
            "document": DOCUMENT_FILENAME,
            "page": None,
            "url": f"file://{DOCUMENT_FILENAME}",
        },
    },
    "retrieval_result": {
        "results": [
            {
                "chunk_id": CHUNK_ID,
                "text": (
                    "Jam kunjungan pengunjung adalah pukul 11.00–13.00 "
                    "dan 17.00–20.00."
                ),
                "score": 0.91,
                "source": {
                    "document": DOCUMENT_FILENAME,
                    "page": None,
                    "url": f"file://{DOCUMENT_FILENAME}",
                },
            }
        ],
        "query_used": QUERY_VISIT,
        "latency_ms": 42,
        "top_k": 5,
        "reranked": False,
    },
    "source": {
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "chunk_index": 0,
        "score": 0.91,
    },
    "rag_response": {
        "answer": (
            "Jam kunjungan pengunjung rawat inap di RS Sehat Sentosa "
            "adalah pukul 11.00–13.00 dan 17.00–20.00."
        ),
        "sources": [
            {
                "chunk_id": CHUNK_ID,
                "document_id": DOCUMENT_ID,
                "chunk_index": 0,
                "score": 0.91,
            }
        ],
    },
    "health_ok": {"status": "ok"},
    "health_error": {"status": "error"},
    "unauthorized": {
        "error": "unauthorized",
        "message": "Invalid API key",
        "request_id": REQUEST_ID,
    },
    "not_found": {
        "error": "not_found",
        "message": f"Collection '{COLLECTION_ID}' not found",
        "request_id": REQUEST_ID,
    },
    "conflict": {
        "error": "conflict",
        "message": f"Collection with slug '{COLLECTION_SLUG}' already exists",
        "request_id": REQUEST_ID,
    },
    "validation_error": {
        "error": "validation_error",
        "message": "Exactly one of collection_id or collection_slug is required",
        "request_id": REQUEST_ID,
    },
}

OPENAPI_EXAMPLES: dict[str, dict[str, dict[str, Any]]] = {
    "create_collection": {
        "with_slug": {
            "summary": "Create with an explicit slug",
            "value": EXAMPLES["create_collection_with_slug"],
        },
        "name_only": {
            "summary": "Create and auto-generate the slug",
            "value": EXAMPLES["create_collection_name_only"],
        },
    },
    "update_collection": {
        "rename": {
            "summary": "Update name and slug",
            "value": EXAMPLES["update_collection_rename"],
        },
    },
    "upload_document_json": {
        "by_slug": {
            "summary": "Upload markdown into a collection by slug",
            "value": EXAMPLES["upload_document_json_by_slug"],
        },
        "by_id": {
            "summary": "Upload markdown into a collection by ID",
            "value": EXAMPLES["upload_document_json_by_id"],
        },
    },
    "update_document": {
        "title": {
            "summary": "Update the document title",
            "value": EXAMPLES["update_document_title"],
        },
        "content": {
            "summary": "Update markdown content and re-ingest",
            "value": EXAMPLES["update_document_content"],
        },
    },
    "retrieve": {
        "simple": {
            "summary": "Retrieve chunks for visiting hours",
            "value": EXAMPLES["retrieve_simple"],
        },
        "filtered": {
            "summary": "Retrieve with collection, rerank, and metadata filter",
            "value": EXAMPLES["retrieve_filtered"],
        },
    },
    "query": {
        "simple": {
            "summary": "Grounded answer for visiting hours",
            "value": EXAMPLES["query_simple"],
        },
        "with_cap": {
            "summary": "Grounded answer with an LLM context token cap",
            "value": EXAMPLES["query_grounded"],
        },
    },
}
