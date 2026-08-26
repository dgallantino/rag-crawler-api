"""Contract tests for OpenAPI operationId, summary, and examples."""

from fastapi.testclient import TestClient

from app.schemas.examples import COLLECTION_SLUG, QUERY_VISIT

EXPECTED_OPERATIONS = {
    ("/health", "get"): (
        "health",
        "Check API liveness",
    ),
    ("/health/db", "get"): (
        "health_db",
        "Check database connectivity",
    ),
    ("/v1/collections", "post"): (
        "create_collection",
        "Create a named collection for the authenticated tenant",
    ),
    ("/v1/collections", "get"): (
        "list_collections",
        "List collections owned by the authenticated tenant",
    ),
    ("/v1/collections/{collection_id}", "get"): (
        "get_collection",
        "Get a collection by ID",
    ),
    ("/v1/collections/{collection_id}", "patch"): (
        "update_collection",
        "Update a collection name or slug",
    ),
    ("/v1/collections/{collection_id}", "delete"): (
        "delete_collection",
        "Delete a collection and its documents",
    ),
    ("/v1/documents", "post"): (
        "upload_document",
        "Upload a markdown file and enqueue ingestion",
    ),
    ("/v1/documents/json", "post"): (
        "upload_document_json",
        "Upload markdown as JSON and enqueue ingestion",
    ),
    ("/v1/documents", "get"): (
        "list_documents",
        "List documents, optionally filtered by collection",
    ),
    ("/v1/documents/{document_id}", "get"): (
        "get_document",
        "Get a document including its markdown content",
    ),
    ("/v1/documents/{document_id}", "patch"): (
        "update_document",
        "Update a document title or content and re-ingest",
    ),
    ("/v1/documents/{document_id}", "delete"): (
        "delete_document",
        "Delete a document and its chunks",
    ),
    ("/v1/documents/{document_id}/status", "get"): (
        "document_status",
        "Get ingestion status for a document",
    ),
    ("/v1/retrieve", "post"): (
        "retrieve",
        "Retrieve ranked chunks for a query",
    ),
    ("/v1/query", "post"): (
        "query",
        "Retrieve chunks and generate a grounded answer",
    ),
}

JSON_BODY_OPERATIONS = (
    ("/v1/collections", "post"),
    ("/v1/collections/{collection_id}", "patch"),
    ("/v1/documents/json", "post"),
    ("/v1/documents/{document_id}", "patch"),
    ("/v1/retrieve", "post"),
    ("/v1/query", "post"),
)

NAMED_EXAMPLE_KEYS = {
    ("/v1/collections", "post"): {"with_slug", "name_only"},
    ("/v1/retrieve", "post"): {"simple", "filtered"},
    ("/v1/query", "post"): {"simple", "with_cap"},
    ("/v1/documents/json", "post"): {"by_slug", "by_id"},
}


def _json_content(operation: dict) -> dict:
    return operation["requestBody"]["content"]["application/json"]


def test_openapi_operation_ids_and_summaries(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]

    actual = {}
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.startswith("x-") or method == "parameters":
                continue
            actual[(path, method)] = (
                operation.get("operationId"),
                operation.get("summary"),
            )

    assert actual == EXPECTED_OPERATIONS


def test_json_request_bodies_have_examples(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path, method in JSON_BODY_OPERATIONS:
        content = _json_content(paths[path][method])
        named = content.get("examples") or {}
        schema_examples = content.get("schema", {}).get("examples") or []
        assert named or schema_examples, f"{method.upper()} {path} has no examples"


def test_named_request_examples(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for (path, method), expected_keys in NAMED_EXAMPLE_KEYS.items():
        named = _json_content(paths[path][method]).get("examples") or {}
        assert expected_keys <= set(named), (
            f"{method.upper()} {path} missing named examples"
        )


def test_example_values_use_hospital_domain(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    create_examples = _json_content(paths["/v1/collections"]["post"])["examples"]
    assert create_examples["with_slug"]["value"]["slug"] == COLLECTION_SLUG

    retrieve_examples = _json_content(paths["/v1/retrieve"]["post"])["examples"]
    assert retrieve_examples["simple"]["value"]["query"] == QUERY_VISIT
    assert COLLECTION_SLUG in str(retrieve_examples["filtered"]["value"])
