"""API tests for indexing on upload, search and delete."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import vector_store
from tests.pdf_factory import make_pdf

client = TestClient(app)


def upload(pages: list[str] | bytes, filename: str = "doc.pdf") -> dict:
    pdf = pages if isinstance(pages, bytes) else make_pdf(pages)
    response = client.post("/api/documents/upload", files={"file": (filename, pdf, "application/pdf")})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def two_documents() -> tuple[str, str]:
    policy = upload(["Office hours are nine to five.", "Refunds are issued within thirty days."], "policy.pdf")
    manual = upload(["To reset the router, hold the power button for ten seconds."], "manual.pdf")
    return policy["document_id"], manual["document_id"]


def test_upload_reports_indexing() -> None:
    body = upload(["Some page text.", "More text."])
    assert body["indexing_status"] == "indexed"
    assert body["indexed_chunk_count"] == body["chunk_count"] == 2


def test_global_search_returns_page_references(two_documents) -> None:
    _, manual_id = two_documents

    response = client.post("/api/search", json={"question": "How do I reset the router?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 2
    top = body["results"][0]
    assert top["document_id"] == manual_id
    assert top["filename"] == "manual.pdf"
    assert top["page_number"] == 1
    assert "reset the router" in top["text"]
    assert set(top) == {
        "chunk_id", "document_id", "filename", "page_number", "chunk_index", "text", "distance", "similarity",
        "dense_rank", "sparse_rank", "bm25_score", "fusion_score", "rerank_score",
    }


def test_global_search_with_document_filter(two_documents) -> None:
    policy_id, _ = two_documents

    body = client.post(
        "/api/search", json={"question": "router reset", "top_k": 5, "document_id": policy_id}
    ).json()

    assert body["document_id"] == policy_id
    assert {r["document_id"] for r in body["results"]} == {policy_id}


def test_document_search_only_returns_that_document(two_documents) -> None:
    policy_id, _ = two_documents

    response = client.post(f"/api/documents/{policy_id}/search", json={"question": "refunds"})

    assert response.status_code == 200
    body = response.json()
    assert body["top_k"] == 5  # default
    assert {r["document_id"] for r in body["results"]} == {policy_id}
    assert body["results"][0]["page_number"] == 2


def test_document_search_unknown_document_is_404() -> None:
    response = client.post(f"/api/documents/{'f' * 32}/search", json={"question": "refunds"})
    assert response.status_code == 404


@pytest.mark.parametrize("bad_id", ["not-a-valid-id", "A" * 32, "a" * 31])
def test_invalid_document_id_is_422(bad_id: str) -> None:
    assert client.post(f"/api/documents/{bad_id}/search", json={"question": "q"}).status_code == 422
    assert client.post("/api/search", json={"question": "q", "document_id": bad_id}).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "   "}, {"question": "q", "top_k": 0}, {"question": "q", "top_k": 21}],
)
def test_invalid_search_input_is_422(payload: dict) -> None:
    assert client.post("/api/search", json=payload).status_code == 422


def test_search_with_nothing_indexed_returns_empty_results() -> None:
    body = client.post("/api/search", json={"question": "anything"}).json()
    assert body["result_count"] == 0
    assert body["results"] == []


def test_delete_document_removes_it_from_search(two_documents) -> None:
    policy_id, manual_id = two_documents

    response = client.delete(f"/api/documents/{policy_id}")

    assert response.status_code == 200
    assert response.json() == {"document_id": policy_id, "deleted_chunks": 2}
    assert client.post(f"/api/documents/{policy_id}/search", json={"question": "refunds"}).status_code == 404
    remaining = client.post("/api/search", json={"question": "refunds", "top_k": 10}).json()["results"]
    assert {r["document_id"] for r in remaining} == {manual_id}
    assert client.delete(f"/api/documents/{policy_id}").status_code == 404


def test_reuploading_does_not_duplicate_chunks() -> None:
    # Build the PDF once: each generated PDF embeds a unique ID, so the bytes must be reused.
    pdf = make_pdf(["Same content every time."])
    first = upload(pdf)
    second = upload(pdf)

    assert first["document_id"] == second["document_id"]
    assert vector_store.get_collection().count() == 1


def test_indexing_failure_returns_error_and_leaves_nothing_searchable(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = vector_store.get_collection()

    def broken_upsert(**kwargs):
        raise RuntimeError("database locked")

    monkeypatch.setattr(collection, "upsert", broken_upsert)

    response = client.post(
        "/api/documents/upload", files={"file": ("doc.pdf", make_pdf(["Will not be indexed."]), "application/pdf")}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to store the document in the vector store."}
    assert collection.count() == 0


def test_vector_store_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable():
        raise vector_store.VectorStoreError("The vector store is unavailable.", 503)

    monkeypatch.setattr(vector_store, "get_collection", unavailable)

    assert client.post("/api/search", json={"question": "anything"}).status_code == 503
