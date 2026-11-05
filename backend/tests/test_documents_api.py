"""API tests for POST /api/documents/upload."""

import dataclasses

import pytest
from fastapi.testclient import TestClient

from app.api import documents
from app.core.config import get_settings
from app.main import app
from tests.pdf_factory import make_pdf

client = TestClient(app)
UPLOAD_URL = "/api/documents/upload"


def upload(data: bytes, filename: str = "test.pdf", params: dict | None = None):
    return client.post(UPLOAD_URL, files={"file": (filename, data, "application/pdf")}, params=params)


def override_settings(monkeypatch: pytest.MonkeyPatch, **changes) -> None:
    patched = dataclasses.replace(get_settings(), **changes)
    monkeypatch.setattr(documents, "get_settings", lambda: patched)


def test_valid_pdf_returns_summary_without_text_by_default() -> None:
    response = upload(make_pdf(["Introduction to the document", "Terms and conditions"]), "contract.pdf")

    assert response.status_code == 200
    body = response.json()
    assert len(body["document_id"]) == 32
    assert body["filename"] == "contract.pdf"
    assert body["page_count"] == 2
    assert body["pages_with_text"] == 2
    assert body["empty_pages"] == []
    assert [p["page_number"] for p in body["pages"]] == [1, 2]
    assert all(p["text_preview"] is None for p in body["pages"])
    assert body["chunk_count"] == 2  # two short pages -> one chunk each
    assert body["chunk_size"] == 800
    assert body["chunk_overlap"] == 120
    assert body["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert body["embedding_dimensions"] == 384
    assert body["chunk_previews"] is None


def test_same_file_gets_the_same_document_id() -> None:
    pdf = make_pdf(["Identical content"])
    assert upload(pdf).json()["document_id"] == upload(pdf).json()["document_id"]


def test_include_text_returns_limited_chunk_previews() -> None:
    long_page = "Retrieval-augmented generation grounds answers in source documents. " * 12
    pdf = make_pdf([long_page] * 4)

    body = upload(pdf, params={"include_text": "true"}).json()

    assert body["chunk_count"] > documents.CHUNK_PREVIEW_COUNT
    previews = body["chunk_previews"]
    assert len(previews) == documents.CHUNK_PREVIEW_COUNT
    assert [p["chunk_index"] for p in previews] == list(range(documents.CHUNK_PREVIEW_COUNT))
    assert all(len(p["text_preview"]) <= documents.CHUNK_PREVIEW_CHARS for p in previews)
    assert previews[0]["chunk_id"] == f"{body['document_id']}:p1:c0"


def test_embedding_failure_returns_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import embeddings

    def unavailable():
        raise embeddings.EmbeddingError("The embedding model is unavailable.", 503)

    monkeypatch.setattr(embeddings, "get_embedding_model", unavailable)

    response = upload(make_pdf(["Some text"]))

    assert response.status_code == 503
    assert response.json() == {"detail": "The embedding model is unavailable."}


def test_include_text_returns_preview_in_development() -> None:
    response = upload(make_pdf(["Introduction"]), params={"include_text": "true"})

    assert response.status_code == 200
    assert response.json()["pages"][0]["text_preview"] == "Introduction"


def test_include_text_is_forbidden_outside_development(monkeypatch: pytest.MonkeyPatch) -> None:
    override_settings(monkeypatch, app_env="production")

    response = upload(make_pdf(["Introduction"]), params={"include_text": "true"})

    assert response.status_code == 403


def test_pages_without_text_are_reported() -> None:
    response = upload(make_pdf(["First page", "", "Third page"]))

    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 3
    assert body["pages_with_text"] == 2
    assert body["empty_pages"] == [2]
    assert [p["page_number"] for p in body["pages"]] == [1, 3]


def test_empty_file_is_rejected() -> None:
    assert upload(b"").status_code == 400


def test_pdf_with_no_text_is_rejected() -> None:
    response = upload(make_pdf(["", ""]))
    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_non_pdf_renamed_to_pdf_is_rejected() -> None:
    response = upload(b"This is a plain text file pretending to be a PDF.", "fake.pdf")
    assert response.status_code == 415


def test_non_pdf_extension_is_rejected() -> None:
    response = upload(make_pdf(["Real PDF content"]), "notes.txt")
    assert response.status_code == 415


def test_corrupt_pdf_is_rejected() -> None:
    response = upload(b"%PDF-1.7\n" + b"\x00garbage" * 50)
    assert response.status_code == 422
    assert response.json()["detail"] == "The PDF is corrupt or unreadable."


def test_password_protected_pdf_is_rejected() -> None:
    response = upload(make_pdf(["secret"], user_password="pw"))
    assert response.status_code == 422
    assert "Password-protected" in response.json()["detail"]


def test_oversized_upload_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    override_settings(monkeypatch, max_upload_bytes=1024)

    response = upload(make_pdf(["x" * 50] * 30))

    assert response.status_code == 413


def test_filename_path_is_stripped() -> None:
    response = upload(make_pdf(["Hello"]), "C:\\Users\\someone\\secret\\report.pdf")
    assert response.json()["filename"] == "report.pdf"
