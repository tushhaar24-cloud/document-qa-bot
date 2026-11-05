"""Document metadata and processing status in the database (upload, list, detail, delete)."""

import pytest
from fastapi.testclient import TestClient

from app.api import documents
from app.db.models import Document
from app.db.session import DatabaseError, session_scope
from app.main import app
from app.services import vector_store
from app.services.ingestion import make_document_id
from tests.db_helpers import database_down  # noqa: F401  (fixture)
from tests.pdf_factory import make_pdf

client = TestClient(app)


def upload(data: bytes, filename: str = "test.pdf"):
    return client.post("/api/documents/upload", files={"file": (filename, data, "application/pdf")})


def stored(document_id: str) -> Document | None:
    with session_scope() as db:
        return db.get(Document, document_id)


def test_list_is_empty_without_uploads() -> None:
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == {"document_count": 0, "indexed_count": 0, "total_chunks": 0, "documents": []}


def test_successful_upload_is_recorded_as_indexed() -> None:
    pdf = make_pdf(["Refund policy text", "Shipping policy text", ""])

    body = upload(pdf, "policies.pdf").json()

    document = stored(body["document_id"])
    assert document.status == "indexed"
    assert document.filename == "policies.pdf"
    assert document.file_size_bytes == len(pdf)
    assert document.page_count == 3
    assert document.pages_with_text == 2
    assert document.chunk_count == body["indexed_chunk_count"] == vector_store.count_chunks()
    assert document.embedding_model == body["embedding_model"]
    assert (document.chunk_size, document.chunk_overlap) == (800, 120)
    assert document.indexed_at is not None
    assert document.error_message is None


def test_document_is_marked_processing_while_it_is_indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    real_ingest = documents.ingest_pdf

    def spy(data, filename, settings):
        seen["status"] = stored(make_document_id(data)).status
        return real_ingest(data, filename, settings)

    monkeypatch.setattr(documents, "ingest_pdf", spy)
    body = upload(make_pdf(["Some text"])).json()

    assert seen["status"] == "processing"
    assert stored(body["document_id"]).status == "indexed"


def test_failed_processing_is_recorded_with_the_error() -> None:
    data = b"this is not really a pdf"

    response = upload(data, "fake.pdf")

    assert response.status_code == 415
    document = stored(make_document_id(data))
    assert document.status == "failed"
    assert document.error_message == response.json()["detail"]
    assert document.chunk_count == 0


@pytest.mark.parametrize(
    ("data", "filename", "status"),
    [(b"", "empty.pdf", 400), (b"plain text", "notes.txt", 415)],
)
def test_requests_rejected_before_processing_create_no_record(data: bytes, filename: str, status: int) -> None:
    assert upload(data, filename).status_code == status
    assert client.get("/api/documents").json()["documents"] == []


def test_uploading_the_same_file_again_keeps_one_record() -> None:
    pdf = make_pdf(["Identical content"])
    upload(pdf)
    upload(pdf)

    body = client.get("/api/documents").json()

    assert body["document_count"] == 1
    assert body["documents"][0]["status"] == "indexed"


def test_list_reports_status_counts_and_newest_first() -> None:
    first = upload(make_pdf(["First document"]), "first.pdf").json()
    upload(b"not a pdf", "broken.pdf")
    second = upload(make_pdf(["Second document", "More text"]), "second.pdf").json()

    body = client.get("/api/documents").json()

    assert [d["filename"] for d in body["documents"]] == ["second.pdf", "broken.pdf", "first.pdf"]
    assert body["document_count"] == 3
    assert body["indexed_count"] == 2
    assert body["total_chunks"] == first["indexed_chunk_count"] + second["indexed_chunk_count"]
    broken = body["documents"][1]
    assert broken["status"] == "failed"
    assert broken["error_message"] == "The file is not a valid PDF."
    assert broken["chunk_count"] == 0


def test_document_detail() -> None:
    body = upload(make_pdf(["Detail text"]), "detail.pdf").json()

    response = client.get(f"/api/documents/{body['document_id']}")

    assert response.status_code == 200
    detail = response.json()
    assert detail["filename"] == "detail.pdf"
    assert detail["status"] == "indexed"
    assert detail["embedding_model"] == body["embedding_model"]
    assert detail["indexed_page_count"] == 1


def test_document_detail_404() -> None:
    assert client.get(f"/api/documents/{'a' * 32}").status_code == 404


def test_delete_removes_chunks_and_record() -> None:
    body = upload(make_pdf(["Temporary content"]), "temp.pdf").json()

    response = client.delete(f"/api/documents/{body['document_id']}")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == body["indexed_chunk_count"]
    assert stored(body["document_id"]) is None
    assert vector_store.count_chunks() == 0


def test_a_failed_document_can_be_deleted() -> None:
    data = b"not a pdf"
    upload(data, "broken.pdf")
    document_id = make_document_id(data)

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {"document_id": document_id, "deleted_chunks": 0}
    assert stored(document_id) is None


def test_delete_unknown_document_404() -> None:
    assert client.delete(f"/api/documents/{'b' * 32}").status_code == 404


def test_upload_is_refused_and_nothing_indexed_when_the_database_is_down(database_down) -> None:
    response = upload(make_pdf(["Some text"]))

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is unavailable."}
    assert vector_store.count_chunks() == 0


def test_status_save_failure_after_indexing_asks_for_a_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise DatabaseError("The database is unavailable.", 503)

    monkeypatch.setattr(documents, "_record_indexed", fail)
    pdf = make_pdf(["Some text"])

    response = upload(pdf)

    assert response.status_code == 503
    assert "upload it again" in response.json()["detail"]
    monkeypatch.undo()
    # Uploading again (same file, same ID) completes the record.
    assert upload(pdf).status_code == 200
    assert stored(make_document_id(pdf)).status == "indexed"


def test_listing_returns_503_when_the_database_is_down(database_down) -> None:
    response = client.get("/api/documents")

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is unavailable."}
