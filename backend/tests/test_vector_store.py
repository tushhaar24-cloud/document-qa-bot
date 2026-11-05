"""Tests for ChromaDB storage, using a temporary database per test."""

import dataclasses

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_core.documents import Document

from app.services import vector_store
from app.services.chunking import chunk_pages
from app.services.embeddings import embed_documents
from app.services.ingestion import PageText
from app.services.vector_store import (
    VectorStoreError,
    delete_document,
    document_exists,
    get_chunk_ids,
    get_collection,
    replace_document_chunks,
    upsert_chunks,
)

DOC_A = "a" * 32
DOC_B = "b" * 32
LONG_TEXT = "Refunds are issued within thirty days of purchase. " * 40


def make_chunks(document_id: str, pages: list[str], chunk_size: int = 800) -> list[Document]:
    page_texts = [PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)]
    return chunk_pages(page_texts, document_id, f"{document_id[:4]}.pdf", chunk_size, chunk_size // 8)


def index(document_id: str, pages: list[str], chunk_size: int = 800) -> list[Document]:
    chunks = make_chunks(document_id, pages, chunk_size)
    replace_document_chunks(document_id, chunks, embed_documents([c.page_content for c in chunks]))
    return chunks


def test_stores_ids_text_embeddings_and_metadata() -> None:
    chunks = index(DOC_A, ["Page one text.", "Page two text."])

    stored = get_collection().get(ids=[chunks[1].id], include=["documents", "metadatas", "embeddings"])

    assert stored["documents"] == ["Page two text."]
    assert len(stored["embeddings"][0]) == 384
    meta = stored["metadatas"][0]
    assert meta["document_id"] == DOC_A
    assert meta["filename"] == "aaaa.pdf"
    assert meta["page_number"] == 2
    assert meta["chunk_index"] == 1
    assert meta["chunk_id"] == chunks[1].id


def test_empty_chunks_are_not_stored() -> None:
    chunks = [
        Document(id="x:p1:c0", page_content="Real text", metadata={"document_id": "x"}),
        Document(id="x:p1:c1", page_content="   ", metadata={"document_id": "x"}),
    ]
    assert upsert_chunks(chunks, [[1.0] + [0.0] * 383, [0.0, 1.0] + [0.0] * 382]) == 1
    assert get_collection().count() == 1


def test_mismatched_chunks_and_embeddings_are_rejected() -> None:
    with pytest.raises(ValueError):
        upsert_chunks(make_chunks(DOC_A, ["text"]), [])


def test_no_chunks_stores_nothing() -> None:
    assert upsert_chunks([], []) == 0


def test_data_persists_after_client_reinitialization() -> None:
    chunks = index(DOC_A, [LONG_TEXT])

    # Drop every in-memory reference, then reopen from disk.
    vector_store.reset_collection_cache()
    SharedSystemClient.clear_system_cache()

    assert sorted(get_chunk_ids(DOC_A)) == sorted(c.id for c in chunks)


def test_reindexing_the_same_document_is_idempotent() -> None:
    first = index(DOC_A, [LONG_TEXT, "Second page."])
    second = index(DOC_A, [LONG_TEXT, "Second page."])

    assert [c.id for c in first] == [c.id for c in second]
    assert get_collection().count() == len(first)


def test_reindexing_with_fewer_chunks_removes_stale_ones() -> None:
    many = index(DOC_A, [LONG_TEXT], chunk_size=200)
    few = index(DOC_A, [LONG_TEXT], chunk_size=800)

    assert len(few) < len(many)
    assert sorted(get_chunk_ids(DOC_A)) == sorted(c.id for c in few)


def test_delete_document_removes_only_its_chunks() -> None:
    index(DOC_A, [LONG_TEXT])
    b_chunks = index(DOC_B, ["Other document."])

    deleted = delete_document(DOC_A)

    assert deleted > 0
    assert not document_exists(DOC_A)
    assert get_chunk_ids(DOC_B) == [b_chunks[0].id]
    assert delete_document(DOC_A) == 0


def test_failed_write_leaves_no_partial_document(monkeypatch: pytest.MonkeyPatch) -> None:
    other = index(DOC_B, ["Unrelated document stays indexed."])
    collection = get_collection()
    real_upsert = collection.upsert
    calls = {"n": 0}

    def fail_on_second_batch(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real_upsert(**kwargs)

    monkeypatch.setattr(vector_store, "_max_batch_size", 2)
    monkeypatch.setattr(collection, "upsert", fail_on_second_batch)

    chunks = make_chunks(DOC_A, [LONG_TEXT], chunk_size=200)
    assert len(chunks) > 2  # needs at least two batches

    with pytest.raises(VectorStoreError) as exc_info:
        replace_document_chunks(DOC_A, chunks, embed_documents([c.page_content for c in chunks]))

    assert exc_info.value.status_code == 500
    assert get_chunk_ids(DOC_A) == []  # the first batch was rolled back
    assert get_chunk_ids(DOC_B) == [other[0].id]


def test_embedding_model_change_is_detected(isolated_vector_store, monkeypatch: pytest.MonkeyPatch) -> None:
    index(DOC_A, ["Indexed with the original model."])

    other_model = dataclasses.replace(isolated_vector_store, embedding_model_name="some/other-model")
    monkeypatch.setattr(vector_store, "get_settings", lambda: other_model)
    vector_store.reset_collection_cache()

    with pytest.raises(VectorStoreError) as exc_info:
        get_collection()
    assert "different embedding model" in exc_info.value.message
