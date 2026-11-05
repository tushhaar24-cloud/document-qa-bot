"""Tests for semantic retrieval over a temporary ChromaDB."""

import pytest

from app.services.chunking import chunk_pages
from app.services.embeddings import embed_documents
from app.services.ingestion import PageText
from app.services.retrieval import retrieve_chunks
from app.services.vector_store import replace_document_chunks

POLICY_DOC = "1" * 32
MANUAL_DOC = "2" * 32


def index(document_id: str, filename: str, pages: list[str]) -> None:
    page_texts = [PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)]
    chunks = chunk_pages(page_texts, document_id, filename, 800, 120)
    replace_document_chunks(document_id, chunks, embed_documents([c.page_content for c in chunks]))


@pytest.fixture
def indexed_documents() -> None:
    index(POLICY_DOC, "policy.pdf", [
        "Our office hours are nine to five on weekdays.",
        "Refunds are issued within thirty days of purchase with a receipt.",
    ])
    index(MANUAL_DOC, "manual.pdf", [
        "To reset the router, hold the power button for ten seconds.",
        "Refunds for hardware require the original packaging.",
    ])


def test_most_relevant_chunk_ranks_first(indexed_documents) -> None:
    results = retrieve_chunks("How do I reset the router?", top_k=3)

    top = results[0]
    assert top.document_id == MANUAL_DOC
    assert top.filename == "manual.pdf"
    assert top.page_number == 1
    assert "reset the router" in top.text
    assert top.chunk_id == f"{MANUAL_DOC}:p1:c0"
    # Results are ordered from closest to farthest.
    assert [r.distance for r in results] == sorted(r.distance for r in results)
    assert top.similarity == pytest.approx(1 - top.distance, abs=1e-3)


def test_document_filter_restricts_results(indexed_documents) -> None:
    results = retrieve_chunks("refunds", top_k=10, document_id=POLICY_DOC)

    assert results
    assert {r.document_id for r in results} == {POLICY_DOC}
    assert results[0].page_number == 2


def test_top_k_limits_result_count(indexed_documents) -> None:
    assert len(retrieve_chunks("refunds", top_k=1)) == 1


def test_empty_store_returns_no_results() -> None:
    assert retrieve_chunks("anything at all", top_k=5) == []


def test_unknown_document_filter_returns_no_results(indexed_documents) -> None:
    assert retrieve_chunks("refunds", top_k=5, document_id="f" * 32) == []


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question: str) -> None:
    with pytest.raises(ValueError):
        retrieve_chunks(question)


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_top_k_out_of_range_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError):
        retrieve_chunks("refunds", top_k=top_k)
