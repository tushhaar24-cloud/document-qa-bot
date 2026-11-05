"""Tests for BM25 keyword retrieval and keeping its index in sync with ChromaDB."""

import pytest

from app.services import sparse_retrieval, vector_store
from app.services.chunking import chunk_pages
from app.services.embeddings import embed_documents
from app.services.ingestion import PageText
from app.services.sparse_retrieval import BM25Index, sparse_search, tokenize
from app.services.vector_store import StoredChunk, delete_document, replace_document_chunks

POLICY_DOC = "1" * 32
CONTRACT_DOC = "2" * 32


def stored(chunk_id: str, text: str, document_id: str = POLICY_DOC, page: int = 1) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"chunk_id": chunk_id, "document_id": document_id, "filename": "policy.pdf", "page_number": page},
    )


def index(document_id: str, filename: str, pages: list[str]) -> None:
    page_texts = [PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)]
    chunks = chunk_pages(page_texts, document_id, filename, 800, 120)
    replace_document_chunks(document_id, chunks, embed_documents([c.page_content for c in chunks]))


@pytest.fixture(autouse=True)
def fresh_sparse_index():
    sparse_retrieval.reset_sparse_index()
    yield
    sparse_retrieval.reset_sparse_index()


# --- tokenizer -------------------------------------------------------------------


def test_tokenizer_keeps_identifiers_whole_and_splits_hyphenated_ones() -> None:
    assert tokenize("See clause 7.3 and form HR-27.") == ["see", "clause", "7.3", "form", "hr-27", "hr", "27"]


def test_clause_number_does_not_match_its_reverse() -> None:
    bm25 = BM25Index([stored("a", "Section 3.7 covers delivery within 3 days."), stored("b", "Section 7.3 covers notice.")])

    assert [r.chunk.chunk_id for r in bm25.search("7.3", top_k=5)] == ["b"]


def test_tokenizer_drops_stopwords_and_single_letters_but_keeps_digits() -> None:
    assert tokenize("What is the child's leave? 3 days") == ["child", "leave", "3", "days"]


def test_tokenizer_keeps_negations() -> None:
    assert "not" in tokenize("Alcohol is not reimbursed")


def test_tokenizer_handles_accented_words() -> None:
    assert tokenize("Café prices in München") == ["café", "prices", "münchen"]


# --- BM25 scoring (no database) --------------------------------------------------


def test_exact_identifier_ranks_first() -> None:
    bm25 = BM25Index([
        stored("a", "Error ERR-4021 means the card was declined."),
        stored("b", "Error ERR-4012 means the session expired."),
        stored("c", "Office hours are nine to five."),
    ])

    results = bm25.search("What does ERR-4012 mean?", top_k=3)

    assert results[0].chunk.chunk_id == "b"
    assert results[0].score > results[1].score


def test_rare_words_count_more_than_common_ones() -> None:
    bm25 = BM25Index([
        stored("a", "leave policy for leave requests"),
        stored("b", "leave rules for maternity"),
        stored("c", "leave and holidays"),
    ])

    # "leave" is everywhere, "maternity" only in b, so b wins despite a repeating "leave".
    assert bm25.search("maternity leave", top_k=1)[0].chunk.chunk_id == "b"


def test_term_in_half_of_the_chunks_still_scores() -> None:
    # rank_bm25.BM25Okapi gives this term an IDF of 0, so its only match would score 0.
    bm25 = BM25Index([stored("a", "refunds within thirty days"), stored("b", "office hours")])

    results = bm25.search("refunds", top_k=5)

    assert [r.chunk.chunk_id for r in results] == ["a"]
    assert results[0].score > 0


def test_shorter_chunk_wins_for_the_same_match() -> None:
    bm25 = BM25Index([
        stored("long", "probation " + "filler words about other unrelated topics " * 10),
        stored("short", "probation lasts six months"),
    ])

    assert bm25.search("probation", top_k=1)[0].chunk.chunk_id == "short"


def test_only_chunks_sharing_a_word_are_returned() -> None:
    bm25 = BM25Index([stored("a", "annual leave"), stored("b", "router reset")])

    assert [r.chunk.chunk_id for r in bm25.search("leave", top_k=10)] == ["a"]
    assert bm25.search("pension contribution", top_k=10) == []


def test_metadata_is_preserved() -> None:
    chunk = stored("doc:p4:c2", "Maternity leave is 26 weeks.", document_id=CONTRACT_DOC, page=4)

    match = BM25Index([chunk]).search("maternity", top_k=1)[0]

    assert match.chunk == chunk
    assert match.chunk.metadata["page_number"] == 4
    assert match.chunk.metadata["document_id"] == CONTRACT_DOC


def test_document_filter() -> None:
    bm25 = BM25Index([
        stored("p", "notice period is 60 days", document_id=POLICY_DOC),
        stored("c", "notice period is 30 days", document_id=CONTRACT_DOC),
    ])

    results = bm25.search("notice period", top_k=10, document_id=CONTRACT_DOC)

    assert [r.chunk.chunk_id for r in results] == ["c"]


def test_top_k_limits_results_and_ties_are_deterministic() -> None:
    bm25 = BM25Index([stored(str(i), "same words here") for i in range(5)])

    assert [r.chunk.chunk_id for r in bm25.search("words", top_k=3)] == ["0", "1", "2"]


def test_empty_index_returns_nothing() -> None:
    assert BM25Index([]).search("anything", top_k=5) == []


# --- sparse_search over ChromaDB ------------------------------------------------


def test_sparse_search_reads_chunks_from_chromadb() -> None:
    index(CONTRACT_DOC, "contract.pdf", ["Clause 7.3: termination needs 30 days notice.", "Clause 3.7: delivery."])

    results = sparse_search("clause 7.3", top_k=5)

    top = results[0].chunk
    assert top.chunk_id == f"{CONTRACT_DOC}:p1:c0"
    assert top.metadata["filename"] == "contract.pdf"
    assert top.metadata["page_number"] == 1
    assert top.metadata["document_id"] == CONTRACT_DOC


def test_empty_store_returns_nothing() -> None:
    assert sparse_search("refunds", top_k=5) == []


@pytest.mark.parametrize("query", ["", "   ", "what is the", "?!"])
def test_queries_without_meaningful_words_return_nothing(query: str) -> None:
    index(POLICY_DOC, "policy.pdf", ["What is the refund policy? Refunds take thirty days."])

    assert sparse_search(query, top_k=5) == []


def test_index_is_not_rebuilt_while_the_store_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    index(POLICY_DOC, "policy.pdf", ["Refunds take thirty days."])
    reads = []
    real_get_all = vector_store.get_all_chunks
    monkeypatch.setattr(vector_store, "get_all_chunks", lambda: reads.append(1) or real_get_all())

    for _ in range(3):
        sparse_search("refunds", top_k=5)

    assert len(reads) == 1


def test_upload_is_searchable_immediately() -> None:
    index(POLICY_DOC, "policy.pdf", ["Refunds take thirty days."])
    assert sparse_search("router", top_k=5) == []

    index(CONTRACT_DOC, "manual.pdf", ["Hold the router button to reset it."])

    assert [m.chunk.metadata["document_id"] for m in sparse_search("router", top_k=5)] == [CONTRACT_DOC]


def test_deleted_document_disappears_from_results() -> None:
    index(POLICY_DOC, "policy.pdf", ["Refunds take thirty days."])
    index(CONTRACT_DOC, "contract.pdf", ["Refunds need a receipt."])
    assert len(sparse_search("refunds", top_k=5)) == 2

    delete_document(CONTRACT_DOC)

    assert [m.chunk.metadata["document_id"] for m in sparse_search("refunds", top_k=5)] == [POLICY_DOC]


def test_changes_made_by_another_process_are_detected() -> None:
    index(POLICY_DOC, "policy.pdf", ["Refunds take thirty days."])
    sparse_search("refunds", top_k=5)

    # Written straight to Chroma, bypassing this process's version counter (like another worker would).
    vector_store.get_collection().add(
        ids=["other:p1:c0"], documents=["The router reset button."], embeddings=[[0.1] * 384],
        metadatas=[{"document_id": CONTRACT_DOC, "page_number": 1}],
    )

    assert [m.chunk.chunk_id for m in sparse_search("router", top_k=5)] == ["other:p1:c0"]
