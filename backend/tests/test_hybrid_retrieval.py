"""Tests for Reciprocal Rank Fusion, hybrid retrieval and a dense / sparse / hybrid comparison.

The comparison uses `ConceptEmbedding`, a fake embedding model that mimics two
known traits of real sentence embeddings, so results are deterministic and offline:
- synonyms land close together ("vacation" ~ "leave"), which keyword search cannot do;
- numbers and codes barely register ("ERR-4021" ~ "ERR-4012"), which keyword search handles.
The same comparison with the real model is in test_retrieval_comparison_integration.py.
"""

import math
import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings

from app.core.config import get_settings
from app.services import embeddings, rag, sparse_retrieval
from app.services.chunking import chunk_pages
from app.services.embeddings import embed_documents, embed_query
from app.services.ingestion import PageText
from app.services.rag import NOT_FOUND_MESSAGE, answer_question
from app.services.sparse_retrieval import STOPWORDS
from app.services.retrieval import (
    reciprocal_rank_fusion,
    retrieve,
    retrieve_chunks,
    retrieve_hybrid,
    retrieve_sparse,
)
from app.services.vector_store import replace_document_chunks

HANDBOOK = "a" * 32
CONTRACT = "b" * 32


def index(document_id: str, filename: str, pages: list[str]) -> None:
    page_texts = [PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)]
    chunks = chunk_pages(page_texts, document_id, filename, 800, 120)
    replace_document_chunks(document_id, chunks, embed_documents([c.page_content for c in chunks]))


@pytest.fixture(autouse=True)
def fresh_sparse_index():
    sparse_retrieval.reset_sparse_index()
    yield
    sparse_retrieval.reset_sparse_index()


# --- Reciprocal Rank Fusion -------------------------------------------------------


def test_rrf_scores_are_the_sum_of_reciprocal_ranks() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_between_retrievers() -> None:
    # "b" is never first, but both retrievers rank it well, so it beats each list's winner.
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "b", "e"]], k=60)

    assert fused[0][0] == "b"


def test_rrf_uses_ranks_not_raw_scores() -> None:
    # Only the order of each list matters: the same ranks give the same fusion,
    # however different the underlying score scales are (0..1 cosine vs unbounded BM25).
    assert reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_merges_duplicates_into_one_entry() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "a"], ["a"]], k=60)

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    # The repeated "a" in the first list only counts once, at its best rank.
    assert dict(fused)["a"] == pytest.approx(2 / 61)


def test_rrf_ties_are_broken_deterministically() -> None:
    fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]


def test_smaller_rrf_k_favours_top_ranks_more() -> None:
    lists = [["a", "x1", "x2", "x3"], ["y1", "y2", "b", "a"]]

    def gap(k: int) -> float:
        fused = dict(reciprocal_rank_fusion(lists, k))
        return fused["a"] - fused["b"]

    assert gap(1) > gap(60)


def test_rrf_of_empty_lists_is_empty() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == []


# --- hybrid retrieval over a real temporary ChromaDB (keyword fake embeddings) ---


@pytest.fixture
def two_documents() -> None:
    index(HANDBOOK, "handbook.pdf", [
        "Office hours are nine to five on weekdays.",
        "The notice period is 60 days for permanent staff.",
    ])
    index(CONTRACT, "contract.pdf", [
        "Clause 7.3: the notice period for termination is 30 days.",
        "Refunds are issued within 14 days.",
    ])


def test_hybrid_results_keep_their_original_metadata(two_documents) -> None:
    results = retrieve_hybrid("clause 7.3 notice period", top_k=4)

    top = results[0]
    assert top.chunk_id == f"{CONTRACT}:p1:c0"
    assert (top.document_id, top.filename, top.page_number, top.chunk_index) == (CONTRACT, "contract.pdf", 1, 0)
    assert top.text == "Clause 7.3: the notice period for termination is 30 days."
    assert len({r.chunk_id for r in results}) == len(results)  # no duplicates


def test_hybrid_records_both_ranks_and_the_fusion_score(two_documents) -> None:
    k = get_settings().rrf_k
    for chunk in retrieve_hybrid("notice period", top_k=4):
        expected = sum(1 / (k + rank) for rank in (chunk.dense_rank, chunk.sparse_rank) if rank is not None)
        assert chunk.fusion_score == pytest.approx(expected, abs=1e-6)
        assert chunk.similarity == pytest.approx(1 - chunk.distance, abs=1e-3)
        assert (chunk.bm25_score is None) == (chunk.sparse_rank is None)


def test_chunks_found_only_by_keywords_get_their_real_similarity(two_documents, monkeypatch) -> None:
    # Dense search finds nothing, so every result was found by BM25 alone.
    monkeypatch.setattr("app.services.retrieval.query_chunks", lambda *args, **kwargs: [])

    results = retrieve_hybrid("refunds within 14 days", top_k=4)

    keyword_only = [r for r in results if r.dense_rank is None]
    assert keyword_only and len(keyword_only) == len(results)
    query_vector = embed_query("refunds within 14 days")
    for chunk in keyword_only:
        stored_vector = embed_documents([chunk.text])[0]
        cosine = sum(a * b for a, b in zip(query_vector, stored_vector))
        assert chunk.similarity == pytest.approx(cosine, abs=1e-3)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_document_filter_applies_to_every_mode(two_documents, mode: str) -> None:
    results = retrieve("notice period", top_k=10, document_id=HANDBOOK, mode=mode)

    assert results
    assert {r.document_id for r in results} == {HANDBOOK}


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_top_k_is_respected_in_every_mode(two_documents, mode: str) -> None:
    assert len(retrieve("notice period days", top_k=1, mode=mode)) == 1


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_empty_store_returns_nothing_in_every_mode(mode: str) -> None:
    assert retrieve("notice period", top_k=5, mode=mode) == []


@pytest.mark.parametrize("retriever", [retrieve_chunks, retrieve_sparse, retrieve_hybrid])
@pytest.mark.parametrize("question, top_k", [("", 5), ("   ", 5), ("refunds", 0), ("refunds", 21)])
def test_invalid_input_is_rejected(retriever, question: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        retriever(question, top_k)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        retrieve("refunds", mode="fuzzy")


def test_dense_mode_is_unchanged(two_documents) -> None:
    dense = retrieve("notice period", top_k=4, mode="dense")

    assert [r.chunk_id for r in dense] == [r.chunk_id for r in retrieve_chunks("notice period", top_k=4)]
    assert [r.distance for r in dense] == sorted(r.distance for r in dense)
    assert all(r.sparse_rank is None and r.bm25_score is None and r.fusion_score is None for r in dense)


# --- dense vs sparse vs hybrid comparison ----------------------------------------

SYNONYMS = {
    "vacation": "leave", "holiday": "leave", "holidays": "leave", "leave": "leave",
    "home": "remote", "remotely": "remote", "remote": "remote",
    "end": "terminate", "terminate": "terminate", "termination": "terminate", "cancel": "terminate",
    "accommodation": "hotel", "hotel": "hotel",
}


class ConceptEmbedding(Embeddings):
    """Bag of concepts: synonyms share a slot; filler words and tokens containing digits are ignored."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 384
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOPWORDS or any(ch.isdigit() for ch in word):
                continue
            concept = SYNONYMS.get(word, word)
            vector[zlib.crc32(concept.encode()) % 384] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


ERRORS = "c" * 32


@pytest.fixture
def comparison_corpus(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Each target has look-alike decoys that differ only in their numbers or codes."""
    model = ConceptEmbedding()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: model)
    index(HANDBOOK, "handbook.pdf", [
        "Employees receive 25 days of paid annual leave each calendar year.",
        "Staff may work remotely up to 2 days per week with manager approval.",
        "Sick leave requires a medical certificate after 3 consecutive days.",
        "The maximum hotel rate is 150 EUR per night, or 200 EUR in London.",
    ])
    index(CONTRACT, "contract.pdf", [
        "Clause 7.1: this agreement starts on the signing date.",
        "Clause 7.2: prices are fixed for the first year.",
        "Clause 7.3: either party may terminate this agreement with 30 days written notice.",
        "Clause 3.7: the supplier shall deliver all goods within 14 days of the order date.",
        "Invoices are payable within 45 days of delivery.",
        "Returns are accepted within 60 days of delivery.",
    ])
    index(ERRORS, "errors.pdf", [
        "Error ERR-4011 means the card number is invalid.",
        "Error ERR-4012 means the login session expired and the user must sign in again.",
        "Error ERR-4021 means the payment gateway declined the card.",
        "Error ERR-4022 means the card has expired.",
    ])
    return {
        "semantic": f"{HANDBOOK}:p1:c0",
        "clause": f"{CONTRACT}:p3:c2",
        "identifier": f"{ERRORS}:p2:c1",
        "number": f"{CONTRACT}:p4:c3",
    }


COMPARISON_QUESTIONS = {
    "semantic": "How much vacation time do I get?",  # different words, same meaning
    "clause": "What does clause 7.3 say?",           # exact clause number (7.1, 7.2 and 3.7 are decoys)
    "identifier": "What does ERR-4012 mean?",        # technical identifier (ERR-4011/4021/4022 are decoys)
    "number": "Which rule mentions 14 days?",        # numerical value (other "N days" rules are decoys)
}
MODES = ("dense", "sparse", "hybrid")


def rank_of(chunk_id: str, results) -> int | None:
    return next((i for i, r in enumerate(results, start=1) if r.chunk_id == chunk_id), None)


def compare(expected: dict[str, str], top_k: int = 3) -> dict[str, dict[str, int | None]]:
    """Rank of the right chunk for each question and mode; None = not in the top_k passages."""
    return {
        kind: {mode: rank_of(expected[kind], retrieve(q, top_k, mode=mode)) for mode in MODES}
        for kind, q in COMPARISON_QUESTIONS.items()
    }


def test_semantic_wording_is_found_by_dense_and_hybrid_but_not_sparse(comparison_corpus) -> None:
    ranks = compare(comparison_corpus)["semantic"]

    assert ranks["dense"] is not None
    assert ranks["sparse"] is None  # "vacation" never appears in the handbook
    assert ranks["hybrid"] is not None


@pytest.mark.parametrize("kind", ["clause", "identifier", "number"])
def test_exact_terms_are_found_by_sparse_and_hybrid_but_missed_by_dense(comparison_corpus, kind: str) -> None:
    ranks = compare(comparison_corpus)[kind]

    assert ranks["dense"] is None  # the look-alike decoys crowd it out of the top 3
    assert ranks["sparse"] == 1
    assert ranks["hybrid"] is not None


def test_dense_cannot_tell_look_alike_identifiers_apart(comparison_corpus) -> None:
    results = retrieve(COMPARISON_QUESTIONS["identifier"], 4, mode="dense")
    right = next(r for r in results if r.chunk_id == comparison_corpus["identifier"])

    # The right chunk is not the most similar one: only its code differs from the decoys.
    assert right.similarity < results[0].similarity


def test_only_hybrid_finds_every_question_type(comparison_corpus) -> None:
    table = compare(comparison_corpus)
    found = {mode: sum(table[kind][mode] is not None for kind in table) for mode in MODES}

    assert found == {"dense": 1, "sparse": 3, "hybrid": 4}


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
async def test_question_without_evidence_is_not_found_in_every_mode(comparison_corpus, monkeypatch, mode) -> None:
    calls = []

    async def llm(user_message: str, system_prompt: str) -> str:
        calls.append(user_message)
        return "Paris [1]."

    monkeypatch.setattr(rag, "generate_answer", llm)

    result = await answer_question("What is the capital of France?", retrieval_mode=mode)

    assert result.status == "not_found"
    assert result.answer == NOT_FOUND_MESSAGE
    assert calls == [], "no passage was relevant, so the LLM must not be asked"
