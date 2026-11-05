"""Tests for Cohere reranking. Cohere is always replaced by a fake client: no network calls.

`FakeCohere` behaves like a cross-encoder: it reads the question and each passage
together and scores how many of the question's words (including exact numbers and
codes such as "7.3") the passage contains.
"""

import dataclasses
import logging
import re

import httpx
import pytest
from cohere.errors import (
    BadRequestError,
    GatewayTimeoutError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
)
from cohere.v2.types import V2RerankResponse, V2RerankResponseResultsItem
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services import rag, reranking, sparse_retrieval
from app.services.rag import NOT_FOUND_MESSAGE, answer_question
from app.services.reranking import RerankError, apply_reranking, rerank_chunks
from app.services.retrieval import RetrievedChunk
from tests.pdf_factory import make_pdf

DOC = "a" * 32
SECRET = "co_live_SECRET_KEY_123"
client = TestClient(app)


def chunk(text: str, page: int, similarity: float = 0.5, document_id: str = DOC, filename: str = "contract.pdf"):
    return RetrievedChunk(
        chunk_id=f"{document_id}:p{page}:c{page - 1}", document_id=document_id, filename=filename,
        page_number=page, chunk_index=page - 1, text=text, distance=round(1 - similarity, 4),
        similarity=similarity, dense_rank=page, fusion_score=round(1 / (60 + page), 6),
    )


CLAUSE_37 = chunk("Clause 3.7: goods are delivered within 14 days.", page=1)
CLAUSE_72 = chunk("Clause 7.2: prices are fixed for the first year.", page=2)
CLAUSE_73 = chunk("Clause 7.3: either party may terminate with 30 days notice.", page=3)
CANDIDATES = [CLAUSE_37, CLAUSE_72, CLAUSE_73]  # retrieval order: the right answer is last


class FakeCohere:
    def __init__(self, error: Exception | None = None, results: list[tuple[int, float]] | None = None) -> None:
        self.error, self.results = error, results
        self.calls: list[dict] = []

    @staticmethod
    def score(query: str, document: str) -> float:
        words = set(re.findall(r"[a-z0-9.]+[a-z0-9]|[a-z0-9]", query.lower()))
        text = document.lower()
        return round(sum(word in text for word in words) / (len(words) or 1), 4)

    async def rerank(self, *, model, query, documents, top_n, max_tokens_per_doc, request_options=None):
        self.calls.append({"model": model, "query": query, "documents": list(documents), "top_n": top_n,
                           "max_tokens_per_doc": max_tokens_per_doc, "request_options": request_options})
        if self.error is not None:
            raise self.error
        results = self.results or sorted(
            ((i, self.score(query, d)) for i, d in enumerate(documents)), key=lambda r: -r[1]
        )[:top_n]
        return V2RerankResponse(
            id="test", results=[V2RerankResponseResultsItem(index=i, relevance_score=s) for i, s in results]
        )


@pytest.fixture
def cohere(monkeypatch: pytest.MonkeyPatch):
    """Turn reranking on with a fake Cohere client; returns a function to swap in a failing one."""
    monkeypatch.setenv("COHERE_API_KEY", SECRET)
    monkeypatch.setenv("RERANK_ENABLED", "true")
    get_settings.cache_clear()

    def install(fake: FakeCohere | None = None) -> FakeCohere:
        fake = fake or FakeCohere()
        monkeypatch.setattr(reranking, "get_cohere_client", lambda: fake)
        return fake

    install()
    yield install
    get_settings.cache_clear()


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch):
    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def __call__(self, user_message: str, system_prompt: str) -> str:
            self.calls.append(user_message)
            return "The answer is in the first passage [1]."

    fake = FakeLLM()
    monkeypatch.setattr(rag, "generate_answer", fake)
    return fake


# --- rerank_chunks -----------------------------------------------------------------


async def test_candidates_are_reordered_by_relevance_score(cohere) -> None:
    results = await rerank_chunks("What does clause 7.3 say?", CANDIDATES, top_n=3)

    # Retrieval ranked the right clause last; the reranker moves it to the top.
    assert results[0].chunk_id == CLAUSE_73.chunk_id
    assert {r.chunk_id for r in results} == {c.chunk_id for c in CANDIDATES}
    assert [r.rerank_score for r in results] == sorted((r.rerank_score for r in results), reverse=True)


async def test_top_n_limits_the_results(cohere) -> None:
    fake = cohere()

    results = await rerank_chunks("What does clause 7.3 say?", CANDIDATES, top_n=1)

    assert [r.chunk_id for r in results] == [CLAUSE_73.chunk_id]
    assert fake.calls[0]["top_n"] == 1


async def test_results_are_sorted_even_if_cohere_returns_them_unsorted(cohere) -> None:
    cohere(FakeCohere(results=[(0, 0.2), (2, 0.9), (1, 0.5)]))

    results = await rerank_chunks("q", CANDIDATES, top_n=3)

    assert [r.chunk_id for r in results] == [CLAUSE_73.chunk_id, CLAUSE_72.chunk_id, CLAUSE_37.chunk_id]


async def test_metadata_is_preserved_exactly(cohere) -> None:
    results = await rerank_chunks("What does clause 7.3 say?", CANDIDATES, top_n=3)

    for result in results:
        original = next(c for c in CANDIDATES if c.chunk_id == result.chunk_id)
        assert result.rerank_score is not None
        # Everything except the new score is the original object's data, untouched.
        assert dataclasses.replace(result, rerank_score=None) == original


async def test_only_passage_texts_are_sent_to_cohere(cohere) -> None:
    fake = cohere()

    await rerank_chunks("What does clause 7.3 say?", CANDIDATES, top_n=2)

    call = fake.calls[0]
    assert call["documents"] == [c.text for c in CANDIDATES]
    assert call["query"] == "What does clause 7.3 say?"
    assert call["model"] == get_settings().rerank_model
    assert call["max_tokens_per_doc"] == get_settings().rerank_max_tokens_per_doc
    assert call["request_options"]["timeout_in_seconds"] == get_settings().rerank_timeout_seconds
    sent = repr(call)
    assert DOC not in sent and "contract.pdf" not in sent  # no IDs or filenames leave the server


async def test_empty_candidates_return_empty_without_calling_cohere(cohere) -> None:
    fake = cohere()

    assert await rerank_chunks("q", [], top_n=5) == []
    assert fake.calls == []


async def test_invalid_top_n_is_rejected(cohere) -> None:
    with pytest.raises(ValueError):
        await rerank_chunks("q", CANDIDATES, top_n=0)


@pytest.mark.parametrize("results", [[(7, 0.9)], [(0, 0.9), (0, 0.8)], [(-1, 0.5)]])
async def test_invalid_response_indexes_are_an_error(cohere, results) -> None:
    cohere(FakeCohere(results=results))

    with pytest.raises(RerankError, match="invalid response"):
        await rerank_chunks("q", CANDIDATES, top_n=3)


# --- apply_reranking: status, fallback and failures ----------------------------------

FAILURES = [
    (UnauthorizedError(body={"message": f"invalid api token {SECRET}"}), "The reranking service is misconfigured."),
    (TooManyRequestsError(body={"message": "trial key limit"}), "The reranking service rate limit was reached."),
    (GatewayTimeoutError(body=None), "The reranking service timed out."),
    (httpx.ReadTimeout("read timed out"), "The reranking service timed out."),
    (httpx.ConnectError("connection refused"), "The reranking service is unreachable."),
    (ServiceUnavailableError(body=None), "The reranking service returned an error."),
    (BadRequestError(body={"message": "model not found"}), "The reranking service returned an error."),
]


@pytest.mark.parametrize("error, message", FAILURES, ids=lambda v: type(v).__name__ if isinstance(v, Exception) else "")
async def test_api_failures_fall_back_to_retrieval_order_and_say_so(cohere, caplog, error, message) -> None:
    cohere(FakeCohere(error=error))
    caplog.set_level(logging.INFO)

    outcome = await apply_reranking("What does clause 7.3 say?", CANDIDATES, top_n=2)

    assert outcome.status == "failed"
    assert outcome.error == message
    assert outcome.chunks == CANDIDATES[:2]  # retrieval order, unchanged
    assert all(c.rerank_score is None for c in outcome.chunks)  # never pretends to be reranked
    assert outcome.latency_ms is not None
    assert SECRET not in caplog.text and SECRET not in repr(outcome)
    assert "Clause" not in caplog.text  # no document text in the logs


async def test_missing_api_key_is_reported_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("COHERE_API_KEY", "")
    get_settings.cache_clear()
    reranking.get_cohere_client.cache_clear()
    try:
        outcome = await apply_reranking("q", CANDIDATES, top_n=2)
    finally:
        get_settings.cache_clear()

    assert outcome.status == "failed"
    assert outcome.error == "The reranking service is not configured."


async def test_disabled_keeps_retrieval_order_without_calling_cohere(cohere) -> None:
    fake = cohere()

    outcome = await apply_reranking("q", CANDIDATES, top_n=2, enabled=False)

    assert (outcome.status, outcome.chunks, outcome.latency_ms) == ("disabled", CANDIDATES[:2], None)
    assert fake.calls == []


@pytest.mark.parametrize("candidates", [[], [CLAUSE_73]])
async def test_zero_or_one_candidate_is_skipped_without_calling_cohere(cohere, candidates) -> None:
    fake = cohere()

    outcome = await apply_reranking("q", candidates, top_n=5)

    assert outcome.status == "skipped"
    assert outcome.chunks == candidates
    assert fake.calls == []


async def test_successful_rerank_reports_model_and_latency(cohere) -> None:
    outcome = await apply_reranking("What does clause 7.3 say?", CANDIDATES, top_n=2)

    assert outcome.status == "applied"
    assert outcome.model == get_settings().rerank_model
    assert outcome.latency_ms is not None and outcome.latency_ms >= 0
    assert outcome.error is None


# --- RAG pipeline -------------------------------------------------------------------


@pytest.fixture
def retrieval(monkeypatch: pytest.MonkeyPatch):
    class FakeRetrieval:
        chunks: list[RetrievedChunk] = list(CANDIDATES)
        calls: list[tuple] = []

        def __call__(self, question, top_k, document_id, mode):
            self.calls.append((question, top_k, document_id, mode))
            return self.chunks[:top_k]

    fake = FakeRetrieval()
    monkeypatch.setattr(rag, "retrieve", fake)
    return fake


async def test_rag_uses_reranked_chunks_as_context(cohere, retrieval, llm) -> None:
    result = await answer_question("What does clause 7.3 say?", top_k=2)

    assert result.reranking.status == "applied"
    assert len(result.contexts) == 2
    assert result.contexts[0].chunk_id == CLAUSE_73.chunk_id
    # Passage [1] shown to the LLM is the reranker's top choice; only top_k passages are sent.
    assert llm.calls[0].index(CLAUSE_73.text) < llm.calls[0].index(result.contexts[1].text)
    assert llm.calls[0].count("<passage number=") == 2
    # ...and its citation metadata comes from ChromaDB, not from Cohere or the LLM.
    source = result.sources[0].chunk
    assert (source.chunk_id, source.filename, source.page_number) == (CLAUSE_73.chunk_id, "contract.pdf", 3)
    assert source.rerank_score is not None


async def test_rag_retrieves_a_wider_candidate_pool_only_when_reranking(cohere, retrieval, llm) -> None:
    await answer_question("q", document_id=DOC, top_k=2, retrieval_mode="hybrid")
    await answer_question("q", document_id=DOC, top_k=2, retrieval_mode="hybrid", rerank=False)

    candidates = get_settings().rerank_candidates
    assert retrieval.calls == [("q", candidates, DOC, "hybrid"), ("q", 2, DOC, "hybrid")]


async def test_rag_without_reranking_is_unchanged(cohere, retrieval, llm) -> None:
    fake = cohere()

    result = await answer_question("What does clause 7.3 say?", top_k=2, rerank=False)

    assert result.reranking.status == "disabled"
    assert [c.chunk_id for c in result.contexts] == [CLAUSE_37.chunk_id, CLAUSE_72.chunk_id]
    assert fake.calls == []


async def test_irrelevant_chunks_are_never_sent_to_cohere(cohere, retrieval, llm) -> None:
    fake = cohere()
    unrelated = chunk("The cafeteria opens at eight.", page=4, similarity=0.05)
    retrieval.chunks = [*CANDIDATES, unrelated]

    await answer_question("What does clause 7.3 say?", top_k=2)

    assert unrelated.text not in fake.calls[0]["documents"]


async def test_no_relevant_evidence_is_not_found_without_cohere_or_llm(cohere, retrieval, llm) -> None:
    fake = cohere()
    retrieval.chunks = [chunk("The cafeteria opens at eight.", page=1, similarity=0.05)]

    result = await answer_question("What is the capital of France?")

    assert result.status == "not_found"
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.reranking.status == "skipped"
    assert fake.calls == [] and llm.calls == []


async def test_rerank_failure_still_answers_from_retrieval_order(cohere, retrieval, llm) -> None:
    cohere(FakeCohere(error=TooManyRequestsError(body=None)))

    result = await answer_question("What does clause 7.3 say?", top_k=2)

    assert result.status == "answered"
    assert result.reranking.status == "failed"
    assert [c.chunk_id for c in result.contexts] == [CLAUSE_37.chunk_id, CLAUSE_72.chunk_id]


@pytest.mark.parametrize("top_k", [0, 21])
async def test_invalid_top_k_is_rejected_with_reranking(cohere, retrieval, llm, top_k) -> None:
    with pytest.raises(ValueError):
        await answer_question("q", top_k=top_k)


# --- API: real upload, ChromaDB and hybrid retrieval; only Cohere and Groq are fake ---


@pytest.fixture
def contracts() -> dict[str, str]:
    sparse_retrieval.reset_sparse_index()

    def upload(pages: list[str], filename: str) -> str:
        response = client.post("/api/documents/upload", files={"file": (filename, make_pdf(pages), "application/pdf")})
        assert response.status_code == 200, response.text
        return response.json()["document_id"]

    return {
        "supply": upload([
            "Clause 3.7: the supplier delivers the goods within 14 days.",
            "Clause 7.2: prices for the goods are fixed for the first year.",
            "Clause 7.3: either party may terminate the agreement with 30 days notice.",
        ], "supply-contract.pdf"),
        "lease": upload([
            "Clause 7.1: the rent for the lease is paid monthly.",
            "Clause 7.3: the tenant may terminate the lease with 60 days notice.",
        ], "lease.pdf"),
    }


def search(**payload) -> dict:
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_hybrid_only_vs_reranked_comparison(contracts, cohere) -> None:
    question = "What does clause 7.3 say about terminating the agreement?"
    target = f"{contracts['supply']}:p3:c2"

    hybrid = search(question=question, top_k=3, document_id=contracts["supply"], rerank=False)
    reranked = search(question=question, top_k=3, document_id=contracts["supply"], rerank=True)

    assert hybrid["reranking"]["status"] == "disabled"
    assert all(r["rerank_score"] is None for r in hybrid["results"])
    assert reranked["reranking"]["status"] == "applied"
    assert reranked["reranking"]["latency_ms"] is not None
    assert reranked["results"][0]["chunk_id"] == target
    scores = [r["rerank_score"] for r in reranked["results"]]
    assert scores == sorted(scores, reverse=True)
    # Same candidates, so reranking can only change the order (and keeps all metadata).
    by_id = {r["chunk_id"]: r for r in hybrid["results"]}
    for result in reranked["results"]:
        before = by_id[result["chunk_id"]]
        assert {k: v for k, v in result.items() if k != "rerank_score"} == {
            k: v for k, v in before.items() if k != "rerank_score"
        }


def test_document_filter_is_kept_with_reranking(contracts, cohere, llm) -> None:
    fake = cohere()

    body = client.post("/api/ask", json={
        "question": "What does clause 7.3 say about terminating the agreement?",
        "document_id": contracts["lease"], "rerank": True,
    }).json()

    assert body["status"] == "answered"
    assert {s["document_id"] for s in body["sources"]} == {contracts["lease"]}
    assert body["sources"][0]["filename"] == "lease.pdf"
    assert body["reranking"]["status"] == "applied"
    # Only the lease's passages were sent to Cohere; the other document's never were.
    assert fake.calls[0]["documents"]
    assert all("tenant" in d or "rent" in d for d in fake.calls[0]["documents"])


def test_ask_reports_reranking_and_scores(contracts, cohere, llm) -> None:
    body = client.post("/api/ask", json={
        "question": "What does clause 7.3 say about terminating the agreement?",
        "document_id": contracts["supply"], "top_k": 2,
    }).json()

    assert body["reranking"]["status"] == "applied"
    assert body["reranking"]["model"] == get_settings().rerank_model
    assert body["sources"][0]["chunk_id"] == f"{contracts['supply']}:p3:c2"
    assert body["sources"][0]["rerank_score"] is not None


def test_ask_rerank_failure_is_visible_and_safe(contracts, cohere, llm) -> None:
    cohere(FakeCohere(error=UnauthorizedError(body={"message": f"bad key {SECRET}"})))

    response = client.post("/api/ask", json={"question": "What does clause 7.3 say?", "rerank": True})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["reranking"]["status"] == "failed"
    assert body["reranking"]["error"] == "The reranking service is misconfigured."
    assert all(s["rerank_score"] is None for s in body["sources"])
    assert SECRET not in response.text
