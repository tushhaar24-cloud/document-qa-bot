"""Dense vs sparse vs hybrid with the real embedding model (all-MiniLM-L6-v2).

Slow and needs the model cached locally, so it only runs with: pytest -m integration -s
It prints the rank of the right passage for every question and mode.
"""

import pytest

from app.services import sparse_retrieval
from app.services.retrieval import retrieve
from tests.test_hybrid_retrieval import COMPARISON_QUESTIONS, CONTRACT, ERRORS, HANDBOOK, MODES, index, rank_of

pytestmark = pytest.mark.integration

TOP_K = 3


@pytest.fixture
def corpus() -> dict[str, str]:
    sparse_retrieval.reset_sparse_index()
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
    yield {
        "semantic": f"{HANDBOOK}:p1:c0",
        "clause": f"{CONTRACT}:p3:c2",
        "identifier": f"{ERRORS}:p2:c1",
        "number": f"{CONTRACT}:p4:c3",
    }
    sparse_retrieval.reset_sparse_index()


def test_real_model_comparison(corpus) -> None:
    table = {
        kind: {mode: rank_of(corpus[kind], retrieve(question, TOP_K, mode=mode)) for mode in MODES}
        for kind, question in COMPARISON_QUESTIONS.items()
    }
    print(f"\nRank of the right passage in the top {TOP_K} (None = missed):")
    for kind, ranks in table.items():
        print(f"  {kind:<11} " + "  ".join(f"{mode}={ranks[mode]}" for mode in MODES))
    found = {mode: sum(table[kind][mode] is not None for kind in table) for mode in MODES}
    print(f"  found       {found}")

    assert found["hybrid"] >= found["dense"]
    assert found["hybrid"] >= found["sparse"]
    assert table["semantic"]["sparse"] is None  # no shared word with "vacation"
    for kind in ("clause", "identifier", "number"):
        assert table[kind]["sparse"] == 1
        assert table[kind]["hybrid"] is not None


def test_real_model_question_without_evidence(corpus) -> None:
    question = "What is the capital of France?"

    dense = retrieve(question, 5, mode="dense")
    print("\nUnrelated question, best dense similarity:", max(r.similarity for r in dense))

    assert retrieve(question, 5, mode="sparse") == []
    assert all(r.similarity < 0.2 for r in dense)  # below RAG_MIN_SIMILARITY, so the LLM is never asked
    assert all(r.bm25_score is None for r in retrieve(question, 5, mode="hybrid"))
