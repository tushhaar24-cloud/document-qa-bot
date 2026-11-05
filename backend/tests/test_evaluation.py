"""Tests for the evaluation script. The RAG pipeline and Ragas are mocked; no Groq, Cohere or judge calls."""

import csv
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pymupdf
import pytest

from evaluation import evaluate_rag
from evaluation.evaluate_rag import (
    DEFAULT_DATASET,
    DOCUMENTS_DIR,
    NO_CONTEXT,
    CaseResult,
    ConfigRun,
    DatasetError,
    EvalCase,
    build_ragas_rows,
    index_documents,
    install_ragas_compat_shim,
    load_dataset,
    parse_args,
    plan_runs,
    run_case,
    run_cases,
    summarize,
    verify_reference_contexts,
    write_reports,
)
from app.core.config import get_settings
from app.services.embeddings import EmbeddingError
from app.services.generation import AnswerGenerationError
from app.services.rag import NOT_FOUND_MESSAGE, Citation, RagAnswer
from app.services.reranking import RerankOutcome
from app.services.retrieval import RetrievedChunk
from tests.pdf_factory import make_pdf

DOC_ID = "d" * 32
NOTICE_TEXT = "After the first year the notice period is two months."
ANSWERABLE = EvalCase(
    id="notice", question_type="numerical", question="What is the notice period?", reference="Two months.",
    reference_contexts=[NOTICE_TEXT], document="handbook.pdf", answerable=True, expected_pages=[6],
)
UNANSWERABLE = EvalCase(
    id="pension", question_type="unanswerable", question="What is the pension contribution?",
    reference="Not stated.", document="handbook.pdf", answerable=False,
)


def src(page: int, filename: str = "handbook.pdf") -> tuple[str, int]:
    return (filename, page)


def chunk(page: int, text: str = "passage text", filename: str = "handbook.pdf") -> RetrievedChunk:
    return RetrievedChunk(f"{DOC_ID}:p{page}:c0", DOC_ID, filename, page, 0, text, 0.3, 0.7)


def rag_answer(status: str, answer: str, cited: list[int], context: list[int], **extra) -> RagAnswer:
    contexts = [chunk(page) for page in context]
    sources = [Citation(number=i + 1, chunk=chunk(page)) for i, page in enumerate(cited)]
    return RagAnswer("q", answer, status, DOC_ID, sources, len(contexts), len(contexts), contexts, **extra)


def write_jsonl(tmp_path, *rows) -> object:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows), encoding="utf-8")
    return path


VALID_ROW = {
    "id": "q1", "question_type": "factual", "question": "Q?", "reference": "A.",
    "reference_contexts": ["Exact sentence."], "document": "doc.pdf", "answerable": True, "expected_pages": [1],
}
VALID_UNANSWERABLE = {
    "id": "q2", "question_type": "unanswerable", "question": "Q?", "reference": "Not stated.",
    "document": "doc.pdf", "answerable": False,
}


# --- dataset -------------------------------------------------------------


def test_shipped_dataset_covers_every_question_type() -> None:
    cases = load_dataset(DEFAULT_DATASET)

    assert len(cases) >= 20
    assert {c.question_type for c in cases} >= {
        "factual", "numerical", "date", "semantic", "multi_passage", "unanswerable",
    }
    assert any(len(c.expected_pages) > 1 for c in cases)  # needs passages from several pages
    for case in cases:
        with pymupdf.open(DOCUMENTS_DIR / case.document) as pdf:
            assert all(1 <= page <= pdf.page_count for page in case.expected_pages), case.id


def test_shipped_reference_passages_are_quoted_from_the_pdfs() -> None:
    # Raises if any reference passage is not word for word on its expected page.
    verify_reference_contexts(load_dataset(DEFAULT_DATASET), DOCUMENTS_DIR)


def test_invented_reference_passage_is_rejected(tmp_path) -> None:
    (tmp_path / "doc.pdf").write_bytes(make_pdf(["The notice period is two months."]))
    case = EvalCase.model_validate({**VALID_ROW, "reference_contexts": ["The notice period is three months."]})

    with pytest.raises(DatasetError, match="q1: a reference context is not on pages"):
        verify_reference_contexts([case], tmp_path)


def test_reference_passage_on_the_wrong_page_is_rejected(tmp_path) -> None:
    (tmp_path / "doc.pdf").write_bytes(make_pdf(["Intro.", "The notice period is two months."]))
    case = EvalCase.model_validate({**VALID_ROW, "reference_contexts": ["notice period is two   months"]})

    with pytest.raises(DatasetError):
        verify_reference_contexts([case], tmp_path)  # expected page 1, text is on page 2
    verify_reference_contexts([replace_pages(case, [2])], tmp_path)  # whitespace and case are ignored


def replace_pages(case: EvalCase, pages: list[int]) -> EvalCase:
    return case.model_copy(update={"expected_pages": pages})


def test_page_beyond_the_document_is_rejected(tmp_path) -> None:
    (tmp_path / "doc.pdf").write_bytes(make_pdf(["Only page."]))

    with pytest.raises(DatasetError, match="has no page 3"):
        verify_reference_contexts([EvalCase.model_validate({**VALID_ROW, "expected_pages": [3]})], tmp_path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("{not json", "line 1"),
        ({**VALID_ROW, "question": "  "}, "question"),
        ({k: v for k, v in VALID_ROW.items() if k != "reference"}, "reference"),  # missing reference answer
        ({**VALID_ROW, "reference": "   "}, "reference"),  # blank reference answer
        ({**VALID_ROW, "expected_pages": []}, "need expected_pages and reference_contexts"),
        ({**VALID_ROW, "reference_contexts": []}, "need expected_pages and reference_contexts"),
        ({**VALID_ROW, "reference_contexts": ["  "]}, "reference_contexts"),
        ({**VALID_ROW, "question_type": "unanswerable"}, "cannot have question_type 'unanswerable'"),
        ({**VALID_UNANSWERABLE, "expected_pages": [1]}, "must have empty expected_pages"),
        ({**VALID_UNANSWERABLE, "question_type": "factual"}, "must have question_type 'unanswerable'"),
        ({**VALID_ROW, "question_type": "trivia"}, "question_type"),
        ({**VALID_ROW, "id": "Bad ID"}, "id"),
        ({**VALID_ROW, "unknown_field": 1}, "unknown_field"),
    ],
)
def test_invalid_dataset_rows_are_reported_with_line_numbers(tmp_path, row, message: str) -> None:
    with pytest.raises(DatasetError) as exc_info:
        load_dataset(write_jsonl(tmp_path, row))
    assert "line 1" in str(exc_info.value)
    assert message in str(exc_info.value)


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    with pytest.raises(DatasetError, match="duplicate id 'q1'"):
        load_dataset(write_jsonl(tmp_path, VALID_ROW, VALID_ROW))


def test_blank_lines_are_ignored_and_empty_datasets_rejected(tmp_path) -> None:
    assert len(load_dataset(write_jsonl(tmp_path, VALID_ROW, "", "  ", VALID_UNANSWERABLE))) == 2
    with pytest.raises(DatasetError, match="no questions"):
        load_dataset(write_jsonl(tmp_path, ""))


def test_missing_dataset_file(tmp_path) -> None:
    with pytest.raises(DatasetError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.jsonl")


# --- running questions -----------------------------------------------------


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Replace answer_question with a fake that returns (or raises) queued responses."""
    calls: list[tuple] = []
    responses: list = []

    async def fake_answer_question(question, document_id, top_k, retrieval_mode=None, rerank=None):
        calls.append((question, document_id, top_k, retrieval_mode, rerank))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(evaluate_rag, "answer_question", fake_answer_question)
    monkeypatch.setattr(evaluate_rag, "RATE_LIMIT_WAIT_SECONDS", 0)
    return calls, responses


async def test_answered_question_records_sources_contexts_and_timings(fake_pipeline) -> None:
    calls, responses = fake_pipeline
    timings = {"retrieval": 12.5, "generation": 800.0}
    responses.append(rag_answer("answered", "Two months [1].", cited=[6], context=[2, 6], timings=timings))

    result = await run_case(ANSWERABLE, DOC_ID, top_k=4, retrieval_mode="hybrid", rerank=False)

    assert calls == [("What is the notice period?", DOC_ID, 4, "hybrid", False)]
    assert result.status == "answered"
    assert result.context_sources == [src(2), src(6)]
    assert result.cited_sources == [src(6)]
    assert result.timings == timings
    assert result.outcome_correct and result.retrieval_hit and result.citation_hit


async def test_run_cases_searches_all_documents_unless_per_document(fake_pipeline) -> None:
    calls, responses = fake_pipeline
    responses += [rag_answer("not_found", NOT_FOUND_MESSAGE, [], [])] * 2

    await run_cases([UNANSWERABLE], None, None, "dense", False)
    await run_cases([UNANSWERABLE], {"handbook.pdf": DOC_ID}, None, "dense", False)

    assert [call[1] for call in calls] == [None, DOC_ID]


async def test_rate_limit_is_retried(fake_pipeline) -> None:
    calls, responses = fake_pipeline
    responses += [AnswerGenerationError("Too many requests.", 429), rag_answer("not_found", NOT_FOUND_MESSAGE, [], [])]

    result = await run_case(UNANSWERABLE, DOC_ID, top_k=None)

    assert len(calls) == 2
    assert result.status == "not_found" and result.outcome_correct


@pytest.mark.parametrize(
    "error",
    [AnswerGenerationError("The answer service timed out.", 504), EmbeddingError("The embedding model is unavailable.", 503)],
)
async def test_failures_are_recorded_instead_of_stopping_the_run(fake_pipeline, error) -> None:
    _, responses = fake_pipeline
    responses.append(error)

    result = await run_case(ANSWERABLE, DOC_ID, top_k=None)

    assert result.status == "error"
    assert result.error == error.message
    assert not result.outcome_correct
    assert result.reference_context_recall is None  # an error is not a retrieval score


async def test_rerank_status_is_recorded(fake_pipeline) -> None:
    _, responses = fake_pipeline
    answer = rag_answer("answered", "Two months [1].", cited=[6], context=[6])
    responses.append(replace(answer, reranking=RerankOutcome([], "applied", "rerank-v5.0-fast", 123.4)))

    result = await run_case(ANSWERABLE, DOC_ID, top_k=None, rerank=True)

    assert (result.rerank_status, result.rerank_latency_ms) == ("applied", 123.4)


def test_wrong_page_cited_is_a_citation_miss() -> None:
    result = CaseResult(ANSWERABLE, "answered", answer="x [1]", cited_sources=[src(2)], context_sources=[src(2), src(6)])
    assert result.outcome_correct
    assert result.retrieval_hit is True
    assert result.citation_hit is False


def test_right_page_of_the_wrong_document_is_not_a_hit() -> None:
    other = src(6, "it_guide.pdf")
    result = CaseResult(ANSWERABLE, "answered", cited_sources=[other], context_sources=[other])
    assert result.retrieval_hit is False
    assert result.citation_hit is False


def test_answering_an_unanswerable_question_is_a_failure() -> None:
    result = CaseResult(UNANSWERABLE, "answered", answer="Invented [1]", cited_sources=[src(3)])
    assert not result.outcome_correct
    assert result.retrieval_hit is None and result.citation_hit is None


def test_reference_context_recall_and_rank() -> None:
    case = ANSWERABLE.model_copy(update={"reference_contexts": [NOTICE_TEXT, "Notice must be in writing."]})
    contexts = ["Office hours are nine to five.", f"Intro. {NOTICE_TEXT.upper()}  More text."]

    result = CaseResult(case, "answered", contexts=contexts)

    assert result.reference_context_recall == 0.5  # one of two passages retrieved (case and spacing ignored)
    assert result.first_relevant_rank == 2
    assert CaseResult(case, "not_found", contexts=["unrelated"]).first_relevant_rank is None


# --- scoring ----------------------------------------------------------------


def test_summary_rates() -> None:
    results = [
        CaseResult(ANSWERABLE, "answered", cited_sources=[src(6)], context_sources=[src(6)], contexts=[NOTICE_TEXT],
                   ragas={"faithfulness": 1.0}, timings={"retrieval": 10.0, "generation": 900.0}, seconds=1.0),
        CaseResult(ANSWERABLE.model_copy(update={"id": "b"}), "answered", cited_sources=[src(2)],
                   context_sources=[src(2), src(6)], contexts=["other", NOTICE_TEXT], ragas={"faithfulness": 0.5},
                   timings={"retrieval": 30.0, "generation": 1100.0}, seconds=1.0),
        CaseResult(ANSWERABLE.model_copy(update={"id": "c"}), "not_found", context_sources=[src(1)],
                   contexts=["other"], ragas={"faithfulness": None}, seconds=1.0),
        CaseResult(UNANSWERABLE, "not_found", seconds=1.0),
        CaseResult(UNANSWERABLE.model_copy(update={"id": "e"}), "answered", cited_sources=[src(1)], seconds=1.0),
        CaseResult(UNANSWERABLE.model_copy(update={"id": "f"}), "error", error="timeout", seconds=1.0),
    ]

    summary = summarize(results)

    assert summary["answer_rate"] == 0.667  # 2 of 3 answerable answered
    assert summary["refusal_accuracy"] == 0.333  # 1 of 3 unanswerable refused
    assert summary["unsupported_answers"] == 1
    assert summary["retrieval_hit_rate"] == 0.667
    assert summary["reference_context_recall"] == 0.667  # 1, 1, 0
    assert summary["mrr"] == 0.5  # ranks 1, 2, missed -> (1 + 0.5 + 0) / 3
    assert summary["citation_accuracy"] == 0.5  # 1 of 2 answers cited page 6
    assert summary["errors"] == 1
    assert summary["latency_ms"] == {"retrieval": 20.0, "rerank": None, "generation": 1000.0, "total": 1000.0}
    assert summary["ragas"] == {"faithfulness": {"mean": 0.75, "scored": 2, "missing": 1}}
    assert summary["by_question_type"]["numerical"] == {"cases": 3, "passed": 1}  # b cited the wrong page


def test_summary_counts_rerank_failures_and_latency() -> None:
    results = [
        CaseResult(ANSWERABLE, "answered", rerank_status="applied", rerank_latency_ms=100.0),
        CaseResult(ANSWERABLE.model_copy(update={"id": "b"}), "answered", rerank_status="applied", rerank_latency_ms=300.0),
        CaseResult(UNANSWERABLE, "not_found", rerank_status="failed", rerank_latency_ms=10000.0),
    ]

    summary = summarize(results)

    assert summary["reranked"] == 2
    assert summary["rerank_failures"] == 1
    assert summary["rerank_mean_latency_ms"] == 200.0  # failed calls are not counted as rerank latency


def test_incomplete_ragas_scores_are_flagged() -> None:
    complete = [CaseResult(ANSWERABLE, "answered", ragas={"faithfulness": 1.0})]
    partial = complete + [CaseResult(ANSWERABLE.model_copy(update={"id": "b"}), "answered", ragas={"faithfulness": None})]

    assert summarize(complete)["ragas_complete"] is True
    assert summarize(partial)["ragas_complete"] is False
    assert summarize([CaseResult(UNANSWERABLE, "not_found")])["ragas_complete"] is None  # Ragas skipped


def test_ragas_rows_cover_answerable_questions_only() -> None:
    results = [
        CaseResult(ANSWERABLE, "answered", answer="Two months [1][2].", contexts=["p6 text", "p2 text"]),
        CaseResult(ANSWERABLE.model_copy(update={"id": "missed"}), "not_found", answer=NOT_FOUND_MESSAGE),
        CaseResult(ANSWERABLE.model_copy(update={"id": "failed"}), "error", error="timeout"),
        CaseResult(UNANSWERABLE, "not_found", answer=NOT_FOUND_MESSAGE),
    ]

    rows = build_ragas_rows(results)

    assert [result.case.id for result, _ in rows] == ["notice", "missed"]
    assert rows[0][1] == {
        "user_input": "What is the notice period?",
        "response": "Two months.",
        "retrieved_contexts": ["p6 text", "p2 text"],
        "reference": "Two months.",
        "reference_contexts": [NOTICE_TEXT],
    }
    # A missed answer is still scored (it should get low scores), with a placeholder context.
    assert rows[1][1]["retrieved_contexts"] == [NO_CONTEXT]


class FakeEvaluation:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


def test_ragas_scores_are_mapped_back_to_their_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metric integration with Ragas' evaluate() mocked: no judge calls."""
    install_ragas_compat_shim()
    import ragas

    seen = {}

    def fake_evaluate(dataset, metrics, llm, embeddings, run_config, raise_exceptions):
        seen["metrics"] = [m.name for m in metrics]
        seen["questions"] = [sample.user_input for sample in dataset.samples]
        seen["raise_exceptions"] = raise_exceptions
        # The second answer's faithfulness failed (NaN), as happens when the judge errors out.
        return FakeEvaluation([
            {"faithfulness": 0.8, "answer_relevancy": 0.91234, "llm_context_precision_with_reference": 1.0,
             "context_recall": 1.0},
            {"faithfulness": float("nan"), "answer_relevancy": 0.4, "llm_context_precision_with_reference": 0.5,
             "context_recall": 0.0},
        ])

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluate_rag, "get_embedding_model", lambda: object())
    results = [
        CaseResult(ANSWERABLE, "answered", answer="Two months [1].", contexts=[NOTICE_TEXT]),
        CaseResult(UNANSWERABLE, "not_found"),  # never sent to the judge
        CaseResult(ANSWERABLE.model_copy(update={"id": "b", "question": "Other?"}), "answered", answer="x", contexts=["y"]),
    ]

    usage = evaluate_rag.run_ragas(results, "judge-model", "test-key")

    # Ragas' own name for context precision differs; reports use this script's short names.
    assert seen["metrics"] == ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
    assert seen["questions"] == ["What is the notice period?", "Other?"]
    assert seen["raise_exceptions"] is False
    assert results[0].ragas == {"faithfulness": 0.8, "answer_relevancy": 0.912, "context_precision": 1.0, "context_recall": 1.0}
    assert results[1].ragas == {}
    assert results[2].ragas["faithfulness"] is None  # missing, never counted as 0
    assert usage == {"calls": 0, "tokens": 0}  # the fake made no judge calls
    summary = summarize(results)
    assert summary["ragas"]["faithfulness"] == {"mean": 0.8, "scored": 1, "missing": 1}
    assert summary["ragas_complete"] is False


def test_token_counter_adds_up_judge_usage() -> None:
    counter = evaluate_rag.make_token_counter()

    counter.on_llm_end(SimpleNamespace(llm_output={"token_usage": {"total_tokens": 120}}))
    counter.on_llm_end(SimpleNamespace(llm_output=None))

    assert (counter.calls, counter.total_tokens) == (2, 120)


# --- reports ------------------------------------------------------------------


def config_run(name: str, results: list[CaseResult], **extra) -> ConfigRun:
    run = ConfigRun(name, "hybrid" if name.startswith("hybrid") else "dense", name.endswith("rerank"), results, **extra)
    run.summary = summarize(results)
    return run


def test_reports_compare_configs_without_document_text(tmp_path) -> None:
    dense = config_run("dense", [
        CaseResult(ANSWERABLE, "answered", answer="Two months [1].", cited_sources=[src(2)], context_sources=[src(2)],
                   contexts=["SECRET PASSAGE TEXT"], ragas={"faithfulness": 0.3}, timings={"retrieval": 5.0}),
        CaseResult(UNANSWERABLE, "not_found", answer=NOT_FOUND_MESSAGE),
    ], judge_usage={"calls": 4, "tokens": 1234})
    hybrid = config_run("hybrid", [
        CaseResult(ANSWERABLE, "answered", answer="Two months [1].", cited_sources=[src(6)], context_sources=[src(6)],
                   contexts=[NOTICE_TEXT], ragas={"faithfulness": 1.0}),
        CaseResult(UNANSWERABLE, "not_found", answer=NOT_FOUND_MESSAGE),
    ])
    rerank = ConfigRun("hybrid+rerank", "hybrid", True, skipped_reason="COHERE_API_KEY is not set")

    json_path, csv_path, markdown_path = write_reports([dense, hybrid, rerank], {"answer model": "m"}, tmp_path)

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert [c["name"] for c in report["configs"]] == ["dense", "hybrid", "hybrid+rerank"]
    assert report["configs"][0]["summary"]["citation_accuracy"] == 0.0
    assert report["configs"][1]["summary"]["citation_accuracy"] == 1.0
    assert report["configs"][2]["skipped_reason"] == "COHERE_API_KEY is not set"
    assert "created_at" in report and report["setup"] == {"answer model": "m"}

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert [(r["config"], r["id"]) for r in rows] == [
        ("dense", "notice"), ("dense", "pension"), ("hybrid", "notice"), ("hybrid", "pension"),
    ]
    assert rows[0]["ragas_faithfulness"] == "0.3"
    assert rows[0]["retrieval_ms"] == "5.0"
    assert "expected page not cited" in rows[0]["needs_review"] and "low faithfulness (0.30)" in rows[0]["needs_review"]
    assert rows[2]["needs_review"] == ""

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| Check | dense | hybrid |" in markdown
    assert "| Citation accuracy | 0.00 | 1.00 |" in markdown
    assert "| Ragas faithfulness | 0.30 (1 scored) | 1.00 (1 scored) |" in markdown
    assert "| Judge tokens used | 1234 | - |" in markdown
    assert "**hybrid+rerank**: not run (COHERE_API_KEY is not set)" in markdown
    assert "#### notice: expected page not retrieved; expected page not cited; low faithfulness (0.30)" in markdown
    for text in (json_path.read_text(encoding="utf-8"), csv_path.read_text(encoding="utf-8"), markdown):
        assert "SECRET PASSAGE TEXT" not in text


def test_incomplete_ragas_scores_are_flagged_in_the_report(tmp_path) -> None:
    run = config_run("hybrid", [
        CaseResult(ANSWERABLE, "answered", ragas={"faithfulness": 1.0}),
        CaseResult(ANSWERABLE.model_copy(update={"id": "b"}), "answered", ragas={"faithfulness": None}),
    ])

    _, _, markdown_path = write_reports([run], {}, tmp_path)

    assert "**Incomplete Ragas scores:**" in markdown_path.read_text(encoding="utf-8")


def test_retrieval_and_generation_failures_are_listed_for_review(tmp_path) -> None:
    run = config_run("dense", [
        CaseResult(ANSWERABLE, "error", error="The answer service timed out."),
        CaseResult(ANSWERABLE.model_copy(update={"id": "missed"}), "not_found", context_sources=[src(1)]),
        CaseResult(UNANSWERABLE, "answered", answer="Invented [1]", cited_sources=[src(3)]),
    ])

    _, _, markdown_path = write_reports([run], {}, tmp_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "#### notice: error: The answer service timed out." in markdown
    assert "#### missed: refused; expected page not retrieved" in markdown
    assert "#### pension: answered although the documents do not say" in markdown


# --- setup and command line -------------------------------------------------------


def test_documents_are_indexed_with_the_app_pipeline(tmp_path) -> None:
    (tmp_path / "handbook.pdf").write_bytes(make_pdf(["Page one.", "Page two."]))

    document_ids = index_documents([ANSWERABLE, UNANSWERABLE], tmp_path)

    assert list(document_ids) == ["handbook.pdf"]
    from app.services.vector_store import get_chunk_ids
    assert len(get_chunk_ids(document_ids["handbook.pdf"])) == 2


def test_missing_document_is_a_dataset_error(tmp_path) -> None:
    with pytest.raises(DatasetError, match="handbook.pdf"):
        index_documents([ANSWERABLE], tmp_path)
    with pytest.raises(DatasetError, match="handbook.pdf"):
        verify_reference_contexts([ANSWERABLE], tmp_path)


def test_ragas_column_names_match_what_ragas_returns() -> None:
    # Ragas stores mode metrics as "name(mode=...)"; reading plain "name" lost every score once.
    assert evaluate_rag.ragas_column(SimpleNamespace(name="factual_correctness", mode="f1")) == "factual_correctness(mode=f1)"
    assert evaluate_rag.ragas_column(SimpleNamespace(name="faithfulness")) == "faithfulness"


def test_compat_shim_adds_only_the_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, name, raising=False)

    install_ragas_compat_shim()

    assert hasattr(sys.modules[name], "ChatVertexAI")
    # An existing module is never replaced.
    existing = sys.modules[name]
    install_ragas_compat_shim()
    assert sys.modules[name] is existing


def test_main_rejects_a_bad_dataset_without_calling_groq(tmp_path, capsys) -> None:
    assert evaluate_rag.main(["--dataset", str(tmp_path / "missing.jsonl")]) == 2
    assert "Dataset error" in capsys.readouterr().out


def test_main_rejects_invented_reference_passages(tmp_path, capsys) -> None:
    row = {**VALID_ROW, "document": "northwind_employee_handbook.pdf", "reference_contexts": ["Everyone gets 99 days off."]}

    assert evaluate_rag.main(["--dataset", str(write_jsonl(tmp_path, row))]) == 2
    assert "a reference context is not on pages" in capsys.readouterr().out


def test_main_requires_an_api_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import dataclasses

    no_key = dataclasses.replace(evaluate_rag.get_settings(), groq_api_key="")
    monkeypatch.setattr(evaluate_rag, "get_settings", lambda: no_key)

    assert evaluate_rag.main(["--skip-ragas"]) == 2
    assert "GROQ_API_KEY is not set" in capsys.readouterr().out


def test_metrics_option() -> None:
    assert parse_args([]).metrics == ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    args = parse_args(["--metrics", "faithfulness, factual_correctness,faithfulness"])
    assert args.metrics == ("faithfulness", "factual_correctness")
    for bad in ["", "faithfulness,made_up"]:
        with pytest.raises(SystemExit):
            parse_args(["--metrics", bad])


def test_configs_option() -> None:
    assert parse_args(["--configs", "dense, hybrid+rerank"]).configs == ("dense", "hybrid+rerank")
    with pytest.raises(SystemExit):
        parse_args(["--configs", "dense,fuzzy"])


@pytest.mark.parametrize("argv, expected", [([], None), (["--rerank"], True), (["--no-rerank"], False)])
def test_rerank_option(argv: list[str], expected) -> None:
    assert parse_args(argv).rerank is expected


def test_rerank_config_is_skipped_without_a_cohere_key() -> None:
    import dataclasses

    no_key = dataclasses.replace(get_settings(), cohere_api_key="")
    with_key = dataclasses.replace(get_settings(), cohere_api_key="test-key")
    args = parse_args(["--configs", "dense,hybrid,hybrid+rerank"])

    assert [run.skipped_reason for run in plan_runs(args, no_key)] == [None, None, "COHERE_API_KEY is not set"]
    assert [(r.name, r.retrieval_mode, r.rerank) for r in plan_runs(args, with_key)] == [
        ("dense", "dense", False), ("hybrid", "hybrid", False), ("hybrid+rerank", "hybrid", True),
    ]
    single = plan_runs(parse_args(["--retrieval-mode", "dense", "--no-rerank"]), with_key)
    assert [(r.name, r.retrieval_mode, r.rerank) for r in single] == [("dense", "dense", False)]


def test_document_filter_with_no_matching_questions_is_rejected(capsys) -> None:
    assert evaluate_rag.main(["--document", "unknown.pdf", "--skip-ragas"]) == 2
    assert "no questions about unknown.pdf" in capsys.readouterr().out
