"""Evaluate the RAG pipeline against questions with known correct answers.

Run from the backend folder:
    python -m evaluation.evaluate_rag                         # current settings, pipeline checks + Ragas
    python -m evaluation.evaluate_rag --skip-ragas            # pipeline checks only (no judge tokens)
    python -m evaluation.evaluate_rag --configs dense,hybrid,hybrid+rerank --skip-ragas
    python -m evaluation.evaluate_rag --configs dense,hybrid --metrics faithfulness,answer_relevancy
    python -m evaluation.evaluate_rag --limit 3               # first 3 questions only
    python -m evaluation.evaluate_rag --document northwind_it_service_desk_guide.pdf   # one document's questions

Steps:
1. Check the dataset: every reference passage must appear, word for word, on the expected
   page of its PDF, so reference facts cannot be invented.
2. Index the PDFs in evaluation/documents into a temporary ChromaDB.
   Your real chroma_db/ folder is never read or changed.
3. For each retrieval configuration, ask every question through the real pipeline (the same
   code as POST /api/ask). By default each question searches all test documents, as a user
   would; --per-document restricts it to the question's own document.
4. Score the answers:
   - Pipeline checks, no LLM needed: answered or refused correctly, right page retrieved,
     reference passages retrieved, rank of the first relevant passage, right page cited, latency.
   - Ragas, with a separate Groq model as the judge: faithfulness, answer relevancy, context
     precision and context recall (and optionally factual correctness).
5. Write JSON, CSV and Markdown reports to evaluation/reports/, and store the results in PostgreSQL.
"""

import argparse
import asyncio
import csv
import importlib
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from app.core.config import RETRIEVAL_MODES, get_settings
from app.db.session import DatabaseError, session_scope
from app.repositories import evaluations as evaluation_repo
from app.services import vector_store
from app.services.embeddings import EmbeddingError, get_embedding_model
from app.services.evaluation_reports import load_report_file
from app.services.generation import AnswerGenerationError, close_groq_client
from app.services.ingestion import DocumentIngestionError, extract_pdf_pages, ingest_pdf
from app.services.rag import answer_question
from app.services.reranking import close_cohere_client
from app.services.vector_store import VectorStoreError

logger = logging.getLogger("evaluation")

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
DOCUMENTS_DIR = EVAL_DIR / "documents"
REPORTS_DIR = EVAL_DIR / "reports"

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
# A different model family from the answer model, so the model does not grade its own work.
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"
RATE_LIMIT_ATTEMPTS = 4
RATE_LIMIT_WAIT_SECONDS = 20
NO_CONTEXT = "No passages were retrieved."
# Judge metrics; each costs Groq tokens (the free tier allows about 200,000 judge tokens per day).
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
OPTIONAL_RAGAS_METRICS = ("factual_correctness",)
ALL_RAGAS_METRICS = RAGAS_METRICS + OPTIONAL_RAGAS_METRICS
# Ragas scores below this are listed under "Questions to review".
LOW_SCORE = 0.5

# Named retrieval configurations: (retrieval mode, rerank with Cohere).
CONFIGS = {"dense": ("dense", False), "hybrid": ("hybrid", False), "hybrid+rerank": ("hybrid", True)}

QuestionType = Literal["factual", "numerical", "date", "semantic", "identifier", "multi_passage", "unanswerable"]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DatasetError(Exception):
    pass


class EvalCase(BaseModel):
    """One exam question: what to ask, the correct answer and the exact passages that prove it."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
    question_type: QuestionType
    question: NonEmptyText
    reference: NonEmptyText
    # Sentences copied word for word from the document; checked against the PDF before every run.
    reference_contexts: list[NonEmptyText] = []
    document: Annotated[str, StringConstraints(min_length=1)]
    answerable: bool
    expected_pages: list[Annotated[int, Field(ge=1)]] = []

    @model_validator(mode="after")
    def evidence_matches_answerable(self) -> "EvalCase":
        if self.answerable:
            if not self.expected_pages or not self.reference_contexts:
                raise ValueError("answerable questions need expected_pages and reference_contexts")
            if self.question_type == "unanswerable":
                raise ValueError("answerable questions cannot have question_type 'unanswerable'")
        else:
            if self.expected_pages or self.reference_contexts:
                raise ValueError("unanswerable questions must have empty expected_pages and reference_contexts")
            if self.question_type != "unanswerable":
                raise ValueError("unanswerable questions must have question_type 'unanswerable'")
        return self


def normalize(text: str) -> str:
    """Lowercase with single spaces, so PDF line breaks do not affect text matching."""
    return " ".join(text.split()).lower()


@dataclass
class CaseResult:
    case: EvalCase
    status: str  # "answered", "not_found" or "error"
    answer: str = ""
    # (filename, page) of each passage given to the LLM, in the order it saw them, and of each cited one.
    context_sources: list[tuple[str, int]] = field(default_factory=list)
    cited_sources: list[tuple[str, int]] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # passage texts, only kept in memory
    error: str | None = None
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    rerank_status: str = "disabled"
    rerank_latency_ms: float | None = None
    ragas: dict[str, float | None] = field(default_factory=dict)

    @property
    def outcome_correct(self) -> bool:
        """Answerable questions should be answered; unanswerable ones refused."""
        return self.status == ("answered" if self.case.answerable else "not_found")

    def _is_expected(self, source: tuple[str, int]) -> bool:
        return source[0] == self.case.document and source[1] in self.case.expected_pages

    @property
    def retrieval_hit(self) -> bool | None:
        """An expected page (of the right document) was among the passages given to the LLM."""
        if not self.case.answerable:
            return None
        return any(self._is_expected(source) for source in self.context_sources)

    @property
    def citation_hit(self) -> bool | None:
        if not self.case.answerable or self.status != "answered":
            return None
        return any(self._is_expected(source) for source in self.cited_sources)

    @property
    def reference_context_recall(self) -> float | None:
        """Share of the reference passages found word for word in the passages given to the LLM."""
        if not self.case.answerable or self.status == "error":
            return None
        found = [normalize(ref) for ref in self.case.reference_contexts]
        texts = [normalize(text) for text in self.contexts]
        return round(sum(any(ref in text for text in texts) for ref in found) / len(found), 3)

    @property
    def first_relevant_rank(self) -> int | None:
        """Position (1 = first) of the first passage containing a reference passage; None if none did."""
        if not self.case.answerable:
            return None
        references = [normalize(ref) for ref in self.case.reference_contexts]
        for rank, text in enumerate(self.contexts, start=1):
            if any(ref in normalize(text) for ref in references):
                return rank
        return None


def load_dataset(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path.name}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except ValidationError as exc:
            error = exc.errors()[0]
            location = ".".join(str(part) for part in error["loc"]) or "line"
            raise DatasetError(f"{path.name} line {line_number}: {location}: {error['msg']}") from exc
        if case.id in seen:
            raise DatasetError(f"{path.name} line {line_number}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"{path.name} contains no questions")
    return cases


def verify_reference_contexts(cases: list[EvalCase], documents_dir: Path) -> None:
    """Every reference passage must appear on one of its expected pages; otherwise it was not derived from the PDF."""
    pages_by_document: dict[str, dict[int, str]] = {}
    for case in cases:
        if case.document not in pages_by_document:
            path = documents_dir / case.document
            if not path.is_file():
                raise DatasetError(f"Document not found in {documents_dir.name}/: {case.document}")
            extracted = extract_pdf_pages(path.read_bytes(), max_pages=get_settings().max_pdf_pages)
            pages_by_document[case.document] = {page.page_number: normalize(page.text) for page in extracted.pages}
        pages = pages_by_document[case.document]
        for page in case.expected_pages:
            if page not in pages:
                raise DatasetError(f"{case.id}: {case.document} has no page {page}")
        for reference in case.reference_contexts:
            if not any(normalize(reference) in pages[page] for page in case.expected_pages):
                raise DatasetError(
                    f"{case.id}: a reference context is not on pages {case.expected_pages} of {case.document}"
                )


def index_documents(cases: list[EvalCase], documents_dir: Path) -> dict[str, str]:
    """Upload each referenced PDF through the normal ingestion pipeline; returns filename -> document_id."""
    settings = get_settings()
    document_ids: dict[str, str] = {}
    for name in sorted({case.document for case in cases}):
        path = documents_dir / name
        if not path.is_file():
            raise DatasetError(f"Document not found in {documents_dir.name}/: {name}")
        ingested = ingest_pdf(path.read_bytes(), name, settings)
        document_ids[name] = ingested.document_id
        print(f"Indexed {name}: {ingested.extracted.page_count} pages, {ingested.indexed_chunk_count} chunks")
    return document_ids


async def run_case(
    case: EvalCase,
    document_id: str | None,
    top_k: int | None,
    retrieval_mode: str | None = None,
    rerank: bool | None = None,
) -> CaseResult:
    started = time.perf_counter()
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        try:
            result = await answer_question(case.question, document_id, top_k, retrieval_mode, rerank)
            break
        except AnswerGenerationError as exc:
            if exc.status_code == 429 and attempt < RATE_LIMIT_ATTEMPTS:
                wait = RATE_LIMIT_WAIT_SECONDS * attempt
                print(f"  Groq rate limit reached; waiting {wait}s before retrying")
                await asyncio.sleep(wait)
                continue
            return CaseResult(case, "error", error=exc.message, seconds=time.perf_counter() - started)
        except (EmbeddingError, VectorStoreError) as exc:
            return CaseResult(case, "error", error=exc.message, seconds=time.perf_counter() - started)

    return CaseResult(
        case,
        result.status,
        answer=result.answer,
        context_sources=[(chunk.filename, chunk.page_number) for chunk in result.contexts],
        cited_sources=[(source.chunk.filename, source.chunk.page_number) for source in result.sources],
        contexts=[chunk.text for chunk in result.contexts],
        seconds=time.perf_counter() - started,
        timings=dict(result.timings),
        rerank_status=result.reranking.status,
        rerank_latency_ms=result.reranking.latency_ms,
    )


async def run_cases(
    cases: list[EvalCase],
    document_ids: dict[str, str] | None,
    top_k: int | None,
    retrieval_mode: str | None = None,
    rerank: bool | None = None,
) -> list[CaseResult]:
    """Ask every question. With `document_ids`, each search is limited to the question's document."""
    results = []
    try:
        for number, case in enumerate(cases, start=1):
            document_id = document_ids[case.document] if document_ids else None
            result = await run_case(case, document_id, top_k, retrieval_mode, rerank)
            mark = "ok " if result.outcome_correct else "BAD"
            print(f"[{number:>2}/{len(cases)}] {mark} {case.id:<28} {result.status:<9} ({result.seconds:.1f}s)")
            results.append(result)
    finally:
        # Close the Groq and Cohere clients inside the event loop that created them.
        await close_groq_client()
        await close_cohere_client()
    return results


def strip_citations(answer: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", answer).strip()


def build_ragas_rows(results: list[CaseResult]) -> list[tuple[CaseResult, dict]]:
    """Ragas scores answerable questions only; correct refusals are already measured by the pipeline checks."""
    return [
        (
            result,
            {
                "user_input": result.case.question,
                "response": strip_citations(result.answer),
                "retrieved_contexts": result.contexts or [NO_CONTEXT],
                "reference": result.case.reference,
                "reference_contexts": result.case.reference_contexts,
            },
        )
        for result in results
        if result.case.answerable and result.status != "error"
    ]


def install_ragas_compat_shim() -> None:
    """Let Ragas import with langchain-community 0.4+.

    Ragas 0.3/0.4 imports `langchain_community.chat_models.vertexai`, which was
    removed from langchain-community. Ragas only uses that class in an isinstance()
    check, so an empty stand-in class is enough. Nothing else is patched.
    """
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        module = types.ModuleType(name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = module


def clean_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(score) else round(score, 3)


def ragas_column(metric) -> str:
    """Ragas names the result column of metrics with a mode like "factual_correctness(mode=f1)"."""
    mode = getattr(metric, "mode", None)
    return f"{metric.name}(mode={mode})" if mode else metric.name


def make_token_counter():
    """A LangChain callback that adds up the tokens every judge call used."""
    from langchain_core.callbacks import BaseCallbackHandler

    class TokenCounter(BaseCallbackHandler):
        def __init__(self) -> None:
            self.total_tokens = 0
            self.calls = 0

        def on_llm_end(self, response, **kwargs) -> None:
            self.calls += 1
            usage = (response.llm_output or {}).get("token_usage") or {}
            self.total_tokens += int(usage.get("total_tokens") or 0)

    return TokenCounter()


def run_ragas(
    results: list[CaseResult], judge_model: str, api_key: str, metric_names: tuple[str, ...] = RAGAS_METRICS
) -> dict:
    """Score answers in place; returns how many judge calls and tokens were used."""
    rows = build_ragas_rows(results)
    if not rows:
        return {"calls": 0, "tokens": 0}

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no usage analytics
    install_ragas_compat_shim()
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    counter = make_token_counter()
    # Groq exposes an OpenAI-compatible API, so no extra Groq/LangChain package is needed.
    judge = ChatOpenAI(
        model=judge_model, api_key=api_key, base_url=GROQ_OPENAI_BASE_URL,
        temperature=0, timeout=60, max_retries=6, callbacks=[counter],
    )
    factories = {
        "faithfulness": Faithfulness,
        "factual_correctness": FactualCorrectness,
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        # Groq rejects n > 1, so generate one question per answer instead of the default 3.
        "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
    }
    metrics = [factories[name]() for name in metric_names]
    print(f"\nScoring {len(rows)} answers with Ragas (judge: {judge_model}). This can take several minutes.")
    evaluation = evaluate(
        EvaluationDataset.from_list([row for _, row in rows]),
        metrics=metrics,
        llm=LangchainLLMWrapper(judge),
        # The app's own local embedding model; no text is sent to an embedding API.
        embeddings=LangchainEmbeddingsWrapper(get_embedding_model()),
        # Few parallel calls: the Groq free tier allows about 8,000 tokens per minute. The timeout
        # is per judge call and includes time spent waiting out rate limits.
        run_config=RunConfig(max_workers=2, timeout=360, max_retries=6, max_wait=60),
        raise_exceptions=False,
    )
    frame = evaluation.to_pandas()
    for metric in metrics:
        if ragas_column(metric) not in frame.columns:
            print(f"WARNING: Ragas returned no column for {metric.name}; its scores are recorded as missing.")
    # Scores are stored under this script's metric names: Ragas' own names differ for some metrics
    # (context_precision is "llm_context_precision_with_reference" inside Ragas).
    for (result, _), (_, scores) in zip(rows, frame.iterrows()):
        result.ragas = {
            name: clean_score(scores.get(ragas_column(metric))) for name, metric in zip(metric_names, metrics)
        }
    print(f"Judge used {counter.total_tokens} tokens in {counter.calls} calls.")
    return {"calls": counter.calls, "tokens": counter.total_tokens}


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def summarize(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]
    answered = [r for r in answerable if r.status == "answered"]
    scored = [r for r in answerable if r.status != "error"]

    ragas_summary = {}
    for name in sorted({name for r in results for name in r.ragas}):
        scores = [r.ragas[name] for r in results if r.ragas.get(name) is not None]
        ragas_summary[name] = {
            "mean": mean(scores),
            "scored": len(scores),
            "missing": sum(1 for r in results if name in r.ragas and r.ragas[name] is None),
        }

    by_type: dict[str, dict] = {}
    for result in results:
        item = by_type.setdefault(result.case.question_type, {"cases": 0, "passed": 0})
        item["cases"] += 1
        item["passed"] += result.outcome_correct and result.citation_hit is not False

    def timing(stage: str) -> float | None:
        values = [r.timings[stage] for r in results if stage in r.timings]
        return round(statistics.fmean(values), 1) if values else None

    rerank_latencies = [r.rerank_latency_ms for r in results if r.rerank_status == "applied" and r.rerank_latency_ms]
    return {
        "cases": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "errors": sum(r.status == "error" for r in results),
        "answer_rate": rate(len(answered), len(answerable)),
        "refusal_accuracy": rate(sum(r.status == "not_found" for r in unanswerable), len(unanswerable)),
        "unsupported_answers": sum(r.status == "answered" for r in unanswerable),
        "retrieval_hit_rate": rate(sum(bool(r.retrieval_hit) for r in answerable), len(answerable)),
        "reference_context_recall": mean([r.reference_context_recall for r in scored]),
        # Mean of 1/rank of the first passage containing reference text (1.0 = always first).
        "mrr": mean([1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in scored]),
        "citation_accuracy": rate(sum(bool(r.citation_hit) for r in answered), len(answered)),
        "by_question_type": dict(sorted(by_type.items())),
        "latency_ms": {
            "retrieval": timing("retrieval"),
            "rerank": timing("rerank"),
            "generation": timing("generation"),
            "total": round(statistics.fmean(r.seconds for r in results) * 1000, 1) if results else None,
        },
        "reranked": sum(r.rerank_status == "applied" for r in results),
        # Reranking failed and the retrieval order was used: such a run does not measure reranking.
        "rerank_failures": sum(r.rerank_status == "failed" for r in results),
        "rerank_mean_latency_ms": round(statistics.fmean(rerank_latencies), 1) if rerank_latencies else None,
        "ragas": ragas_summary,
        # False when the judge failed on some answers (often the daily token limit); averages then
        # cover only the scored answers and must not be compared with a complete run.
        "ragas_complete": all(item["missing"] == 0 for item in ragas_summary.values()) if ragas_summary else None,
    }


def case_record(result: CaseResult) -> dict:
    # Passage texts are deliberately left out so reports never copy whole document chunks.
    return {
        "id": result.case.id,
        "question_type": result.case.question_type,
        "question": result.case.question,
        "reference": result.case.reference,
        "document": result.case.document,
        "answerable": result.case.answerable,
        "expected_pages": result.case.expected_pages,
        "status": result.status,
        "outcome_correct": result.outcome_correct,
        "answer": result.answer,
        "retrieved_sources": [f"{name} p{page}" for name, page in result.context_sources],
        "cited_sources": [f"{name} p{page}" for name, page in result.cited_sources],
        "retrieval_hit": result.retrieval_hit,
        "reference_context_recall": result.reference_context_recall,
        "first_relevant_rank": result.first_relevant_rank,
        "citation_hit": result.citation_hit,
        "error": result.error,
        "seconds": round(result.seconds, 2),
        "timings_ms": result.timings,
        "rerank_status": result.rerank_status,
        "rerank_latency_ms": result.rerank_latency_ms,
        "ragas": result.ragas,
    }


def review_reasons(record: dict) -> list[str]:
    """Why a question needs a human look; empty when it does not."""
    reasons = []
    if record["status"] == "error":
        reasons.append(f"error: {record['error']}")
    elif not record["outcome_correct"]:
        reasons.append("answered although the documents do not say" if not record["answerable"] else "refused")
    if record["retrieval_hit"] is False:
        reasons.append("expected page not retrieved")
    if record["citation_hit"] is False:
        reasons.append("expected page not cited")
    reasons += [f"low {name} ({score:.2f})" for name, score in record["ragas"].items() if score is not None and score < LOW_SCORE]
    return reasons


@dataclass
class ConfigRun:
    """One retrieval configuration evaluated on the whole dataset."""

    name: str
    retrieval_mode: str
    rerank: bool
    results: list[CaseResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    skipped_reason: str | None = None
    judge_usage: dict | None = None


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


COMPARISON_ROWS = [
    ("Answer rate", "answer_rate"),
    ("Refusal accuracy", "refusal_accuracy"),
    ("Unsupported answers (should be 0)", "unsupported_answers"),
    ("Retrieval hit rate (right page retrieved)", "retrieval_hit_rate"),
    ("Reference passages retrieved", "reference_context_recall"),
    ("MRR (first relevant passage rank)", "mrr"),
    ("Citation accuracy", "citation_accuracy"),
    ("Errors", "errors"),
    ("Rerank failures (should be 0)", "rerank_failures"),
]


def render_markdown(created_at: str, run_info: dict, runs: list[ConfigRun]) -> str:
    lines = [f"# RAG evaluation report ({created_at})", "", "## Setup", ""]
    lines += [f"- **{key}**: {value}" for key, value in run_info.items()]
    done = [run for run in runs if run.skipped_reason is None]
    for run in runs:
        if run.skipped_reason:
            lines += [f"- **{run.name}**: not run ({run.skipped_reason})"]

    if done:
        header = "| Check | " + " | ".join(run.name for run in done) + " |"
        lines += ["", "## Comparison", "", header, "|---" * (len(done) + 1) + "|"]
        for label, key in COMPARISON_ROWS:
            lines.append(f"| {label} | " + " | ".join(fmt(run.summary[key]) for run in done) + " |")
        for stage in ("retrieval", "rerank", "generation", "total"):
            values = [fmt(run.summary["latency_ms"][stage]) for run in done]
            lines.append(f"| Mean {stage} latency (ms) | " + " | ".join(values) + " |")
        for name in sorted({name for run in done for name in run.summary["ragas"]}):
            values = []
            for run in done:
                item = run.summary["ragas"].get(name)
                values.append(f"{fmt(item['mean'])} ({item['scored']} scored)" if item else "-")
            lines.append(f"| Ragas {name} | " + " | ".join(values) + " |")
        if any(run.judge_usage for run in done):
            lines.append("| Judge tokens used | " + " | ".join(
                fmt(run.judge_usage["tokens"]) if run.judge_usage else "-" for run in done) + " |")
        if any(run.summary["ragas_complete"] is False for run in done):
            lines += [
                "",
                "> **Incomplete Ragas scores:** the judge could not score some answers (often the Groq daily "
                "token limit). Those averages cover only the scored answers; do not compare them with complete ones.",
            ]
        if any(run.summary["rerank_failures"] for run in done):
            lines += ["", "> **Reranking failed** for some questions; those used the hybrid order instead."]

    for run in done:
        lines += ["", f"## {run.name} (retrieval: {run.retrieval_mode}, rerank: {'on' if run.rerank else 'off'})", ""]
        lines += ["| Question type | Passed / cases |", "|---|---|"]
        lines += [f"| {kind} | {item['passed']} / {item['cases']} |" for kind, item in run.summary["by_question_type"].items()]
        records = [case_record(result) for result in run.results]
        metric_names = list(run.summary["ragas"])
        lines += ["", "| Result | ID | Type | Got | Expected / retrieved / cited pages | Ref. passages | " + " | ".join(metric_names) + " |"]
        lines += ["|---" * (6 + len(metric_names)) + "|"]
        for record in records:
            pages = f"{record['expected_pages'] or '-'} / {len(record['retrieved_sources'])} passages / {record['cited_sources'] or '-'}"
            scores = [fmt(record["ragas"].get(name)) for name in metric_names]
            flag = "REVIEW" if review_reasons(record) else "PASS"
            lines.append(
                f"| {flag} | {record['id']} | {record['question_type']} | {record['status']} | {pages} | "
                f"{fmt(record['reference_context_recall'])} | " + " | ".join(scores) + " |"
            )
        to_review = [(record, review_reasons(record)) for record in records if review_reasons(record)]
        if to_review:
            lines += ["", f"### Questions to review ({run.name})", ""]
            for record, reasons in to_review:
                lines += [
                    f"#### {record['id']}: {'; '.join(reasons)}",
                    f"- **Question:** {record['question']}",
                    f"- **Correct answer:** {record['reference']}",
                    f"- **Bot answer:** {record['answer'] or record['error']}",
                    f"- **Expected pages:** {record['expected_pages']} of {record['document']}",
                    f"- **Retrieved:** {', '.join(record['retrieved_sources']) or '-'}",
                    "",
                ]
    return "\n".join(lines) + "\n"


CSV_FIELDS = [
    "config", "id", "question_type", "document", "answerable", "expected_pages", "status", "outcome_correct",
    "retrieval_hit", "reference_context_recall", "first_relevant_rank", "citation_hit", "retrieved_sources",
    "cited_sources", "retrieval_ms", "rerank_ms", "generation_ms", "total_ms", "rerank_status", "error",
    *(f"ragas_{name}" for name in ALL_RAGAS_METRICS), "needs_review", "question", "reference", "answer",
]


def write_reports(runs: list[ConfigRun], run_info: dict, reports_dir: Path) -> tuple[Path, Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    created_at = now.isoformat(timespec="seconds")
    stem = f"eval-{now:%Y%m%d-%H%M%S}"

    json_path = reports_dir / f"{stem}.json"
    json_path.write_text(json.dumps({
        "created_at": created_at,
        "setup": run_info,
        "configs": [
            {
                "name": run.name,
                "retrieval_mode": run.retrieval_mode,
                "rerank": run.rerank,
                "skipped_reason": run.skipped_reason,
                "judge_usage": run.judge_usage,
                "summary": run.summary,
                "cases": [case_record(result) for result in run.results],
            }
            for run in runs
        ],
    }, indent=2), encoding="utf-8")

    csv_path = reports_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in runs:
            for result in run.results:
                record = case_record(result)
                writer.writerow({
                    "config": run.name,
                    **{key: record[key] for key in (
                        "id", "question_type", "document", "answerable", "status", "outcome_correct",
                        "retrieval_hit", "reference_context_recall", "first_relevant_rank", "citation_hit",
                        "error", "rerank_status", "question", "reference", "answer",
                    )},
                    "expected_pages": " ".join(map(str, record["expected_pages"])),
                    "retrieved_sources": "; ".join(record["retrieved_sources"]),
                    "cited_sources": "; ".join(record["cited_sources"]),
                    "retrieval_ms": result.timings.get("retrieval"),
                    "rerank_ms": result.timings.get("rerank"),
                    "generation_ms": result.timings.get("generation"),
                    "total_ms": round(result.seconds * 1000, 1),
                    **{f"ragas_{name}": result.ragas.get(name) for name in ALL_RAGAS_METRICS},
                    "needs_review": "; ".join(review_reasons(record)),
                })

    markdown_path = reports_dir / f"{stem}.md"
    markdown_path.write_text(render_markdown(created_at, run_info, runs), encoding="utf-8")
    return json_path, csv_path, markdown_path


def print_summary(run: ConfigRun) -> None:
    summary = run.summary
    print(f"\n=== {run.name} ===")
    print(f"Answer rate (answerable answered):        {fmt(summary['answer_rate'])}")
    print(f"Refusal accuracy (unanswerable refused):  {fmt(summary['refusal_accuracy'])}")
    print(f"Unsupported answers (should be 0):        {summary['unsupported_answers']}")
    print(f"Retrieval hit rate (right page found):    {fmt(summary['retrieval_hit_rate'])}")
    print(f"Reference passages retrieved:             {fmt(summary['reference_context_recall'])}")
    print(f"MRR (rank of first relevant passage):     {fmt(summary['mrr'])}")
    print(f"Citation accuracy (right page cited):     {fmt(summary['citation_accuracy'])}")
    print(f"Errors:                                   {summary['errors']}")
    latency = summary["latency_ms"]
    print(f"Mean latency ms (retrieval/rerank/LLM):   {fmt(latency['retrieval'])} / {fmt(latency['rerank'])} / {fmt(latency['generation'])}")
    if summary["rerank_failures"]:
        print("WARNING: reranking failed for some questions; this run does not fully measure reranking.")
    for name, item in summary["ragas"].items():
        print(f"Ragas {name:<36} {fmt(item['mean'])}  (scored {item['scored']}, missing {item['missing']})")
    if summary["ragas_complete"] is False:
        print("WARNING: Ragas scores are incomplete (often the Groq daily token limit). Do not compare these")
        print("         averages with a complete run; rerun later or use --metrics to score fewer metrics.")


def parse_metrics(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown = [name for name in names if name not in ALL_RAGAS_METRICS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(ALL_RAGAS_METRICS)}")
    return names


def parse_configs(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown = [name for name in names if name not in CONFIGS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(CONFIGS)}")
    return names


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a question set.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL file of questions")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument(
        "--document", default=None,
        help="only ask the questions about this PDF (all test documents are still indexed and searched)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="passages given to the LLM (default: RETRIEVAL_TOP_K)")
    parser.add_argument(
        "--configs", type=parse_configs, default=None,
        help=f"compare named configurations on the same questions: {', '.join(CONFIGS)}",
    )
    parser.add_argument(
        "--retrieval-mode", choices=RETRIEVAL_MODES, default=None,
        help="without --configs: dense, sparse or hybrid (default: RETRIEVAL_MODE)",
    )
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=None,
        help="without --configs: rerank with Cohere (--rerank) or not (--no-rerank); default: RERANK_ENABLED",
    )
    parser.add_argument(
        "--per-document", action="store_true",
        help="search only each question's own document (default: search all test documents)",
    )
    parser.add_argument("--skip-ragas", action="store_true", help="only run the pipeline checks")
    parser.add_argument(
        "--metrics", type=parse_metrics, default=RAGAS_METRICS,
        help=f"comma-separated Ragas metrics (default: {','.join(RAGAS_METRICS)}; "
        f"also available: {','.join(OPTIONAL_RAGAS_METRICS)})",
    )
    parser.add_argument(
        "--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL), help="Groq model used by Ragas"
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args(argv)


def store_report_in_database(json_path: Path) -> bool:
    """Save a written report to PostgreSQL. The report files stay the primary output,
    so a database problem is reported but does not fail the evaluation."""
    try:
        report = load_report_file(json_path)
        with session_scope() as db:
            evaluation_repo.save_report(db, report)
    except DatabaseError as exc:
        print(f"Not saved to the database ({exc.message}). Later run: python -m app.db.cli import-reports")
        return False
    print(f"Saved {report.report_id} to the database.")
    return True


def plan_runs(args: argparse.Namespace, settings) -> list[ConfigRun]:
    if args.configs:
        runs = [ConfigRun(name, *CONFIGS[name]) for name in args.configs]
    else:
        mode = args.retrieval_mode or settings.retrieval_mode
        rerank = settings.rerank_enabled if args.rerank is None else args.rerank
        runs = [ConfigRun(f"{mode}+rerank" if rerank else mode, mode, rerank)]
    for run in runs:
        if run.rerank and not settings.cohere_api_key:
            run.skipped_reason = "COHERE_API_KEY is not set"
    return runs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        all_cases = load_dataset(args.dataset)
        verify_reference_contexts(all_cases, DOCUMENTS_DIR)
        cases = [case for case in all_cases if args.document in (None, case.document)]
        if not cases:
            raise DatasetError(f"no questions about {args.document}")
        if args.limit:
            cases = cases[: args.limit]
    except DatasetError as exc:
        print(f"Dataset error: {exc}")
        return 2

    if not get_settings().groq_api_key:
        print("GROQ_API_KEY is not set; add it to backend/.env.")
        return 2

    # Use a throwaway ChromaDB so the evaluation never touches the real index.
    workdir = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    previous_chroma_dir = os.environ.get("CHROMA_PERSIST_DIR")
    os.environ["CHROMA_PERSIST_DIR"] = str(workdir)
    get_settings.cache_clear()
    vector_store.reset_collection_cache()
    try:
        settings = get_settings()
        if args.top_k is not None and not 1 <= args.top_k <= settings.retrieval_max_top_k:
            print(f"--top-k must be between 1 and {settings.retrieval_max_top_k}")
            return 2
        runs = plan_runs(args, settings)
        if all(run.skipped_reason for run in runs):
            print("Nothing to run: " + "; ".join(f"{run.name}: {run.skipped_reason}" for run in runs))
            return 2

        # Index every document of the dataset, so filtering questions never makes the search easier.
        document_ids = index_documents(all_cases, DOCUMENTS_DIR)
        for run in runs:
            if run.skipped_reason:
                print(f"\nSkipping {run.name}: {run.skipped_reason}")
                continue
            print(f"\n=== {run.name}: asking {len(cases)} questions (answer model: {settings.groq_model}) ===")
            run.results = asyncio.run(run_cases(
                cases, document_ids if args.per_document else None, args.top_k, run.retrieval_mode, run.rerank,
            ))
            if not args.skip_ragas:
                run.judge_usage = run_ragas(run.results, args.judge_model, settings.groq_api_key, args.metrics)
            run.summary = summarize(run.results)
            print_summary(run)

        run_info = {
            "dataset": f"{args.dataset.name} ({len(cases)} of {len(all_cases)} questions)",
            "documents": ", ".join(sorted(document_ids)),
            "search scope": "each question's own document" if args.per_document else "all test documents",
            "answer model": settings.groq_model,
            "judge model": "skipped" if args.skip_ragas else args.judge_model,
            "ragas metrics": "skipped" if args.skip_ragas else ", ".join(args.metrics),
            "embedding model": settings.embedding_model_name,
            "chunking": f"{settings.chunk_size} characters, {settings.chunk_overlap} overlap",
            "top_k": args.top_k or settings.retrieval_top_k,
            "hybrid candidates (dense / sparse), RRF k": (
                f"{settings.hybrid_dense_candidates} / {settings.hybrid_sparse_candidates}, {settings.rrf_k}"
            ),
            "rerank model, candidates": f"{settings.rerank_model}, {settings.rerank_candidates}",
            "rag_min_similarity": settings.rag_min_similarity,
        }
        json_path, csv_path, markdown_path = write_reports(runs, run_info, args.reports_dir)
        print(f"\nReports:\n  {markdown_path}\n  {csv_path}\n  {json_path}")
        store_report_in_database(json_path)
        return 0
    except DatasetError as exc:
        print(f"Dataset error: {exc}")
        return 2
    except (DocumentIngestionError, EmbeddingError, VectorStoreError) as exc:
        print(f"Evaluation failed: {exc.message}")
        return 1
    finally:
        from chromadb.api.client import SharedSystemClient

        vector_store.reset_collection_cache()
        SharedSystemClient.clear_system_cache()
        if previous_chroma_dir is None:
            os.environ.pop("CHROMA_PERSIST_DIR", None)
        else:
            os.environ["CHROMA_PERSIST_DIR"] = previous_chroma_dir
        get_settings.cache_clear()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
