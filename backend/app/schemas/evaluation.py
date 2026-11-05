"""Schemas for reading the saved evaluation reports (written by evaluation/evaluate_rag.py)."""

from typing import Any

from pydantic import BaseModel


class RagasScore(BaseModel):
    mean: float | None  # None when the judge scored no cases
    scored: int
    missing: int


class QuestionTypeResult(BaseModel):
    cases: int
    passed: int


class EvaluationSummary(BaseModel):
    """Every value comes from the report file; checks that a run did not compute are None."""

    cases: int | None = None
    answerable: int | None = None
    unanswerable: int | None = None
    errors: int | None = None
    answer_rate: float | None = None
    refusal_accuracy: float | None = None
    unsupported_answers: int | None = None
    retrieval_hit_rate: float | None = None
    reference_context_recall: float | None = None
    mrr: float | None = None
    citation_accuracy: float | None = None
    latency_ms: dict[str, float | None] | None = None
    # Keyed by Ragas metric name; a metric that was not run is absent.
    ragas: dict[str, RagasScore] = {}
    ragas_complete: bool | None = None
    by_question_type: dict[str, QuestionTypeResult] | None = None


class EvaluationConfigResult(BaseModel):
    name: str  # e.g. "dense", "hybrid", "hybrid+rerank"
    retrieval_mode: str | None = None
    rerank: bool | None = None
    skipped_reason: str | None = None
    summary: EvaluationSummary | None = None  # None when the configuration was skipped


class EvaluationReport(BaseModel):
    report_id: str
    created_at: str
    setup: dict[str, Any]
    configs: list[EvaluationConfigResult]


class EvaluationReportInfo(BaseModel):
    report_id: str
    created_at: str
    config_names: list[str]
    ragas_metrics: list[str]  # Ragas metrics scored in at least one configuration


class EvaluationReportList(BaseModel):
    reports: list[EvaluationReportInfo]  # newest first


class EvaluationImportResponse(BaseModel):
    imported: list[str]  # report IDs stored by this call
    skipped: list[str]  # already in the database
    failed: list[str]  # files that could not be read
