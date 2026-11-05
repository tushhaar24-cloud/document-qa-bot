"""CRUD for stored evaluation results (written by evaluation/evaluate_rag.py)."""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import EvaluationConfigResult, EvaluationRun, RagasMetricScore
from app.schemas.evaluation import EvaluationReport


def get_run(db: Session, run_id: str) -> EvaluationRun | None:
    query = (
        select(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .options(selectinload(EvaluationRun.config_results).selectinload(EvaluationConfigResult.ragas_scores))
    )
    return db.scalars(query).first()


def list_runs(db: Session) -> Sequence[EvaluationRun]:
    query = (
        select(EvaluationRun)
        .order_by(EvaluationRun.run_at.desc(), EvaluationRun.id.desc())
        .options(selectinload(EvaluationRun.config_results).selectinload(EvaluationConfigResult.ragas_scores))
    )
    return db.scalars(query).all()


def run_exists(db: Session, run_id: str) -> bool:
    return db.get(EvaluationRun, run_id) is not None


def save_report(db: Session, report: EvaluationReport) -> EvaluationRun:
    """Store a parsed report, replacing an earlier copy with the same ID."""
    existing = db.get(EvaluationRun, report.report_id)
    if existing is not None:
        db.delete(existing)
        db.flush()
    run = EvaluationRun(id=report.report_id, run_at=parse_report_time(report.created_at, report.report_id), setup=report.setup)
    for position, config in enumerate(report.configs):
        summary = config.summary.model_dump(exclude={"ragas"}) if config.summary else None
        run.config_results.append(
            EvaluationConfigResult(
                position=position,
                name=config.name,
                retrieval_mode=config.retrieval_mode,
                rerank=config.rerank,
                skipped_reason=config.skipped_reason,
                summary=summary,
                ragas_scores=[
                    RagasMetricScore(metric_name=name, mean=score.mean, scored=score.scored, missing=score.missing)
                    for name, score in (config.summary.ragas.items() if config.summary else [])
                ],
            )
        )
    db.add(run)
    db.flush()
    return run


def delete_run(db: Session, run_id: str) -> bool:
    run = db.get(EvaluationRun, run_id)
    if run is None:
        return False
    db.delete(run)
    db.flush()
    return True


def parse_report_time(created_at: str, report_id: str) -> datetime:
    """The report's created_at, falling back to the timestamp in its ID (eval-YYYYMMDD-HHMMSS)."""
    try:
        return datetime.fromisoformat(created_at)
    except ValueError:
        return datetime.strptime(report_id.removeprefix("eval-"), "%Y%m%d-%H%M%S")
