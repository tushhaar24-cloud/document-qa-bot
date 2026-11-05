"""Read evaluation report files and convert between files, API schemas and database rows."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import BACKEND_DIR
from app.db.models import EvaluationRun
from app.repositories import evaluations as evaluation_repo
from app.schemas.evaluation import (
    EvaluationConfigResult,
    EvaluationReport,
    EvaluationReportInfo,
    EvaluationSummary,
    RagasScore,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = BACKEND_DIR / "evaluation" / "reports"
# Matches the file names evaluate_rag.py writes.
REPORT_ID_PATTERN = r"^eval-\d{8}-\d{6}$"

# Older reports used Ragas' internal metric name.
RAGAS_NAME_ALIASES = {"llm_context_precision_with_reference": "context_precision"}

READ_ERRORS = (OSError, ValueError, KeyError, TypeError, ValidationError)


def _normalize_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    # evaluate_rag.py writes an empty summary for a skipped configuration.
    if not summary:
        return None
    ragas = {RAGAS_NAME_ALIASES.get(name, name): score for name, score in (summary.get("ragas") or {}).items()}
    return {**summary, "ragas": ragas}


def parse_report(report_id: str, raw: dict[str, Any]) -> EvaluationReport:
    if "configs" in raw:
        setup = raw.get("setup") or {}
        configs = [
            EvaluationConfigResult(
                name=config["name"],
                retrieval_mode=config.get("retrieval_mode"),
                rerank=config.get("rerank"),
                skipped_reason=config.get("skipped_reason"),
                summary=_normalize_summary(config.get("summary")),
            )
            for config in raw["configs"]
        ]
    else:
        # Reports from before the dense / hybrid / rerank comparison hold a single run.
        setup = raw.get("config") or {}
        mode = setup.get("retrieval_mode")
        configs = [
            EvaluationConfigResult(
                name=mode or "default", retrieval_mode=mode, summary=_normalize_summary(raw.get("summary"))
            )
        ]
    return EvaluationReport(report_id=report_id, created_at=raw.get("created_at", ""), setup=setup, configs=configs)


def load_report_file(path: Path) -> EvaluationReport:
    return parse_report(path.stem, json.loads(path.read_text(encoding="utf-8")))


def run_to_report(run: EvaluationRun) -> EvaluationReport:
    return EvaluationReport(
        report_id=run.id,
        created_at=run.run_at.isoformat(timespec="seconds"),
        setup=run.setup or {},
        configs=[
            EvaluationConfigResult(
                name=config.name,
                retrieval_mode=config.retrieval_mode,
                rerank=config.rerank,
                skipped_reason=config.skipped_reason,
                summary=None
                if config.summary is None
                else EvaluationSummary(
                    **config.summary,
                    ragas={
                        score.metric_name: RagasScore(mean=score.mean, scored=score.scored, missing=score.missing)
                        for score in config.ragas_scores
                    },
                ),
            )
            for config in run.config_results
        ],
    )


def run_to_info(run: EvaluationRun) -> EvaluationReportInfo:
    return EvaluationReportInfo(
        report_id=run.id,
        created_at=run.run_at.isoformat(timespec="seconds"),
        config_names=[config.name for config in run.config_results],
        ragas_metrics=sorted({score.metric_name for config in run.config_results for score in config.ragas_scores}),
    )


@dataclass
class ImportResult:
    imported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already in the database
    failed: list[str] = field(default_factory=list)  # unreadable files


def import_report_files(db: Session, reports_dir: Path, *, replace: bool = False) -> ImportResult:
    """Store every eval-*.json report of `reports_dir` that is not in the database yet."""
    result = ImportResult()
    for path in sorted(reports_dir.glob("eval-*.json")):
        if not replace and evaluation_repo.run_exists(db, path.stem):
            result.skipped.append(path.stem)
            continue
        try:
            report = load_report_file(path)
        except READ_ERRORS:
            logger.warning("Skipping unreadable evaluation report %s", path.name)
            result.failed.append(path.stem)
            continue
        evaluation_repo.save_report(db, report)
        result.imported.append(path.stem)
    return result
