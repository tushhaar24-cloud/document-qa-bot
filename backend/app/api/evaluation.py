"""Evaluation results stored in PostgreSQL.

Reports are produced offline by `python -m evaluation.evaluate_rag`, which also saves
them to the database. Reports written before that (or while the database was down)
are loaded with POST /api/evaluation/reports/import. This API never runs an
evaluation itself (a run takes minutes and spends LLM judge tokens).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi import Path as PathParam
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories import evaluations as evaluation_repo
from app.schemas.evaluation import EvaluationImportResponse, EvaluationReport, EvaluationReportList
from app.services import evaluation_reports
from app.services.evaluation_reports import REPORT_ID_PATTERN, import_report_files, run_to_info, run_to_report

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

DB_ERROR_RESPONSES = {500: {"description": "Database error"}, 503: {"description": "Database unavailable"}}

ReportIdPath = Annotated[str, PathParam(pattern=REPORT_ID_PATTERN, description="e.g. eval-20261104-124423")]
DbSession = Annotated[Session, Depends(get_db)]


@router.get("/reports", response_model=EvaluationReportList, responses=DB_ERROR_RESPONSES)
def list_reports(db: DbSession) -> EvaluationReportList:
    """Stored evaluation reports, newest first. Empty when none have been stored."""
    return EvaluationReportList(reports=[run_to_info(run) for run in evaluation_repo.list_runs(db)])


@router.get(
    "/reports/{report_id}",
    response_model=EvaluationReport,
    responses={404: {"description": "Report not found"}, **DB_ERROR_RESPONSES},
)
def get_report(report_id: ReportIdPath, db: DbSession) -> EvaluationReport:
    """Summary metrics of one report, per retrieval configuration (per-question details are not stored)."""
    run = evaluation_repo.get_run(db, report_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation report not found.")
    return run_to_report(run)


@router.post("/reports/import", response_model=EvaluationImportResponse, responses=DB_ERROR_RESPONSES)
def import_reports(db: DbSession) -> EvaluationImportResponse:
    """Store the JSON reports in evaluation/reports/ that are not in the database yet."""
    # Read the module attribute at call time so tests can point it at another folder.
    result = import_report_files(db, evaluation_reports.REPORTS_DIR)
    db.commit()
    return EvaluationImportResponse(imported=result.imported, skipped=result.skipped, failed=result.failed)


@router.delete(
    "/reports/{report_id}",
    status_code=204,
    responses={404: {"description": "Report not found"}, **DB_ERROR_RESPONSES},
)
def delete_report(report_id: ReportIdPath, db: DbSession) -> Response:
    """Remove a report from the database (its JSON file is kept and can be imported again)."""
    if not evaluation_repo.delete_run(db, report_id):
        raise HTTPException(status_code=404, detail="Evaluation report not found.")
    db.commit()
    return Response(status_code=204)
