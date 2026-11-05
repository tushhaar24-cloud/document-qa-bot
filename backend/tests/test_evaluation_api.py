"""Evaluation results in the database: importing report files and reading them through the API."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.models import EvaluationConfigResult, RagasMetricScore
from app.db.session import session_scope
from app.main import app
from app.services import evaluation_reports
from evaluation import evaluate_rag
from tests.db_helpers import database_down  # noqa: F401  (fixture)

client = TestClient(app)

COMPARISON_REPORT = {
    "created_at": "2026-11-04T12:44:23",
    "setup": {"top_k": 5, "answer model": "openai/gpt-oss-120b"},
    "configs": [
        {
            "name": "dense",
            "retrieval_mode": "dense",
            "rerank": False,
            "skipped_reason": None,
            "summary": {
                "cases": 35,
                "mrr": 0.977,
                "latency_ms": {"retrieval": 20.3, "rerank": None},
                "ragas": {
                    "faithfulness": {"mean": 1.0, "scored": 29, "missing": 0},
                    "answer_relevancy": {"mean": 0.836, "scored": 29, "missing": 0},
                },
            },
            "cases": [{"id": "q1"}],
        },
        # evaluate_rag.py writes an empty summary (not null) for a skipped configuration.
        {"name": "hybrid+rerank", "retrieval_mode": "hybrid", "rerank": True,
         "skipped_reason": "COHERE_API_KEY is not set", "summary": {}, "cases": []},
    ],
}

LEGACY_REPORT = {
    "created_at": "2026-10-28T13:49:25",
    "config": {"top_k": 5},
    "summary": {"cases": 18, "ragas": {
        "llm_context_precision_with_reference": {"mean": 0.964, "scored": 14, "missing": 0},
        "context_recall": {"mean": 1.0, "scored": 11, "missing": 3},
    }},
    "cases": [],
}


@pytest.fixture
def reports_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(evaluation_reports, "REPORTS_DIR", tmp_path)
    return tmp_path


def write_report(directory, name: str, data: dict | str) -> None:
    text = data if isinstance(data, str) else json.dumps(data)
    (directory / f"{name}.json").write_text(text, encoding="utf-8")


def import_reports() -> dict:
    response = client.post("/api/evaluation/reports/import")
    assert response.status_code == 200
    return response.json()


def test_no_reports_returns_an_empty_list(reports_dir) -> None:
    assert client.get("/api/evaluation/reports").json() == {"reports": []}


def test_import_stores_reports_and_lists_them_newest_first(reports_dir) -> None:
    write_report(reports_dir, "eval-20261028-134925", LEGACY_REPORT)
    write_report(reports_dir, "eval-20261104-124423", COMPARISON_REPORT)

    assert import_reports() == {"imported": ["eval-20261028-134925", "eval-20261104-124423"], "skipped": [], "failed": []}

    reports = client.get("/api/evaluation/reports").json()["reports"]
    assert [r["report_id"] for r in reports] == ["eval-20261104-124423", "eval-20261028-134925"]
    assert reports[0] == {
        "report_id": "eval-20261104-124423",
        "created_at": "2026-11-04T12:44:23",
        "config_names": ["dense", "hybrid+rerank"],
        "ragas_metrics": ["answer_relevancy", "faithfulness"],
    }


def test_import_is_idempotent_and_reports_unreadable_files(reports_dir) -> None:
    write_report(reports_dir, "eval-20261104-124423", COMPARISON_REPORT)
    write_report(reports_dir, "eval-20261101-100000", "not json")
    import_reports()

    second = import_reports()

    assert second == {"imported": [], "skipped": ["eval-20261104-124423"], "failed": ["eval-20261101-100000"]}
    assert len(client.get("/api/evaluation/reports").json()["reports"]) == 1


def test_report_detail_comes_from_the_database(reports_dir) -> None:
    write_report(reports_dir, "eval-20261104-124423", COMPARISON_REPORT)
    import_reports()
    # Removing the file proves the API reads PostgreSQL, not the file.
    (reports_dir / "eval-20261104-124423.json").unlink()

    body = client.get("/api/evaluation/reports/eval-20261104-124423").json()

    assert body["setup"] == COMPARISON_REPORT["setup"]
    dense, rerank = body["configs"]
    assert dense["summary"]["mrr"] == 0.977
    assert dense["summary"]["latency_ms"] == {"retrieval": 20.3, "rerank": None}
    assert dense["summary"]["ragas"] == {
        "answer_relevancy": {"mean": 0.836, "scored": 29, "missing": 0},
        "faithfulness": {"mean": 1.0, "scored": 29, "missing": 0},
    }
    assert dense["summary"]["reference_context_recall"] is None  # not in the report: never invented
    assert "cases" not in dense
    assert rerank["summary"] is None
    assert rerank["skipped_reason"] == "COHERE_API_KEY is not set"


def test_legacy_report_is_normalized(reports_dir) -> None:
    write_report(reports_dir, "eval-20261028-134925", LEGACY_REPORT)
    import_reports()

    body = client.get("/api/evaluation/reports/eval-20261028-134925").json()

    assert body["setup"] == {"top_k": 5}
    assert [c["name"] for c in body["configs"]] == ["default"]
    assert body["configs"][0]["summary"]["ragas"]["context_precision"] == {"mean": 0.964, "scored": 14, "missing": 0}


def test_ragas_scores_are_stored_one_row_per_metric(reports_dir) -> None:
    write_report(reports_dir, "eval-20261104-124423", COMPARISON_REPORT)
    import_reports()

    with session_scope() as db:
        rows = db.execute(select(RagasMetricScore.metric_name, RagasMetricScore.mean).order_by(RagasMetricScore.metric_name)).all()
        configs = db.scalar(select(func.count()).select_from(EvaluationConfigResult))

    assert [tuple(r) for r in rows] == [("answer_relevancy", 0.836), ("faithfulness", 1.0)]
    assert configs == 2


def test_delete_report(reports_dir) -> None:
    write_report(reports_dir, "eval-20261104-124423", COMPARISON_REPORT)
    import_reports()

    assert client.delete("/api/evaluation/reports/eval-20261104-124423").status_code == 204
    assert client.get("/api/evaluation/reports/eval-20261104-124423").status_code == 404
    assert client.delete("/api/evaluation/reports/eval-20261104-124423").status_code == 404
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(RagasMetricScore)) == 0


def test_missing_report_returns_404(reports_dir) -> None:
    assert client.get("/api/evaluation/reports/eval-20990101-000000").status_code == 404


@pytest.mark.parametrize("report_id", ["latest", "eval-1", "..%2Fsecrets"])
def test_report_id_must_match_the_report_file_pattern(reports_dir, report_id: str) -> None:
    assert client.get(f"/api/evaluation/reports/{report_id}").status_code in (404, 422)


def test_reports_503_when_the_database_is_down(database_down) -> None:
    response = client.get("/api/evaluation/reports")

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is unavailable."}


def test_evaluation_script_saves_its_report(tmp_path, capsys) -> None:
    write_report(tmp_path, "eval-20261104-124423", COMPARISON_REPORT)

    assert evaluate_rag.store_report_in_database(tmp_path / "eval-20261104-124423.json") is True

    assert "Saved eval-20261104-124423 to the database." in capsys.readouterr().out
    body = client.get("/api/evaluation/reports/eval-20261104-124423").json()
    assert body["configs"][0]["summary"]["ragas"]["faithfulness"]["mean"] == 1.0


def test_evaluation_script_keeps_going_when_the_database_is_down(tmp_path, capsys, database_down) -> None:
    write_report(tmp_path, "eval-20261104-124423", COMPARISON_REPORT)

    assert evaluate_rag.store_report_in_database(tmp_path / "eval-20261104-124423.json") is False

    assert "import-reports" in capsys.readouterr().out
