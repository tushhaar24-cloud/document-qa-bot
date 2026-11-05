"""Repository (CRUD) functions, session handling and the maintenance CLI, on SQLite."""

import pytest
from sqlalchemy import func, select

from app.db import cli
from app.db.models import ChatMessage, Conversation, Document, MessageCitation
from app.db.session import DatabaseError, session_scope
from app.repositories import conversations as conversation_repo
from app.repositories import documents as document_repo
from app.repositories import evaluations as evaluation_repo
from app.repositories.conversations import AnswerData, CitationData
from app.schemas.evaluation import EvaluationConfigResult, EvaluationReport, EvaluationSummary, RagasScore
from app.services import vector_store
from tests.db_helpers import database_down  # noqa: F401  (fixture)

DOC = "0123456789abcdef0123456789abcdef"


# --- Sessions ---

def test_session_scope_commits_on_success() -> None:
    with session_scope() as db:
        document_repo.mark_processing(db, DOC, "a.pdf", 10)
    with session_scope() as db:
        assert db.get(Document, DOC) is not None


def test_session_scope_rolls_back_on_error() -> None:
    with pytest.raises(RuntimeError):
        with session_scope() as db:
            document_repo.mark_processing(db, DOC, "a.pdf", 10)
            raise RuntimeError("boom")
    with session_scope() as db:
        assert db.get(Document, DOC) is None


def test_constraint_violation_becomes_a_safe_500() -> None:
    with pytest.raises(DatabaseError) as error:
        with session_scope() as db:
            db.add(Document(id=DOC, filename="a.pdf", status="not-a-status"))
    assert (error.value.status_code, error.value.message) == (500, "A database error occurred.")


def test_unreachable_database_becomes_a_503(database_down) -> None:
    with pytest.raises(DatabaseError) as error:
        with session_scope() as db:
            db.get(Document, DOC)
    assert (error.value.status_code, error.value.message) == (503, "The database is unavailable.")


# --- Documents ---

def test_document_lifecycle() -> None:
    with session_scope() as db:
        document_repo.mark_processing(db, DOC, "a.pdf", 1234)
        document = document_repo.mark_failed(db, DOC, "Embedding failed.")
        assert (document.status, document.error_message, document.chunk_count) == ("failed", "Embedding failed.", 0)

        # A re-upload resets the error.
        document = document_repo.mark_processing(db, DOC, "a-renamed.pdf", 1234)
        assert (document.status, document.error_message, document.filename) == ("processing", None, "a-renamed.pdf")

        document = document_repo.mark_indexed(
            db, DOC, page_count=3, pages_with_text=2, chunk_count=4,
            embedding_model="m", chunk_size=800, chunk_overlap=120,
        )
        assert document.status == "indexed" and document.indexed_at is not None


def test_mark_indexed_requires_an_existing_record() -> None:
    with pytest.raises(LookupError):
        with session_scope() as db:
            document_repo.mark_indexed(
                db, DOC, page_count=1, pages_with_text=1, chunk_count=1,
                embedding_model="m", chunk_size=800, chunk_overlap=120,
            )


def test_list_documents_filters_by_status() -> None:
    with session_scope() as db:
        document_repo.mark_processing(db, "a" * 32, "a.pdf", 1)
        document_repo.mark_processing(db, "b" * 32, "b.pdf", 1)
        document_repo.mark_failed(db, "b" * 32, "bad")
    with session_scope() as db:
        assert [d.id for d in document_repo.list_documents(db, status="failed")] == ["b" * 32]
        assert len(document_repo.list_documents(db)) == 2
        assert document_repo.delete_document(db, "a" * 32) is True
        assert document_repo.delete_document(db, "a" * 32) is False


def test_upsert_from_vector_store_creates_and_repairs_but_keeps_indexed_rows() -> None:
    with session_scope() as db:
        assert document_repo.upsert_indexed_from_vector_store(db, DOC, "a.pdf", 5, 3) is True
        assert document_repo.upsert_indexed_from_vector_store(db, DOC, "a.pdf", 99, 99) is False
        assert db.get(Document, DOC).chunk_count == 5  # an indexed row is left alone

        document_repo.mark_processing(db, "c" * 32, "stuck.pdf", 1)  # e.g. a crash mid-upload
        assert document_repo.upsert_indexed_from_vector_store(db, "c" * 32, "stuck.pdf", 2, 1) is False
        assert db.get(Document, "c" * 32).status == "indexed"


# --- Conversations ---

def answer(text: str = "Answer [1].", citations: int = 1) -> AnswerData:
    return AnswerData(
        content=text, answer_status="answered", document_id=None, retrieval_mode="hybrid",
        rerank_status="disabled", rerank_model=None, retrieved_chunk_count=5, context_chunk_count=3,
        citations=[
            CitationData(n, f"chunk-{n}", DOC, "a.pdf", n, f"excerpt {n}", 0.5, None) for n in range(1, citations + 1)
        ],
    )


def test_make_title_shortens_long_questions() -> None:
    assert conversation_repo.make_title("  What   is\nthis? ") == "What is this?"
    title = conversation_repo.make_title("word " * 50)
    assert len(title) == conversation_repo.TITLE_LENGTH and title.endswith("…")


def test_exchanges_get_increasing_positions_and_citations() -> None:
    with session_scope() as db:
        conversation = conversation_repo.create_conversation(db, "First?", None)
        conversation_repo.add_exchange(db, conversation, "First?", answer(citations=2))
        conversation_repo.add_exchange(db, conversation, "Second?", answer("No.", citations=0))
        conversation_id = conversation.id

    with session_scope() as db:
        loaded = conversation_repo.get_conversation(db, conversation_id, with_messages=True)
        assert [(m.position, m.role) for m in loaded.messages] == [
            (1, "user"), (2, "assistant"), (3, "user"), (4, "assistant")
        ]
        assert [c.citation_number for c in loaded.messages[1].citations] == [1, 2]
        assert loaded.messages[3].citations == []


def test_list_rename_and_delete_conversations() -> None:
    with session_scope() as db:
        first = conversation_repo.create_conversation(db, "First?", None)
        conversation_repo.add_exchange(db, first, "First?", answer())
        conversation_repo.create_conversation(db, "Empty", DOC)

    with session_scope() as db:
        rows, total = conversation_repo.list_conversations(db, limit=10, offset=0)
        assert total == 2
        assert sorted(count for _, count in rows) == [0, 2]

        conversation = conversation_repo.get_conversation(db, first.id)
        conversation_repo.rename_conversation(db, conversation, "Renamed")
        conversation_repo.delete_conversation(db, conversation)

    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Conversation)) == 1
        # ON DELETE CASCADE removed the messages and citations.
        assert db.scalar(select(func.count()).select_from(ChatMessage)) == 0
        assert db.scalar(select(func.count()).select_from(MessageCitation)) == 0


# --- Evaluations ---

def report(faithfulness: float) -> EvaluationReport:
    return EvaluationReport(
        report_id="eval-20261104-124423",
        created_at="2026-11-04T12:44:23",
        setup={"top_k": 5},
        configs=[
            EvaluationConfigResult(
                name="dense", retrieval_mode="dense", rerank=False,
                summary=EvaluationSummary(cases=3, ragas={"faithfulness": RagasScore(mean=faithfulness, scored=3, missing=0)}),
            ),
            EvaluationConfigResult(name="hybrid+rerank", skipped_reason="COHERE_API_KEY is not set"),
        ],
    )


def test_save_report_replaces_an_earlier_copy() -> None:
    with session_scope() as db:
        evaluation_repo.save_report(db, report(0.5))
    with session_scope() as db:
        evaluation_repo.save_report(db, report(0.9))

    with session_scope() as db:
        runs = evaluation_repo.list_runs(db)
        assert len(runs) == 1
        dense, skipped = runs[0].config_results
        assert [(s.metric_name, s.mean) for s in dense.ragas_scores] == [("faithfulness", 0.9)]
        assert "ragas" not in dense.summary  # stored once, in ragas_metric_scores
        assert skipped.summary is None and skipped.ragas_scores == []


def test_report_time_falls_back_to_the_report_id() -> None:
    assert evaluation_repo.parse_report_time("", "eval-20261104-124423").isoformat() == "2026-11-04T12:44:23"


# --- CLI ---

def test_sync_documents_adds_records_for_chunks_already_in_chromadb(capsys) -> None:
    from langchain_core.documents import Document as Chunk

    chunks = [
        Chunk(id=f"c{n}", page_content=f"text {n}", metadata={"document_id": DOC, "filename": "old.pdf", "page_number": page, "chunk_index": n})
        for n, page in enumerate([1, 1, 2])
    ]
    vector_store.upsert_chunks(chunks, [[1.0] + [0.0] * 383] * 3)

    assert cli.main(["sync-documents"]) == 0
    assert cli.main(["sync-documents"]) == 0  # running it again changes nothing

    with session_scope() as db:
        document = db.get(Document, DOC)
        assert (document.filename, document.status, document.chunk_count, document.pages_with_text) == ("old.pdf", "indexed", 3, 2)
    out = capsys.readouterr().out
    assert "1 new database records created" in out and "0 new database records created" in out
