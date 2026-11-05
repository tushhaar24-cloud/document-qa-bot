"""Chat history: /api/ask saves questions, answers and citations; /api/conversations manages them.

Upload, retrieval and ChromaDB are real (fake embeddings); Groq is replaced by a fake.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api import questions
from app.db.models import ChatMessage, MessageCitation
from app.db.session import DatabaseError, session_scope
from app.main import app
from app.services import rag
from app.services.rag import NOT_FOUND_MESSAGE
from tests.db_helpers import database_down  # noqa: F401  (fixture)
from tests.pdf_factory import make_pdf

client = TestClient(app)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch):
    """Answers citing passage 1 when the question mentions "notice"; otherwise NOT_FOUND."""
    calls = []

    async def fake(user_message: str, system_prompt: str) -> str:
        calls.append(user_message)
        question = user_message.rsplit("Question:", 1)[-1]  # the passages come before it
        return "Two months of notice [1]." if "notice" in question.lower() else "NOT_FOUND"

    monkeypatch.setattr(rag, "generate_answer", fake)
    return calls


@pytest.fixture
def handbook() -> str:
    pdf = make_pdf(["Holiday policy: 25 days.", "Resignation notice period is two months after the first year."])
    response = client.post("/api/documents/upload", files={"file": ("handbook.pdf", pdf, "application/pdf")})
    return response.json()["document_id"]


def ask(**body):
    return client.post("/api/ask", json=body)


def count(model) -> int:
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(model))


def test_question_without_conversation_starts_a_new_one(llm, handbook) -> None:
    body = ask(question="What is the notice period?").json()

    assert body["history_saved"] is True
    conversation = client.get(f"/api/conversations/{body['conversation_id']}").json()
    assert conversation["title"] == "What is the notice period?"
    assert conversation["document_id"] is None
    question, answer = conversation["messages"]
    assert question["role"] == "user" and question["content"] == "What is the notice period?"
    assert question["sources"] == [] and question["status"] is None
    assert answer["id"] == body["message_id"]
    assert answer["role"] == "assistant"
    assert answer["content"] == body["answer"]
    assert answer["status"] == "answered"
    assert answer["retrieval_mode"] == body["retrieval_mode"]
    assert answer["rerank_status"] == body["reranking"]["status"]
    assert answer["context_chunk_count"] == body["context_chunk_count"]
    # Stored citations are exactly the ones returned with the answer.
    assert answer["sources"] == body["sources"]
    assert answer["sources"][0]["filename"] == "handbook.pdf"
    assert answer["sources"][0]["page_number"] == 2


def test_follow_up_questions_are_appended_in_order(llm, handbook) -> None:
    first = ask(question="What is the notice period?").json()
    second = ask(question="And the holidays?", conversation_id=first["conversation_id"]).json()

    assert second["conversation_id"] == first["conversation_id"]
    messages = client.get(f"/api/conversations/{first['conversation_id']}").json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "What is the notice period?"),
        ("assistant", first["answer"]),
        ("user", "And the holidays?"),
        ("assistant", NOT_FOUND_MESSAGE),
    ]
    assert messages[3]["status"] == "not_found" and messages[3]["sources"] == []


def test_document_scope_is_saved(llm, handbook) -> None:
    body = ask(question="What is the notice period?", document_id=handbook).json()

    conversation = client.get(f"/api/conversations/{body['conversation_id']}").json()
    assert conversation["document_id"] == handbook
    assert conversation["messages"][1]["document_id"] == handbook


def test_unknown_conversation_is_rejected_before_calling_the_llm(llm, handbook) -> None:
    response = ask(question="What is the notice period?", conversation_id=str(uuid.uuid4()))

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}
    assert llm == []


def test_invalid_conversation_id_is_a_validation_error(llm) -> None:
    assert ask(question="Hi", conversation_id="not-a-uuid").status_code == 422


def test_answer_is_returned_when_saving_history_fails(llm, handbook, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args):
        raise DatabaseError("The database is unavailable.", 503)

    monkeypatch.setattr(questions, "_save_exchange", fail)

    response = ask(question="What is the notice period?")

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Two months of notice [1]."
    assert body["history_saved"] is False
    assert body["conversation_id"] is None and body["message_id"] is None


def test_asking_still_works_when_the_database_is_down(llm, handbook, database_down) -> None:
    # The RAG pipeline does not depend on PostgreSQL; only the history is lost.
    body = ask(question="What is the notice period?").json()

    assert body["status"] == "answered"
    assert body["history_saved"] is False


def test_continuing_a_conversation_needs_the_database(llm, handbook, database_down) -> None:
    response = ask(question="What is the notice period?", conversation_id=str(uuid.uuid4()))

    assert response.status_code == 503
    assert llm == []


def test_list_conversations_newest_first_with_counts(llm, handbook) -> None:
    older = ask(question="What is the notice period?").json()["conversation_id"]
    newer = ask(question="Notice during probation?").json()["conversation_id"]
    ask(question="And after a year?", conversation_id=older)  # makes "older" the most recent

    body = client.get("/api/conversations").json()

    assert body["total"] == 2
    assert [c["id"] for c in body["conversations"]] == [older, newer]
    assert [c["message_count"] for c in body["conversations"]] == [4, 2]


def test_list_conversations_pagination(llm, handbook) -> None:
    for n in range(3):
        ask(question=f"Notice question {n}?")

    page = client.get("/api/conversations", params={"limit": 2, "offset": 2}).json()

    assert page["total"] == 3
    assert len(page["conversations"]) == 1
    assert client.get("/api/conversations", params={"limit": 0}).status_code == 422


def test_rename_conversation(llm, handbook) -> None:
    conversation_id = ask(question="What is the notice period?").json()["conversation_id"]

    response = client.patch(f"/api/conversations/{conversation_id}", json={"title": "  Notice periods  "})

    assert response.status_code == 200
    assert response.json()["title"] == "Notice periods"
    assert client.get(f"/api/conversations/{conversation_id}").json()["title"] == "Notice periods"


@pytest.mark.parametrize("title", ["", "   ", "x" * 201])
def test_rename_validates_the_title(llm, handbook, title: str) -> None:
    conversation_id = ask(question="What is the notice period?").json()["conversation_id"]

    assert client.patch(f"/api/conversations/{conversation_id}", json={"title": title}).status_code == 422


def test_delete_conversation_removes_messages_and_citations(llm, handbook) -> None:
    conversation_id = ask(question="What is the notice period?").json()["conversation_id"]
    assert count(MessageCitation) == 1

    response = client.delete(f"/api/conversations/{conversation_id}")

    assert response.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    assert count(ChatMessage) == 0
    assert count(MessageCitation) == 0


@pytest.mark.parametrize("method", ["get", "patch", "delete"])
def test_unknown_conversation_404(method: str) -> None:
    kwargs = {"json": {"title": "x"}} if method == "patch" else {}
    response = getattr(client, method)(f"/api/conversations/{uuid.uuid4()}", **kwargs)

    assert response.status_code == 404


def test_conversation_list_503_when_the_database_is_down(database_down) -> None:
    response = client.get("/api/conversations")

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is unavailable."}


def test_history_is_kept_after_its_document_is_deleted(llm, handbook) -> None:
    conversation_id = ask(question="What is the notice period?").json()["conversation_id"]
    client.delete(f"/api/documents/{handbook}")

    messages = client.get(f"/api/conversations/{conversation_id}").json()["messages"]

    assert messages[1]["sources"][0]["filename"] == "handbook.pdf"
