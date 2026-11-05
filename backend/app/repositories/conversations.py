"""CRUD for chat conversations, messages and their citations. Callers own the transaction."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.base import utcnow
from app.db.models import ChatMessage, Conversation, MessageCitation

TITLE_LENGTH = 80


@dataclass(frozen=True)
class CitationData:
    citation_number: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    excerpt: str
    similarity: float
    rerank_score: float | None


@dataclass(frozen=True)
class AnswerData:
    content: str
    answer_status: str
    document_id: str | None
    retrieval_mode: str
    rerank_status: str
    rerank_model: str | None
    retrieved_chunk_count: int
    context_chunk_count: int
    citations: list[CitationData]


def make_title(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= TITLE_LENGTH else text[: TITLE_LENGTH - 1].rstrip() + "…"


def get_conversation(db: Session, conversation_id: uuid.UUID, *, with_messages: bool = False) -> Conversation | None:
    if not with_messages:
        return db.get(Conversation, conversation_id)
    query = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages).selectinload(ChatMessage.citations))
    )
    return db.scalars(query).first()


def list_conversations(db: Session, limit: int, offset: int) -> tuple[Sequence[tuple[Conversation, int]], int]:
    """Newest first, each with its message count; also returns the total number of conversations."""
    counts = (
        select(ChatMessage.conversation_id, func.count().label("message_count"))
        .group_by(ChatMessage.conversation_id)
        .subquery()
    )
    query = (
        select(Conversation, func.coalesce(counts.c.message_count, 0))
        .outerjoin(counts, counts.c.conversation_id == Conversation.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id)
        .limit(limit)
        .offset(offset)
    )
    rows = [(conversation, int(count)) for conversation, count in db.execute(query).all()]
    total = db.scalar(select(func.count()).select_from(Conversation)) or 0
    return rows, total


def create_conversation(db: Session, title: str, document_id: str | None) -> Conversation:
    now = utcnow()
    conversation = Conversation(title=make_title(title), document_id=document_id, created_at=now, updated_at=now)
    db.add(conversation)
    db.flush()
    return conversation


def add_exchange(db: Session, conversation: Conversation, question: str, answer: AnswerData) -> ChatMessage:
    """Append a question and its answer (with citations); returns the assistant message."""
    next_position = (
        db.scalar(select(func.max(ChatMessage.position)).where(ChatMessage.conversation_id == conversation.id)) or 0
    ) + 1
    now = utcnow()
    db.add(
        ChatMessage(
            conversation_id=conversation.id,
            position=next_position,
            role="user",
            content=question,
            document_id=answer.document_id,
            created_at=now,
        )
    )
    assistant = ChatMessage(
        conversation_id=conversation.id,
        position=next_position + 1,
        role="assistant",
        content=answer.content,
        answer_status=answer.answer_status,
        document_id=answer.document_id,
        retrieval_mode=answer.retrieval_mode,
        rerank_status=answer.rerank_status,
        rerank_model=answer.rerank_model,
        retrieved_chunk_count=answer.retrieved_chunk_count,
        context_chunk_count=answer.context_chunk_count,
        created_at=now,
        citations=[MessageCitation(**vars(citation)) for citation in answer.citations],
    )
    db.add(assistant)
    conversation.updated_at = now
    db.flush()
    return assistant


def rename_conversation(db: Session, conversation: Conversation, title: str) -> Conversation:
    conversation.title = title
    conversation.updated_at = utcnow()
    db.flush()
    return conversation


def delete_conversation(db: Session, conversation: Conversation) -> None:
    db.delete(conversation)
    db.flush()
