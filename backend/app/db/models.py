"""Database tables.

Column types are portable (no PostgreSQL-only types), so the test suite can run
on SQLite while production uses PostgreSQL.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

# JSONB on PostgreSQL (binary, indexable), plain JSON elsewhere (SQLite in tests).
JsonType = JSON().with_variant(JSONB(), "postgresql")

DOCUMENT_STATUSES = ("processing", "indexed", "failed")


class Document(Base):
    """Metadata and processing status of an uploaded PDF. Its chunks and vectors live in ChromaDB."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(f"status IN {DOCUMENT_STATUSES}", name="status"),
    )

    # Same ID as in ChromaDB: the first 32 hex characters of the file's SHA-256.
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    pages_with_text: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    chunk_size: Mapped[int | None] = mapped_column(Integer)
    chunk_overlap: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, index=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200))
    # The document the chat was limited to when it started (None = all documents).
    # Not a foreign key: history is kept after the document is deleted.
    document_id: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.position",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role"),
        UniqueConstraint("conversation_id", "position"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Indexed through the (conversation_id, position) unique constraint.
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    # Order within the conversation (timestamps can tie).
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    # Assistant messages only:
    answer_status: Mapped[str | None] = mapped_column(String(20))  # "answered" | "not_found"
    document_id: Mapped[str | None] = mapped_column(String(32))  # search scope of this question
    retrieval_mode: Mapped[str | None] = mapped_column(String(20))
    rerank_status: Mapped[str | None] = mapped_column(String(20))
    rerank_model: Mapped[str | None] = mapped_column(String(100))
    retrieved_chunk_count: Mapped[int | None] = mapped_column(Integer)
    context_chunk_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    citations: Mapped[list["MessageCitation"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageCitation.citation_number",
    )


class MessageCitation(Base):
    """A source passage an answer cites. Copied from ChromaDB metadata at answer time."""

    __tablename__ = "message_citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), index=True)
    citation_number: Mapped[int] = mapped_column(Integer)  # the [n] marker in the answer
    chunk_id: Mapped[str] = mapped_column(String(100))
    document_id: Mapped[str] = mapped_column(String(32), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    page_number: Mapped[int] = mapped_column(Integer)
    excerpt: Mapped[str] = mapped_column(Text)
    similarity: Mapped[float] = mapped_column(Float)
    rerank_score: Mapped[float | None] = mapped_column(Float)

    message: Mapped[ChatMessage] = relationship(back_populates="citations")


class EvaluationRun(Base):
    """One evaluation report written by evaluation/evaluate_rag.py."""

    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # e.g. "eval-20261104-124423"
    # When the evaluation ran, in the evaluating machine's local time (as written in the report).
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    setup: Mapped[dict] = mapped_column(JsonType, default=dict)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    config_results: Mapped[list["EvaluationConfigResult"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="EvaluationConfigResult.position",
    )


class EvaluationConfigResult(Base):
    """Results of one retrieval configuration (e.g. dense, hybrid, hybrid+rerank) within a run."""

    __tablename__ = "evaluation_config_results"
    __table_args__ = (UniqueConstraint("run_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(50))
    retrieval_mode: Mapped[str | None] = mapped_column(String(20))
    rerank: Mapped[bool | None] = mapped_column(Boolean)
    skipped_reason: Mapped[str | None] = mapped_column(Text)
    # Deterministic checks (answer rate, MRR, latency, ...) as written in the report; None when skipped.
    # Ragas scores are stored in ragas_metric_scores instead.
    summary: Mapped[dict | None] = mapped_column(JsonType)

    run: Mapped[EvaluationRun] = relationship(back_populates="config_results")
    ragas_scores: Mapped[list["RagasMetricScore"]] = relationship(
        back_populates="config_result",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="RagasMetricScore.metric_name",
    )


class RagasMetricScore(Base):
    __tablename__ = "ragas_metric_scores"
    __table_args__ = (UniqueConstraint("config_result_id", "metric_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_result_id: Mapped[int] = mapped_column(ForeignKey("evaluation_config_results.id", ondelete="CASCADE"))
    metric_name: Mapped[str] = mapped_column(String(100))  # e.g. "faithfulness", "context_precision"
    mean: Mapped[float | None] = mapped_column(Float)  # None when the judge scored no question
    scored: Mapped[int] = mapped_column(Integer)
    missing: Mapped[int] = mapped_column(Integer)

    config_result: Mapped[EvaluationConfigResult] = relationship(back_populates="ragas_scores")
