"""Schemas for the chat history API."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from app.schemas.common import RetrievalMode

ConversationTitle = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class StoredCitation(BaseModel):
    citation: int  # the [n] marker in the answer
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    excerpt: str
    similarity: float
    rerank_score: float | None


class StoredMessage(BaseModel):
    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    # Assistant messages only (None for questions):
    status: Literal["answered", "not_found"] | None
    document_id: str | None
    retrieval_mode: RetrievalMode | None
    rerank_status: Literal["applied", "disabled", "skipped", "failed"] | None
    rerank_model: str | None
    retrieved_chunk_count: int | None
    context_chunk_count: int | None
    sources: list[StoredCitation]


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    document_id: str | None  # the document the chat started with, None = all documents
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str
    document_id: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[StoredMessage]  # oldest first


class ConversationList(BaseModel):
    total: int
    conversations: list[ConversationSummary]  # most recently active first


class ConversationUpdate(BaseModel):
    title: ConversationTitle
