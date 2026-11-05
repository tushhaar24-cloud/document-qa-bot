"""Request and response schemas for the question-answering API."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.schemas.common import DocumentId, QuestionText, RerankInfo, RerankOption, RetrievalMode, TopK

_settings = get_settings()


class QuestionRequest(BaseModel):
    question: QuestionText
    # Continue a saved conversation; omit to start a new one. Earlier messages are
    # stored for history only: every question is answered from the documents alone.
    conversation_id: uuid.UUID | None = None
    # Omit to search every indexed document.
    document_id: DocumentId | None = None
    top_k: TopK = _settings.retrieval_top_k
    # Omit to use the RETRIEVAL_MODE setting; set it to compare dense, sparse and hybrid.
    retrieval_mode: RetrievalMode | None = None
    rerank: RerankOption = None

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What is the resignation notice period?", "document_id": None}]
        }
    }


class SourceCitation(BaseModel):
    """A passage the answer relies on. Every field comes from ChromaDB metadata, never from the LLM."""

    citation: int = Field(description="Matches the [n] marker in the answer")
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    excerpt: str
    similarity: float
    rerank_score: float | None = None  # Cohere relevance (0..1); None when not reranked


class QuestionResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "not_found"]
    document_id: str | None
    sources: list[SourceCitation]
    retrieved_chunk_count: int
    # Retrieved chunks that passed the relevance floor and were shown to the LLM.
    context_chunk_count: int
    retrieval_mode: RetrievalMode
    reranking: RerankInfo
    # Where the exchange was saved. When saving fails the answer is still returned,
    # with history_saved=False (and conversation_id echoing the request's, if any).
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    history_saved: bool = False
