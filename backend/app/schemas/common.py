"""Validated field types shared by several API schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.core.config import get_settings

_settings = get_settings()

# Leading/trailing whitespace is stripped first, so "   " fails the min_length check.
QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=_settings.max_question_length),
]

# Document IDs are the first 32 hex characters of the file's SHA-256 hash.
DOCUMENT_ID_PATTERN = r"^[0-9a-f]{32}$"
DocumentId = Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]

TopK = Annotated[int, Field(ge=1, le=_settings.retrieval_max_top_k)]

# Same values as config.RETRIEVAL_MODES; omit it in a request to use the RETRIEVAL_MODE setting.
RetrievalMode = Literal["dense", "sparse", "hybrid"]

# Omit it in a request to use the RERANK_ENABLED setting; set it to compare with and without reranking.
RerankOption = Annotated[bool | None, Field(description="Rerank candidates with Cohere (default: RERANK_ENABLED)")]


class RerankInfo(BaseModel):
    """What happened at the reranking step. "failed" means the retrieval order was used instead."""

    status: Literal["applied", "disabled", "skipped", "failed"]
    model: str | None = None
    latency_ms: float | None = None
    error: str | None = None  # a safe, generic message when status is "failed"
