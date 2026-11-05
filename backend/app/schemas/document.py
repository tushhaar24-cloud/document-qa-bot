"""Schemas for the document upload, listing, search and delete APIs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.common import (  # noqa: F401
    DOCUMENT_ID_PATTERN,
    DocumentId,
    QuestionText,
    RerankInfo,
    RerankOption,
    RetrievalMode,
    TopK,
)

_settings = get_settings()


class PageSummary(BaseModel):
    page_number: int
    char_count: int
    # Only filled when a development text preview is requested.
    text_preview: str | None = None


class ChunkPreview(BaseModel):
    chunk_id: str
    page_number: int
    chunk_index: int
    char_count: int
    text_preview: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages_with_text: int
    empty_pages: list[int]
    total_characters: int
    pages: list[PageSummary]
    chunk_count: int
    # Only ever "indexed": if indexing fails the API returns an error instead of this response.
    indexing_status: Literal["indexed"]
    indexed_chunk_count: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimensions: int
    # Only filled when a development text preview is requested; limited to the first few chunks.
    chunk_previews: list[ChunkPreview] | None = None


DocumentStatus = Literal["processing", "indexed", "failed"]


class DocumentSummary(BaseModel):
    """A document's metadata from PostgreSQL."""

    document_id: str
    filename: str
    status: DocumentStatus
    error_message: str | None  # why processing failed; None otherwise
    chunk_count: int  # chunks in ChromaDB (0 unless indexed)
    # Pages that produced at least one chunk; pages without text are not stored.
    indexed_page_count: int
    page_count: int | None  # None until the PDF has been read
    file_size_bytes: int | None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None


class DocumentDetail(DocumentSummary):
    embedding_model: str | None
    chunk_size: int | None
    chunk_overlap: int | None


class DocumentListResponse(BaseModel):
    document_count: int  # every listed document, whatever its status
    indexed_count: int
    total_chunks: int  # chunks of indexed documents
    documents: list[DocumentSummary]  # most recently changed first


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int


class DocumentSearchRequest(BaseModel):
    question: QuestionText
    top_k: TopK = _settings.retrieval_top_k
    # Omit to use the RETRIEVAL_MODE setting; set it to compare dense, sparse and hybrid.
    retrieval_mode: RetrievalMode | None = None
    rerank: RerankOption = None

    model_config = {
        "json_schema_extra": {"examples": [{"question": "What is the refund policy?", "top_k": 5}]}
    }


class SearchRequest(DocumentSearchRequest):
    document_id: DocumentId | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What is the refund policy?", "top_k": 5, "document_id": None}]
        }
    }


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    distance: float
    similarity: float
    # 1 = best; None when that retriever did not return the chunk.
    dense_rank: int | None = None
    sparse_rank: int | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None  # Cohere relevance (0..1); None when not reranked


class SearchResponse(BaseModel):
    question: str
    document_id: str | None
    top_k: int
    retrieval_mode: RetrievalMode
    reranking: RerankInfo
    result_count: int
    results: list[SearchResult]
