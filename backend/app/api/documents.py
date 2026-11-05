"""Document upload, listing, search and delete endpoints."""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy.orm import Session

from app.api.search import SEARCH_ERROR_RESPONSES, run_search
from app.core.config import get_settings
from app.db.models import Document
from app.db.session import DatabaseError, get_db, session_scope
from app.repositories import documents as document_repo
from app.schemas.document import (
    DOCUMENT_ID_PATTERN,
    ChunkPreview,
    DocumentDeleteResponse,
    DocumentDetail,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentSummary,
    DocumentUploadResponse,
    PageSummary,
    SearchResponse,
)
from app.services.embeddings import EmbeddingError
from app.services.ingestion import DocumentIngestionError, ingest_pdf, make_document_id
from app.services.vector_store import VectorStoreError, delete_document, document_exists

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

READ_CHUNK_BYTES = 1024 * 1024
PREVIEW_CHARS = 1000
CHUNK_PREVIEW_COUNT = 5
CHUNK_PREVIEW_CHARS = 300

DocumentIdPath = Annotated[
    str, Path(pattern=DOCUMENT_ID_PATTERN, description="32-character document ID returned by upload")
]
DbSession = Annotated[Session, Depends(get_db)]

DB_ERROR_RESPONSES = {500: {"description": "Database error"}, 503: {"description": "Database unavailable"}}


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in chunks, stopping as soon as it exceeds `max_bytes`."""
    data = bytearray()
    while chunk := await file.read(READ_CHUNK_BYTES):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large (maximum {max_bytes // (1024 * 1024)} MB).",
            )
    return bytes(data)


def safe_filename(filename: str | None) -> str:
    # Keep only the base name; browsers and clients sometimes send full paths.
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name[:255] or "document.pdf"


def to_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        error_message=document.error_message,
        chunk_count=document.chunk_count or 0,
        indexed_page_count=document.pages_with_text or 0,
        page_count=document.page_count,
        file_size_bytes=document.file_size_bytes,
        created_at=document.created_at,
        updated_at=document.updated_at,
        indexed_at=document.indexed_at,
    )


@router.get("", response_model=DocumentListResponse, responses=DB_ERROR_RESPONSES)
def list_documents_endpoint(db: DbSession) -> DocumentListResponse:
    """Every uploaded document with its processing status, most recently changed first."""
    documents = [to_summary(document) for document in document_repo.list_documents(db)]
    indexed = [document for document in documents if document.status == "indexed"]
    return DocumentListResponse(
        document_count=len(documents),
        indexed_count=len(indexed),
        total_chunks=sum(document.chunk_count for document in indexed),
        documents=documents,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    responses={404: {"description": "Document not found"}, **DB_ERROR_RESPONSES},
)
def get_document_endpoint(document_id: DocumentIdPath, db: DbSession) -> DocumentDetail:
    document = document_repo.get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentDetail(
        **to_summary(document).model_dump(),
        embedding_model=document.embedding_model,
        chunk_size=document.chunk_size,
        chunk_overlap=document.chunk_overlap,
    )


def _record_processing(document_id: str, filename: str, size: int) -> None:
    with session_scope() as db:
        document_repo.mark_processing(db, document_id, filename, size)


def _record_failure(document_id: str, message: str) -> None:
    try:
        with session_scope() as db:
            document_repo.mark_failed(db, document_id, message)
    except DatabaseError:
        # The upload already failed; the client gets that error, not this one.
        logger.error("Could not record the failed upload of document %s", document_id)


def _record_indexed(document_id: str, **values) -> None:
    with session_scope() as db:
        document_repo.mark_indexed(db, document_id, **values)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"description": "Empty file"},
        403: {"description": "Text preview requested outside development"},
        413: {"description": "File too large or too many pages"},
        415: {"description": "File is not a PDF"},
        422: {"description": "Corrupt, password-protected or text-less PDF"},
        500: {"description": "Embedding generation or indexing failed"},
        503: {"description": "Embedding model, vector store or database unavailable"},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="A PDF file")],
    include_text: Annotated[
        bool,
        Query(
            description=(
                f"Development only: include the first {PREVIEW_CHARS} characters of each page "
                f"and a preview of the first {CHUNK_PREVIEW_COUNT} chunks."
            )
        ),
    ] = False,
) -> DocumentUploadResponse:
    settings = get_settings()
    if include_text and not settings.is_development:
        raise HTTPException(status_code=403, detail="Text preview is only available in development.")

    filename = safe_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are accepted.")

    try:
        data = await read_upload_limited(file, settings.max_upload_bytes)
    finally:
        await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    document_id = make_document_id(data)
    try:
        # Record the upload first: nothing gets indexed that the database does not know about.
        await asyncio.to_thread(_record_processing, document_id, filename, len(data))
    except DatabaseError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    try:
        # Extraction, chunking, embedding and indexing are blocking; run them off the event loop.
        ingested = await asyncio.to_thread(ingest_pdf, data, filename, settings)
    except (DocumentIngestionError, EmbeddingError, VectorStoreError) as exc:
        await asyncio.to_thread(_record_failure, document_id, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception:
        await asyncio.to_thread(_record_failure, document_id, "Unexpected error while processing the document.")
        raise

    extracted = ingested.extracted
    try:
        await asyncio.to_thread(
            _record_indexed,
            document_id,
            page_count=extracted.page_count,
            pages_with_text=len(extracted.pages),
            chunk_count=ingested.indexed_chunk_count,
            embedding_model=settings.embedding_model_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except DatabaseError as exc:
        # Uploads are idempotent (same file, same ID), so uploading again fixes the record.
        raise HTTPException(
            status_code=exc.status_code,
            detail="The document was indexed, but its status could not be saved. Please upload it again.",
        ) from exc

    logger.info(
        "Indexed document %s: %d bytes, %d pages, %d with text, %d chunks",
        ingested.document_id, len(data), extracted.page_count, len(extracted.pages),
        ingested.indexed_chunk_count,
    )

    return DocumentUploadResponse(
        document_id=ingested.document_id,
        filename=filename,
        page_count=extracted.page_count,
        pages_with_text=len(extracted.pages),
        empty_pages=extracted.empty_pages,
        total_characters=sum(len(page.text) for page in extracted.pages),
        pages=[
            PageSummary(
                page_number=page.page_number,
                char_count=len(page.text),
                text_preview=page.text[:PREVIEW_CHARS] if include_text else None,
            )
            for page in extracted.pages
        ],
        chunk_count=len(ingested.chunks),
        indexing_status="indexed",
        indexed_chunk_count=ingested.indexed_chunk_count,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_model=settings.embedding_model_name,
        embedding_dimensions=len(ingested.embeddings[0]) if ingested.embeddings else 0,
        chunk_previews=[
            ChunkPreview(
                chunk_id=chunk.id,
                page_number=chunk.metadata["page_number"],
                chunk_index=chunk.metadata["chunk_index"],
                char_count=len(chunk.page_content),
                text_preview=chunk.page_content[:CHUNK_PREVIEW_CHARS],
            )
            for chunk in ingested.chunks[:CHUNK_PREVIEW_COUNT]
        ]
        if include_text
        else None,
    )


@router.post(
    "/{document_id}/search",
    response_model=SearchResponse,
    responses={404: {"description": "Document not found"}, **SEARCH_ERROR_RESPONSES},
)
async def search_document(document_id: DocumentIdPath, request: DocumentSearchRequest) -> SearchResponse:
    """Search only within one uploaded document."""
    try:
        exists = await asyncio.to_thread(document_exists, document_id)
    except VectorStoreError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    if not exists:
        raise HTTPException(status_code=404, detail="Document not found.")
    return await run_search(request.question, request.top_k, document_id, request.retrieval_mode, request.rerank)


def _delete_record(document_id: str) -> bool:
    with session_scope() as db:
        return document_repo.delete_document(db, document_id)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={404: {"description": "Document not found"}, 500: {}, 503: {}},
)
async def delete_document_endpoint(document_id: DocumentIdPath) -> DocumentDeleteResponse:
    """Remove a document's chunks from the vector store and its record from the database."""
    try:
        # Vector store first: once this succeeds the document is no longer searchable,
        # and a retry after a database failure still removes the record.
        deleted = await asyncio.to_thread(delete_document, document_id)
        had_record = await asyncio.to_thread(_delete_record, document_id)
    except (VectorStoreError, DatabaseError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    if deleted == 0 and not had_record:
        raise HTTPException(status_code=404, detail="Document not found.")
    logger.info("Deleted document %s (%d chunks)", document_id, deleted)
    return DocumentDeleteResponse(document_id=document_id, deleted_chunks=deleted)
