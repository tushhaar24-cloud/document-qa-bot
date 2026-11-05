"""CRUD for document metadata. Callers own the session and the transaction."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import Document


def get_document(db: Session, document_id: str) -> Document | None:
    return db.get(Document, document_id)


def list_documents(db: Session, status: str | None = None) -> Sequence[Document]:
    query = select(Document).order_by(Document.updated_at.desc(), Document.id)
    if status is not None:
        query = query.where(Document.status == status)
    return db.scalars(query).all()


def mark_processing(db: Session, document_id: str, filename: str, file_size_bytes: int) -> Document:
    """Create the row, or reset it for a re-upload of the same file."""
    now = utcnow()
    document = db.get(Document, document_id)
    if document is None:
        document = Document(id=document_id, created_at=now)
        db.add(document)
    document.filename = filename
    document.file_size_bytes = file_size_bytes
    document.status = "processing"
    document.error_message = None
    document.updated_at = now
    db.flush()
    return document


def mark_indexed(
    db: Session,
    document_id: str,
    *,
    page_count: int,
    pages_with_text: int,
    chunk_count: int,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Document:
    document = _require(db, document_id)
    document.status = "indexed"
    document.error_message = None
    document.page_count = page_count
    document.pages_with_text = pages_with_text
    document.chunk_count = chunk_count
    document.embedding_model = embedding_model
    document.chunk_size = chunk_size
    document.chunk_overlap = chunk_overlap
    document.indexed_at = document.updated_at = utcnow()
    db.flush()
    return document


def mark_failed(db: Session, document_id: str, error_message: str) -> Document:
    document = _require(db, document_id)
    document.status = "failed"
    document.error_message = error_message
    # Ingestion removes a failed document's chunks, so nothing of it is searchable.
    document.chunk_count = 0
    document.indexed_at = None
    document.updated_at = utcnow()
    db.flush()
    return document


def delete_document(db: Session, document_id: str) -> bool:
    document = db.get(Document, document_id)
    if document is None:
        return False
    db.delete(document)
    db.flush()
    return True


def upsert_indexed_from_vector_store(
    db: Session, document_id: str, filename: str, chunk_count: int, pages_with_text: int
) -> bool:
    """Record a document found in ChromaDB (indexed before PostgreSQL was added).

    Returns True when a row was created. Existing rows are only corrected when they
    are not already marked indexed (e.g. stuck in "processing" after a crash).
    """
    document = db.get(Document, document_id)
    if document is not None and document.status == "indexed":
        return False
    now = utcnow()
    created = document is None
    if created:
        document = Document(id=document_id, filename=filename, created_at=now)
        db.add(document)
    document.status = "indexed"
    document.error_message = None
    document.chunk_count = chunk_count
    document.pages_with_text = pages_with_text
    document.indexed_at = document.updated_at = now
    db.flush()
    return created


def _require(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise LookupError(f"Document {document_id} has no database record.")
    return document
