"""Persistent ChromaDB storage for document chunks.

Embeddings are computed by `app.services.embeddings` and passed in explicitly,
so ChromaDB never embeds text itself (its collection has no embedding function).
That guarantees stored chunks and search queries use the same model.
"""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_collection: Collection | None = None
_collection_lock = threading.Lock()
# Largest number of records Chroma accepts in one write; set when the client opens.
_max_batch_size = 1000
# Bumped on every write, delete and reset so derived indexes (BM25) know when they are stale.
_index_version = 0
_version_lock = threading.Lock()


class VectorStoreError(Exception):
    """A vector store failure that is safe to show to API clients."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class StoredChunkMatch:
    chunk_id: str
    text: str
    metadata: dict
    distance: float  # cosine distance: 0 = same direction, 2 = opposite


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict


def index_version() -> int:
    return _index_version


def _bump_index_version() -> None:
    global _index_version
    with _version_lock:
        _index_version += 1


def get_collection() -> Collection:
    """Open the client and collection once; reuse them for every request."""
    global _collection, _max_batch_size
    if _collection is not None:
        return _collection

    with _collection_lock:
        if _collection is None:
            settings = get_settings()
            try:
                client = chromadb.PersistentClient(
                    path=str(settings.chroma_persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name=settings.chroma_collection_name,
                    embedding_function=None,
                    configuration={"hnsw": {"space": "cosine"}},
                    metadata={"embedding_model": settings.embedding_model_name},
                )
                _max_batch_size = client.get_max_batch_size()
            except Exception as exc:
                logger.error("Could not open the vector store: %s", type(exc).__name__)
                raise VectorStoreError("The vector store is unavailable.", 503) from exc

            # Vectors from different models cannot be compared, so refuse to mix them.
            stored_model = (collection.metadata or {}).get("embedding_model")
            if stored_model != settings.embedding_model_name:
                logger.error(
                    "Collection %r was built with %r but EMBEDDING_MODEL_NAME is %r",
                    settings.chroma_collection_name, stored_model, settings.embedding_model_name,
                )
                raise VectorStoreError("The vector store was built with a different embedding model.", 500)

            _collection = collection
    return _collection


def reset_collection_cache() -> None:
    """Forget the cached collection (used by tests and after changing settings)."""
    global _collection
    with _collection_lock:
        _collection = None
    _bump_index_version()


def upsert_chunks(chunks: Sequence[Document], embeddings: Sequence[Sequence[float]]) -> int:
    """Insert or replace chunks by ID; returns the number stored."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk needs exactly one embedding.")

    pairs = [(c, e) for c, e in zip(chunks, embeddings) if c.page_content.strip()]
    if not pairs:
        return 0

    collection = get_collection()
    try:
        for start in range(0, len(pairs), _max_batch_size):
            batch = pairs[start : start + _max_batch_size]
            collection.upsert(
                ids=[chunk.id for chunk, _ in batch],
                documents=[chunk.page_content for chunk, _ in batch],
                embeddings=[list(embedding) for _, embedding in batch],
                metadatas=[chunk.metadata for chunk, _ in batch],
            )
    except Exception as exc:
        logger.error("Writing %d chunks failed: %s", len(pairs), type(exc).__name__)
        raise VectorStoreError("Failed to store the document in the vector store.", 500) from exc
    finally:
        # Even a failed batch may have written some records.
        _bump_index_version()
    return len(pairs)


def get_chunk_ids(document_id: str) -> list[str]:
    try:
        return get_collection().get(where={"document_id": document_id}, include=[])["ids"]
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Reading chunk IDs failed: %s", type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc


def document_exists(document_id: str) -> bool:
    try:
        return bool(get_collection().get(where={"document_id": document_id}, limit=1, include=[])["ids"])
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Checking a document failed: %s", type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc


def delete_chunks(chunk_ids: Sequence[str]) -> None:
    if not chunk_ids:
        return
    try:
        get_collection().delete(ids=list(chunk_ids))
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Deleting %d chunks failed: %s", len(chunk_ids), type(exc).__name__)
        raise VectorStoreError("Failed to delete chunks from the vector store.", 500) from exc
    finally:
        _bump_index_version()


def delete_document(document_id: str) -> int:
    """Delete every chunk of a document; returns how many were deleted."""
    chunk_ids = get_chunk_ids(document_id)
    delete_chunks(chunk_ids)
    return len(chunk_ids)


def replace_document_chunks(
    document_id: str, chunks: Sequence[Document], embeddings: Sequence[Sequence[float]]
) -> int:
    """Store a document's chunks so the index ends up matching exactly this upload.

    Upsert makes re-uploads idempotent (same IDs overwrite, no duplicates). Chunks
    left over from an earlier upload with different chunk settings are removed.
    If any write fails, all of the document's chunks are removed so a half-indexed
    document is never searchable; the client can simply retry the upload.
    """
    try:
        stored = upsert_chunks(chunks, embeddings)
        new_ids = {chunk.id for chunk in chunks}
        delete_chunks([cid for cid in get_chunk_ids(document_id) if cid not in new_ids])
    except VectorStoreError:
        try:
            delete_document(document_id)
        except VectorStoreError:
            logger.error("Cleanup after a failed write also failed for document %s", document_id)
        raise
    return stored


def count_chunks() -> int:
    try:
        return get_collection().count()
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Counting chunks failed: %s", type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc


def get_all_chunks() -> list[StoredChunk]:
    """Every stored chunk's text and metadata (no embeddings), read in pages."""
    chunks: list[StoredChunk] = []
    try:
        collection = get_collection()
        while True:
            page = collection.get(include=["documents", "metadatas"], limit=_max_batch_size, offset=len(chunks))
            chunks.extend(
                StoredChunk(chunk_id=cid, text=text or "", metadata=meta or {})
                for cid, text, meta in zip(page["ids"], page["documents"], page["metadatas"])
            )
            if len(page["ids"]) < _max_batch_size:
                return chunks
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Reading all chunks failed: %s", type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    filename: str
    chunk_count: int
    page_numbers: tuple[int, ...]  # pages that produced at least one chunk


def list_documents() -> list[StoredDocument]:
    """One entry per indexed document, built from chunk metadata (texts are not read)."""
    metadatas: list[dict] = []
    try:
        collection = get_collection()
        while True:
            page = collection.get(include=["metadatas"], limit=_max_batch_size, offset=len(metadatas))
            metadatas.extend(meta or {} for meta in page["metadatas"])
            if len(page["ids"]) < _max_batch_size:
                break
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Listing documents failed: %s", type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc

    grouped: dict[str, dict] = {}
    for meta in metadatas:
        document_id = meta.get("document_id")
        if not document_id:
            continue
        entry = grouped.setdefault(document_id, {"filename": meta.get("filename", ""), "chunks": 0, "pages": set()})
        entry["chunks"] += 1
        if isinstance(meta.get("page_number"), int):
            entry["pages"].add(meta["page_number"])

    documents = [
        StoredDocument(
            document_id=document_id,
            filename=entry["filename"],
            chunk_count=entry["chunks"],
            page_numbers=tuple(sorted(entry["pages"])),
        )
        for document_id, entry in grouped.items()
    ]
    return sorted(documents, key=lambda doc: (doc.filename.lower(), doc.document_id))


def get_chunk_embeddings(chunk_ids: Sequence[str]) -> dict[str, list[float]]:
    if not chunk_ids:
        return {}
    try:
        result = get_collection().get(ids=list(chunk_ids), include=["embeddings"])
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Reading %d embeddings failed: %s", len(chunk_ids), type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc
    return {cid: list(embedding) for cid, embedding in zip(result["ids"], result["embeddings"])}


def query_chunks(
    query_embedding: Sequence[float], top_k: int, document_id: str | None = None
) -> list[StoredChunkMatch]:
    try:
        result = get_collection().query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where={"document_id": document_id} if document_id else None,
            include=["documents", "metadatas", "distances"],
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Vector search failed: %s", type(exc).__name__)
        raise VectorStoreError("The vector store is unavailable.", 503) from exc

    # Chroma returns one list per query embedding; we sent exactly one.
    return [
        StoredChunkMatch(chunk_id=cid, text=text, metadata=meta or {}, distance=float(dist))
        for cid, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
