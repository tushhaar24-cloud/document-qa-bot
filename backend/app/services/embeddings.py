"""Local text embeddings with a Sentence Transformers model.

The model runs on this machine, so document text is never sent to an external
API. The model files are downloaded from Hugging Face once and then cached.
Use the same `get_embedding_model()` for documents and questions so their
vectors are comparable.
"""

import logging
import threading

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model: Embeddings | None = None
_model_lock = threading.Lock()


class EmbeddingError(Exception):
    """An embedding failure that is safe to show to API clients."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_embedding_model() -> Embeddings:
    """Load the model once; the lock stops concurrent requests from loading it twice."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            settings = get_settings()
            logger.info("Loading embedding model %s", settings.embedding_model_name)
            try:
                # Imported here because importing sentence-transformers/torch is slow.
                from langchain_huggingface import HuggingFaceEmbeddings

                _model = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model_name,
                    model_kwargs={"device": settings.embedding_device},
                    # Unit-length vectors make cosine similarity a simple dot product.
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": settings.embedding_batch_size,
                    },
                )
            # Model loading can fail in many ways (no network on first download,
            # missing files, bad device name), each with a different exception type.
            except Exception as exc:
                logger.error("Could not load embedding model: %s", type(exc).__name__)
                raise EmbeddingError("The embedding model is unavailable.", 503) from exc
            logger.info("Embedding model loaded")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    try:
        return model.embed_documents(texts)
    except Exception as exc:
        logger.error("Embedding %d texts failed: %s", len(texts), type(exc).__name__)
        raise EmbeddingError("Failed to generate embeddings.", 500) from exc


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    try:
        return model.embed_query(text)
    except Exception as exc:
        logger.error("Embedding a query failed: %s", type(exc).__name__)
        raise EmbeddingError("Failed to generate embeddings.", 500) from exc
