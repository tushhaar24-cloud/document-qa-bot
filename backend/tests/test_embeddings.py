"""Tests for the embedding service.

Most tests use the fake model from conftest.py. The `integration` test loads
the real model; run it with:  pytest -m integration
"""

import math
import sys
import types

import pytest

from app.services import embeddings
from app.services.embeddings import EmbeddingError, embed_documents, embed_query
from tests.conftest import FAKE_EMBEDDING_SIZE


def test_embed_documents_returns_one_vector_per_text() -> None:
    vectors = embed_documents(["first chunk", "second chunk"])

    assert len(vectors) == 2
    assert all(len(v) == FAKE_EMBEDDING_SIZE for v in vectors)


def test_documents_and_queries_use_the_same_model() -> None:
    # With one shared model, the same text gives the same vector either way.
    assert embed_documents(["What is RAG?"])[0] == embed_query("What is RAG?")


def test_empty_input_does_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(embeddings, "get_embedding_model", fail)
    assert embed_documents([]) == []


def test_encoding_failure_becomes_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenModel:
        def embed_documents(self, texts):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: BrokenModel())

    with pytest.raises(EmbeddingError) as exc_info:
        embed_documents(["text"])
    assert exc_info.value.status_code == 500


def test_model_load_failure_becomes_503(monkeypatch: pytest.MonkeyPatch) -> None:
    # Undo the conftest fake and replace the Hugging Face class with one that fails to load.
    monkeypatch.undo()

    def failing_constructor(**kwargs):
        raise OSError("model files not found")

    fake_module = types.SimpleNamespace(HuggingFaceEmbeddings=failing_constructor)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_module)
    monkeypatch.setattr(embeddings, "_model", None)

    with pytest.raises(EmbeddingError) as exc_info:
        embeddings.get_embedding_model()
    assert exc_info.value.status_code == 503
    assert embeddings._model is None  # a failed load is not cached; the next call retries


@pytest.mark.integration
def test_real_model_produces_normalized_similar_vectors() -> None:
    query = embed_query("What does retrieval-augmented generation do?")
    related, unrelated = embed_documents(
        ["RAG retrieves relevant documents before generating an answer.", "The cat sat on the mat."]
    )

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert len(query) == 384
    assert math.isclose(dot(query, query), 1.0, abs_tol=1e-3)  # normalized
    assert dot(query, related) > dot(query, unrelated)
