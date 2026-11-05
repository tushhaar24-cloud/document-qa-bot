"""Shared test setup.

- Ordinary tests use a fake embedding model so they run fast, offline and
  without loading PyTorch. Tests marked `integration` use the real model.
- Cohere reranking is off and has no API key; reranking tests use a fake client.
- Every test gets its own empty ChromaDB folder, so tests never touch the
  real `chroma_db/` and never affect each other.
- Every test gets its own empty in-memory SQLite database instead of PostgreSQL.
  Tests marked `postgres` (table creation) use TEST_DATABASE_URL instead.
"""

import dataclasses
import math
import os
import re
import zlib

# Set before the app reads its settings: tests never use a real Cohere key or call Cohere.
# Tests that need reranking enable it explicitly with a mocked client.
os.environ["COHERE_API_KEY"] = ""
os.environ["RERANK_ENABLED"] = "false"

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db import models  # noqa: F401  (registers the tables on Base.metadata)
from app.db import session as db_session
from app.db.base import Base
from app.services import embeddings, vector_store

FAKE_EMBEDDING_SIZE = 384  # same dimensions as all-MiniLM-L6-v2


class KeywordEmbedding(Embeddings):
    """Bag-of-words vectors: texts sharing words get similar vectors.

    Crude, but enough to test that retrieval ranks matching text first.
    """

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * FAKE_EMBEDDING_SIZE
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % FAKE_EMBEDDING_SIZE] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(autouse=True)
def fake_embedding_model(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("integration"):
        yield None
        return
    model = KeywordEmbedding()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: model)
    yield model


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    test_settings = dataclasses.replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(vector_store, "get_settings", lambda: test_settings)
    vector_store.reset_collection_cache()
    yield test_settings
    vector_store.reset_collection_cache()
    # Release Chroma's file handles so the temporary folder can be cleaned up.
    SharedSystemClient.clear_system_cache()


def make_sqlite_engine():
    # StaticPool: one shared connection, so every thread sees the same in-memory database.
    engine = create_engine(
        "sqlite+pysqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _record) -> None:
        # SQLite ignores ON DELETE CASCADE unless this is on (PostgreSQL always enforces it).
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def isolated_database(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("postgres"):
        yield None
        return
    engine = make_sqlite_engine()
    db_session.set_engine(engine)
    yield engine
    db_session.set_engine(None)
