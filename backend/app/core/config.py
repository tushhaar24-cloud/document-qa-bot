"""Application configuration loaded from environment variables.

Values come from the process environment, with a local `.env` file as a
fallback for development. Secrets are never hardcoded here.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

# Paths are resolved relative to this file so they work from any working directory.
BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"

# dense = embeddings only, sparse = BM25 keywords only, hybrid = both fused with RRF.
RETRIEVAL_MODES = ("dense", "sparse", "hybrid")

# Real environment variables take precedence over values in .env.
load_dotenv(ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    port: int
    groq_api_key: str
    groq_model: str
    groq_timeout_seconds: float
    groq_max_completion_tokens: int
    max_question_length: int
    max_upload_bytes: int
    max_pdf_pages: int
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    embedding_device: str
    embedding_batch_size: int
    embedding_preload: bool
    chroma_persist_dir: Path
    chroma_collection_name: str
    retrieval_top_k: int
    retrieval_max_top_k: int
    rag_min_similarity: float
    retrieval_mode: str
    hybrid_dense_candidates: int
    hybrid_sparse_candidates: int
    rrf_k: int
    cohere_api_key: str
    rerank_enabled: bool
    rerank_model: str
    rerank_candidates: int
    rerank_timeout_seconds: float
    rerank_max_tokens_per_doc: int
    # Browser origins allowed to call the API (the React dev server by default).
    cors_origins: tuple[str, ...]
    # SQLAlchemy URL for PostgreSQL; contains the password, so it is never logged.
    database_url: str
    # Create the database and any missing tables when the API starts.
    db_init_on_startup: bool

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be at least 0 and smaller than CHUNK_SIZE.")
        if not 1 <= self.retrieval_top_k <= self.retrieval_max_top_k:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and RETRIEVAL_MAX_TOP_K.")
        if not -1.0 <= self.rag_min_similarity <= 1.0:
            raise ValueError("RAG_MIN_SIMILARITY must be between -1 and 1.")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE must be one of: {', '.join(RETRIEVAL_MODES)}.")
        if not 1 <= self.hybrid_dense_candidates <= 1000 or not 1 <= self.hybrid_sparse_candidates <= 1000:
            raise ValueError("HYBRID_DENSE_CANDIDATES and HYBRID_SPARSE_CANDIDATES must be between 1 and 1000.")
        if self.rrf_k < 0:
            raise ValueError("RRF_K must not be negative.")
        if not 1 <= self.rerank_candidates <= self.retrieval_max_top_k:
            raise ValueError("RERANK_CANDIDATES must be between 1 and RETRIEVAL_MAX_TOP_K.")
        if self.rerank_timeout_seconds <= 0 or self.rerank_max_tokens_per_doc < 1:
            raise ValueError("RERANK_TIMEOUT_SECONDS and RERANK_MAX_TOKENS_PER_DOC must be positive.")

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    # Keep the API key out of logs and tracebacks.
    def __repr__(self) -> str:
        return (
            f"Settings(app_name={self.app_name!r}, groq_model={self.groq_model!r}, groq_api_key='***', "
            f"rerank_model={self.rerank_model!r}, cohere_api_key='***', "
            f"database_url={make_url(self.database_url).render_as_string(hide_password=True)!r})"
        )


def build_database_url() -> str:
    """DATABASE_URL when set, otherwise built from the DB_* variables."""
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return explicit
    # URL.create escapes special characters in the password.
    return URL.create(
        "postgresql+psycopg",
        username=os.getenv("DB_USER", "postgres").strip(),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost").strip(),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "document_qa_bot").strip(),
    ).render_as_string(hide_password=False)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Evaluated Document QA Bot"),
        app_env=os.getenv("APP_ENV", "development").strip().lower(),
        port=int(os.getenv("PORT", "8030")),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        groq_timeout_seconds=float(os.getenv("GROQ_TIMEOUT_SECONDS", "30")),
        groq_max_completion_tokens=int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2048")),
        max_question_length=int(os.getenv("MAX_QUESTION_LENGTH", "2000")),
        max_upload_bytes=int(float(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024),
        max_pdf_pages=int(os.getenv("MAX_PDF_PAGES", "500")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        embedding_preload=os.getenv("EMBEDDING_PRELOAD", "true").strip().lower() == "true",
        # A relative CHROMA_PERSIST_DIR is relative to the backend folder.
        chroma_persist_dir=BACKEND_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db").strip(),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "document_chunks").strip(),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        retrieval_max_top_k=int(os.getenv("RETRIEVAL_MAX_TOP_K", "20")),
        # Chunks below this cosine similarity are never shown to the LLM. Deliberately low:
        # it only drops clearly unrelated text; the LLM's citation check does the rest.
        rag_min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.2")),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower(),
        # Candidates each retriever contributes before fusion (at least top_k are always fetched).
        hybrid_dense_candidates=int(os.getenv("HYBRID_DENSE_CANDIDATES", "20")),
        hybrid_sparse_candidates=int(os.getenv("HYBRID_SPARSE_CANDIDATES", "20")),
        # Reciprocal Rank Fusion constant: larger values shrink the advantage of the very top ranks.
        rrf_k=int(os.getenv("RRF_K", "60")),
        cohere_api_key=os.getenv("COHERE_API_KEY", "").strip(),
        # On by default only when a Cohere key is configured; RERANK_ENABLED overrides either way.
        rerank_enabled=os.getenv(
            "RERANK_ENABLED", "true" if os.getenv("COHERE_API_KEY", "").strip() else "false"
        ).strip().lower() == "true",
        rerank_model=os.getenv("RERANK_MODEL", "rerank-v5.0-fast").strip(),
        # Retrieved candidates sent to the reranker; the best top_k of them become the LLM context.
        rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "20")),
        rerank_timeout_seconds=float(os.getenv("RERANK_TIMEOUT_SECONDS", "10")),
        # Chunks are ~800 characters (~200 tokens); this cap only guards against unusually long ones.
        rerank_max_tokens_per_doc=int(os.getenv("RERANK_MAX_TOKENS_PER_DOC", "512")),
        cors_origins=tuple(
            origin.strip().rstrip("/")
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
            if origin.strip()
        ),
        database_url=build_database_url(),
        db_init_on_startup=os.getenv("DB_INIT_ON_STARTUP", "true").strip().lower() == "true",
    )
