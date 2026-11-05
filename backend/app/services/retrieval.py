"""Retrieval: find the stored chunks most relevant to a question.

Three modes (setting RETRIEVAL_MODE, or per request):
- dense:  embedding similarity in ChromaDB. Finds the same meaning in different words.
- sparse: BM25 keyword match. Finds exact words, numbers, clause numbers and identifiers.
- hybrid: both, merged with Reciprocal Rank Fusion (RRF).

RRF combines the two result lists by rank, never by score: a cosine similarity
(0..1) and a BM25 score (unbounded, depends on the query) are on different scales
and cannot be added. Each chunk scores sum(1 / (rrf_k + rank)) over the lists it
appears in, so chunks ranked well by both retrievers rise to the top.

Every returned chunk carries its real cosine `similarity` to the question, even
when only BM25 found it, so the RAG relevance floor still has a comparable number.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.core.config import RETRIEVAL_MODES, get_settings
from app.services.embeddings import embed_query
from app.services.sparse_retrieval import sparse_search
from app.services.vector_store import get_chunk_embeddings, query_chunks


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    distance: float  # cosine distance, lower is closer
    similarity: float  # 1 - distance; higher is more relevant
    # How each retriever ranked this chunk (1 = best); None when it was not in that list.
    dense_rank: int | None = None
    sparse_rank: int | None = None
    bm25_score: float | None = None  # raw BM25 score; None when no query word matched
    fusion_score: float | None = None  # RRF score, hybrid mode only
    rerank_score: float | None = None  # Cohere relevance score (0..1); None when not reranked


def _validate(question: str, top_k: int | None) -> tuple[str, int]:
    settings = get_settings()
    question = question.strip()
    if not question:
        raise ValueError("The question must not be empty.")
    top_k = settings.retrieval_top_k if top_k is None else top_k
    if not 1 <= top_k <= settings.retrieval_max_top_k:
        raise ValueError(f"top_k must be between 1 and {settings.retrieval_max_top_k}.")
    return question, top_k


def _make_chunk(chunk_id: str, text: str, metadata: dict, similarity: float, **ranks) -> RetrievedChunk:
    similarity = round(similarity, 4)
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=metadata.get("document_id", ""),
        filename=metadata.get("filename", ""),
        page_number=int(metadata.get("page_number", 0)),
        chunk_index=int(metadata.get("chunk_index", 0)),
        text=text,
        distance=round(1 - similarity, 4),
        similarity=similarity,
        **ranks,
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def _similarities(query_embedding: Sequence[float], chunk_ids: Sequence[str]) -> dict[str, float]:
    """Cosine similarity for chunks that dense search did not return, from their stored embeddings."""
    stored = get_chunk_embeddings(chunk_ids)
    return {cid: _cosine(query_embedding, embedding) for cid, embedding in stored.items()}


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int) -> list[tuple[str, float]]:
    """Merge ranked ID lists into one, best first. Duplicates are merged into a single entry.

    Ties are broken by the best single rank, then by ID, so the order is deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue  # a repeated ID within one list only counts once, at its best rank
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


def retrieve_chunks(question: str, top_k: int | None = None, document_id: str | None = None) -> list[RetrievedChunk]:
    """Dense (embedding) retrieval."""
    question, top_k = _validate(question, top_k)
    # Same model as the stored chunks, so the vectors are comparable.
    query_embedding = embed_query(question)
    matches = query_chunks(query_embedding, top_k=top_k, document_id=document_id)

    return [
        replace(_make_chunk(m.chunk_id, m.text, m.metadata, 1 - m.distance, dense_rank=rank), distance=round(m.distance, 4))
        for rank, m in enumerate(matches, start=1)
    ]


def retrieve_sparse(question: str, top_k: int | None = None, document_id: str | None = None) -> list[RetrievedChunk]:
    """Sparse (BM25 keyword) retrieval. Only chunks sharing a meaningful word with the question are returned."""
    question, top_k = _validate(question, top_k)
    matches = sparse_search(question, top_k, document_id)
    if not matches:
        return []
    similarities = _similarities(embed_query(question), [m.chunk.chunk_id for m in matches])
    return [
        _make_chunk(
            m.chunk.chunk_id, m.chunk.text, m.chunk.metadata, similarities[m.chunk.chunk_id],
            sparse_rank=rank, bm25_score=round(m.score, 4),
        )
        for rank, m in enumerate(matches, start=1)
        if m.chunk.chunk_id in similarities  # deleted between the two reads
    ]


def retrieve_hybrid(question: str, top_k: int | None = None, document_id: str | None = None) -> list[RetrievedChunk]:
    """Dense and sparse candidates merged with Reciprocal Rank Fusion."""
    question, top_k = _validate(question, top_k)
    settings = get_settings()

    query_embedding = embed_query(question)
    dense = query_chunks(query_embedding, max(top_k, settings.hybrid_dense_candidates), document_id)
    sparse = sparse_search(question, max(top_k, settings.hybrid_sparse_candidates), document_id)

    dense_by_id = {m.chunk_id: (rank, m) for rank, m in enumerate(dense, start=1)}
    sparse_by_id = {m.chunk.chunk_id: (rank, m) for rank, m in enumerate(sparse, start=1)}
    fused = reciprocal_rank_fusion([list(dense_by_id), list(sparse_by_id)], settings.rrf_k)[:top_k]

    keyword_only = [cid for cid, _ in fused if cid not in dense_by_id]
    similarities = _similarities(query_embedding, keyword_only)

    results: list[RetrievedChunk] = []
    for chunk_id, fusion_score in fused:
        dense_rank, dense_match = dense_by_id.get(chunk_id, (None, None))
        sparse_rank, sparse_match = sparse_by_id.get(chunk_id, (None, None))
        if dense_match is not None:
            text, metadata, similarity = dense_match.text, dense_match.metadata, 1 - dense_match.distance
        elif chunk_id in similarities:
            text, metadata, similarity = sparse_match.chunk.text, sparse_match.chunk.metadata, similarities[chunk_id]
        else:
            continue  # deleted between the two reads
        results.append(
            _make_chunk(
                chunk_id, text, metadata, similarity,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                bm25_score=round(sparse_match.score, 4) if sparse_match else None,
                fusion_score=round(fusion_score, 6),
            )
        )
    return results


def retrieve(
    question: str, top_k: int | None = None, document_id: str | None = None, mode: str | None = None
) -> list[RetrievedChunk]:
    """Retrieve with `mode`, or with the RETRIEVAL_MODE setting when it is None."""
    mode = mode or get_settings().retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval mode must be one of: {', '.join(RETRIEVAL_MODES)}.")
    retriever = {"dense": retrieve_chunks, "sparse": retrieve_sparse, "hybrid": retrieve_hybrid}[mode]
    return retriever(question, top_k, document_id)
