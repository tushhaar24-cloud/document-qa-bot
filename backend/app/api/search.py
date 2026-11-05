"""Search endpoints (dense, sparse or hybrid retrieval, without generating an answer)."""

import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.common import RerankInfo
from app.schemas.document import SearchRequest, SearchResponse, SearchResult
from app.services.embeddings import EmbeddingError
from app.services.reranking import RerankOutcome, apply_reranking
from app.services.retrieval import retrieve
from app.services.vector_store import VectorStoreError

router = APIRouter(prefix="/api", tags=["search"])

SEARCH_ERROR_RESPONSES = {
    422: {"description": "Invalid question, top_k or document ID"},
    500: {"description": "Embedding or vector store failure"},
    503: {"description": "Embedding model or vector store unavailable"},
}


def rerank_info(outcome: RerankOutcome) -> RerankInfo:
    return RerankInfo(status=outcome.status, model=outcome.model, latency_ms=outcome.latency_ms, error=outcome.error)


async def run_search(
    question: str, top_k: int, document_id: str | None, mode: str | None = None, rerank: bool | None = None
) -> SearchResponse:
    settings = get_settings()
    mode = mode or settings.retrieval_mode
    rerank = settings.rerank_enabled if rerank is None else rerank
    # With reranking, retrieve a wider candidate pool; the reranker keeps the best top_k of it.
    fetch = max(top_k, settings.rerank_candidates) if rerank else top_k
    try:
        # Embedding the question, BM25 and querying Chroma are blocking; run them off the event loop.
        candidates = await asyncio.to_thread(retrieve, question, fetch, document_id, mode)
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    reranking = await apply_reranking(question, candidates, top_k, enabled=rerank)
    chunks = reranking.chunks
    return SearchResponse(
        question=question,
        document_id=document_id,
        top_k=top_k,
        retrieval_mode=mode,
        reranking=rerank_info(reranking),
        result_count=len(chunks),
        results=[SearchResult(**vars(chunk)) for chunk in chunks],
    )


@router.post("/search", response_model=SearchResponse, responses=SEARCH_ERROR_RESPONSES)
async def search(request: SearchRequest) -> SearchResponse:
    """Search all indexed documents, or only `document_id` when it is given."""
    return await run_search(
        request.question, request.top_k, request.document_id, request.retrieval_mode, request.rerank
    )
