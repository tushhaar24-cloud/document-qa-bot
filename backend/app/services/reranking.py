"""Cohere Rerank: reorder retrieved candidates by how well each one answers the question.

Retrieval (dense, BM25, RRF) is fast but approximate: it compares the question and each
chunk separately. A reranker reads the question and a chunk *together*, so it judges
relevance much more precisely; it is too slow to run over a whole index, so it only
reorders the few candidates retrieval already found.

Only chunk texts are sent to Cohere, never filenames, IDs or other metadata. Cohere
returns positions in the list we sent, so every reranked chunk is the original
RetrievedChunk object with only `rerank_score` added: metadata cannot be lost or invented.

A reranking failure never fails the request: `apply_reranking` falls back to the
retrieval order and reports status "failed" (it never claims the results were reranked).
"""

import logging
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

import cohere
import httpx
from cohere.core.api_error import ApiError
from cohere.errors import (
    ForbiddenError,
    GatewayTimeoutError,
    InvalidTokenError,
    TooManyRequestsError,
    UnauthorizedError,
)

from app.core.config import get_settings
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

RerankStatus = Literal["applied", "disabled", "skipped", "failed"]


class RerankError(Exception):
    """A reranking failure whose message is safe to show to API clients."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class RerankOutcome:
    chunks: list[RetrievedChunk]
    # applied:  Cohere reordered the candidates.
    # disabled: reranking is turned off; retrieval order is kept.
    # skipped:  0 or 1 candidates, nothing to reorder; Cohere is not called.
    # failed:   Cohere failed; retrieval order is kept and `error` says why.
    status: RerankStatus
    model: str | None = None
    latency_ms: float | None = None
    error: str | None = None


@lru_cache
def _get_http_client() -> httpx.AsyncClient:
    # Owned here so it can be closed at shutdown (the Cohere client has no public close()).
    return httpx.AsyncClient(timeout=get_settings().rerank_timeout_seconds)


@lru_cache
def get_cohere_client() -> cohere.AsyncClientV2:
    settings = get_settings()
    if not settings.cohere_api_key:
        raise RerankError("The reranking service is not configured.")
    return cohere.AsyncClientV2(
        api_key=settings.cohere_api_key, timeout=settings.rerank_timeout_seconds, httpx_client=_get_http_client()
    )


async def close_cohere_client() -> None:
    # Only close a client that was actually created.
    if _get_http_client.cache_info().currsize:
        await _get_http_client().aclose()
    _get_http_client.cache_clear()
    get_cohere_client.cache_clear()


async def rerank_chunks(
    question: str, candidates: list[RetrievedChunk], top_n: int, client: cohere.AsyncClientV2 | None = None
) -> list[RetrievedChunk]:
    """The `top_n` most relevant candidates, best first, each with its `rerank_score`. Raises RerankError."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if not candidates:
        return []
    settings = get_settings()
    client = client or get_cohere_client()

    try:
        response = await client.rerank(
            model=settings.rerank_model,
            query=question,
            documents=[chunk.text for chunk in candidates],
            top_n=min(top_n, len(candidates)),
            max_tokens_per_doc=settings.rerank_max_tokens_per_doc,
            request_options={"max_retries": 1, "timeout_in_seconds": settings.rerank_timeout_seconds},
        )
    # Exception details can echo request data, so only the type and status code are logged.
    except (httpx.TimeoutException, GatewayTimeoutError) as exc:
        logger.warning("Cohere rerank timed out: %s", type(exc).__name__)
        raise RerankError("The reranking service timed out.") from exc
    except (UnauthorizedError, ForbiddenError, InvalidTokenError) as exc:
        logger.error("Cohere rejected the API key (status %s)", exc.status_code)
        raise RerankError("The reranking service is misconfigured.") from exc
    except TooManyRequestsError as exc:
        logger.warning("Cohere rate limit reached")
        raise RerankError("The reranking service rate limit was reached.") from exc
    except ApiError as exc:
        logger.error("Cohere rerank returned an error (status %s)", exc.status_code)
        raise RerankError("The reranking service returned an error.") from exc
    except httpx.HTTPError as exc:
        logger.warning("Could not reach Cohere: %s", type(exc).__name__)
        raise RerankError("The reranking service is unreachable.") from exc

    reranked: list[RetrievedChunk] = []
    seen: set[int] = set()
    for item in sorted(response.results, key=lambda r: -r.relevance_score):
        if not 0 <= item.index < len(candidates) or item.index in seen:
            logger.error("Cohere rerank returned an invalid result index")
            raise RerankError("The reranking service returned an invalid response.")
        seen.add(item.index)
        reranked.append(replace(candidates[item.index], rerank_score=round(float(item.relevance_score), 4)))
    return reranked[:top_n]


async def apply_reranking(
    question: str, candidates: list[RetrievedChunk], top_n: int, enabled: bool | None = None
) -> RerankOutcome:
    """Rerank when enabled; on failure keep the retrieval order and report status "failed"."""
    settings = get_settings()
    enabled = settings.rerank_enabled if enabled is None else enabled
    if not enabled:
        return RerankOutcome(candidates[:top_n], "disabled")
    if len(candidates) <= 1:
        return RerankOutcome(candidates[:top_n], "skipped", settings.rerank_model)

    started = time.perf_counter()
    try:
        chunks = await rerank_chunks(question, candidates, top_n)
    except RerankError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.warning("Reranking failed after %.0f ms; using the retrieval order", latency_ms)
        return RerankOutcome(candidates[:top_n], "failed", settings.rerank_model, latency_ms, exc.message)

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info("Reranked %d candidates to %d in %.0f ms", len(candidates), len(chunks), latency_ms)
    return RerankOutcome(chunks, "applied", settings.rerank_model, latency_ms)
