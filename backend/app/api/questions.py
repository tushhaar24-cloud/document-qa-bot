"""Question-answering endpoints."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException

from app.api.search import rerank_info
from app.db.session import DatabaseError, session_scope
from app.repositories import conversations as conversation_repo
from app.repositories.conversations import AnswerData, CitationData
from app.schemas.question import QuestionRequest, QuestionResponse, SourceCitation
from app.services.embeddings import EmbeddingError
from app.services.generation import AnswerGenerationError
from app.services.rag import answer_question, make_excerpt
from app.services.vector_store import VectorStoreError, document_exists

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["questions"])


def _conversation_exists(conversation_id: uuid.UUID) -> bool:
    with session_scope() as db:
        return conversation_repo.get_conversation(db, conversation_id) is not None


def _save_exchange(
    conversation_id: uuid.UUID | None, request: QuestionRequest, response: QuestionResponse
) -> tuple[uuid.UUID, uuid.UUID]:
    """Store the question and answer; returns (conversation_id, assistant message id)."""
    answer = AnswerData(
        content=response.answer,
        answer_status=response.status,
        document_id=request.document_id,
        retrieval_mode=response.retrieval_mode,
        rerank_status=response.reranking.status,
        rerank_model=response.reranking.model,
        retrieved_chunk_count=response.retrieved_chunk_count,
        context_chunk_count=response.context_chunk_count,
        citations=[
            CitationData(
                citation_number=source.citation,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                filename=source.filename,
                page_number=source.page_number,
                excerpt=source.excerpt,
                similarity=source.similarity,
                rerank_score=source.rerank_score,
            )
            for source in response.sources
        ],
    )
    with session_scope() as db:
        conversation = (
            conversation_repo.get_conversation(db, conversation_id) if conversation_id is not None else None
        )
        if conversation is None:
            # New chat, or the conversation was deleted while the answer was generated.
            conversation = conversation_repo.create_conversation(db, request.question, request.document_id)
        message = conversation_repo.add_exchange(db, conversation, request.question, answer)
        return conversation.id, message.id


@router.post(
    "/ask",
    response_model=QuestionResponse,
    responses={
        404: {"description": "Document or conversation not found"},
        422: {"description": "Invalid question, top_k or document ID"},
        429: {"description": "LLM provider rate limit reached"},
        500: {"description": "Embedding or vector store failure"},
        502: {"description": "LLM provider returned an error"},
        503: {"description": "LLM provider, embedding model, vector store or database unavailable"},
        504: {"description": "LLM provider timed out"},
    },
)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Answer a question from the uploaded documents, with page-level citations.

    Searches only `document_id` when it is given, otherwise every indexed document.
    The question and answer are saved to `conversation_id`, or to a new conversation.
    """
    try:
        if request.conversation_id is not None:
            # Checked before calling the LLM, so a bad ID costs nothing.
            if not await asyncio.to_thread(_conversation_exists, request.conversation_id):
                raise HTTPException(status_code=404, detail="Conversation not found.")
        if request.document_id is not None:
            if not await asyncio.to_thread(document_exists, request.document_id):
                raise HTTPException(status_code=404, detail="Document not found.")
        result = await answer_question(
            request.question, request.document_id, request.top_k, request.retrieval_mode, request.rerank
        )
    except (AnswerGenerationError, EmbeddingError, VectorStoreError, DatabaseError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = QuestionResponse(
        question=result.question,
        answer=result.answer,
        status=result.status,
        document_id=result.document_id,
        sources=[
            SourceCitation(
                citation=citation.number,
                chunk_id=citation.chunk.chunk_id,
                document_id=citation.chunk.document_id,
                filename=citation.chunk.filename,
                page_number=citation.chunk.page_number,
                excerpt=make_excerpt(citation.chunk.text),
                similarity=citation.chunk.similarity,
                rerank_score=citation.chunk.rerank_score,
            )
            for citation in result.sources
        ],
        retrieved_chunk_count=result.retrieved_chunk_count,
        context_chunk_count=result.context_chunk_count,
        retrieval_mode=result.retrieval_mode,
        reranking=rerank_info(result.reranking),
        conversation_id=request.conversation_id,
    )

    try:
        conversation_id, message_id = await asyncio.to_thread(
            _save_exchange, request.conversation_id, request, response
        )
    except DatabaseError:
        # The answer has already been generated (and paid for): return it, unsaved.
        logger.error("Could not save the chat history; returning the answer without saving it")
        return response
    return response.model_copy(
        update={"conversation_id": conversation_id, "message_id": message_id, "history_saved": True}
    )
