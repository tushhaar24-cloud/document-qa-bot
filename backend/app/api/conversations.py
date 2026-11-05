"""Chat history: saved conversations with their questions, answers and citations.

Conversations are created by POST /api/ask (on a question without `conversation_id`).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, Conversation
from app.db.session import get_db
from app.repositories import conversations as conversation_repo
from app.schemas.conversation import (
    ConversationDetail,
    ConversationList,
    ConversationSummary,
    ConversationUpdate,
    StoredCitation,
    StoredMessage,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

DbSession = Annotated[Session, Depends(get_db)]
DB_ERROR_RESPONSES = {500: {"description": "Database error"}, 503: {"description": "Database unavailable"}}
NOT_FOUND = {404: {"description": "Conversation not found"}, **DB_ERROR_RESPONSES}


def _message(message: ChatMessage) -> StoredMessage:
    return StoredMessage(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        status=message.answer_status,
        document_id=message.document_id,
        retrieval_mode=message.retrieval_mode,
        rerank_status=message.rerank_status,
        rerank_model=message.rerank_model,
        retrieved_chunk_count=message.retrieved_chunk_count,
        context_chunk_count=message.context_chunk_count,
        sources=[
            StoredCitation(
                citation=citation.citation_number,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                filename=citation.filename,
                page_number=citation.page_number,
                excerpt=citation.excerpt,
                similarity=citation.similarity,
                rerank_score=citation.rerank_score,
            )
            for citation in message.citations
        ],
    )


def _detail(conversation: Conversation) -> ConversationDetail:
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        document_id=conversation.document_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_message(message) for message in conversation.messages],
    )


def _get_or_404(db: Session, conversation_id: uuid.UUID, with_messages: bool = False) -> Conversation:
    conversation = conversation_repo.get_conversation(db, conversation_id, with_messages=with_messages)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@router.get("", response_model=ConversationList, responses=DB_ERROR_RESPONSES)
def list_conversations(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationList:
    """Saved conversations, most recently active first."""
    rows, total = conversation_repo.list_conversations(db, limit, offset)
    return ConversationList(
        total=total,
        conversations=[
            ConversationSummary(
                id=conversation.id,
                title=conversation.title,
                document_id=conversation.document_id,
                message_count=count,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
            for conversation, count in rows
        ],
    )


@router.get("/{conversation_id}", response_model=ConversationDetail, responses=NOT_FOUND)
def get_conversation(conversation_id: uuid.UUID, db: DbSession) -> ConversationDetail:
    """A conversation with every message and the sources each answer cited."""
    return _detail(_get_or_404(db, conversation_id, with_messages=True))


@router.patch("/{conversation_id}", response_model=ConversationDetail, responses=NOT_FOUND)
def rename_conversation(conversation_id: uuid.UUID, update: ConversationUpdate, db: DbSession) -> ConversationDetail:
    conversation = _get_or_404(db, conversation_id, with_messages=True)
    conversation_repo.rename_conversation(db, conversation, update.title)
    db.commit()
    return _detail(conversation)


@router.delete("/{conversation_id}", status_code=204, responses=NOT_FOUND)
def delete_conversation(conversation_id: uuid.UUID, db: DbSession) -> Response:
    """Delete a conversation with all its messages and citations."""
    conversation_repo.delete_conversation(db, _get_or_404(db, conversation_id))
    db.commit()
    return Response(status_code=204)
