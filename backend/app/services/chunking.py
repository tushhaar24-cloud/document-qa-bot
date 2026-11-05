"""Split extracted PDF pages into overlapping chunks for retrieval.

Each page is split on its own, so every chunk belongs to exactly one page and
its `page_number` metadata is always accurate for citations.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from app.services.ingestion import PageText


@lru_cache
def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    # Tries to split on paragraphs, then lines, then words, then characters,
    # so chunks break at the most natural boundary that fits chunk_size.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )


def make_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """Deterministic ID: the same document and settings always produce the same IDs."""
    return f"{document_id}:p{page_number}:c{chunk_index}"


def chunk_pages(
    pages: Sequence[PageText],
    document_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []

    for page in pages:
        for piece in splitter.create_documents([page.text]):
            text = piece.page_content.strip()
            if not text:
                continue
            chunk_index = len(chunks)  # position within the whole document
            chunk_id = make_chunk_id(document_id, page.page_number, chunk_index)
            chunks.append(
                Document(
                    id=chunk_id,
                    page_content=text,
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        # Character offset of the chunk within its page.
                        "start_index": piece.metadata.get("start_index", -1),
                    },
                )
            )

    return chunks
