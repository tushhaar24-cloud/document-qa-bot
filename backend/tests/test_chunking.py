"""Unit tests for page-aware chunking."""

import dataclasses

import pytest

from app.core.config import get_settings
from app.services.chunking import chunk_pages, make_chunk_id
from app.services.ingestion import PageText

DOC_ID = "a" * 32
SENTENCE = "Retrieval-augmented generation grounds answers in source documents. "


def chunk(pages: list[PageText], chunk_size: int = 800, chunk_overlap: int = 120):
    return chunk_pages(pages, DOC_ID, "manual.pdf", chunk_size, chunk_overlap)


def test_long_page_is_split_into_multiple_chunks_within_size() -> None:
    page = PageText(page_number=1, text=SENTENCE * 40)  # ~2,700 characters

    chunks = chunk([page])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 800 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    words = " ".join(f"word{i}" for i in range(400))

    chunks = chunk([PageText(page_number=1, text=words)], chunk_size=200, chunk_overlap=50)

    for previous, current in zip(chunks, chunks[1:]):
        last_word_of_previous = previous.page_content.split()[-1]
        assert last_word_of_previous in current.page_content


def test_metadata_preserves_page_document_and_filename() -> None:
    pages = [
        PageText(page_number=2, text=SENTENCE * 20),
        PageText(page_number=5, text="Short final page."),
    ]

    chunks = chunk(pages)

    assert {c.metadata["page_number"] for c in chunks} == {2, 5}
    assert all(c.metadata["document_id"] == DOC_ID for c in chunks)
    assert all(c.metadata["filename"] == "manual.pdf" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # The last chunk comes only from page 5: pages are never merged.
    assert chunks[-1].page_content == "Short final page."
    assert chunks[-1].metadata["page_number"] == 5


def test_chunk_ids_are_deterministic_and_unique() -> None:
    pages = [PageText(page_number=1, text=SENTENCE * 30), PageText(page_number=2, text=SENTENCE * 30)]

    first_run = [c.id for c in chunk(pages)]
    second_run = [c.id for c in chunk(pages)]

    assert first_run == second_run
    assert len(set(first_run)) == len(first_run)
    assert first_run[0] == make_chunk_id(DOC_ID, 1, 0) == f"{DOC_ID}:p1:c0"
    assert all(c.metadata["chunk_id"] == c.id for c in chunk(pages))


def test_no_empty_chunks() -> None:
    text = "First paragraph.\n\n\n\n     \n\n\nSecond paragraph.\n\n   \t  \n"

    chunks = chunk([PageText(page_number=1, text=text)], chunk_size=20, chunk_overlap=5)

    assert chunks
    assert all(c.page_content.strip() for c in chunks)


def test_no_pages_gives_no_chunks() -> None:
    assert chunk([]) == []


def test_start_index_points_into_the_page() -> None:
    page = PageText(page_number=1, text=SENTENCE * 30)

    for c in chunk([page]):
        start = c.metadata["start_index"]
        assert page.text[start : start + len(c.page_content)] == c.page_content


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(get_settings(), chunk_size=100, chunk_overlap=100)
