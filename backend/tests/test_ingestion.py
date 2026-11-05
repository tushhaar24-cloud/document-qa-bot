"""Unit tests for PDF text extraction."""

import pytest

from app.services.ingestion import DocumentIngestionError, extract_pdf_pages
from tests.pdf_factory import make_pdf


def test_extracts_text_with_one_based_page_numbers() -> None:
    result = extract_pdf_pages(make_pdf(["Introduction", "Terms and conditions"]), max_pages=10)

    assert result.page_count == 2
    assert [(p.page_number, p.text) for p in result.pages] == [
        (1, "Introduction"),
        (2, "Terms and conditions"),
    ]
    assert result.empty_pages == []


def test_reports_pages_without_text() -> None:
    result = extract_pdf_pages(make_pdf(["First", "", "Third"]), max_pages=10)

    assert [p.page_number for p in result.pages] == [1, 3]
    assert result.empty_pages == [2]


@pytest.mark.parametrize(
    ("data", "expected_status"),
    [
        (b"just some text, not a pdf", 415),
        (b"%PDF-1.7\n" + b"\x00garbage" * 50, 422),
        (make_pdf(["", ""]), 422),
        (make_pdf(["secret"], user_password="pw"), 422),
    ],
    ids=["not-a-pdf", "corrupt", "no-text", "password-protected"],
)
def test_rejects_unusable_pdfs(data: bytes, expected_status: int) -> None:
    with pytest.raises(DocumentIngestionError) as exc_info:
        extract_pdf_pages(data, max_pages=10)
    assert exc_info.value.status_code == expected_status


def test_rejects_too_many_pages() -> None:
    with pytest.raises(DocumentIngestionError) as exc_info:
        extract_pdf_pages(make_pdf(["a", "b", "c"]), max_pages=2)
    assert exc_info.value.status_code == 413
