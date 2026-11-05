"""PDF ingestion: text extraction with PyMuPDF, chunking, embedding and indexing.

Everything here is CPU-bound and synchronous; call it from async code through
`asyncio.to_thread` so it does not block the event loop.
"""

import hashlib
import logging
from dataclasses import dataclass

import pymupdf
from langchain_core.documents import Document

from app.core.config import Settings
from app.services.chunking import chunk_pages
from app.services.embeddings import embed_documents
from app.services.vector_store import replace_document_chunks

logger = logging.getLogger(__name__)

# MuPDF prints parser errors straight to stderr; we report failures ourselves instead.
pymupdf.TOOLS.mupdf_display_errors(False)

PDF_SIGNATURE = b"%PDF-"
# The PDF spec allows the header to start anywhere in the first 1024 bytes.
SIGNATURE_SEARCH_BYTES = 1024


class DocumentIngestionError(Exception):
    """An ingestion failure that is safe to show to API clients."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, as shown in PDF viewers
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    page_count: int
    pages: list[PageText]  # only pages with usable text
    empty_pages: list[int]  # 1-based numbers of pages with no extractable text


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    extracted: ExtractedDocument
    chunks: list[Document]
    embeddings: list[list[float]]  # embeddings[i] belongs to chunks[i]
    indexed_chunk_count: int


def make_document_id(data: bytes) -> str:
    # Derived from the file content: uploading the same PDF again gives the same
    # ID (and the same chunk IDs), which lets the vector store skip duplicates later.
    return hashlib.sha256(data).hexdigest()[:32]


def has_pdf_signature(data: bytes) -> bool:
    return PDF_SIGNATURE in data[:SIGNATURE_SEARCH_BYTES]


def clean_page_text(text: str) -> str:
    # Drop NUL characters and trailing whitespace on each line, keep the line structure.
    lines = (line.rstrip() for line in text.replace("\x00", "").splitlines())
    return "\n".join(lines).strip()


def extract_pdf_pages(data: bytes, max_pages: int) -> ExtractedDocument:
    if not has_pdf_signature(data):
        raise DocumentIngestionError("The file is not a valid PDF.", 415)

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        logger.info("PDF could not be opened: %s", type(exc).__name__)
        raise DocumentIngestionError("The PDF is corrupt or unreadable.", 422) from exc

    with doc:
        if doc.needs_pass:
            raise DocumentIngestionError("Password-protected PDFs are not supported.", 422)
        if doc.page_count == 0:
            raise DocumentIngestionError("The PDF has no pages.", 422)
        if doc.page_count > max_pages:
            raise DocumentIngestionError(f"The PDF has too many pages (maximum {max_pages}).", 413)

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index, page in enumerate(doc):
            page_number = index + 1
            try:
                text = clean_page_text(page.get_text("text"))
            except RuntimeError:
                # A single damaged page should not fail the whole document.
                logger.warning("Could not read page %d", page_number)
                text = ""
            if text:
                pages.append(PageText(page_number=page_number, text=text))
            else:
                empty_pages.append(page_number)

        page_count = doc.page_count

    if not pages:
        raise DocumentIngestionError(
            "No extractable text found. The PDF may be scanned images; OCR is not supported.", 422
        )

    return ExtractedDocument(page_count=page_count, pages=pages, empty_pages=empty_pages)


def ingest_pdf(data: bytes, filename: str, settings: Settings) -> IngestedDocument:
    extracted = extract_pdf_pages(data, settings.max_pdf_pages)
    document_id = make_document_id(data)
    chunks = chunk_pages(
        extracted.pages,
        document_id=document_id,
        filename=filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = embed_documents([chunk.page_content for chunk in chunks])
    # Last step: nothing is written to the index unless everything before it succeeded.
    indexed_chunk_count = replace_document_chunks(document_id, chunks, embeddings)
    return IngestedDocument(
        document_id=document_id,
        extracted=extracted,
        chunks=chunks,
        embeddings=embeddings,
        indexed_chunk_count=indexed_chunk_count,
    )
