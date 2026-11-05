"""Builds small in-memory PDFs for tests, so no real documents are needed."""

import pymupdf


def make_pdf(page_texts: list[str], user_password: str | None = None) -> bytes:
    """One page per entry; an empty string produces a page with no text."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            # Wrap text inside 1-inch margins so long strings stay on the page.
            margins = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
            overflow = page.insert_textbox(margins, text, fontsize=10)
            assert overflow >= 0, "test text does not fit on one page"

    save_options = {}
    if user_password:
        save_options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "user_pw": user_password,
            "owner_pw": user_password + "-owner",
        }
    data = doc.tobytes(**save_options)
    doc.close()
    return data
