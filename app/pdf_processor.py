"""
PDF -> page-by-page text extraction using PyMuPDF (fitz).
"""
import re
import pymupdf as fitz  # PyMuPDF (the `fitz` import name is deprecated upstream)


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns a list of {"page": int, "text": str} for every non-empty page.
    Page numbers are 1-indexed to match how humans reference documents.
    """
    pages = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            if cleaned:
                pages.append({"page": i + 1, "text": cleaned})
    finally:
        doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Light cleanup: collapse whitespace, drop stray control characters,
    keep it simple so we don't accidentally destroy meaningful content.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
