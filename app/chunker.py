"""
Splits per-page text into overlapping character chunks.
CHUNK_SIZE and CHUNK_OVERLAP come from .env so behavior is tunable
without touching code.
"""
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """
    pages: [{"page": int, "text": str}, ...]
    Returns: [{"text": str, "page": int, "source": str}, ...]

    Chunking is done per-page so every chunk keeps an accurate page number
    for citation in the UI. A page shorter than CHUNK_SIZE becomes one chunk.
    """
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    all_chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]

        if len(text) <= CHUNK_SIZE:
            all_chunks.append({"text": text, "page": page_num, "source": source})
            continue

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                all_chunks.append({"text": chunk_text, "page": page_num, "source": source})
            if end >= len(text):
                break
            start += step

    return all_chunks
