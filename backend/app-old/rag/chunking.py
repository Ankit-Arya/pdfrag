import hashlib

from app.rag.types import PageText, TextChunk


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        for ordinal, text in enumerate(_split_text(page.text, chunk_size, overlap), start=1):
            digest = hashlib.sha1(
                f"{page.filename}:{page.page_number}:{ordinal}:{text}".encode()
            ).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    chunk_id=digest,
                    filename=page.filename,
                    page_number=page.page_number,
                    text=text,
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            candidates = [
                text.rfind("\n\n", start + chunk_size // 2, hard_end),
                text.rfind(". ", start + chunk_size // 2, hard_end),
                text.rfind(" ", start + chunk_size // 2, hard_end),
            ]
            best = max(candidates)
            if best > start:
                end = best + (2 if text[best : best + 2] in {"\n\n", ". "} else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks
