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
        if page.content_type == "table":
            page_chunks = _split_table(page.text, chunk_size)
        else:
            page_chunks = _split_text(page.text, chunk_size, overlap)

        for ordinal, text in enumerate(page_chunks, start=1):
            digest = hashlib.sha1(
                (
                    f"{page.filename}:{page.page_number}:{page.content_type}:"
                    f"{page.table_index}:{ordinal}:{text}"
                ).encode()
            ).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    chunk_id=digest,
                    filename=page.filename,
                    page_number=page.page_number,
                    text=text,
                    content_type=page.content_type,
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
                text.rfind("\n", start + chunk_size // 2, hard_end),
                text.rfind(". ", start + chunk_size // 2, hard_end),
                text.rfind(" ", start + chunk_size // 2, hard_end),
            ]
            best = max(candidates)
            if best > start:
                delimiter = text[best : best + 2]
                end = best + (2 if delimiter in {"\n\n", ". "} else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_table(text: str, chunk_size: int) -> list[str]:
    """Split Markdown tables by complete rows and repeat the header in each chunk."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    table_start = next((index for index, line in enumerate(lines) if line.startswith("|")), -1)
    if table_start < 0 or len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    prefix = lines[:table_start]
    table_lines = lines[table_start:]
    if len(table_lines) < 3:
        return _split_text(text, chunk_size, 0)

    header = table_lines[:2]
    rows = table_lines[2:]
    chunks: list[str] = []
    current = prefix + header

    for row in rows:
        candidate = "\n".join(current + [row])
        if len(candidate) > chunk_size and len(current) > len(prefix) + len(header):
            chunks.append("\n".join(current).strip())
            current = prefix + header + [row]
        else:
            current.append(row)

    if current:
        chunks.append("\n".join(current).strip())
    return chunks
