from app.rag.chunking import chunk_pages
from app.rag.types import PageText


def test_heading_context_is_embedded_in_chunks() -> None:
    pages = [
        PageText(
            filename="manual.pdf",
            page_number=1,
            text=(
                "ROLLING STOCK TYPE A\n"
                "1 Brake Isolation Procedure\n"
                "Warning: isolate supply before work.\n"
                "1.1 Preconditions\n"
                "The train must be secured before isolation.\n"
                "1.2 Steps\n"
                "1. Open the cabinet.\n"
                "2. Move the brake isolation cock to ISOLATE.\n"
            ),
        )
    ]

    chunks = chunk_pages(pages, chunk_size=500, overlap=80)

    assert chunks
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "[PDF CHUNK CONTEXT]" in joined
    assert "Section path: ROLLING STOCK TYPE A > 1 Brake Isolation Procedure" in joined
    assert "Rolling stock / train context: ROLLING STOCK TYPE A" in joined
    assert "Procedure context: Brake Isolation Procedure" in joined


def test_heading_path_carries_across_pages() -> None:
    pages = [
        PageText("manual.pdf", 1, "ROLLING STOCK TYPE B\n2 Door Reset Procedure\nStep one on page one."),
        PageText("manual.pdf", 2, "Step two continues on page two without repeating the heading."),
    ]

    chunks = chunk_pages(pages, chunk_size=700, overlap=80)

    assert any("Pages: 1-2" in chunk.text for chunk in chunks)
    assert all("Door Reset Procedure" in chunk.text for chunk in chunks)
