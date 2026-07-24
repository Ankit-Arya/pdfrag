from app.rag.chunking import chunk_pages
from app.rag.types import PageText


def test_chunking_preserves_source_metadata_and_overlap() -> None:
    text = " ".join(f"word-{index}" for index in range(300))
    chunks = chunk_pages(
        [PageText(filename="policy.pdf", page_number=7, text=text)],
        chunk_size=240,
        overlap=40,
    )

    assert len(chunks) > 2
    assert all(chunk.filename == "policy.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_short_page_becomes_one_chunk() -> None:
    chunks = chunk_pages(
        [PageText(filename="a.pdf", page_number=1, text="A short paragraph.")],
        chunk_size=500,
        overlap=50,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."
