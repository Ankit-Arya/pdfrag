from app.rag.v5.ingestion import _v5_terminology
from app.rag.v5.types import V5Chunk


def _chunk(text: str, *, cells: list[str] | None = None) -> V5Chunk:
    return V5Chunk(
        chunk_id="00000000-0000-0000-0000-000000000001",
        chunk_index=0,
        page_number=1,
        page_end=1,
        content_type="table_row" if cells else "text",
        text=f"[PDF STRUCTURE]\nFile: x.pdf\nPages: 1\nSection path: X\nContent type: text\n[/PDF STRUCTURE]\n\n{text}",
        parent_key="section:x",
        metadata={"cells": cells or []},
    )


def test_terminology_accepts_initials_supported_definition() -> None:
    values = _v5_terminology(_chunk("Brake Isolating Cock (BIC) shall be checked."))
    assert any(alias == "BIC" and canonical == "Brake Isolating Cock" for alias, canonical, _ in values)


def test_terminology_rejects_ocr_table_pollution_when_initials_do_not_match() -> None:
    values = _v5_terminology(_chunk("", cells=["AVAILABLE", "fo book tickets"]))
    assert values == []
