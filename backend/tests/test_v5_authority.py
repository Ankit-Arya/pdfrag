from app.rag.v5.ingestion import _authority_metadata
from app.rag.v5.types import V5Chunk


def _chunk(index: int, page: int, body: str, *, section: list[str] | None = None, kind: str = "text") -> V5Chunk:
    path = section or []
    header = "\n".join(
        [
            "[PDF STRUCTURE]",
            "File: claims.pdf",
            f"Pages: {page}",
            f"Section path: {' > '.join(path) if path else 'Unsectioned content'}",
            f"Content type: {kind}",
            "[/PDF STRUCTURE]",
        ]
    )
    return V5Chunk(
        chunk_id=f"00000000-0000-0000-0000-{index:012d}",
        chunk_index=index,
        page_number=page,
        page_end=page,
        content_type=kind,
        text=f"{header}\n\n{body}",
        parent_key="section:test",
        section_path=path,
        heading=path[-1] if path else "",
    )


def test_section_substitution_marks_multipage_rows_current_and_appended_base_historical() -> None:
    anchor = "For the Second Schedule of the principal rules, the following Schedule shall be substituted, namely:-"
    chunks = [
        _chunk(
            0,
            8,
            'The Metro Railways (Procedure of Claims) Amendment Rules, 2025. '
            'In rule 18, for the words "five lakh", the words "eight lakh" shall be substituted.',
        ),
        _chunk(1, 8, '"The Second Schedule (see rule 17)', section=[anchor]),
        _chunk(2, 9, "PART III continuation", section=[anchor, "PART III"]),
        _chunk(
            3,
            10,
            "Table row 33: Nature of Injury: Fracture of Major Bone-Femur, Tibia of one limb | Amount of Compensation: 80,000",
            section=[anchor, "PART III"],
            kind="table_row",
        ),
        _chunk(4, 11, "The Gazette of India, 2017 PUBLISHED BY AUTHORITY", section=["PUBLISHED BY AUTHORITY"]),
        _chunk(5, 25, "33. Fracture of Major Bone-Femur, Tibia of one limb 32,000", section=["The Second Schedule"]),
    ]

    directives = _authority_metadata(chunks)

    assert any(item["directive_type"] == "replace_section" and item["target_norm"] == "second schedule" for item in directives)
    assert chunks[3].authority_status == "current_replacement"
    assert chunks[5].authority_status == "historical_appended"


def test_authority_extraction_reads_structural_heading_not_only_body() -> None:
    anchor = "For the Second Schedule of the principal rules, the following Schedule shall be substituted, namely:-"
    chunks = [
        _chunk(0, 1, "Metro Railways Claims Amendment Rules, 2025."),
        _chunk(1, 2, "PART I", section=[anchor]),
    ]

    directives = _authority_metadata(chunks)

    assert any(item["directive_type"] == "replace_section" for item in directives)
