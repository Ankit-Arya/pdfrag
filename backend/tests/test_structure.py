from app.rag.structure import (
    major_section_match_score,
    normalize_heading,
    section_match_score,
    section_path_from_text,
)


def test_section_matching_normalizes_case_and_punctuation() -> None:
    path = "CHAPTER X > PERMANENT WAY AND WORKS > 67 General"

    assert section_match_score("permanent-way and works", path) == 1.0
    assert major_section_match_score("permanent way and works", path) == 1.0
    assert normalize_heading("Permanent-Way & Works") == "permanent way works"


def test_section_path_is_read_from_chunk_header_only() -> None:
    chunk = (
        "[PDF CHUNK CONTEXT]\n"
        "Section path: CHAPTER XII > Examination of trains\n"
        "[/PDF CHUNK CONTEXT]\n\n"
        "Body text."
    )

    assert section_path_from_text(chunk) == "CHAPTER XII > Examination of trains"
    assert section_path_from_text("Body mentions Section path informally.") == ""
