from app.rag.evidence import build_evidence_answer
from app.rag.types import PromptSource, RetrievedChunk, TextChunk


def _source(
    chunk_id: str,
    page: int,
    section: str,
    body: str,
    *,
    content_type: str = "text",
) -> PromptSource:
    text = (
        "[PDF CHUNK CONTEXT]\n"
        "File: 02. MRGR 2020.pdf\n"
        f"Pages: {page}\n"
        f"Section path: {section}\n"
        f"Content type: {content_type}\n"
        "[/PDF CHUNK CONTEXT]\n\n"
        f"{body}"
    )
    result = RetrievedChunk(
        TextChunk(
            chunk_id,
            "02. MRGR 2020.pdf",
            page,
            text,
            content_type=content_type,
        ),
        0.9,
    )
    return PromptSource(result=result, excerpt=text)


def test_evidence_answer_includes_multiple_relevant_chunks_from_same_document() -> None:
    sources = [
        _source(
            "definition",
            78,
            "PRELIMINARY",
            '“wake-up process” means process initiated by UTMS.',
        ),
        _source(
            "examination",
            128,
            "Examination of trains",
            "(1)\nUTMS shall perform wake-up test before passenger service.\n"
            "(2)\nWake-up test examination shall check cab signalling and brakes.",
        ),
    ]
    answer, used = build_evidence_answer(
        "wake up process",
        sources,
    )

    assert "02. MRGR 2020.pdf — page 78" in answer
    assert "02. MRGR 2020.pdf — page 128" in answer
    assert "**Heading/subheading:** PRELIMINARY" in answer
    assert "**Heading/subheading:** Examination of trains" in answer
    assert "UTMS shall perform wake-up test before passenger service. [S2]" in answer
    assert "Wake-up test examination shall check cab signalling and brakes. [S2]" in answer
    assert "No further procedural steps" not in answer
    assert used == sources


def test_evidence_answer_filters_unrelated_units_and_overlap() -> None:
    answer, used = build_evidence_answer(
        "wake up process",
        [
            _source(
                "first",
                78,
                "PRELIMINARY",
                '“wake-up process” means process initiated by UTMS.\n\n'
                "Passengers may use an emergency escape door.",
            ),
            _source(
                "overlap",
                78,
                "PRELIMINARY",
                '“wake-up process” means process initiated by UTMS.',
            ),
        ]
    )

    assert answer.count("process initiated by UTMS") == 1
    assert "emergency escape door" not in answer
    assert used and len(used) == 1


def test_evidence_answer_drops_malformed_table_fragments() -> None:
    answer, used = build_evidence_answer(
        "wake up process",
        [
            _source(
                "broken-table",
                121,
                "Unattended Train Management System",
                "Table 1 on page 121 | | wake | -up an | d sle | ep process |",
                content_type="table",
            )
        ],
    )

    assert answer == ""
    assert used == []


def test_evidence_answer_converts_valid_table_row_to_readable_text() -> None:
    answer, used = build_evidence_answer(
        "15 kmph",
        [
            _source(
                "valid-table",
                20,
                "Speed restrictions > Table 1",
                "| Condition | Speed |\n"
                "| --- | --- |\n"
                "| Train operated in RM mode | 15 kmph |\n"
                "| Normal ATO operation | As authorised |",
                content_type="table",
            )
        ],
    )

    assert "Condition: Train operated in RM mode; Speed: 15 kmph [S1]" in answer
    assert "Normal ATO operation" not in answer
    assert "|" not in answer
    assert len(used) == 1
