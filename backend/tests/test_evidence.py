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
            "“wake-up process” means process initiated by UTMS.",
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
    assert "#### PRELIMINARY" in answer
    assert "#### Examination of trains" in answer
    assert "Heading/subheading:" not in answer
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
                "“wake-up process” means process initiated by UTMS.\n\n"
                "Passengers may use an emergency escape door.",
            ),
            _source(
                "overlap",
                78,
                "PRELIMINARY",
                "“wake-up process” means process initiated by UTMS.",
            ),
        ],
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


def test_major_section_query_includes_section_body_and_hides_incidental_mentions() -> None:
    sources = [
        _source(
            "incidental",
            9,
            "Accident definitions",
            "An accident may affect ELECTRICAL SAFETY AND CONTROL equipment.",
        ),
        _source(
            "chapter",
            80,
            "ELECTRICAL SAFETY AND CONTROL > General",
            "(1)\nAll running equipment shall be inspected as scheduled.\n"
            "(2)\nAuthorised persons shall wear the required protective equipment.",
        ),
    ]

    answer, used = build_evidence_answer(
        "ELECTRICAL SAFETY AND CONTROL",
        sources,
    )

    assert "All running equipment shall be inspected as scheduled." in answer
    assert "Authorised persons shall wear the required protective equipment." in answer
    assert "An accident may affect" not in answer
    assert "#### General" in answer
    assert "Heading/subheading:" not in answer
    assert [source.result.chunk.chunk_id for source in used] == ["chapter"]


def test_evidence_answer_rejects_fragmented_prose_disguised_as_table() -> None:
    false_table = (
        "| THE G | AZETTE | OF INDIA | : EXTRAORDI | NARY | [PART II | —SEC. 3(i)] |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| running tracks | shall b | e inspecte | d as per the | schedule sp | "
        "ecified in t | he special |\n"
        "| authorised pers | ons who | se duties r | equire them t | o go on the t | "
        "racks shall b | e trained |\n"
        "| maintenance st | aff requir | ing to carr | y out inspecti | on or repair | "
        "of equipment | which does |"
    )

    answer, used = build_evidence_answer(
        "track work",
        [
            _source(
                "false-table",
                114,
                "67 Track work and track side work in non-traffic hours > Table 1",
                false_table,
                content_type="table",
            )
        ],
    )

    assert answer == ""
    assert used == []


def test_absence_lookup_combines_heading_rule_and_duration_rows() -> None:
    source = _source(
        "absence",
        25,
        "SPEED AND WORKING OF TRAINS > 25 General",
        "A Train Operator who has not worked on a section for three months or more "
        "should be given road learning trips to refresh his knowledge as under:\n\n"
        "DURATION OF ABSENCE | NUMBER OF ROAD LEARNING TRIPS\n"
        "3-6 months | 1 round trip\nOver 6 months | 3 round trips",
    )

    for question in ("3 month absence", "TO absence for months"):
        answer, used = build_evidence_answer(question, [source])

        assert "three months or more" in answer
        assert "3-6 months" in answer
        assert "Over 6 months" in answer
        assert used == [source]
