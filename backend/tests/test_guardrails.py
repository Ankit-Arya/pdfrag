from app.rag.guardrails import (
    grounding_failure_reason,
    validate_grounded_answer,
)
from app.rag.prompts import NO_ANSWER


def test_valid_source_citations_pass() -> None:
    answer, grounded = validate_grounded_answer(
        "The term is 30 days. [S1]",
        source_count=2,
    )
    assert grounded is True
    assert answer.endswith("[S1]")


def test_answer_without_citation_is_rejected_but_preserved() -> None:
    answer, grounded = validate_grounded_answer(
        "The term is 30 days.",
        source_count=2,
    )
    assert grounded is False
    assert answer == "The term is 30 days."
    assert grounding_failure_reason(
        answer,
        2,
    ) == "missing_citations"


def test_unknown_citation_is_rejected_but_preserved() -> None:
    answer, grounded = validate_grounded_answer(
        "The term is 30 days. [S9]",
        source_count=2,
    )
    assert grounded is False
    assert answer.endswith("[S9]")
    assert grounding_failure_reason(
        answer,
        2,
    ) == "citation_out_of_range"


def test_no_answer_response_is_not_marked_grounded() -> None:
    answer, grounded = validate_grounded_answer(
        NO_ANSWER,
        source_count=3,
    )
    assert grounded is False
    assert answer == NO_ANSWER
    assert grounding_failure_reason(
        answer,
        3,
    ) == "model_reported_no_answer"


def test_each_paragraph_must_be_cited() -> None:
    answer, grounded = validate_grounded_answer(
        "First supported point. [S1]\n\n"
        "Second unsupported point.",
        source_count=2,
    )
    assert grounded is False
    assert grounding_failure_reason(
        answer,
        2,
    ) == "uncited_claim_unit"


def test_each_bullet_must_be_cited() -> None:
    answer, grounded = validate_grounded_answer(
        "- Supported item. [S1]\n"
        "- Unsupported item.",
        source_count=2,
    )
    assert grounded is False
    assert grounding_failure_reason(
        answer,
        2,
    ) == "uncited_claim_unit"


def test_markdown_heading_does_not_need_citation() -> None:
    answer, grounded = validate_grounded_answer(
        "## FDS Operating Procedure\n\n"
        "The system powers on with AUX-ON. [S1]",
        source_count=1,
    )
    assert grounded is True


def test_bold_heading_does_not_need_citation() -> None:
    answer, grounded = validate_grounded_answer(
        "**FDS Operating Procedure**\n\n"
        "- Power the system with AUX-ON. [S1]\n"
        "- Inform OCC before reset. [S1]",
        source_count=1,
    )
    assert grounded is True
