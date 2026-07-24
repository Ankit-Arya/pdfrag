from app.rag.guardrails import validate_grounded_answer
from app.rag.prompts import NO_ANSWER


def test_valid_source_citations_pass() -> None:
    answer, grounded = validate_grounded_answer("The term is 30 days. [S1]", source_count=2)
    assert grounded is True
    assert answer.endswith("[S1]")


def test_answer_without_citation_is_rejected() -> None:
    answer, grounded = validate_grounded_answer("The term is 30 days.", source_count=2)
    assert grounded is False
    assert answer == NO_ANSWER


def test_unknown_citation_is_rejected() -> None:
    answer, grounded = validate_grounded_answer("The term is 30 days. [S9]", source_count=2)
    assert grounded is False
    assert answer == NO_ANSWER


def test_no_answer_response_is_not_marked_grounded() -> None:
    answer, grounded = validate_grounded_answer(NO_ANSWER, source_count=3)
    assert grounded is False
    assert answer == NO_ANSWER


def test_each_paragraph_must_be_cited() -> None:
    answer, grounded = validate_grounded_answer(
        "First supported point. [S1]\n\nSecond unsupported point.",
        source_count=2,
    )
    assert grounded is False
    assert answer == NO_ANSWER


def test_each_bullet_must_be_cited() -> None:
    answer, grounded = validate_grounded_answer(
        "- Supported item. [S1]\n- Unsupported item.",
        source_count=2,
    )
    assert grounded is False
    assert answer == NO_ANSWER
