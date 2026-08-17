# ruff: noqa: E501
import sys
import types
from dataclasses import dataclass

if "app.rag.types" not in sys.modules:
    stub = types.ModuleType("app.rag.types")

    @dataclass
    class TextChunk:
        chunk_id: str
        filename: str
        page_number: int
        text: str
        content_type: str
        document_id: str | None = None
        chunk_index: int | None = None

    @dataclass
    class RetrievedChunk:
        chunk: TextChunk
        score: float
        method: str
        vector_score: float = 0.0
        keyword_score: float = 0.0

    stub.TextChunk = TextChunk
    stub.RetrievedChunk = RetrievedChunk
    sys.modules["app.rag.types"] = stub

from app.rag.authority import (
    extract_authority_directives,
    looks_like_subdocument_boundary,
    replacement_span_end,
)
from app.rag.smart_retrieval import retrieval_answerable, structured_lookup_request, value_lookup_request
from app.rag.terminology import definition_for_token, definition_request_aliases
from app.rag.types import RetrievedChunk, TextChunk


def _result(
    text: str,
    *,
    chunk_id: str,
    index: int,
    method: str = "test",
    vector: float = 0.0,
    content_type: str = "text",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            filename="manual.pdf",
            page_number=max(1, index),
            text=text,
            content_type=content_type,
            document_id="d1",
            chunk_index=index,
        ),
        score=0.8,
        method=method,
        vector_score=vector,
        keyword_score=0.0,
    )


def test_multi_acronym_definition_request_extracts_all_targets() -> None:
    assert definition_request_aliases("What is BIC and TCMS?") == ["BIC", "TCMS"]
    assert definition_request_aliases("What are ATP, ATO and ATS?") == ["ATP", "ATO", "ATS"]
    assert definition_request_aliases("BIC/TCMS") == ["BIC", "TCMS"]


def test_multi_acronym_procedure_question_is_not_definition_request() -> None:
    assert definition_request_aliases("What is BIC and TCMS procedure during brake failure?") == []
    assert definition_request_aliases("What should TO do if BIC is isolated?") == []


def test_definition_parser_preserves_exact_corpus_long_form() -> None:
    assert definition_for_token("Brake Isolating Cock (BIC) - saloon", "BIC") == "Brake Isolating Cock"
    assert definition_for_token("Train Control and Monitoring System (TCMS)", "TCMS") == "Train Control and Monitoring System"


def test_false_parenthetical_usage_is_not_accepted_as_definition() -> None:
    assert definition_for_token("Some unrelated words near the device (BIC) during a test", "BIC") == ""


def test_value_lookup_is_general_not_compensation_specific() -> None:
    assert value_lookup_request("What is the maximum brake pressure?")
    assert value_lookup_request("At what speed can the train proceed?")
    assert value_lookup_request("How much compensation is payable?")
    assert structured_lookup_request("speed limits for different operating modes")
    assert not structured_lookup_request("types of train signals")


def test_answerability_can_bridge_split_table_header_and_value_row() -> None:
    header = _result(
        "Maximum Brake Pressure (bar)",
        chunk_id="h",
        index=10,
        method="smart-structured-value-neighbor",
    )
    row = _result(
        "Emergency brake 5.0",
        chunk_id="r",
        index=11,
        method="smart-structured-value",
        vector=0.61,
        content_type="table",
    )
    assert retrieval_answerable("What is maximum emergency brake pressure?", [header, row])


def test_answerability_does_not_require_colloquial_words_to_exist_in_same_row() -> None:
    header = _result(
        "Amount of Compensation (in Rs.)",
        chunk_id="h",
        index=20,
        method="smart-structured-value-neighbor",
    )
    row = _result(
        "Fracture of Major Bone - Femur, Tibia of one limb 80,000",
        chunk_id="r",
        index=21,
        method="smart-structured-value",
        vector=0.58,
        content_type="table",
    )
    assert retrieval_answerable("what is compensation in case of broken leg", [header, row])


def test_word_substitution_directive_is_extracted_generically() -> None:
    directives = extract_authority_directives(
        'In rule 18, for the words "five lakh", the words "eight lakh" shall be substituted.',
        chunk_index=4,
        page_number=8,
    )
    assert any(
        item.directive_type == "replace_words"
        and item.target.casefold() == "rule 18"
        and item.old_text.casefold() == "five lakh"
        and item.new_text.casefold() == "eight lakh"
        for item in directives
    )


def test_whole_section_substitution_directive_is_extracted_generically() -> None:
    directives = extract_authority_directives(
        "For the Second Schedule of the principal rules, the following Schedule shall be substituted, namely:-",
        chunk_index=10,
        page_number=8,
    )
    assert any(item.directive_type == "replace_section" and "second schedule" in item.target.casefold() for item in directives)


def test_replacement_span_stops_before_appended_older_instrument() -> None:
    @dataclass
    class Row:
        chunk_index: int
        page_number: int
        text: str

    rows = [
        Row(10, 8, "Amendment Rules, 2025. Second Schedule shall be substituted."),
        Row(11, 9, "Replacement table rows"),
        Row(12, 10, "More replacement table rows"),
        Row(13, 10, "Note: The principal rules were published in 2017."),
        Row(14, 11, "The Metro Railways Rules, 2017"),
    ]
    assert replacement_span_end(rows, anchor_chunk_index=10, anchor_year=2025) == 12
    assert looks_like_subdocument_boundary(rows[3].text, anchor_year=2025)


def test_ocr_noisy_substitution_still_keeps_each_rule_and_value_separate() -> None:
    text = '''
    (a) in sub-rule (2), for the words "four lakh", the words "eight lakh" shall be substituted;
    (b) in sub-rule (3), for the words "eighty thousand", the words .'one lakh sixty thousand,, shall be substituted.
    (2) ln rule l8 of the principal rules, for the words "five lakh", the words "eight lakh" shall be substituted.
    '''
    directives = extract_authority_directives(text, chunk_index=4, page_number=8)
    pairs = {(item.target.casefold(), item.old_text.casefold(), item.new_text.casefold()) for item in directives if item.directive_type == "replace_words"}
    assert ("rule 18", "five lakh", "eight lakh") in pairs
    assert any(old == "eighty thousand" and new == "one lakh sixty thousand" for _target, old, new in pairs)
    assert not any("five lakh" in old and "eighty thousand" in old for _target, old, _new in pairs)


def test_single_case_value_lookup_stays_single_fact() -> None:
    assert value_lookup_request("what is compensation in case of a broken leg")
    assert not structured_lookup_request("what is compensation in case of a broken leg")


def test_replacement_span_keeps_table_tail_when_boundary_note_shares_chunk() -> None:
    @dataclass
    class Row:
        chunk_index: int
        page_number: int
        text: str

    rows = [
        Row(10, 8, "Amendment Rules, 2025. Second Schedule shall be substituted."),
        Row(11, 9, "Replacement table rows"),
        Row(12, 10, "32 Fracture of pelvis 80,000. 33 Fracture of major bone 80,000. " * 8 + " Note: The principal rules were published in 2017."),
        Row(13, 11, "The Metro Railways Rules, 2017"),
    ]
    assert replacement_span_end(rows, anchor_chunk_index=10, anchor_year=2025) == 12
