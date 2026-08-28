from types import SimpleNamespace

from app.rag.v5.layout import _plumber_first_row_is_header
from app.rag.v5.retrieval_completeness import (
    completeness_policy,
    is_conditional_procedure_query,
    is_entity_enumeration_query,
)


def interpretation(intent: str):
    return SimpleNamespace(intent=intent)


def test_membership_list_is_entity_enumeration_not_scope_fanout():
    ctx = interpretation("list")
    assert completeness_policy("members of DMT", ctx) == "entity_enumeration"
    assert is_entity_enumeration_query("members of DMT", ctx) is True
    assert is_conditional_procedure_query("members of DMT", ctx) is False


def test_generic_types_components_and_duties_are_entity_enumeration():
    assert completeness_policy("What are the types of signals?", interpretation("list")) == "entity_enumeration"
    assert completeness_policy("List components of the system", interpretation("list")) == "entity_enumeration"
    assert completeness_policy("duties of controller", interpretation("list")) == "entity_enumeration"


def test_conditional_semantics_override_generic_list_intent():
    ctx = interpretation("list")
    assert completeness_policy("List steps to follow if a door fails to close", ctx) == "cross_scope_procedure"
    assert completeness_policy("What will happen if equipment fails?", ctx) == "cross_scope_procedure"


def test_procedure_intent_gets_cross_scope_coverage_without_hardcoded_subjects():
    assert completeness_policy("isolation procedure", interpretation("procedure")) == "cross_scope_procedure"
    assert completeness_policy("how to recover after failure", interpretation("troubleshooting")) == "cross_scope_procedure"


def test_definition_request_keeps_definition_enumeration():
    assert completeness_policy("What is BIC?", interpretation("definition")) == "definition_enumeration"
    assert completeness_policy("full form of ABC", interpretation("definition")) == "definition_enumeration"


def test_operational_explicit_scope_comparison_gets_cross_scope_coverage():
    assert completeness_policy("Compare RS3 and RS-10 procedure", interpretation("comparison")) == "cross_scope_procedure"


def test_non_operational_comparison_does_not_force_rs_line_fanout():
    assert completeness_policy("Compare the definitions of two terms", interpretation("comparison")) == "direct_lookup"


def test_headerless_numbered_table_preserves_first_data_row():
    rows = [
        ["1", "Chairman/DMT", "HOD/Operations"],
        ["2", "Member", "Concerned DY.HOD/Operations"],
        ["3", "Member", "Concerned DY.HOD/RS"],
    ]
    assert _plumber_first_row_is_header(rows) is False


def test_normal_numbered_table_header_is_still_header():
    rows = [
        ["No.", "Designation", "Department"],
        ["1", "Chairman", "Operations"],
        ["2", "Member", "Rolling Stock"],
    ]
    assert _plumber_first_row_is_header(rows) is True


def test_uncertain_unstructured_first_row_keeps_legacy_header_behavior():
    rows = [
        ["Description", "Value"],
        ["Mode", "Normal"],
        ["State", "Ready"],
    ]
    assert _plumber_first_row_is_header(rows) is True


def test_legacy_headerless_first_row_can_be_recovered_from_source_metadata():
    from app.rag.types import RetrievedChunk, TextChunk
    from app.rag.v5.retrieval_completeness import _headerless_first_row_recovery

    item = RetrievedChunk(
        chunk=TextChunk(
            chunk_id="chunk-2",
            filename="manual.pdf",
            page_number=10,
            page_end=10,
            content_type="table_row",
            section_path=("Section",),
            heading="Section",
            document_id="00000000-0000-0000-0000-000000000001",
            chunk_index=11,
            text=(
                "[PDF STRUCTURE]\n"
                "File: manual.pdf\n"
                "Pages: 10\n"
                "Section path: Section\n"
                "Content type: table_row\n"
                "Heading: Section\n"
                "Columns: 1 | Chairman/DMT | HOD/Operations\n"
                "[/PDF STRUCTURE]\n\n"
                "Table row 1: 1: 2 | Chairman/DMT: Member | HOD/Operations: Dy.HOD/Operations"
            ),
        ),
        score=0.7,
        method="v5.2-synthesis:governing",
        vector_score=0.4,
        keyword_score=0.5,
    )
    recovered = _headerless_first_row_recovery(item)
    assert recovered is not None
    assert "1 | Chairman/DMT | HOD/Operations" in recovered.chunk.text
    assert "v5.4-headerless-first-row" in recovered.method
