from types import SimpleNamespace

from app.rag.types import RetrievedChunk, TextChunk
from app.rag.v5.retrieval_completeness import (
    enrich_retrieval_summary,
    explicit_scope_labels,
    required_coverage_documents,
    response_diagnostics,
    scope_labels_from_text,
)
from app.rag.v5.synthesis_retrieval import SynthesisCoverage, select_results_for_answer


def _chunk(filename: str, chunk_id: str, role: str = "governing") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            filename=filename,
            page_number=10,
            page_end=10,
            text=f"Source evidence from {filename}",
            content_type="text",
            section_path=("Procedure",),
            heading="Procedure",
            document_id=chunk_id,
            chunk_index=1,
        ),
        score=0.8,
        method=f"v5.5-balanced-vector-0+v5.2-synthesis:{role}",
        vector_score=0.7,
        keyword_score=0.4,
    )


def test_source_defined_rs_code_and_named_line_are_scope_labels():
    labels = scope_labels_from_text(
        "SUBJECT: OPERATING AND TROUBLESHOOTING MANUAL, RS-CAF TRAINS (Airport Line)"
    )
    assert "RS-CAF" in labels
    assert "Airport Line" in labels


def test_multiple_rs_scopes_on_identity_page_are_preserved():
    labels = scope_labels_from_text("Operating Manual RS-1/RS-13 (ROTEM TRAINS)")
    assert "RS-1" in labels
    assert "RS-13" in labels


def test_explicit_scope_query_is_case_insensitive_for_source_codes_and_named_lines():
    labels = explicit_scope_labels("procedure for rs-caf on airport line only")
    assert "RS-CAF" in labels
    assert "Airport Line" in labels


def test_generic_main_line_phrase_is_not_promoted_to_named_scope():
    labels = scope_labels_from_text("General troubleshooting on Main Line")
    assert "Main Line" not in labels


def test_non_contributing_final_evidence_is_not_required_answer_scope():
    diagnostics = [
        {
            "filename": "rs3.pdf",
            "decision": "FINAL_EVIDENCE",
            "final_evidence": True,
            "contributing": False,
            "rerank_role": "governing",
            "scope_type": "rs",
            "scope_label": "RS-3",
            "scope_labels": ["RS-3"],
        },
        {
            "filename": "line7.pdf",
            "decision": "FINAL_EVIDENCE",
            "final_evidence": True,
            "contributing": False,
            "rerank_role": "supporting",
            "scope_type": "line",
            "scope_label": "Line-7",
            "scope_labels": ["Line-7"],
        },
    ]
    assert required_coverage_documents(diagnostics) == []
    summary = enrich_retrieval_summary({}, diagnostics)
    assert summary["required_answer_documents"] == []
    assert summary["required_scope_labels"] == []


def test_only_validated_contributing_governing_scope_becomes_required():
    diagnostics = [
        {
            "filename": "rs3.pdf",
            "decision": "FINAL_EVIDENCE",
            "final_evidence": True,
            "contributing": True,
            "rerank_role": "governing",
            "scope_type": "rs",
            "scope_label": "RS-3",
            "scope_labels": ["RS-3"],
        },
        {
            "filename": "rs10.pdf",
            "decision": "FINAL_EVIDENCE",
            "final_evidence": True,
            "contributing": True,
            "rerank_role": "supporting",
            "scope_type": "rs",
            "scope_label": "RS-10",
            "scope_labels": ["RS-10"],
        },
    ]
    assert required_coverage_documents(diagnostics) == ["rs3.pdf"]
    summary = enrich_retrieval_summary({}, diagnostics)
    assert summary["required_answer_documents"] == ["rs3.pdf"]
    assert summary["required_scope_labels"] == ["RS-3"]


def test_response_diagnostics_keeps_rejected_route_diagnostics_only():
    diagnostics = [
        {
            "filename": "accepted.pdf",
            "decision": "FINAL_EVIDENCE",
            "deep_searched": True,
            "final_evidence": True,
            "rerank_role": "governing",
            "scope_type": "rs",
            "scope_label": "RS-1",
            "scope_labels": ["RS-1"],
        },
        {
            "filename": "rejected.pdf",
            "decision": "FINAL_EVIDENCE",
            "deep_searched": True,
            "final_evidence": True,
            "rerank_role": "supporting",
            "scope_type": "line",
            "scope_label": "Line-5",
            "scope_labels": ["Line-5"],
        },
    ]
    out, summary = response_diagnostics(diagnostics, {}, ["accepted.pdf"])
    accepted = next(item for item in out if item["filename"] == "accepted.pdf")
    rejected = next(item for item in out if item["filename"] == "rejected.pdf")
    assert accepted["answer_eligible"] is True
    assert accepted["decision"] == "CONTRIBUTING_GOVERNING_EVIDENCE"
    assert rejected["answer_eligible"] is False
    assert rejected["decision"] == "REVIEWED_NON_CONTRIBUTOR"
    assert summary["required_scope_labels"] == ["RS-1"]


def test_successful_coverage_critic_filters_even_overoptimistic_reranker_route():
    accepted = _chunk("accepted.pdf", "00000000-0000-0000-0000-000000000001")
    rejected = _chunk("rejected.pdf", "00000000-0000-0000-0000-000000000002")
    review = SynthesisCoverage(
        sufficient=True,
        ai_used=True,
        answer_strategy="multi_document_synthesis",
        contributing_documents=("accepted.pdf",),
        evidence_coverage_status="complete",
    )
    selected = select_results_for_answer([accepted, rejected], review)
    assert [item.chunk.filename for item in selected] == ["accepted.pdf"]


def test_reranker_strong_role_is_fallback_when_coverage_critic_has_no_contributors():
    accepted = _chunk("accepted.pdf", "00000000-0000-0000-0000-000000000001")
    weak = _chunk("weak.pdf", "00000000-0000-0000-0000-000000000002", role="supporting")
    review = SynthesisCoverage(
        sufficient=False,
        ai_used=False,
        answer_strategy="multi_document_synthesis",
        contributing_documents=(),
        evidence_coverage_status="incomplete",
    )
    selected = select_results_for_answer([accepted, weak], review)
    assert [item.chunk.filename for item in selected] == ["accepted.pdf"]
