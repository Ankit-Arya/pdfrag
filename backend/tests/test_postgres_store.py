from app.rag.postgres_store import (
    _document_diverse_results,
    _keyword_or_query,
    _phrase_match,
    _section_phrase_match,
    _terms,
)
from app.rag.types import RetrievedChunk, TextChunk


def _result(chunk_id: str, document_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            filename=f"{document_id}.pdf",
            page_number=1,
            text="wake-up test",
            document_id=document_id,
        ),
        score=score,
    )


def test_phrase_match_normalizes_hyphens() -> None:
    assert _phrase_match("wake up test", "The automatic wake-up test is required.") == 1.0
    assert {"wake", "up", "test"}.issubset(_terms("wake-up test"))


def test_candidate_cap_keeps_one_result_per_document_first() -> None:
    results = [
        _result("a-1", "a", 0.95),
        _result("a-2", "a", 0.94),
        _result("a-3", "a", 0.93),
        _result("b-1", "b", 0.70),
        _result("c-1", "c", 0.65),
    ]

    selected = _document_diverse_results(results, limit=3)

    assert [item.chunk.chunk_id for item in selected] == ["a-1", "b-1", "c-1"]


def test_section_phrase_match_reads_only_structural_context() -> None:
    chapter = (
        "[PDF CHUNK CONTEXT]\n"
        "Section path: ELECTRICAL SAFETY AND CONTROL > General\n"
        "[/PDF CHUNK CONTEXT]\n\n"
        "All equipment shall be inspected."
    )
    incidental = "An accident can affect electrical safety and control equipment."

    assert _section_phrase_match("electrical safety and control", chapter) == 1.0
    assert _section_phrase_match("electrical safety and control", incidental) == 0.0


def test_keyword_candidate_query_uses_normalized_or_prefixes() -> None:
    query = _keyword_or_query("3 month absence")

    assert "3:*" in query
    assert "three:*" in query
    assert "month:*" in query
    assert "absence:*" in query
