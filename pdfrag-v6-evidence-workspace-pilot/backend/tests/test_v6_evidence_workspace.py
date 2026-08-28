from types import SimpleNamespace

from app.rag.types import RetrievedChunk, TextChunk
from app.rag.v6 import evidence_workspace as ew


def _interpretation(intent: str = "fact_lookup"):
    return SimpleNamespace(
        intent=intent,
        resolved_question="What should be the permitted speed if 25% brakes are isolated?",
        evidence_needs=("permitted speed", "applicable condition"),
        concepts=("brakes", "isolated", "speed"),
        scope={},
        ambiguity_note="",
    )


def _result(chunk_id: str, filename: str, score: float, page: int = 1, content_type: str = "text"):
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            filename=filename,
            page_number=page,
            page_end=page,
            text=f"Evidence from {filename} on page {page}",
            content_type=content_type,
            section_path=("Section",),
            heading="Section",
            document_id="00000000-0000-0000-0000-000000000001" if filename == "A.pdf" else "00000000-0000-0000-0000-000000000002",
            chunk_index=page,
        ),
        score=score,
        method="v5.2-synthesis:governing",
        vector_score=score,
        keyword_score=score,
    )


def test_query_frame_preserves_quantity_basis_and_explicit_scope():
    frame = ew.build_query_frame(
        "If 50% brakes are isolated in RS-3 train what is its speed?",
        _interpretation(),
    )
    assert "50%" in frame["quantities_and_units"]
    assert "RS-3" in frame["explicit_scopes"]


def test_workspace_source_selection_is_document_balanced():
    results = [
        _result("a1", "A.pdf", 0.99, 1),
        _result("a2", "A.pdf", 0.98, 2),
        _result("a3", "A.pdf", 0.97, 3),
        _result("b1", "B.pdf", 0.80, 4),
    ]
    sources = ew.select_workspace_sources(results, limit=3)
    names = [source.result.chunk.filename for source in sources]
    assert names[:2] == ["A.pdf", "B.pdf"]


def test_workspace_compiler_drops_invented_source_ids(monkeypatch):
    sources = [
        ew.PromptSource(result=_result("a1", "A.pdf", 0.9), excerpt="Applicable governing rule"),
        ew.PromptSource(result=_result("b1", "B.pdf", 0.8), excerpt="Unrelated rescue rule"),
    ]
    response = """{
      "refined_query_frame": {"answer_type":"direct_fact","subject":"brakes","requested_attributes":["speed"],"scenario":"normal operation","conditions":["25 percent isolated"],"explicit_scopes":[],"quantities_and_units":["25%"],"material_ambiguity":""},
      "answer_mode":"direct_fact",
      "coverage":"complete",
      "confidence":0.95,
      "answer_source_ids":["S1","S99"],
      "claims":[{"claim":"Supported rule","scope":"","scenario":"normal","condition":"25%","action_or_value":"value","importance":"primary","source_ids":["S1","S99"]}],
      "rejected_sources":[{"source_id":"S2","reason":"scenario_mismatch","detail":"rescue only"}],
      "ambiguities":[],"conflicts":[],"missing_facets":[],"retry_queries":[],"negative_claim_safe":false,
      "answer_outline":["Give the supported value first"]
    }"""
    monkeypatch.setattr(ew.llm_service, "generate", lambda *args, **kwargs: response)
    workspace = ew.compile_evidence_workspace(
        question="25% brakes isolated speed",
        interpretation=_interpretation(),
        query_frame=ew.build_query_frame("25% brakes isolated speed", _interpretation()),
        sources=sources,
        search_round=1,
    )
    assert workspace["answer_source_ids"] == ["S1"]
    assert workspace["claims"][0]["source_ids"] == ["S1"]
    assert workspace["rejected_sources"][0]["source_id"] == "S2"


def test_partial_workspace_can_request_targeted_retry():
    workspace = {
        "coverage": "partial",
        "retry_queries": ["brake isolation percentage speed", "BIC isolated permitted speed"],
    }
    assert ew.workspace_retry_queries(workspace) == [
        "brake isolation percentage speed",
        "BIC isolated permitted speed",
    ]
    workspace["coverage"] = "complete"
    assert ew.workspace_retry_queries(workspace) == []


def test_writer_prompt_contains_only_answer_eligible_sources():
    sources = [
        ew.PromptSource(result=_result("a1", "A.pdf", 0.9), excerpt="Applicable rule"),
        ew.PromptSource(result=_result("b1", "B.pdf", 0.8), excerpt="Wrong scenario"),
    ]
    workspace = {
        "answer_source_ids": ["S1"],
        "refined_query_frame": {},
        "answer_mode": "direct_fact",
        "coverage": "complete",
        "confidence": 0.9,
        "claims": [{"claim": "Applicable rule", "source_ids": ["S1"]}],
        "ambiguities": [],
        "conflicts": [],
        "missing_facets": [],
        "negative_claim_safe": False,
        "answer_outline": ["Answer first"],
    }
    prompt = ew.premium_answer_prompt(question="Question", workspace=workspace, sources=sources)
    assert "A.pdf" in prompt
    assert "B.pdf" not in prompt
    assert "REQUIRED RS/LINE" not in prompt
    assert "Required answer scopes" not in prompt


def test_pre_expansion_gate_rejects_wrong_scenario_seed(monkeypatch):
    from app.rag.v5 import synthesis_retrieval as sr

    first = _result("a1", "A.pdf", 0.95)
    second = _result("b1", "B.pdf", 0.90)
    monkeypatch.setattr(
        sr.llm_service,
        "generate",
        lambda *args, **kwargs: '{"keep_ids":["a1"],"reject_ids":["b1"],"reason":"B is a different scenario"}',
    )
    kept = sr._v6_applicable_structure_seeds(
        "ordinary fault handling",
        _interpretation("procedure"),
        [first, second],
    )
    assert [item.chunk.chunk_id for item in kept] == ["a1"]
