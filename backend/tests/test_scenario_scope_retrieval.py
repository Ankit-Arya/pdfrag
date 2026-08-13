from app.rag.postgres_store import _direct_line_alias_target, _named_line_aliases
from app.rag.query import QueryPlanner
from app.rag.relevance import (
    filter_hard_context_candidates,
    rank_scenario_documents,
    select_context_chunks,
)
from app.rag.types import RetrievedChunk, TextChunk


def _chunk(
    chunk_id: str,
    document_id: str,
    filename: str,
    chunk_index: int,
    text: str,
    score: float = 0.82,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            filename=filename,
            page_number=6,
            text=text,
            document_id=document_id,
            chunk_index=chunk_index,
        ),
        score=score,
        method="vector+corpus-fts",
        vector_score=score,
        keyword_score=0.7,
    )


def _line1_plan():
    return QueryPlanner().plan(
        "In Line 1 if train stops after passing VCB open board but before NSCZ board "
        "and both VCBs are in open condition, what should TO do to bring the train to next station",
        enabled=False,
    )


def _scenario_candidates() -> list[RetrievedChunk]:
    return [
        _chunk(
            "sc30-10",
            "doc-sc30",
            "30. SC-30 Rev-09 Procedure Order for Negotiating neutral section by train.pdf",
            10,
            "Section path: 6 PROCEDURE TO BE FOLLOWED > 6.2 Train Stops in neutral section\n"
            "A. In Line - 1, 2, 3 & 4:-\n"
            "Whenever a train stops, TO shall inform TC/OCC stating Train No., location and status of all VCBs.",
        ),
        _chunk(
            "sc30-11",
            "doc-sc30",
            "30. SC-30 Rev-09 Procedure Order for Negotiating neutral section by train.pdf",
            11,
            "Section path: 6 PROCEDURE TO BE FOLLOWED > 6.2 Train Stops in neutral section\n"
            "(ii) Train stops at or after the VCB Open Board but before NSCZ. "
            "Move the 6 Car Train with Rear Pantograph up or 8 Car train with Rear two Pantographs up.",
            0.94,
        ),
        _chunk(
            "sc30-12",
            "doc-sc30",
            "30. SC-30 Rev-09 Procedure Order for Negotiating neutral section by train.pdf",
            12,
            "Section path: 6 PROCEDURE TO BE FOLLOWED > 6.2 Train Stops in neutral section\n"
            "At NSCZ deselect rear pantograph(s), select the first two Front Pantographs and move the Train "
            "to next Station NSP. If VCB opens before NSP, close VCB and move up to NSP.",
            0.91,
        ),
        _chunk(
            "bhs-1",
            "doc-bhs",
            "Airport Line BHS Procedure.pdf",
            30,
            "If a train undershoots or overshoots, BHS operator informs TO on TETRA and TO informs OCC. "
            "Realign the train to the stopping window at the station.",
            0.89,
        ),
    ]


def test_line_scope_is_inherited_across_continuation_chunks() -> None:
    plan = _line1_plan()
    candidates = _scenario_candidates()
    selected = select_context_chunks(plan, candidates, max_chunks=20)
    ids = {item.chunk.chunk_id for item in selected}

    assert "sc30-11" in ids
    assert "sc30-12" in ids


def test_scenario_body_router_promotes_sc30() -> None:
    routes = rank_scenario_documents(_line1_plan(), _scenario_candidates(), max_documents=3)
    assert routes
    assert routes[0].document_id == "doc-sc30"
    assert routes[0].reason == "scenario-body"


def test_hard_context_fallback_rejects_unrelated_bhs_procedure() -> None:
    safe = filter_hard_context_candidates(_line1_plan(), _scenario_candidates())
    assert safe
    assert {item.chunk.document_id for item in safe} == {"doc-sc30"}


def test_preferred_wrong_line_cannot_bypass_mandatory_scope() -> None:
    plan = _line1_plan()
    candidates = _scenario_candidates() + [
        _chunk(
            "line7-1",
            "doc-line7",
            "Line 7 VCB procedure.pdf",
            20,
            "Section path: 6 Procedure\nIn Line 7, VCB opens before NSCZ and TO moves the train to the station.",
            0.97,
        )
    ]
    selected = select_context_chunks(
        plan,
        candidates,
        max_chunks=20,
        preferred_document_ids={"doc-sc30", "doc-line7"},
    )
    assert "line7-1" not in {item.chunk.chunk_id for item in selected}


def test_named_line_mapping_requires_explicit_corpus_mapping() -> None:
    assert _named_line_aliases("In Red Line if train stops") == ["Red Line"]
    assert (
        _direct_line_alias_target(
            "Red Line (Line 1) operating instructions",
            "Red Line",
        )
        == "Line 1"
    )
    assert _direct_line_alias_target("Red Line operating instructions", "Red Line") is None
