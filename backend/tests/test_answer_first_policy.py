import json

from app.rag.query import QueryPlanner, deterministic_search_mode
from app.rag.service import _use_primary_document_backbone
from app.rag.synthesis import list_answer_needs_repair
from app.rag.types import QueryPlan


def _plan(question: str) -> QueryPlan:
    return QueryPlanner().plan(question, enabled=False)


def test_signal_types_fragment_is_synthesized_list() -> None:
    plan = _plan("types of signals for train operation")

    assert plan.intent == "list"
    assert plan.search_mode == "answer"
    assert plan.response_mode == "concise"
    assert "signals for train operation" in plan.search_queries
    assert any("classification" in query for query in plan.search_queries)



def test_ai_rewrite_cannot_flip_signal_types_into_reference_mode(monkeypatch) -> None:
    payload = {
        "contextual_question": "signals train operation",
        "search_queries": ["signals train operation"],
        "keywords": ["signals", "train"],
        "intent": "fact_lookup",
        "focus_terms": ["signals", "train"],
        "context_terms": [],
        "search_mode": "references",
    }
    monkeypatch.setattr(
        "app.rag.query.llm_service.plan",
        lambda *args, **kwargs: json.dumps(payload),
    )

    plan = QueryPlanner().plan("types of signals for train operation", enabled=True)

    assert plan.intent == "list"
    assert plan.search_mode == "answer"


def test_signal_types_without_question_mark_is_still_answer_mode() -> None:
    assert (
        deterministic_search_mode(
            "types of signals for train operation",
            "types of signals for train operation",
            "list",
        )
        == "answer"
    )


def test_multiword_informational_fragment_defaults_to_answer() -> None:
    plan = _plan("signals for train operation")
    assert plan.search_mode == "answer"


def test_bare_concept_and_explicit_navigation_stay_reference_mode() -> None:
    assert _plan("alcohol").search_mode == "references"
    assert _plan("kit bag").search_mode == "references"
    assert _plan("SC-06").search_mode == "references"
    assert _plan("find mentions of SC-06").search_mode == "references"


def test_existing_natural_list_request_stays_answer_mode() -> None:
    plan = _plan("items to be kept in kit bag")
    assert plan.intent == "list"
    assert plan.search_mode == "answer"


def test_verbose_list_or_evidence_dump_requires_compaction() -> None:
    plan = _plan("types of signals for train operation")
    evidence_dump = "## Information found in the documents\n\n### x.pdf - page 1\n- A [S1]"
    assert list_answer_needs_repair(plan, evidence_dump)


def test_small_complete_list_does_not_require_compaction() -> None:
    plan = _plan("types of signals for train operation")
    answer = "\n".join(
        [
            "- Cab signals. [S1]",
            "- Fixed signals and equipment. [S1]",
            "- Hand signals. [S1]",
            "- Virtual signals. [S1]",
        ]
    )
    assert not list_answer_needs_repair(plan, answer)

def test_general_signal_taxonomy_does_not_force_title_primary_backbone() -> None:
    plan = _plan("types of signals for train operation")
    assert plan.intent == "list"
    assert not _use_primary_document_backbone(plan)


def test_procedure_and_explicit_document_code_allow_primary_backbone() -> None:
    procedure_plan = _plan("what to do if someone obstruct train movement")
    assert procedure_plan.intent == "procedure"
    assert _use_primary_document_backbone(procedure_plan)

    explicit_doc_plan = _plan("SC-06 high wind speed")
    assert _use_primary_document_backbone(explicit_doc_plan)

