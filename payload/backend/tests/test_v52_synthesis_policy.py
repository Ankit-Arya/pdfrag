from app.rag.smart_understanding import SmartInterpretation
from app.rag.v5.synthesis_retrieval import build_synthesis_plan


def interpretation(*, intent: str, needs: tuple[str, ...], act: str = "question") -> SmartInterpretation:
    return SmartInterpretation(
        raw_question="x",
        resolved_question="x",
        intent=intent,
        conversation_act=act,
        evidence_needs=needs,
        search_queries=("x",),
    )


def test_definition_stays_direct() -> None:
    plan = build_synthesis_plan(
        "What is ATP?",
        interpretation(intent="definition", needs=("definition",)),
    )
    assert plan.answer_strategy == "direct_lookup"


def test_duties_are_synthesis() -> None:
    plan = build_synthesis_plan(
        "What are the duties of Station Controller?",
        interpretation(
            intent="list",
            needs=("core duties", "operational responsibilities", "exceptions"),
        ),
    )
    assert plan.answer_strategy == "multi_document_synthesis"
    assert plan.search_scope == "broad_relevant_corpus"


def test_when_must_speed_apply_is_synthesis_even_if_fact_lookup() -> None:
    plan = build_synthesis_plan(
        "When must a speed of 25 km/h be followed?",
        interpretation(intent="fact_lookup", needs=("triggering condition", "applicability")),
    )
    assert plan.answer_strategy == "multi_document_synthesis"


def test_navigation_is_direct() -> None:
    plan = build_synthesis_plan(
        "On which page is Rule 39?",
        interpretation(intent="fact_lookup", needs=("page",), act="navigation"),
    )
    assert plan.answer_strategy == "direct_lookup"
