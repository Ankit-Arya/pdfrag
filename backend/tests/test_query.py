from app.rag.query import QueryPlanner, _extract_context_terms, _validate_plan


def test_domain_acronym_is_not_expanded_from_general_knowledge() -> None:
    plan = _validate_plan(
        "FDS operating procedure",
        {
            "rewritten_question": (
                "Fire Dynamics Simulator operating procedure"
            ),
            "search_queries": [
                "FDS operating procedure",
                "Fire Dynamics Simulator operating procedure",
                "FDS operating procedure document",
                "Fire Dynamics Simulator user manual",
            ],
            "keywords": [
                "FDS",
                "Fire Dynamics Simulator",
                "procedure",
            ],
        },
        max_variants=4,
    )

    assert plan.rewritten_question == "FDS operating procedure"
    assert (
        "Fire Dynamics Simulator operating procedure"
        not in plan.search_queries
    )
    assert (
        "Fire Dynamics Simulator user manual"
        not in plan.search_queries
    )
    assert plan.search_queries[0] == "FDS operating procedure"
    assert "FDS operating steps instructions checks" in plan.search_queries
    assert "FDS operating procedure document" in plan.search_queries
    assert "FDS" in plan.keywords
    assert plan.intent == "procedure"
    assert "FDS" in plan.context_terms


def test_non_acronym_question_can_be_rewritten() -> None:
    plan = _validate_plan(
        "door reset procedure",
        {
            "rewritten_question": (
                "procedure to reset a train door"
            ),
            "search_queries": [
                "train door reset procedure",
            ],
            "keywords": [
                "door",
                "reset",
            ],
        },
        max_variants=4,
    )

    assert (
        plan.rewritten_question
        == "procedure to reset a train door"
    )
    assert plan.search_queries[0] == "door reset procedure"
    assert "train door reset procedure" in plan.search_queries
    assert plan.intent == "procedure"


def test_model_cannot_invent_context_terms() -> None:
    plan = _validate_plan(
        "door reset procedure",
        {
            "rewritten_question": "door reset procedure",
            "search_queries": ["door reset procedure"],
            "keywords": ["door", "reset"],
            "intent": "procedure",
            "focus_terms": ["door", "reset", "procedure"],
            "context_terms": ["Type B rolling stock"],
        },
        max_variants=4,
    )

    assert plan.context_terms == []


def test_context_terms_are_extracted_without_ai_rewrite() -> None:
    assert _extract_context_terms("Reset doors for rolling stock Type A on Line 2") == [
        "Type A",
        "Line 2",
        "2",
    ]


def test_process_query_searches_for_associated_test_evidence() -> None:
    plan = QueryPlanner().plan("wake up process", enabled=False)

    assert plan.intent == "procedure"
    assert "wake up test" in plan.search_queries
    assert any("examination" in query for query in plan.search_queries)
    assert plan.response_mode == "evidence"


def test_specific_question_uses_concise_response_mode() -> None:
    plan = QueryPlanner().plan("When do we operate the train at 15 kmph?", enabled=False)

    assert plan.response_mode == "concise"
