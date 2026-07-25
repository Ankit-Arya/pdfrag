from app.rag.query import _validate_plan


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
    assert plan.search_queries == [
        "FDS operating procedure",
        "FDS operating procedure document",
    ]
    assert "FDS" in plan.keywords


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
