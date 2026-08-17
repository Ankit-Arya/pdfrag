# ruff: noqa: E501
from app.rag.scenario_reasoning import (
    compile_scenario,
    extract_numeric_rules,
    logical_match_score,
    scenario_relevance_question,
    scenario_relaxed_query,
)


def test_four_brakes_matches_two_or_more_rule() -> None:
    scenario = compile_scenario("Four brakes have failed. What should SC do?")
    source = "If two or more brakes are failed, the Station Controller shall follow the degraded-mode procedure."
    boost, notes = logical_match_score(scenario, source)
    assert boost > 0
    assert notes
    assert "4 >= 2" in notes[0]


def test_one_brake_does_not_match_two_or_more_rule() -> None:
    scenario = compile_scenario("One brake has failed. What should be done?")
    source = "If two or more brakes are failed, the train shall be dealt with under the degraded-mode procedure."
    boost, notes = logical_match_score(scenario, source)
    assert boost < 0
    assert notes
    assert "false" in notes[0]


def test_abbreviation_hint_is_added_to_canonical_scenario() -> None:
    scenario = compile_scenario(
        "What should SC do if the train cannot move?",
        ["SC = Station Controller — organisation terminology index, 8 supporting definition(s)"],
    )
    assert "SC means Station Controller" in scenario.canonical_question
    assert scenario.inferred_states["train_movement"] == "unable_to_proceed"


def test_relaxed_query_drops_exact_user_number_but_keeps_concept() -> None:
    scenario = compile_scenario("4 brakes are isolated and the train cannot move")
    relaxed = scenario_relaxed_query(scenario).casefold()
    assert "4" not in relaxed
    assert "brake" in relaxed
    assert "isolate" in relaxed


def test_rule_extractor_handles_at_least_and_more_than() -> None:
    rules = extract_numeric_rules(
        "At least 2 brakes are isolated before this restriction applies. Speed is above 25 km/h only under test conditions."
    )
    assert any(rule.operator == ">=" and rule.threshold == 2 for rule in rules)
    assert any(rule.operator == ">" and rule.threshold == 25 for rule in rules)


def test_indicator_states_boost_matching_procedure_and_penalize_conflict() -> None:
    scenario = compile_scenario("VCB is open and line voltage is available. Train cannot move. What should I do?")
    matching = "If VCB remains open while line voltage is available, carry out the following checks."
    conflicting = "If VCB is closed and line voltage is unavailable, use the alternate procedure."
    match_boost, _ = logical_match_score(scenario, matching)
    conflict_boost, _ = logical_match_score(scenario, conflicting)
    assert match_boost > conflict_boost
    assert match_boost > 0
    assert conflict_boost < 0


def test_failed_brakes_do_not_satisfy_isolated_brakes_threshold() -> None:
    scenario = compile_scenario("Four brakes have failed. What should be done?")
    source = "If two or more brakes are isolated, the isolation restriction shall apply."
    boost, notes = logical_match_score(scenario, source)
    assert boost <= 0
    assert not any("is true" in note for note in notes)


def test_failure_of_two_or_more_brakes_rule_is_extracted() -> None:
    rules = extract_numeric_rules("In case of failure of two or more brakes, the following restriction shall apply.")
    assert any(rule.operator == ">=" and rule.threshold == 2 and "fail" in rule.tokens for rule in rules)


def test_not_working_is_normalized_to_failed() -> None:
    scenario = compile_scenario("Four brakes are not working. What should I do?")
    source = "If two or more brakes have failed, follow the degraded-mode procedure."
    boost, notes = logical_match_score(scenario, source)
    assert boost > 0
    assert any("4 >= 2" in note for note in notes)


def test_ambiguous_terminology_hint_is_not_forced_into_canonical_scenario() -> None:
    scenario = compile_scenario(
        "What should SC do?",
        [
            "SC = Station Controller — organisation terminology index, 4 supporting definition(s) (ambiguous corpus meaning; use context)",
            "SC = Section Controller — organisation terminology index, 3 supporting definition(s) (ambiguous corpus meaning; use context)",
        ],
    )
    assert "SC means Station Controller" not in scenario.canonical_question
    assert "SC means Section Controller" not in scenario.canonical_question


def test_failed_brakes_do_not_match_failed_doors_threshold() -> None:
    scenario = compile_scenario("Four brakes have failed. What should I do?")
    source = "If two or more doors have failed, isolate the affected doors."
    boost, notes = logical_match_score(scenario, source)
    assert boost <= 0
    assert not any("is true" in note for note in notes)


def test_relevance_question_removes_user_threshold_value_but_keeps_line_scope() -> None:
    scenario = compile_scenario("On Line 1, 4 brakes have failed. What should be done?")
    relevance = scenario_relevance_question(scenario).casefold()
    assert " 4 " not in f" {relevance} "
    assert "line 1" in relevance
    assert "brakes" in relevance


def test_relevance_question_removes_measurement_value_for_threshold_matching() -> None:
    scenario = compile_scenario("Train speed is 28 km/h. What restriction applies?")
    relevance = scenario_relevance_question(scenario).casefold()
    assert "28" not in relevance
    assert "speed" in relevance
    assert "km/h" in relevance


def test_line_number_is_not_treated_as_failed_train_count() -> None:
    scenario = compile_scenario("On Line 1 the train has failed and cannot move. What should be done?")
    assert not any(fact.value == 1 and "train" in fact.tokens for fact in scenario.numeric_facts)
