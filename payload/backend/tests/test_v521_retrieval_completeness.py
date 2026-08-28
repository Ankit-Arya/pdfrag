from app.rag.v5.retrieval_completeness import (
    _extract_alias_expansions,
    aliases_in_question,
)


def test_case_degraded_parenthetical_alias_is_recovered():
    assert "Brake Isolation Cock" in _extract_alias_expansions(
        "Bic (Brake Isolation Cock)", "BIC"
    )


def test_glossary_style_alias_is_recovered():
    assert "Battery Isolation Contactor" in _extract_alias_expansions(
        "BIC Battery Isolation Contactor BLCB Brake Loop Circuit Breaker", "BIC"
    )


def test_bogie_isolating_cock_is_distinct_supported_meaning():
    assert "Bogie Isolating Cock" in _extract_alias_expansions(
        "BIC (Bogie Isolating Cock)", "BIC"
    )


def test_unrelated_operational_usage_is_not_a_definition():
    assert _extract_alias_expansions(
        "Isolate BIC of the affected bogie and proceed as instructed.", "BIC"
    ) == []


def test_initials_mismatch_is_rejected():
    assert _extract_alias_expansions("Bic (Brake Control Unit)", "BIC") == []


def test_lowercase_alias_is_accepted_only_when_explicit_full_form_request():
    assert aliases_in_question("what is full form of bic") == ["BIC"]
    assert aliases_in_question("what is wake up process") == []
