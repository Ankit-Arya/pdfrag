from app.rag.v5.retrieval_completeness import (
    _extract_alias_expansions,
    _fuzzy_known_expansions,
    aliases_in_question,
    explicit_scope_labels,
    explicit_scope_only,
    is_conditional_procedure_query,
    scope_from_text,
)


def test_scope_parser_handles_common_rs_filename_forms():
    assert scope_from_text("2. RS 3 OTM 2.0 searchable.pdf") == ("rs", "RS-3")
    assert scope_from_text("RS-10 Operating Manual.pdf") == ("rs", "RS-10")
    assert scope_from_text("Procedure for UTO mode in Line-7.pdf") == ("line", "Line-7")


def test_explicit_scope_labels_and_only_scope():
    assert explicit_scope_labels("Compare RS3 and RS-10 on Line 7") == ["RS-3", "RS-10", "Line-7"]
    assert explicit_scope_only("Only RS-3: what should be done if brakes fail?") is True
    assert explicit_scope_only("RS-3: what should be done if brakes fail?") is False


def test_conditional_procedure_detection():
    assert is_conditional_procedure_query("What will happen if 50% brakes are isolated?") is True
    assert is_conditional_procedure_query("What happens if a door fails to close?") is True
    assert is_conditional_procedure_query("What to do if a door fails to close?") is True
    assert is_conditional_procedure_query("What is BIC?") is False


def test_clean_case_degraded_definition_recovery():
    assert "Brake Isolation Cock" in _extract_alias_expansions(
        "Bic (Brake Isolation Cock)", "BIC"
    )


def test_table_flattened_known_definition_recovery():
    flattened = """Bic Inside Saloon EP-BCU to brake cylinder 3
    (Brake Drain beside DRL2 Pipe. It isolates all brakes except parking Brake.
    Red Isolation Cock) each bogie."""
    recovered = _fuzzy_known_expansions(
        flattened,
        "BIC",
        ["Brake Isolation Cock", "Battery Isolation Contactor"],
        context="Cut-Out Cocks Inside cab/saloon VCB COCK",
    )
    assert "Brake Isolation Cock" in recovered
    assert "Battery Isolation Contactor" not in recovered


def test_operational_usage_is_not_invented_as_definition():
    recovered = _fuzzy_known_expansions(
        "Inform OCC and isolate BIC of the affected bogie.",
        "BIC",
        ["Brake Isolation Cock"],
        context="General Brake Faults",
    )
    assert recovered == []


def test_lowercase_short_definition_alias_is_detected():
    assert aliases_in_question("What is bic?") == ["BIC"]
    assert aliases_in_question("What does bic stand for?") == ["BIC"]
    assert aliases_in_question("What is door isolation?") == []
