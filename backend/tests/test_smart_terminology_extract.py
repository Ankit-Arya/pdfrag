# ruff: noqa: E501
from app.rag.smart_index import extract_terminology


def _pairs(value: str) -> set[tuple[str, str]]:
    return {(alias, canonical) for alias, canonical, _kind, _confidence in extract_terminology(value)}


def test_parenthetical_long_form_uses_initialism_suffix() -> None:
    assert ("SC", "Station Controller") in _pairs(
        "As directed by the Station Controller (SC), the train shall be held."
    )


def test_table_abbreviation_definition_is_extracted() -> None:
    assert ("OCC", "Operations Control Centre") in _pairs(
        "| OCC | Operations Control Centre |"
    )


def test_unrelated_parenthetical_text_does_not_poison_alias_map() -> None:
    assert not extract_terminology("Some unrelated words around (SC) without a definition.")


def test_uppercase_to_is_kept_as_possible_organisation_abbreviation() -> None:
    from app.rag.terminology import candidate_aliases

    aliases = candidate_aliases("What should TO do if the train cannot proceed?")
    assert "TO" in aliases


def test_lowercase_to_remains_a_stopword() -> None:
    from app.rag.terminology import candidate_aliases

    aliases = candidate_aliases("What should the operator do to proceed?")
    assert "to" not in aliases


def test_bic_definition_is_extracted_from_manual_style_text() -> None:
    assert ("BIC", "Brake Isolating cock") in _pairs(
        "Brake Isolating cock(BIC)- Saloon (B05)"
    )


def test_bare_uppercase_acronym_is_definition_request() -> None:
    from app.rag.terminology import definition_request_aliases, is_definition_request

    assert is_definition_request("BIC")
    assert definition_request_aliases("BIC") == ["BIC"]


def test_full_form_wording_targets_the_acronym_only() -> None:
    from app.rag.terminology import definition_request_aliases, is_definition_request

    assert is_definition_request("BIC full form")
    assert definition_request_aliases("BIC full form") == ["BIC"]
    assert definition_request_aliases("full form of BIC") == ["BIC"]


def test_explicit_reference_request_is_not_definition_request() -> None:
    from app.rag.terminology import is_definition_request

    assert not is_definition_request("find BIC")
    assert not is_definition_request("show references to BIC")


def test_document_code_is_not_treated_as_acronym_definition() -> None:
    from app.rag.terminology import is_definition_request

    assert not is_definition_request("SC-06")


def test_bic_procedure_question_is_not_reclassified_as_definition() -> None:
    from app.rag.terminology import is_definition_request

    assert not is_definition_request("what is BIC procedure")


def test_what_is_bic_is_definition_request() -> None:
    from app.rag.terminology import is_definition_request

    assert is_definition_request("what is BIC")


def test_initialism_ignores_connective_words_in_long_form() -> None:
    assert ("TCMS", "Train Control and Monitoring System") in _pairs(
        "Train Control and Monitoring System (TCMS)"
    )
