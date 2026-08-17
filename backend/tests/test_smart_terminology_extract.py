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
