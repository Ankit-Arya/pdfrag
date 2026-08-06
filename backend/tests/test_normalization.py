from app.rag.normalization import canonical_phrase, number_word_variant, search_terms


def test_search_terms_normalize_plural_and_number_words() -> None:
    assert {"month", "months"}.issubset(search_terms("months"))
    assert {"3", "three"}.issubset(search_terms("three", keep_single=True))
    assert {"3", "three"}.issubset(search_terms("3", keep_single=True))


def test_phrase_and_embedding_variants_preserve_readable_order() -> None:
    assert canonical_phrase("Three months") == "3 months"
    assert number_word_variant("3 month absence") == "three month absence"
