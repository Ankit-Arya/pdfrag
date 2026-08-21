from app.rag.v5.assistant_retrieval import _extract_expansions, _text_affinity


def test_grounded_acronym_expansion_patterns():
    assert "Station Controller" in _extract_expansions("Station Controller (SC) shall remain available.", "SC")
    assert "Station Controller" in _extract_expansions("SC (Station Controller) is responsible for the station.", "SC")


def test_heading_affinity_rewards_semantic_heading_variant_when_query_variant_contains_it():
    score = _text_affinity(
        "39 Responsibilities of Station Controller",
        ["duties of station controller", "responsibilities of station controller"],
        ["station controller", "responsibilities"],
    )
    assert score > 0.55


def test_heading_affinity_rejects_incidental_unrelated_heading():
    good = _text_affinity(
        "Hand Signals",
        ["hand signal provision", "hand signals"],
        ["hand signals"],
    )
    bad = _text_affinity(
        "Engineer possession",
        ["hand signal provision", "hand signals"],
        ["hand signals"],
    )
    assert good > bad
