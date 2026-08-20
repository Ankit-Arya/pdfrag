from app.rag.v5.retrieval import _authority_adjust, _explicit_year


def test_current_authority_is_default_even_without_ai_flag() -> None:
    assert _authority_adjust(0.5, "current_replacement", authority_sensitive=False, explicit_year=None) > 0.5
    assert _authority_adjust(0.5, "historical_appended", authority_sensitive=False, explicit_year=None) < 0.5


def test_amendment_year_mention_does_not_disable_current_precedence() -> None:
    assert _explicit_year("claims rules 2017 as amended in 2025") is None
    assert _explicit_year("current 2025 amendment compensation") is None


def test_explicit_historical_request_can_access_old_state() -> None:
    assert _explicit_year("what was the compensation in 2017 before the amendment") == 2017
