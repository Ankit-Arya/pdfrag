from app.rag.pdf import (
    _choose_best_native_text,
    _choose_text,
    _native_corruption_score,
    _needs_ocr,
    _ocr_orientations,
    _ocr_recovers_missing_content,
    _page_rotation_ratio,
    _text_quality,
)
from app.rag.table_quality import is_plausible_table


def test_readable_native_page_does_not_require_ocr() -> None:
    text = (
        "Track work in non-traffic hours\n"
        "No maintenance staff shall enter a running line without permission from "
        "the Traffic Controller. The permission, location and completion time shall "
        "be recorded before normal service resumes."
    )

    assert _text_quality(text) >= 0.62
    assert not _needs_ocr(text, min_chars=80, min_quality=0.62)


def test_sparse_or_fragmented_native_page_requests_ocr() -> None:
    assert _needs_ocr("T r a c k  w o r k", min_chars=80, min_quality=0.62)


def test_disagreement_between_native_extractors_requests_ocr() -> None:
    readable = (
        "All track work shall be performed under the permission and supervision "
        "specified in the applicable operating rule."
    )

    assert _needs_ocr(
        readable,
        min_chars=80,
        min_quality=0.62,
        consensus=0.2,
        min_consensus=0.42,
    )


def test_rotated_text_is_detected_from_layout_metadata() -> None:
    class PlumberPage:
        chars = [
            {"text": "normal text", "upright": True},
            {"text": "rotated text content", "upright": False},
        ]

    class FitzPage:
        @staticmethod
        def get_text(_: str) -> dict:
            return {
                "blocks": [
                    {
                        "lines": [
                            {
                                "dir": (0.0, -1.0),
                                "spans": [{"text": "rotated text content"}],
                            }
                        ]
                    }
                ]
            }

    assert _page_rotation_ratio(PlumberPage(), FitzPage()) > 0.08
    assert _ocr_orientations(True) == (0, 90, 180, 270)
    assert _ocr_orientations(False) == (0,)


def test_rotation_forces_ocr_result_even_when_native_text_is_nonempty() -> None:
    corrupted_native = "LLN ILN form marks and repeated corrupted native text"
    readable_ocr = "A Train Operator absent for three months requires road learning trips."

    assert _choose_text(corrupted_native, readable_ocr, force_ocr=True) == readable_ocr


def test_repeated_glyph_fragments_are_detected_as_corrupted_native_text() -> None:
    corrupted = """
    Date:
    Name LS/ALS:
    Signature of Nominated LS/ALS
    met tt tT | tT | | ET TT TT TT
    Hw TL | TE cE | rT | | | TT
    Pt et tT tt tT rT TT TT TT TT
    Pt tt tt tT tT tT | tT | dT TT
    Pt tT dT dEcd T | rT rE rT TT TT
    """

    assert _native_corruption_score(corrupted) >= 0.12


def test_normal_short_acronyms_do_not_trigger_corruption_detection() -> None:
    readable = """
    Wake-up test examination
    (a) cab signalling;
    (b) safety brake circuits including brake system;
    (c) train radio communication;
    (d) CCTV of the train and CCTV communication link;
    The train may operate in RM, ATP, ATO, DTO or UTO mode as instructed by OCC.
    """

    assert _native_corruption_score(readable) == 0.0


def test_completeness_ocr_detects_visible_content_missing_from_native_layer() -> None:
    native = (
        "Performance factors and overall health and alertness. "
        "Name and employee number of Train Operator."
    )
    ocr = (
        "A Train Operator who has not worked on a section for three months or more "
        "shall be given road learning trips to refresh his knowledge. Duration of "
        "absence: 3-6 months, one round trip; over 6 months, three round trips."
    )

    assert _ocr_recovers_missing_content(
        native,
        ocr,
        min_novel_terms=10,
        novelty_threshold=0.20,
    )


def test_completeness_ocr_ignores_minor_ocr_wording_differences() -> None:
    native = (
        "Crew Controller shall check the serviceability of the breath analyzer "
        "before the first sign on and record its functionality."
    )
    ocr = (
        "Crew Controller shall check serviceability of the breath analyser before "
        "first sign on and record the functionality."
    )

    assert not _ocr_recovers_missing_content(
        native,
        ocr,
        min_novel_terms=10,
        novelty_threshold=0.20,
    )


def test_best_native_candidate_is_selected_per_page() -> None:
    readable = (
        "All authorised persons whose duties require them to go on the tracks shall "
        "be properly trained and shall wear appropriate protective clothing."
    )
    corrupted = "A \x01 B \x02 C \x03 " * 50

    assert _choose_best_native_text([corrupted, readable]) == readable
    assert _choose_text(readable, corrupted, force_ocr=False) == readable


def test_fragmented_running_text_is_not_accepted_as_a_table() -> None:
    false_rows = [
        ["THE G", "AZETTE", "OF INDIA", ": EXTRAORDI", "NARY", "PART II"],
        ["running tracks", "shall b", "e inspecte", "d as per the", "schedule sp", "ecified"],
        [
            "maintenance st",
            "aff requir",
            "ing to carr",
            "y out inspecti",
            "on or repair",
            "of equipment",
        ],
    ]
    valid_rows = [
        ["Condition", "Speed"],
        ["Train operated in RM mode", "15 kmph"],
        ["Normal operation", "As authorised"],
    ]

    assert not is_plausible_table(false_rows)
    assert is_plausible_table(valid_rows)
