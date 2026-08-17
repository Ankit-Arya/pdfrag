# ruff: noqa: E501
import sys
import types
from dataclasses import dataclass

# smart_retrieval imports the production RetrievedChunk/TextChunk classes. This
# isolated patch test supplies the minimal interface needed by pure query-shape
# functions without copying the application's real types.py into the hotfix ZIP.
if "app.rag.types" not in sys.modules:
    stub = types.ModuleType("app.rag.types")

    @dataclass
    class TextChunk:
        chunk_id: str
        filename: str
        page_number: int
        text: str
        content_type: str
        document_id: str | None = None
        chunk_index: int | None = None

    @dataclass
    class RetrievedChunk:
        chunk: TextChunk
        score: float
        method: str
        vector_score: float = 0.0
        keyword_score: float = 0.0

    stub.TextChunk = TextChunk
    stub.RetrievedChunk = RetrievedChunk
    sys.modules["app.rag.types"] = stub

from app.rag.smart_retrieval import (
    is_amendment_anchor_text,
    retrieval_answerable,
    structured_lookup_request,
)
from app.rag.types import RetrievedChunk, TextChunk


def _result(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id="c1",
            filename="claims.pdf",
            page_number=9,
            text=text,
            content_type="table",
            document_id="d1",
            chunk_index=9,
        ),
        score=0.9,
        method="test",
        keyword_score=0.9,
    )


def test_compensation_by_different_injuries_is_structured_lookup() -> None:
    assert structured_lookup_request(
        "what is compensation amount in case of different injuries and accidents"
    )


def test_single_eye_compensation_is_not_forced_to_list_intent() -> None:
    assert not structured_lookup_request(
        "how much compensation is to be given for loss of one eye"
    )


def test_related_accident_prose_is_not_monetary_answerability() -> None:
    assert not retrieval_answerable(
        "what is compensation amount for different injuries",
        [_result("The controller shall record injuries and report the accident immediately.")],
    )


def test_compensation_schedule_row_is_answerable() -> None:
    assert retrieval_answerable(
        "how much compensation for loss of one eye",
        [_result("Amount of Compensation: For loss of one eye without complications - 3,20,000")],
    )


def test_second_schedule_substitution_is_authority_anchor() -> None:
    assert is_amendment_anchor_text(
        "For the Second Schedule of the principal rules, the following Schedule shall be substituted, namely:-"
    )


def test_old_schedule_row_is_not_an_authority_anchor_by_itself() -> None:
    assert not is_amendment_anchor_text(
        "For loss of one eye without complications the other being normal. 1,60,000"
    )
