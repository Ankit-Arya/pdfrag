from app.rag.service import RagService
from app.rag.types import QueryPlan, RetrievedChunk, TextChunk


def result(
    chunk_id: str,
    filename: str,
    page: int,
    text: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            filename=filename,
            page_number=page,
            text=text,
        ),
        score=score,
        method="vector",
    )


def test_acronym_query_removes_unrelated_manual_sections() -> None:
    plan = QueryPlan(
        original_question="FDS operating procedures",
        rewritten_question="FDS operating procedures",
        search_queries=["FDS operating procedures"],
        keywords=["FDS"],
    )

    retrieved = [
        result(
            "fds-main",
            "manual.pdf",
            3,
            "1.1.5 FDS OPERATING PROCEDURE. Powering ON/OFF and alarm reset.",
            0.62,
        ),
        result(
            "cover",
            "manual.pdf",
            8,
            "PART A OPERATING MANUAL",
            0.49,
        ),
        result(
            "fds-continuation",
            "manual.pdf",
            3,
            "Continue the operating procedure by informing OCC, muting, verifying, and resetting.",
            0.48,
        ),
        result(
            "door",
            "manual.pdf",
            46,
            "Door isolation procedure and degraded movement instructions.",
            0.46,
        ),
    ]

    selected = RagService._select_context_chunks(plan, retrieved)
    selected_ids = [item.chunk.chunk_id for item in selected]

    assert "fds-main" in selected_ids
    assert "fds-continuation" in selected_ids
    assert "cover" not in selected_ids
    assert "door" not in selected_ids


def test_short_request_limits_context_size() -> None:
    plan = QueryPlan(
        original_question="door reset procedure",
        rewritten_question="door reset procedure",
        search_queries=["door reset procedure"],
        keywords=["door", "reset"],
    )
    retrieved = [
        result(
            str(index),
            "manual.pdf",
            index,
            "door reset procedure",
            0.9 - index * 0.02,
        )
        for index in range(8)
    ]

    selected = RagService._select_context_chunks(plan, retrieved)
    assert len(selected) <= 4
