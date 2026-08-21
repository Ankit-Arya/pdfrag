from __future__ import annotations

from app.rag.progress import emit_progress, progress_context


def test_structured_progress_is_backward_compatible_and_safe() -> None:
    events: list[dict[str, object]] = []
    with progress_context(lambda event: events.append(dict(event))):
        emit_progress("search_round_1", "Searching", "Corpus search", current=1, total=3, actor="search", status="running", operation_id="round-1")
        emit_progress(
            "search_round_1",
            "Search complete",
            "Candidates collected",
            current=1,
            total=3,
            actor="search",
            status="complete",
            operation_id="round-1",
            metrics={"candidates": 42},
            document="MRGR.pdf",
            page=99,
            heading="Responsibilities of Station Controller",
        )

    assert len(events) == 2
    assert events[0]["actor"] == "search"
    assert events[0]["status"] == "running"
    assert events[1]["status"] == "complete"
    assert events[1]["metrics"] == {"candidates": 42}
    assert events[1]["document"] == "MRGR.pdf"
    assert events[1]["page"] == 99
    assert isinstance(events[1]["duration_ms"], int)
    assert isinstance(events[1]["total_elapsed_ms"], int)
