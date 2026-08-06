from app.rag.relevance import select_context_chunks
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

    selected = select_context_chunks(plan, retrieved)
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

    selected = select_context_chunks(plan, retrieved)
    assert len(selected) <= 4


def test_intent_reranking_rejects_irrelevant_high_vector_match() -> None:
    plan = QueryPlan(
        original_question="What is the annual leave policy?",
        rewritten_question="What is the annual leave policy?",
        search_queries=["annual leave policy"],
        intent="definition",
        focus_terms=["annual", "leave", "policy"],
    )
    retrieved = [
        result(
            "generic-high-vector",
            "handbook.pdf",
            1,
            "General company introduction, purpose, and values.",
            0.91,
        ),
        result(
            "answerable",
            "handbook.pdf",
            14,
            "Annual leave policy means eligible employees receive 20 days of leave.",
            0.56,
        ),
    ]

    selected = select_context_chunks(plan, retrieved)

    assert [item.chunk.chunk_id for item in selected] == ["answerable"]


def test_context_constraint_prevents_cross_procedure_mixing() -> None:
    plan = QueryPlan(
        original_question="FDS reset procedure",
        rewritten_question="FDS reset procedure",
        search_queries=["FDS reset procedure"],
        keywords=["FDS"],
        intent="procedure",
        focus_terms=["FDS", "reset", "procedure"],
        context_terms=["FDS"],
    )
    retrieved = [
        result(
            "fds",
            "manual.pdf",
            10,
            "FDS reset procedure: acknowledge the alarm and verify the indication.",
            0.61,
        ),
        result(
            "doors",
            "manual.pdf",
            80,
            "Passenger door reset procedure: isolate the affected doorway.",
            0.74,
        ),
    ]

    selected = select_context_chunks(plan, retrieved)

    assert [item.chunk.chunk_id for item in selected] == ["fds"]


def test_multiword_context_constraint_requires_the_complete_context() -> None:
    plan = QueryPlan(
        original_question="How do I reset doors for rolling stock Type A?",
        rewritten_question="How do I reset doors for rolling stock Type A?",
        search_queries=["door reset rolling stock Type A"],
        intent="procedure",
        focus_terms=["reset", "doors", "rolling", "stock", "Type", "A"],
        context_terms=["rolling stock Type A"],
    )
    retrieved = [
        result(
            "wrong-type",
            "manual.pdf",
            20,
            "ROLLING STOCK TYPE B - Door reset procedure.",
            0.82,
        ),
        result(
            "right-type",
            "manual.pdf",
            30,
            "ROLLING STOCK TYPE A - Door reset procedure.",
            0.59,
        ),
    ]

    selected = select_context_chunks(plan, retrieved)

    assert [item.chunk.chunk_id for item in selected] == ["right-type"]


def test_short_query_keeps_evidence_from_every_relevant_document() -> None:
    plan = QueryPlan(
        original_question="wake up test",
        rewritten_question="wake up test",
        search_queries=["wake up test"],
        intent="fact_lookup",
        focus_terms=["wake", "up", "test"],
    )
    retrieved = [
        result(
            f"primary-{index}",
            "frequent-source.pdf",
            index,
            "Wake up test requirements and conditions.",
            0.94 - index * 0.01,
        )
        for index in range(6)
    ]
    retrieved.extend(
        result(
            f"document-{index}",
            f"document-{index}.pdf",
            index,
            "Wake-up test examination and safety devices.",
            0.70 - index * 0.01,
        )
        for index in range(1, 6)
    )

    selected = select_context_chunks(plan, retrieved, max_chunks=12)
    filenames = {item.chunk.filename for item in selected}

    assert filenames == {
        "frequent-source.pdf",
        "document-1.pdf",
        "document-2.pdf",
        "document-3.pdf",
        "document-4.pdf",
        "document-5.pdf",
    }


def test_process_query_keeps_wake_up_test_but_rejects_process_only_noise() -> None:
    plan = QueryPlan(
        original_question="wake up process",
        rewritten_question="wake up process",
        search_queries=["wake up process", "wake up test"],
        intent="procedure",
        focus_terms=["wake", "up", "process"],
    )
    retrieved = [
        result(
            "examination",
            "gazette.pdf",
            128,
            "Wake-up test examination shall ensure train safety devices are working.",
            0.71,
        ),
        result(
            "noise",
            "unrelated.pdf",
            4,
            "The approval process for station signage is described here.",
            0.86,
        ),
    ]

    selected = select_context_chunks(plan, retrieved, max_chunks=12)

    assert [item.chunk.chunk_id for item in selected] == ["examination"]
