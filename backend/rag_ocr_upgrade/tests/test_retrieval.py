import numpy as np

from app.models import FileSummary
from app.rag.store import CollectionStore
from app.rag.types import QueryPlan, TextChunk


def test_fuzzy_keyword_search_recovers_misspelling() -> None:
    chunks = [
        TextChunk("a", "report.pdf", 1, "Quarterly revenue increased to 125 million."),
        TextChunk("b", "report.pdf", 2, "Employee leave policy and holiday schedule."),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store = CollectionStore()
    collection = store.create(
        chunks,
        vectors,
        [FileSummary(name="report.pdf", pages=2, chunks=2)],
        total_pages=2,
    )
    plan = QueryPlan(
        original_question="What is the reveneu?",
        rewritten_question="What is the reveneu?",
        search_queries=["reveneu"],
    )

    results = collection.keyword_search(plan, top_k=2)

    assert results
    assert results[0].chunk.chunk_id == "a"
    assert results[0].keyword_score > 0.5


def test_hybrid_search_can_use_multiple_semantic_queries() -> None:
    chunks = [
        TextChunk("a", "report.pdf", 1, "Revenue for Q2 was 125 million."),
        TextChunk("b", "report.pdf", 2, "The office opens at nine."),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store = CollectionStore()
    collection = store.create(
        chunks,
        vectors,
        [FileSummary(name="report.pdf", pages=2, chunks=2)],
        total_pages=2,
    )
    plan = QueryPlan(
        original_question="q2 reveneu?",
        rewritten_question="What was Q2 revenue?",
        search_queries=["q2 reveneu", "Q2 revenue"],
        keywords=["Q2", "revenue"],
    )
    query_vectors = np.asarray([[0.9, 0.1], [1.0, 0.0]], dtype="float32")

    results = collection.hybrid_search(plan, query_vectors, top_k=2)

    assert results[0].chunk.chunk_id == "a"
    assert "keyword" in results[0].method
    assert "vector" in results[0].method
