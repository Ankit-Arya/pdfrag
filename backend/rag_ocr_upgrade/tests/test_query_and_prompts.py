from app.rag.prompts import build_user_prompt
from app.rag.query import _parse_json_object, _validate_plan
from app.rag.types import RetrievedChunk, TextChunk


def test_query_rewrite_json_parser_accepts_fenced_json() -> None:
    parsed = _parse_json_object(
        '```json\n{"rewritten_question":"What is revenue?",'
        '"search_queries":["revenue","sales income"],"keywords":["revenue"]}\n```'
    )
    plan = _validate_plan("wat is reveneu?", parsed, 4)

    assert plan.rewritten_question == "What is revenue?"
    assert plan.search_queries[0] == "wat is reveneu?"
    assert "revenue" in plan.keywords


def test_prompt_returns_exact_included_excerpt() -> None:
    chunk = TextChunk("a", "file.pdf", 3, "A" * 500, content_type="text")
    result = RetrievedChunk(chunk=chunk, score=0.8, method="vector", vector_score=0.8)

    _, sources = build_user_prompt("Question?", "Question?", [result], max_context_chars=180)

    assert sources
    assert len(sources[0].excerpt) < len(chunk.text)
