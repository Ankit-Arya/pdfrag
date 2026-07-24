from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PageText:
    filename: str
    page_number: int
    text: str
    content_type: str = "text"
    table_index: int | None = None


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    filename: str
    page_number: int
    text: str
    content_type: str = "text"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float
    method: str = "vector"
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass(frozen=True, slots=True)
class PromptSource:
    result: RetrievedChunk
    excerpt: str


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_question: str
    rewritten_question: str
    search_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    used_ai_rewrite: bool = False
