from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageText:
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float
    method: str = "vector"
