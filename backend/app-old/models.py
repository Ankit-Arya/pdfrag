from pydantic import BaseModel, Field, field_validator


class FileSummary(BaseModel):
    name: str
    pages: int
    chunks: int


class CollectionResponse(BaseModel):
    collection_id: str
    files: list[FileSummary]
    total_pages: int
    total_chunks: int
    expires_in_minutes: int


class CollectionInfo(CollectionResponse):
    created_at: str
    last_accessed_at: str


class QuestionRequest(BaseModel):
    collection_id: str = Field(min_length=8, max_length=128)
    question: str = Field(min_length=2, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Question cannot be empty")
        return normalized


class SourceResult(BaseModel):
    id: str
    filename: str
    page: int
    score: float
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    grounded: bool
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
