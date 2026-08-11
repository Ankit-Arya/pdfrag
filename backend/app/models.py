import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    chat_session_id: uuid.UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    rewrite_question: bool | None = None

    @field_validator("question")
    @classmethod
    def normalize(cls, value: str) -> str:
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
    content_type: str = "text"
    retrieval_method: str = "vector"


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    grounded: bool
    grounding_status: str = "insufficient_evidence"
    interpreted_question: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    request_id: str | None = None
    chat_session_id: uuid.UUID | None = None
    question_created_at: datetime | None = None
    response_created_at: datetime | None = None


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_ready: bool
    embedding_backend: str | None = None
    embedding_fallback: bool = False
    embedding_error: str | None = None
    llm_model: str
    ocr_mode: str
    ocr_available: bool
    table_extraction: bool
    table_extraction_available: bool
    query_rewrite: bool


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    size_bytes: int
    page_count: int
    chunk_count: int
    warnings: list[Any] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime


class DocumentBatchRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)

    @field_validator("document_ids")
    @classmethod
    def deduplicate_document_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        return list(dict.fromkeys(value))


class DocumentBatchResponse(BaseModel):
    queued_document_ids: list[uuid.UUID] = Field(default_factory=list)
    already_processing: int = 0
    missing: int = 0


class KnowledgeStatus(BaseModel):
    ready_documents: int
    total_chunks: int


class AdminUserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    role: Literal["admin", "user"] = "user"


class AdminUserStatusUpdate(BaseModel):
    is_active: bool


class AdminUserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime


class ChatSessionOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChatDetail(ChatSessionOut):
    messages: list[ChatMessageOut]


class AuditLogOut(BaseModel):
    id: uuid.UUID
    event_type: str
    success: bool
    user_id: uuid.UUID | None = None
    actor_email: str | None = None
    chat_session_id: uuid.UUID | None = None
    question: str | None = None
    response: str | None = None
    error_message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    created_at: datetime
