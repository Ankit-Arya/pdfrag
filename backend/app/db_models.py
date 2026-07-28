from __future__ import annotations
import enum, uuid
from datetime import UTC, datetime
from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.db import Base
from app.config import get_settings

def now() -> datetime: return datetime.now(UTC)
class UserRole(str, enum.Enum): admin="admin"; user="user"
class DocumentStatus(str, enum.Enum): uploaded="uploaded"; processing="processing"; ready="ready"; failed="failed"

class User(Base):
    __tablename__="users"
    id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str]=mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str]=mapped_column(String(255))
    role: Mapped[UserRole]=mapped_column(Enum(UserRole), default=UserRole.user, index=True)
    is_active: Mapped[bool]=mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)

class UserSession(Base):
    __tablename__="user_sessions"
    id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    refresh_token_hash: Mapped[str]=mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime|None]=mapped_column(DateTime(timezone=True), nullable=True)
    user_agent: Mapped[str|None]=mapped_column(String(500), nullable=True)
    ip_address: Mapped[str|None]=mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)
    user: Mapped[User]=relationship()

class Document(Base):
    __tablename__="documents"
    id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str]=mapped_column(String(500))
    mime_type: Mapped[str]=mapped_column(String(100), default="application/pdf")
    size_bytes: Mapped[int]=mapped_column(Integer)
    content: Mapped[bytes]=mapped_column(LargeBinary)
    status: Mapped[DocumentStatus]=mapped_column(Enum(DocumentStatus), default=DocumentStatus.uploaded, index=True)
    page_count: Mapped[int]=mapped_column(Integer, default=0)
    chunk_count: Mapped[int]=mapped_column(Integer, default=0)
    warnings: Mapped[list]=mapped_column(JSONB, default=list)
    error: Mapped[str|None]=mapped_column(Text, nullable=True)
    uploaded_by: Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)
    processed_at: Mapped[datetime|None]=mapped_column(DateTime(timezone=True), nullable=True)

class DocumentChunk(Base):
    __tablename__="document_chunks"
    __table_args__=(UniqueConstraint("document_id","chunk_index"), Index("ix_document_chunks_document_page","document_id","page_number"))
    id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID]=mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int]=mapped_column(Integer)
    page_number: Mapped[int]=mapped_column(Integer)
    content_type: Mapped[str]=mapped_column(String(30), default="text")
    text: Mapped[str]=mapped_column(Text)
    embedding: Mapped[list[float]]=mapped_column(Vector(get_settings().embedding_dimensions))
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)

class ChatSession(Base):
    __tablename__="chat_sessions"
    id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str]=mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)

class ChatMessage(Base):
    __tablename__="chat_messages"
    id: Mapped[uuid.UUID]=mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_session_id: Mapped[uuid.UUID]=mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str]=mapped_column(String(20))
    content: Mapped[str]=mapped_column(Text)
    message_metadata: Mapped[dict]=mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=now)
