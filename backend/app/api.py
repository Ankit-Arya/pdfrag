import uuid
from datetime import UTC, datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import admin_user, current_user
from app.auth.security import hash_password
from app.config import get_settings
from app.db import get_db
from app.db_models import (
    ChatMessage,
    ChatSession,
    Document,
    DocumentStatus,
    User,
    UserRole,
)
from app.models import (
    AdminUserCreate,
    AdminUserOut,
    AdminUserStatusUpdate,
    AnswerResponse,
    ChatDetail,
    ChatMessageOut,
    ChatSessionOut,
    DocumentOut,
    HealthResponse,
    KnowledgeStatus,
    QuestionRequest,
)
from app.rag.embeddings import EmbeddingUnavailableError, embedding_service
from app.rag.llm import LlmConfigurationError
from app.rag.pdf import ocr_available, safe_filename, table_extraction_available
from app.rag.service import rag_service

router = APIRouter(prefix="/api")


def _document_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        status=document.status.value,
        size_bytes=document.size_bytes,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        warnings=document.warnings,
        error=document.error,
        created_at=document.created_at,
    )


def _document_download_headers(document: Document) -> dict[str, str]:
    fallback = document.filename.replace("\\", "_").replace('"', "") or "document.pdf"
    return {
        "Content-Disposition": (
            f"attachment; filename=\"{fallback}\"; "
            f"filename*=UTF-8''{quote(document.filename)}"
        ),
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
    }


def _admin_user_out(user: User) -> AdminUserOut:
    return AdminUserOut(
        id=user.id,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        created_at=user.created_at,
    )


def _chat_session_out(chat: ChatSession) -> ChatSessionOut:
    return ChatSessionOut(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok" if embedding_service.ready else "degraded",
        embedding_model=settings.embedding_model,
        embedding_ready=embedding_service.ready,
        embedding_backend=embedding_service.backend,
        embedding_fallback=embedding_service.using_fallback,
        embedding_error=embedding_service.last_error,
        llm_model=settings.llm_model,
        ocr_mode=settings.ocr_mode,
        ocr_available=ocr_available(),
        table_extraction=settings.extract_tables,
        table_extraction_available=table_extraction_available(),
        query_rewrite=settings.query_rewrite_enabled,
    )


@router.get("/knowledge/status", response_model=KnowledgeStatus)
def knowledge_status(
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> KnowledgeStatus:
    documents = list(
        db.scalars(select(Document).where(Document.status == DocumentStatus.ready))
    )
    return KnowledgeStatus(
        ready_documents=len(documents),
        total_chunks=sum(document.chunk_count for document in documents),
    )


@router.get("/documents", response_model=list[DocumentOut])
def shared_documents(
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> list[DocumentOut]:
    rows = db.scalars(select(Document).order_by(Document.created_at.desc()))
    return [_document_out(document) for document in rows]


@router.get("/documents/{document_id}/download")
def download_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Response:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    return Response(
        content=document.content,
        media_type=document.mime_type or "application/pdf",
        headers=_document_download_headers(document),
    )


@router.post("/admin/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    process: bool = True,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_user),
) -> DocumentOut:
    settings = get_settings()
    name = safe_filename(file.filename)
    data = await file.read(settings.max_file_size_bytes + 1)
    await file.close()

    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are accepted")
    if len(data) > settings.max_file_size_bytes:
        raise HTTPException(413, "File too large")

    document = Document(
        filename=name,
        size_bytes=len(data),
        content=data,
        uploaded_by=admin.id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    if process:
        try:
            await run_in_threadpool(rag_service.process_document, db, document)
        except EmbeddingUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    db.refresh(document)
    return _document_out(document)


@router.get("/admin/documents", response_model=list[DocumentOut])
def documents(
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> list[DocumentOut]:
    rows = db.scalars(select(Document).order_by(Document.created_at.desc()))
    return [_document_out(document) for document in rows]


@router.post("/admin/documents/{document_id}/process", response_model=DocumentOut)
async def process_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> DocumentOut:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    try:
        await run_in_threadpool(rag_service.process_document, db, document)
    except EmbeddingUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    db.refresh(document)
    return _document_out(document)


@router.delete(
    "/admin/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> None:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    db.delete(document)
    db.commit()


@router.get("/admin/users", response_model=list[AdminUserOut])
def users(
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> list[AdminUserOut]:
    rows = db.scalars(select(User).order_by(User.created_at.desc()))
    return [_admin_user_out(user) for user in rows]


@router.post("/admin/users", response_model=AdminUserOut, status_code=201)
def create_user(
    payload: AdminUserCreate,
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> AdminUserOut:
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "A user with this email already exists")
    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        role=UserRole(payload.role),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _admin_user_out(user)


@router.patch("/admin/users/{user_id}", response_model=AdminUserOut)
def update_user_status(
    user_id: uuid.UUID,
    payload: AdminUserStatusUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_user),
) -> AdminUserOut:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id and not payload.is_active:
        raise HTTPException(400, "You cannot deactivate your own account")
    user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    return _admin_user_out(user)


@router.get("/chats", response_model=list[ChatSessionOut])
def chats(
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[ChatSessionOut]:
    rows = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    return [_chat_session_out(chat) for chat in rows]


@router.get("/chats/{chat_id}", response_model=ChatDetail)
def chat_detail(
    chat_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ChatDetail:
    chat = db.get(ChatSession, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(404, "Chat not found")
    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.chat_session_id == chat.id)
        .order_by(ChatMessage.created_at.asc())
    )
    return ChatDetail(
        **_chat_session_out(chat).model_dump(),
        messages=[
            ChatMessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                metadata=message.message_metadata or {},
                created_at=message.created_at,
            )
            for message in messages
        ],
    )


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> None:
    chat = db.get(ChatSession, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(404, "Chat not found")
    db.delete(chat)
    db.commit()


@router.post("/chat", response_model=AnswerResponse)
async def chat(
    payload: QuestionRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> AnswerResponse:
    chat_session = (
        db.get(ChatSession, payload.chat_session_id)
        if payload.chat_session_id
        else None
    )
    if chat_session and chat_session.user_id != user.id:
        raise HTTPException(404, "Chat not found")
    if not chat_session:
        chat_session = ChatSession(user_id=user.id, title=payload.question[:100])
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

    db.add(
        ChatMessage(
            chat_session_id=chat_session.id,
            role="user",
            content=payload.question,
        )
    )
    chat_session.updated_at = datetime.now(UTC)
    db.commit()

    try:
        response = await run_in_threadpool(
            rag_service.ask,
            db,
            payload.question,
            payload.top_k,
            payload.rewrite_question,
        )
    except EmbeddingUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except LlmConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc

    response.chat_session_id = chat_session.id
    response.request_id = getattr(request.state, "request_id", None)
    db.add(
        ChatMessage(
            chat_session_id=chat_session.id,
            role="assistant",
            content=response.answer,
            message_metadata={
                "sources": [source.model_dump() for source in response.sources],
                "grounded": response.grounded,
                "grounding_status": response.grounding_status,
                "interpreted_question": response.interpreted_question,
                "search_queries": response.search_queries,
                "request_id": response.request_id,
            },
        )
    )
    chat_session.updated_at = datetime.now(UTC)
    db.commit()
    return response
