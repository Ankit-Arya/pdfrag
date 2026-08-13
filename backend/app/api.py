import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from urllib.parse import quote

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.audit import add_audit_event
from app.auth.dependencies import admin_user, current_user
from app.auth.security import hash_password
from app.config import get_settings
from app.db import get_db
from app.db_models import (
    AuditLog,
    ChatMessage,
    ChatSession,
    Document,
    DocumentStatus,
    User,
    UserRole,
)
from app.document_processing import process_document_ids
from app.models import (
    AdminUserCreate,
    AdminUserOut,
    AdminUserStatusUpdate,
    AnswerResponse,
    AuditLogOut,
    ChatDetail,
    ChatMessageOut,
    ChatSessionOut,
    DocumentBatchRequest,
    DocumentBatchResponse,
    DocumentOut,
    HealthResponse,
    KnowledgeStatus,
    QuestionRequest,
)
from app.rag.embeddings import EmbeddingUnavailableError, embedding_service
from app.rag.llm import LlmConfigurationError, LlmRateLimitError
from app.rag.pdf import ocr_available, safe_filename, table_extraction_available
from app.rag.query import ANSWER_POLICY_VERSION
from app.rag.service import rag_service

logger = logging.getLogger(__name__)
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


def _audit_log_out(row: AuditLog) -> AuditLogOut:
    return AuditLogOut(
        id=row.id,
        event_type=row.event_type,
        success=row.success,
        user_id=row.user_id,
        actor_email=row.actor_email,
        chat_session_id=row.chat_session_id,
        question=row.question,
        response=row.response,
        error_message=row.error_message,
        details=row.event_metadata or {},
        ip_address=row.ip_address,
        user_agent=row.user_agent,
        request_id=row.request_id,
        created_at=row.created_at,
    )


def _record_chat_failure(
    db: Session,
    *,
    request: Request,
    user: User,
    chat_session: ChatSession,
    question: str,
    error: Exception,
) -> None:
    db.rollback()
    add_audit_event(
        db,
        event_type="chat_error",
        request=request,
        success=False,
        user=user,
        chat_session_id=chat_session.id,
        question=question,
        error_message=str(error)[:4000],
        details={"error_type": type(error).__name__},
    )
    db.commit()


def _conversation_context(db: Session, chat_session_id: uuid.UUID) -> list[dict[str, str]]:
    """Return recent chat turns for intent resolution, prioritizing newest turns."""
    settings = get_settings()
    if settings.chat_context_messages <= 0 or settings.chat_context_chars <= 0:
        return []

    rows = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.chat_session_id == chat_session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(settings.chat_context_messages)
        )
    )
    newest_first: list[dict[str, str]] = []
    used = 0
    for row in rows:
        if row.role not in {"user", "assistant"}:
            continue
        text = " ".join(row.content.split())
        if not text:
            continue
        text = text[: settings.chat_context_per_message_chars]
        remaining = settings.chat_context_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        turn: dict[str, str] = {"role": row.role, "content": text}
        if row.role == "assistant":
            metadata = row.message_metadata or {}
            contextual = metadata.get("contextual_question")
            if isinstance(contextual, str) and contextual.strip():
                turn["context_hint"] = " ".join(contextual.split())[: settings.chat_context_per_message_chars]
            routing = metadata.get("routing_hints")
            if isinstance(routing, list):
                clean_routing = [
                    " ".join(str(item).split())
                    for item in routing
                    if str(item).strip()
                ]
                if clean_routing:
                    turn["routing_hint"] = " | ".join(clean_routing[:12])
        newest_first.append(turn)
        used += len(text)
    return list(reversed(newest_first))


def _queue_documents(
    db: Session,
    background_tasks: BackgroundTasks,
    document_ids: list[uuid.UUID],
) -> DocumentBatchResponse:
    unique_ids = list(dict.fromkeys(document_ids))
    if not unique_ids:
        return DocumentBatchResponse(
            queued_document_ids=[],
            already_processing=0,
            missing=0,
        )

    known_ids = set(
        db.scalars(select(Document.id).where(Document.id.in_(unique_ids))).all()
    )
    missing = sum(1 for document_id in unique_ids if document_id not in known_ids)

    claimed = set(
        db.scalars(
            update(Document)
            .where(
                Document.id.in_(known_ids),
                Document.status != DocumentStatus.processing,
            )
            .values(status=DocumentStatus.processing, error=None)
            .returning(Document.id)
        ).all()
    )
    db.commit()

    queued_ids = [document_id for document_id in unique_ids if document_id in claimed]
    already_processing = len(known_ids) - len(claimed)
    if queued_ids:
        background_tasks.add_task(process_document_ids, queued_ids)

    return DocumentBatchResponse(
        queued_document_ids=queued_ids,
        already_processing=already_processing,
        missing=missing,
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
        query_model=settings.query_model,
        summary_model=settings.summary_model,
        answer_policy_version=ANSWER_POLICY_VERSION,
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
    background_tasks: BackgroundTasks,
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
        _queue_documents(db, background_tasks, [document.id])
        db.refresh(document)

    return _document_out(document)


@router.get("/admin/documents", response_model=list[DocumentOut])
def documents(
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> list[DocumentOut]:
    rows = db.scalars(select(Document).order_by(Document.created_at.desc()))
    return [_document_out(document) for document in rows]


@router.post(
    "/admin/documents/process-batch",
    response_model=DocumentBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def process_documents_batch(
    payload: DocumentBatchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> DocumentBatchResponse:
    return _queue_documents(db, background_tasks, payload.document_ids)


@router.post(
    "/admin/documents/{document_id}/process",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
)
def process_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> DocumentOut:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    _queue_documents(db, background_tasks, [document_id])
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


@router.get("/admin/audit-logs", response_model=list[AuditLogOut])
def audit_logs(
    event_type: str | None = None,
    user_id: uuid.UUID | None = None,
    email: str | None = None,
    success: bool | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(admin_user),
) -> list[AuditLogOut]:
    statement = select(AuditLog)
    if event_type:
        statement = statement.where(AuditLog.event_type == event_type)
    if user_id:
        statement = statement.where(AuditLog.user_id == user_id)
    if email:
        statement = statement.where(
            AuditLog.actor_email.ilike(f"%{email.strip()}%")
        )
    if success is not None:
        statement = statement.where(AuditLog.success == success)

    rows = db.scalars(
        statement.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit)
    )
    return [_audit_log_out(row) for row in rows]


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


def _prepare_chat_exchange(
    payload: QuestionRequest,
    db: Session,
    user: User,
) -> tuple[ChatSession, list[dict[str, str]], ChatMessage]:
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

    # Capture history before writing the current question. History is intent
    # context only; every factual statement is re-retrieved from the PDFs.
    conversation_context = _conversation_context(db, chat_session.id)
    user_message = ChatMessage(
        chat_session_id=chat_session.id,
        role="user",
        content=payload.question,
    )
    db.add(user_message)
    chat_session.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(user_message)
    return chat_session, conversation_context, user_message


def _finalize_chat_exchange(
    db: Session,
    *,
    request: Request,
    user: User,
    chat_session: ChatSession,
    user_message: ChatMessage,
    response: AnswerResponse,
) -> AnswerResponse:
    response.chat_session_id = chat_session.id
    response.request_id = getattr(request.state, "request_id", None)
    response.answer_policy_version = ANSWER_POLICY_VERSION
    response.question_created_at = user_message.created_at
    source_details = [source.model_dump() for source in response.sources]
    evidence_details = [source.model_dump() for source in response.evidence]
    assistant_message = ChatMessage(
        chat_session_id=chat_session.id,
        role="assistant",
        content=response.answer,
        message_metadata={
            "sources": source_details,
            "evidence": evidence_details,
            "formatted_sources": response.formatted_sources,
            "formatted_evidence": response.formatted_evidence,
            "grounded": response.grounded,
            "grounding_status": response.grounding_status,
            "interpreted_question": response.interpreted_question,
            "contextual_question": response.contextual_question,
            "retrieval_mode": response.retrieval_mode,
            "resolved_abbreviations": response.resolved_abbreviations,
            "routing_hints": response.routing_hints,
            "primary_documents": response.primary_documents,
            "candidate_chunks": response.candidate_chunks,
            "evidence_chunks": response.evidence_chunks,
            "search_queries": response.search_queries,
            "answer_policy_version": response.answer_policy_version,
            "request_id": response.request_id,
        },
    )
    db.add(assistant_message)
    add_audit_event(
        db,
        event_type="chat_exchange",
        request=request,
        success=True,
        user=user,
        chat_session_id=chat_session.id,
        question=user_message.content,
        response=response.answer,
        details={
            "grounded": response.grounded,
            "grounding_status": response.grounding_status,
            "interpreted_question": response.interpreted_question,
            "contextual_question": response.contextual_question,
            "retrieval_mode": response.retrieval_mode,
            "resolved_abbreviations": response.resolved_abbreviations,
            "routing_hints": response.routing_hints,
            "primary_documents": response.primary_documents,
            "candidate_chunks": response.candidate_chunks,
            "evidence_chunks": response.evidence_chunks,
            "search_queries": response.search_queries,
            "answer_policy_version": response.answer_policy_version,
            "sources": source_details,
        },
    )
    chat_session.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(assistant_message)
    response.response_created_at = assistant_message.created_at
    return response


def _sse_event(event: str, payload: dict[str, object]) -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def _stream_error_payload(error: Exception) -> dict[str, object]:
    if isinstance(error, LlmRateLimitError):
        return {
            "detail": str(error),
            "code": "rate_limit",
            "retry_after": 5,
        }
    if isinstance(error, EmbeddingUnavailableError):
        return {"detail": str(error), "code": "embedding_unavailable"}
    if isinstance(error, LlmConfigurationError):
        return {"detail": str(error), "code": "llm_configuration"}
    return {
        "detail": "The language model request failed",
        "code": "chat_failed",
    }


@router.post("/chat", response_model=AnswerResponse)
async def chat(
    payload: QuestionRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> AnswerResponse:
    chat_session, conversation_context, user_message = _prepare_chat_exchange(
        payload, db, user
    )

    try:
        response = await run_in_threadpool(
            rag_service.ask,
            db,
            payload.question,
            payload.top_k,
            payload.rewrite_question,
            conversation_context,
        )
    except EmbeddingUnavailableError as exc:
        _record_chat_failure(
            db,
            request=request,
            user=user,
            chat_session=chat_session,
            question=payload.question,
            error=exc,
        )
        raise HTTPException(503, str(exc)) from exc
    except LlmRateLimitError as exc:
        _record_chat_failure(
            db,
            request=request,
            user=user,
            chat_session=chat_session,
            question=payload.question,
            error=exc,
        )
        raise HTTPException(
            503,
            str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    except LlmConfigurationError as exc:
        _record_chat_failure(
            db,
            request=request,
            user=user,
            chat_session=chat_session,
            question=payload.question,
            error=exc,
        )
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        _record_chat_failure(
            db,
            request=request,
            user=user,
            chat_session=chat_session,
            question=payload.question,
            error=exc,
        )
        logger.exception("Q&A request failed")
        raise HTTPException(502, "The language model request failed") from exc

    return _finalize_chat_exchange(
        db,
        request=request,
        user=user,
        chat_session=chat_session,
        user_message=user_message,
        response=response,
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: QuestionRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> StreamingResponse:
    """Stream truthful RAG workflow progress, then the persisted final answer.

    Events expose only operational stages and counts. They never expose model
    chain-of-thought, hidden reasoning, prompts, or private scratch work.
    """
    chat_session, conversation_context, user_message = _prepare_chat_exchange(
        payload, db, user
    )
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()

    def publish_progress(progress: dict[str, object]) -> None:
        # rag_service runs in a worker thread. Schedule queue writes safely on the
        # request event loop rather than touching asyncio.Queue from that thread.
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", dict(progress)))

    async def run_chat_job() -> None:
        try:
            response = await run_in_threadpool(
                rag_service.ask,
                db,
                payload.question,
                payload.top_k,
                payload.rewrite_question,
                conversation_context,
                publish_progress,
            )
            await queue.put(
                (
                    "progress",
                    {
                        "stage": "save",
                        "label": "Saving the grounded answer",
                        "detail": "Recording the answer and its cited evidence in chat history",
                    },
                )
            )
            response = _finalize_chat_exchange(
                db,
                request=request,
                user=user,
                chat_session=chat_session,
                user_message=user_message,
                response=response,
            )
            await queue.put(("answer", response.model_dump(mode="json")))
        except (EmbeddingUnavailableError, LlmRateLimitError, LlmConfigurationError) as exc:
            _record_chat_failure(
                db,
                request=request,
                user=user,
                chat_session=chat_session,
                question=payload.question,
                error=exc,
            )
            await queue.put(("error", _stream_error_payload(exc)))
        except Exception as exc:
            _record_chat_failure(
                db,
                request=request,
                user=user,
                chat_session=chat_session,
                question=payload.question,
                error=exc,
            )
            logger.exception("Streaming Q&A request failed")
            await queue.put(("error", _stream_error_payload(exc)))
        finally:
            await queue.put(("done", {}))

    task = asyncio.create_task(run_chat_job())

    async def event_stream():
        yield _sse_event(
            "start",
            {
                "chat_session_id": str(chat_session.id),
                "question_created_at": user_message.created_at.isoformat(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )
        try:
            while True:
                try:
                    event, event_payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # SSE comment heartbeat keeps Nginx/browser connections alive
                    # during a long model call without adding a visible UI event.
                    yield ": keepalive\n\n"
                    continue
                if event == "done":
                    break
                yield _sse_event(event, event_payload)
        finally:
            # Keep the DB dependency alive until the worker is finished even if
            # the browser disconnects. In-flight OpenAI calls cannot be forcefully
            # interrupted safely; finishing also preserves the saved chat result.
            if not task.done():
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
