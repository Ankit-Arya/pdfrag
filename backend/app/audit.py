from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.db_models import AuditLog, User


def client_ip(request: Request) -> str | None:
    """Return the originating client IP when a trusted proxy forwards it."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()[:64] or None
    if request.client:
        return request.client.host[:64]
    return None


def add_audit_event(
    db: Session,
    *,
    event_type: str,
    request: Request,
    success: bool,
    user: User | None = None,
    actor_email: str | None = None,
    chat_session_id: uuid.UUID | None = None,
    question: str | None = None,
    response: str | None = None,
    error_message: str | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """Stage an audit event in the caller's current database transaction.

    The caller deliberately controls commit/rollback so the audit event can be
    stored atomically with the login session or assistant response it describes.
    """
    request_id = getattr(request.state, "request_id", None)
    email = actor_email or (user.email if user else None)
    row = AuditLog(
        event_type=event_type[:50],
        success=success,
        user_id=user.id if user else None,
        actor_email=email.lower()[:320] if email else None,
        chat_session_id=chat_session_id,
        question=question,
        response=response,
        error_message=error_message,
        event_metadata=details or {},
        ip_address=client_ip(request),
        user_agent=(request.headers.get("user-agent") or "")[:500] or None,
        request_id=str(request_id)[:100] if request_id else None,
    )
    db.add(row)
    return row
