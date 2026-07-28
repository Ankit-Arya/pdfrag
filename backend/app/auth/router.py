import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import current_user
from app.auth.security import (
    create_access_token,
    hash_password,
    hash_token,
    new_refresh_token,
    verify_password,
)
from app.config import get_settings
from app.db import get_db
from app.db_models import User, UserRole, UserSession

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Login(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)


class Refresh(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: str


class SessionOut(BaseModel):
    id: uuid.UUID
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    user_agent: str | None
    ip_address: str | None


def _pair(db: Session, user: User, request: Request) -> TokenPair:
    settings = get_settings()
    refresh = new_refresh_token()
    session = UserSession(
        user_id=user.id,
        refresh_token_hash=hash_token(refresh),
        expires_at=datetime.now(UTC)
        + timedelta(days=settings.refresh_token_ttl_days),
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return TokenPair(
        access_token=create_access_token(user.id, user.role.value, session.id),
        refresh_token=refresh,
    )


@router.post("/login", response_model=TokenPair)
def login(payload: Login, request: Request, db: Session = Depends(get_db)) -> TokenPair:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if (
        not user
        or not verify_password(payload.password, user.password_hash)
        or not user.is_active
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return _pair(db, user, request)


@router.post("/refresh", response_model=TokenPair)
def refresh(
    payload: Refresh,
    request: Request,
    db: Session = Depends(get_db),
) -> TokenPair:
    session = db.scalar(
        select(UserSession).where(
            UserSession.refresh_token_hash == hash_token(payload.refresh_token)
        )
    )
    if (
        not session
        or session.revoked_at
        or session.expires_at <= datetime.now(UTC)
    ):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.get(User, session.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User is not active")

    session.revoked_at = datetime.now(UTC)
    db.commit()
    return _pair(db, user, request)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: Refresh, db: Session = Depends(get_db)) -> None:
    session = db.scalar(
        select(UserSession).where(
            UserSession.refresh_token_hash == hash_token(payload.refresh_token)
        )
    )
    if session:
        session.revoked_at = datetime.now(UTC)
        db.commit()


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return UserOut(id=user.id, email=user.email, role=user.role.value)


@router.get("/sessions", response_model=list[SessionOut])
def sessions(
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[SessionOut]:
    rows = db.scalars(
        select(UserSession)
        .where(UserSession.user_id == user.id)
        .order_by(UserSession.created_at.desc())
    )
    return [
        SessionOut(
            id=row.id,
            created_at=row.created_at,
            expires_at=row.expires_at,
            revoked_at=row.revoked_at,
            user_agent=row.user_agent,
            ip_address=row.ip_address,
        )
        for row in rows
    ]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> None:
    session = db.get(UserSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.revoked_at:
        session.revoked_at = datetime.now(UTC)
        db.commit()


def bootstrap_admin(db: Session) -> None:
    settings = get_settings()
    if not settings.bootstrap_admin_email or not settings.bootstrap_admin_password:
        return
    if db.scalar(
        select(User).where(User.email == settings.bootstrap_admin_email.lower())
    ):
        return
    db.add(
        User(
            email=settings.bootstrap_admin_email.lower(),
            password_hash=hash_password(settings.bootstrap_admin_password),
            role=UserRole.admin,
        )
    )
    db.commit()
