import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from app.auth.security import decode_access_token
from app.db import get_db
from app.db_models import User, UserRole, UserSession
bearer=HTTPBearer(auto_error=False)
def current_user(credentials:HTTPAuthorizationCredentials|None=Depends(bearer), db:Session=Depends(get_db))->User:
    if not credentials: raise HTTPException(status_code=401,detail="Authentication required")
    try: payload=decode_access_token(credentials.credentials); uid=uuid.UUID(payload["sub"]); sid=uuid.UUID(payload["sid"])
    except Exception as exc: raise HTTPException(status_code=401,detail="Invalid or expired token") from exc
    session=db.get(UserSession,sid); user=db.get(User,uid)
    if not session or session.revoked_at or not user or not user.is_active: raise HTTPException(status_code=401,detail="Session is not active")
    return user
def admin_user(user:User=Depends(current_user))->User:
    if user.role != UserRole.admin: raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="Admin access required")
    return user
