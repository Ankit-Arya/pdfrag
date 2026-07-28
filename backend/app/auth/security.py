import hashlib, secrets, uuid
from datetime import UTC, datetime, timedelta
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from app.config import get_settings
_ph=PasswordHasher()
def hash_password(v:str)->str: return _ph.hash(v)
def verify_password(v:str,h:str)->bool:
    try: return _ph.verify(h,v)
    except VerifyMismatchError: return False
def hash_token(v:str)->str: return hashlib.sha256(v.encode()).hexdigest()
def create_access_token(user_id:uuid.UUID, role:str, session_id:uuid.UUID)->str:
    s=get_settings(); now=datetime.now(UTC)
    return jwt.encode({"sub":str(user_id),"role":role,"sid":str(session_id),"iat":now,"exp":now+timedelta(minutes=s.access_token_ttl_minutes)},s.jwt_secret,algorithm=s.jwt_algorithm)
def decode_access_token(token:str)->dict:
    s=get_settings(); return jwt.decode(token,s.jwt_secret,algorithms=[s.jwt_algorithm])
def new_refresh_token()->str: return secrets.token_urlsafe(48)
