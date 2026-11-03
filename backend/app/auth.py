import time

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

bearer = HTTPBearer(auto_error=False)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except ValueError:
        return False


def create_token(user: User) -> str:
    payload = {"sub": user.id, "role": user.role, "exp": int(time.time()) + settings.JWT_EXPIRE_MINUTES * 60}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if not creds:
        raise HTTPException(401, "Sign in to continue.")
    try:
        data = jwt.decode(creds.credentials, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Your session expired. Sign in again.")
    user = db.get(User, data["sub"])
    if not user:
        raise HTTPException(401, "Account not found.")
    return user


def _require(role: str):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role != role:
            raise HTTPException(403, f"This area is for {role}s only.")
        return user
    return dep


student_only = _require("student")
teacher_only = _require("teacher")
