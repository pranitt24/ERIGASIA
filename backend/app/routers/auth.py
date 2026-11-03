from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import create_token, current_user, verify_password
from ..db import get_db
from ..models import User
from ..services.serialize import user_dict

router = APIRouter()


class LoginIn(BaseModel):
    email: str
    password: str
    role: str | None = None  # optional: the UI's role toggle. Rejected if it doesn't match the account.


@router.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.strip().lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Email or password is incorrect.")
    if body.role and body.role != user.role:
        raise HTTPException(403, f"This account is a {user.role} account. Switch the role above.")
    return {"token": create_token(user), "user": user_dict(user)}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return user_dict(user)
