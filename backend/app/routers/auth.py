"""Auth router — register, login, me."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator

from app.core.db import create_user, get_connection, get_user_by_email, get_user_by_id, get_user_by_login
from app.core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    login: str
    email: EmailStr
    password: str

    @field_validator("login")
    @classmethod
    def login_length(cls, v: str) -> str:
        if len(v) < 3 or len(v) > 64:
            raise ValueError("Login must be 3–64 characters")
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class UserOut(BaseModel):
    id: int
    login: str
    email: str
    role: str
    created_at: str

    @classmethod
    def from_row(cls, row: dict) -> "UserOut":
        return cls(
            id=row["id"],
            login=row["login"],
            email=row["email"],
            role=row["role"],
            created_at=str(row["created_at"]),
        )


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── Dependency — current user from JWT ───────────────────────────────────────
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user_id: int = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    with get_connection() as conn:
        user = get_user_by_id(conn, int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    with get_connection() as conn:
        if get_user_by_login(conn, body.login):
            raise HTTPException(status_code=400, detail="Login already taken")
        if get_user_by_email(conn, body.email):
            raise HTTPException(status_code=400, detail="Email already registered")
        user = create_user(conn, body.login, body.email, hash_password(body.password))

    token = create_access_token(
        {"sub": str(user["id"]), "role": user["role"]},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenOut(access_token=token, user=UserOut.from_row(user))


@router.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends()):
    with get_connection() as conn:
        user = get_user_by_login(conn, form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid login or password")

    token = create_access_token(
        {"sub": str(user["id"]), "role": user["role"]},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenOut(access_token=token, user=UserOut.from_row(dict(user)))


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    return UserOut.from_row(dict(user))
