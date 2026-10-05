import re

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.security import create_access_token, get_current_user, hash_password, verify_password
from app.db.database import get_db
from app.db.models import User
from app.services.email import send_registration_emails

router = APIRouter(prefix="/auth", tags=["auth"])
_EMAIL = re.compile(r"^[^\s@\r\n]+@[^\s@\r\n]+\.[^\s@\r\n]+$")
_PHONE = re.compile(r"^\+?[0-9\s-]+$")


class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    full_name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    phone: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=8, max_length=72)

    @field_validator("full_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip() or "\r" in value or "\n" in value:
            raise ValueError("Name must be non-empty and cannot contain line breaks")
        return value.strip()

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL.fullmatch(normalized):
            raise ValueError("Enter a valid email address")
        return normalized

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, value: str) -> str:
        if "\r" in value or "\n" in value or not _PHONE.fullmatch(value):
            raise ValueError("Phone must contain digits, spaces or dashes and an optional leading +")
        normalized = re.sub(r"[\s-]", "", value)
        digits = normalized.removeprefix("+")
        if not 10 <= len(digits) <= 15:
            raise ValueError("Phone must contain 10 to 15 digits")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.lower()
        if not _EMAIL.fullmatch(normalized):
            raise ValueError("Enter a valid email address")
        return normalized


class AuthUser(BaseModel):
    id: int
    full_name: str
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


def _token_for(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        user=AuthUser(id=user.id, full_name=user.full_name, email=user.email),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == request.email)):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = User(
        full_name=request.full_name,
        email=request.email,
        phone=request.phone,
        hashed_password=hash_password(request.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    db.refresh(user)
    background_tasks.add_task(
        send_registration_emails,
        user.full_name,
        user.email,
        user.phone,
        user.created_at.isoformat(),
    )
    return _token_for(user)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == request.email))
    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_for(user)


@router.get("/me", response_model=AuthUser)
def me(user: User = Depends(get_current_user)):
    return AuthUser(id=user.id, full_name=user.full_name, email=user.email)
