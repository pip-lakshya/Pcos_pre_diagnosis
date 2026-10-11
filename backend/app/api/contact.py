import re

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.services.email import send_contact_email

router = APIRouter(tags=["contact"])
_EMAIL = re.compile(r"^[^\s@\r\n]+@[^\s@\r\n]+\.[^\s@\r\n]+$")


def _deliver_contact_email(name: str, email: str, subject: str, message: str) -> bool:
    """Keep provider exceptions inside the worker thread and expose no internals."""
    try:
        send_contact_email(name, email, subject, message)
        return True
    except Exception:
        return False


class ContactRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=2, max_length=140)
    message: str = Field(min_length=10, max_length=4000)

    @field_validator("name", "subject")
    @classmethod
    def clean_single_line(cls, value: str) -> str:
        if not value.strip() or "\r" in value or "\n" in value:
            raise ValueError("This field must be a non-empty single line")
        return value.strip()

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL.fullmatch(value):
            raise ValueError("Enter a valid email address")
        return value

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please enter a message")
        return value.strip()


@router.post("/contact", status_code=status.HTTP_200_OK)
async def submit_contact(request: ContactRequest):
    provider_ready = (
        settings.admin_notify_email
        and (
            (settings.email_provider == "resend" and settings.resend_api_key and settings.email_from)
            or (
                settings.email_provider == "gmail_api"
                and settings.gmail_sender_email
                and settings.gmail_client_id
                and settings.gmail_client_secret
                and settings.gmail_refresh_token
            )
            or (settings.email_provider == "smtp" and settings.smtp_user and settings.smtp_app_password)
        )
    )
    if not provider_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "The contact form email service is not configured yet. Please use the social links instead."},
        )
    try:
        delivered = await run_in_threadpool(
            _deliver_contact_email, request.name, request.email, request.subject, request.message
        )
    except Exception:
        delivered = False
    if not delivered:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "We could not deliver your message right now. Please try again later or use the social links."},
        )
    return {"message": "Thanks for reaching out. Your message was sent to the NariSaarthi team."}
