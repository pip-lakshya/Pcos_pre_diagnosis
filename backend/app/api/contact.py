import re

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings
from app.services.email import send_contact_email

router = APIRouter(tags=["contact"])
_EMAIL = re.compile(r"^[^\s@\r\n]+@[^\s@\r\n]+\.[^\s@\r\n]+$")


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


@router.post("/contact", status_code=status.HTTP_202_ACCEPTED)
def submit_contact(request: ContactRequest, background_tasks: BackgroundTasks):
    if not (settings.smtp_user and settings.smtp_app_password and settings.admin_notify_email):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The contact form email service is not configured yet. Please use the social links instead.",
        )
    background_tasks.add_task(
        send_contact_email, request.name, request.email, request.subject, request.message
    )
    return {"message": "Thanks for reaching out. Your message has been queued for the NariSaarthi team."}
