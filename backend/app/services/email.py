import logging
import json
import smtplib
import base64
from email.message import EmailMessage
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import settings

logger = logging.getLogger(__name__)


class EmailProviderError(Exception):
    """A safe, credential-free email provider failure suitable for logs."""


def _safe_body_value(value: str) -> str:
    # Messages are plain text, but remove control characters to avoid spoofed lines.
    return "".join(ch for ch in value if ch in "\t " or ord(ch) >= 32).strip()


def _message(to_address: str, subject: str, body: str) -> EmailMessage:
    sender = settings.gmail_sender_email if settings.email_provider == "gmail_api" else settings.smtp_user
    if any("\r" in value or "\n" in value for value in (sender, to_address, subject)):
        raise ValueError("Unsafe email header value")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)
    return message


def _send_smtp(message: EmailMessage) -> None:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(settings.smtp_user, settings.smtp_app_password)
        smtp.send_message(message)


def _send_resend(to_address: str, subject: str, body: str, reply_to: str | None = None) -> None:
    payload = {
        "from": settings.email_from,
        "to": [to_address],
        "subject": subject,
        "text": body,
    }
    if reply_to:
        payload["reply_to"] = reply_to
    request = Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.resend_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            if not 200 <= response.status < 300:
                raise EmailProviderError(f"Email API returned HTTP {response.status}")
            response.read()
    except HTTPError as error:
        # Do not log response bodies: providers can echo recipients or message data.
        raise EmailProviderError(f"Email API returned HTTP {error.code}") from None
    except URLError:
        # Avoid logging the request, API key, or user-submitted message.
        raise EmailProviderError("Email API connection failed") from None


def _send_gmail_api(message: EmailMessage) -> None:
    """Exchange the stored refresh token and send a MIME message over HTTPS."""
    token_request = Request(
        "https://oauth2.googleapis.com/token",
        data=urlencode({
            "client_id": settings.gmail_client_id,
            "client_secret": settings.gmail_client_secret,
            "refresh_token": settings.gmail_refresh_token,
            "grant_type": "refresh_token",
        }).encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(token_request, timeout=15) as response:
            token_data = json.loads(response.read())
        access_token = token_data.get("access_token")
        if not access_token:
            raise EmailProviderError("Google OAuth did not return an access token")
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
        send_request = Request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            data=json.dumps({"raw": raw_message}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(send_request, timeout=15) as response:
            if not 200 <= response.status < 300:
                raise EmailProviderError(f"Gmail API returned HTTP {response.status}")
            response.read()
    except HTTPError as error:
        # Google error bodies can contain identifying data; keep logs generic.
        endpoint = "OAuth" if error.url.startswith("https://oauth2.googleapis.com/") else "Gmail API"
        raise EmailProviderError(f"Google {endpoint} returned HTTP {error.code}") from None
    except URLError:
        raise EmailProviderError("Google email service connection failed") from None
    except (ValueError, KeyError, json.JSONDecodeError):
        raise EmailProviderError("Google email service returned an invalid response") from None


def _send_one(to_address: str, subject: str, body: str, reply_to: str | None = None) -> None:
    if settings.email_provider == "resend":
        _send_resend(to_address, subject, body, reply_to)
        return
    if settings.email_provider == "gmail_api":
        message = _message(to_address, subject, body)
        if reply_to:
            message["Reply-To"] = reply_to
        _send_gmail_api(message)
        return
    message = _message(to_address, subject, body)
    if reply_to:
        message["Reply-To"] = reply_to
    _send_smtp(message)


def _email_is_configured() -> bool:
    if not settings.admin_notify_email:
        return False
    if settings.email_provider == "resend":
        return bool(settings.resend_api_key and settings.email_from)
    if settings.email_provider == "gmail_api":
        return bool(
            settings.gmail_sender_email
            and settings.gmail_client_id
            and settings.gmail_client_secret
            and settings.gmail_refresh_token
        )
    return bool(settings.smtp_user and settings.smtp_app_password)


def send_registration_emails(full_name: str, email: str, phone: str, created_at: str) -> None:
    """Send admin notification and welcome email; safe to invoke as a background task."""
    if not _email_is_configured():
        logger.warning("Registration emails skipped: email provider settings are incomplete")
        return
    clean_name = _safe_body_value(full_name)
    clean_email = _safe_body_value(email)
    clean_phone = _safe_body_value(phone)
    clean_created = _safe_body_value(created_at)
    try:
        messages = (
            (
                settings.admin_notify_email,
                "New PCOS screening account registered",
                f"A new account was created.\n\nName: {clean_name}\nEmail: {clean_email}\nPhone: {clean_phone}\nRegistered at: {clean_created}\n",
            ),
            (
                clean_email,
                "Welcome to the PCOS screening companion",
                f"Hello {clean_name},\n\nYour account is ready. You can use it to complete a screening and review your screening history.\n\nPlease remember: the screening tool provides an estimate, not a diagnosis. It does not replace medical care. Please consult a doctor about your health concerns.\n",
            ),
        )
    except Exception as error:
        logger.warning("Registration emails skipped (%s)", type(error).__name__)
        return
    for label, (to_address, subject, body) in zip(("admin notification", "welcome message"), messages):
        try:
            _send_one(to_address, subject, body)
        except Exception as error:  # A mail outage must never affect registration.
            reason = str(error) if isinstance(error, EmailProviderError) else type(error).__name__
            logger.warning("Could not send registration %s (%s)", label, reason)


def send_contact_email(name: str, email: str, subject: str, body: str) -> None:
    """Forward a public website enquiry to the configured team inbox."""
    if not _email_is_configured():
        logger.warning("Website contact email skipped: email provider settings are incomplete")
        return
    clean_name = _safe_body_value(name)
    clean_email = _safe_body_value(email)
    clean_subject = _safe_body_value(subject)
    clean_body = "".join(ch for ch in body if ch in "\t\n\r " or ord(ch) >= 32).strip()
    try:
        _send_one(
            settings.admin_notify_email,
            "NariSaarthi website contact form submission",
            f"New website enquiry\n\nName: {clean_name}\nEmail: {clean_email}\nSubject: {clean_subject}\n\nMessage:\n{clean_body}\n",
            reply_to=clean_email,
        )
    except Exception as error:
        # Never log submitted message contents or SMTP credentials.
        reason = str(error) if isinstance(error, EmailProviderError) else type(error).__name__
        logger.warning("Could not send website contact email (%s)", reason)
