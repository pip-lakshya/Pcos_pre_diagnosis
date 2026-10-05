import logging
import smtplib
from email.message import EmailMessage

from app.config import settings

logger = logging.getLogger(__name__)


def _safe_body_value(value: str) -> str:
    # Messages are plain text, but remove control characters to avoid spoofed lines.
    return "".join(ch for ch in value if ch in "\t " or ord(ch) >= 32).strip()


def _message(to_address: str, subject: str, body: str) -> EmailMessage:
    if any("\r" in value or "\n" in value for value in (settings.smtp_user, to_address, subject)):
        raise ValueError("Unsafe email header value")
    message = EmailMessage()
    message["From"] = settings.smtp_user
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)
    return message


def _send_one(message: EmailMessage) -> None:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(settings.smtp_user, settings.smtp_app_password)
        smtp.send_message(message)


def send_registration_emails(full_name: str, email: str, phone: str, created_at: str) -> None:
    """Send admin notification and welcome email; safe to invoke as a background task."""
    if not settings.smtp_user or not settings.smtp_app_password or not settings.admin_notify_email:
        logger.warning("Registration emails skipped: SMTP settings are incomplete")
        return
    clean_name = _safe_body_value(full_name)
    clean_email = _safe_body_value(email)
    clean_phone = _safe_body_value(phone)
    clean_created = _safe_body_value(created_at)
    try:
        messages = (
            _message(
                settings.admin_notify_email,
                "New PCOS screening account registered",
                f"A new account was created.\n\nName: {clean_name}\nEmail: {clean_email}\nPhone: {clean_phone}\nRegistered at: {clean_created}\n",
            ),
            _message(
                clean_email,
                "Welcome to the PCOS screening companion",
                f"Hello {clean_name},\n\nYour account is ready. You can use it to complete a screening and review your screening history.\n\nPlease remember: the screening tool provides an estimate, not a diagnosis. It does not replace medical care. Please consult a doctor about your health concerns.\n",
            ),
        )
    except Exception as error:
        logger.warning("Registration emails skipped (%s)", type(error).__name__)
        return
    for label, message in zip(("admin notification", "welcome message"), messages):
        try:
            _send_one(message)
        except Exception as error:  # A mail outage must never affect registration.
            logger.warning("Could not send registration %s (%s)", label, type(error).__name__)


def send_contact_email(name: str, email: str, subject: str, body: str) -> None:
    """Forward a public website enquiry to the configured team inbox."""
    if not settings.smtp_user or not settings.smtp_app_password or not settings.admin_notify_email:
        logger.warning("Website contact email skipped: SMTP settings are incomplete")
        return
    clean_name = _safe_body_value(name)
    clean_email = _safe_body_value(email)
    clean_subject = _safe_body_value(subject)
    clean_body = "".join(ch for ch in body if ch in "\t\n\r " or ord(ch) >= 32).strip()
    try:
        message = _message(
            settings.admin_notify_email,
            "NariSaarthi website contact form submission",
            f"New website enquiry\n\nName: {clean_name}\nEmail: {clean_email}\nSubject: {clean_subject}\n\nMessage:\n{clean_body}\n",
        )
        message["Reply-To"] = clean_email
        _send_one(message)
    except Exception as error:
        # Never log submitted message contents or SMTP credentials.
        logger.warning("Could not send website contact email (%s)", type(error).__name__)
