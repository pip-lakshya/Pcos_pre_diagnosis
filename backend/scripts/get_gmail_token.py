"""Run a one-time local OAuth flow and save Gmail's refresh token to backend/.env.

Use the OAuth client ID and secret for a Desktop app, with gmail.send added to
the Google Auth Platform Data access scopes. Never commit or share the token.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_DIR / ".env"
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def save_refresh_token(token: str) -> None:
    """Set or append GMAIL_REFRESH_TOKEN without printing or exposing its value."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    replacement = f"GMAIL_REFRESH_TOKEN={token}"
    pattern = re.compile(r"^\s*GMAIL_REFRESH_TOKEN\s*=.*$")
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = replacement
            break
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(replacement)

    ENV_FILE.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        # Some filesystems do not support POSIX permission bits.
        pass


def main() -> int:
    if not settings.gmail_client_id or not settings.gmail_client_secret:
        print("Missing GMAIL_CLIENT_ID or GMAIL_CLIENT_SECRET in backend/.env")
        return 1
    if not settings.gmail_sender_email:
        print("Missing GMAIL_SENDER_EMAIL in backend/.env")
        return 1

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Install the backend requirements first: python -m pip install -r requirements.txt")
        return 1

    client_config = {
        "installed": {
            "client_id": settings.gmail_client_id,
            "client_secret": settings.gmail_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    try:
        flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
        credentials = flow.run_local_server(
            host="localhost",
            port=0,
            access_type="offline",
            prompt="consent",
        )
    except Exception as error:
        # Do not print exception text: auth errors can include sensitive URLs/data.
        print(f"Google authorization failed ({type(error).__name__}). Check the Desktop OAuth client and consent settings.")
        return 1

    if not credentials.refresh_token:
        print("Google did not return a refresh token. Revoke this app's access in your Google Account and run again.")
        return 1

    save_refresh_token(credentials.refresh_token)
    print(f"Refresh token saved to {ENV_FILE} (value hidden).")
    print("Now add GMAIL_REFRESH_TOKEN to the Render backend Environment settings as a secret, then redeploy.")
    print("If the OAuth app is still in Testing, this token may expire after 7 days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
