from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    nvidia_api_key: str = ""
    nvidia_model: str = "openai/gpt-oss-20b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_followup_timeout_seconds: float = Field(default=15, gt=0, le=120)
    cors_origins: str = "http://localhost:5173"
    jwt_secret: str = ""
    jwt_expire_minutes: int = Field(default=60, gt=0)
    admin_api_key: str = ""
    smtp_user: str = ""
    smtp_app_password: str = ""
    admin_notify_email: str = ""
    email_provider: Literal["smtp", "resend", "gmail_api"] = "smtp"
    resend_api_key: str = ""
    email_from: str = ""
    gmail_sender_email: str = ""
    gmail_client_id: str = ""
    gmail_client_secret: str = ""
    gmail_refresh_token: str = ""
    tts_provider: Literal["browser", "edge", "magpie"] = "magpie"
    edge_tts_voice: str = "en-IN-NeerjaNeural"
    magpie_tts_url: str = "https://877104f7-e885-42b9-8de8-f6e4c6303969.invocation.api.nvcf.nvidia.com/v1/audio/synthesize"
    magpie_tts_voice: str = "Magpie-Multilingual.EN-US.Aria"
    osm_contact_email: str = ""
    database_url: str = f"sqlite:///{ROOT / 'pcos_agent.sqlite3'}"
    turso_auth_token: str = ""
    model_path: str = str(ROOT / "ml" / "pcos_rf_model.pkl")
    features_path: str = str(ROOT / "ml" / "pcos_model_features.json")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
