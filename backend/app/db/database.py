from collections.abc import Generator
from urllib.parse import urlsplit

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

def _build_engine():
    database_url = settings.database_url.strip()
    if database_url.startswith("libsql://") or database_url.startswith("https://"):
        if not settings.turso_auth_token:
            raise RuntimeError("TURSO_AUTH_TOKEN is required when DATABASE_URL points to Turso")
        parsed = urlsplit(database_url)
        if not parsed.netloc:
            raise RuntimeError("DATABASE_URL must contain a valid Turso database host")
        # Turso's SQLAlchemy dialect expects the `sqlite+libsql` driver URL;
        # the endpoint supplied by Turso is commonly `libsql://<host>`.
        database_url = f"sqlite+libsql://{parsed.netloc}?secure=true"
        connect_args = {"auth_token": settings.turso_auth_token}
    else:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # Import registers the mapped tables with Base.metadata.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    # Safe additive migration for existing development databases. Existing accounts
    # remain valid and are marked with blank profile values until they update them.
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("users")}
    with engine.begin() as connection:
        if "full_name" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN full_name VARCHAR(200) NOT NULL DEFAULT ''"))
        if "phone" not in columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL DEFAULT ''"))
