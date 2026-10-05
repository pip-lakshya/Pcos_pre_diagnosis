from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args, pool_pre_ping=True)
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
