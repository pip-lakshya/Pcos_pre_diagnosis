#!/usr/bin/env python3
"""Export account profile fields (never password hashes) from the configured database."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.db.database import SessionLocal, init_db
from app.services.user_export import export_users


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("csv", "xlsx"), default="xlsx")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    init_db()
    output = args.output or Path("exports") / f"users-{datetime.now().strftime('%Y%m%d-%H%M%S')}.{args.format}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        content, _content_type, _extension = export_users(db, args.format)
    output.write_bytes(content)
    print(f"Exported user profile fields to {output}")


if __name__ == "__main__":
    main()
