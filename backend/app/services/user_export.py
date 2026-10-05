import csv
from io import BytesIO, StringIO

from openpyxl import Workbook
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User

EXPORT_COLUMNS = ("full_name", "email", "phone", "created_at")


def export_users(db: Session, file_format: str) -> tuple[bytes, str, str]:
    users = db.scalars(select(User).order_by(User.created_at.asc(), User.id.asc())).all()
    rows = [[getattr(user, column) for column in EXPORT_COLUMNS] for user in users]
    if file_format == "csv":
        output = StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(EXPORT_COLUMNS)
        writer.writerows(rows)
        return output.getvalue().encode("utf-8"), "text/csv; charset=utf-8", "csv"
    if file_format == "xlsx":
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Users"
        sheet.append(EXPORT_COLUMNS)
        for row in rows:
            sheet.append([value.isoformat() if hasattr(value, "isoformat") else value for value in row])
        buffer = BytesIO()
        workbook.save(buffer)
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    raise ValueError("format must be csv or xlsx")
