import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.services.user_export import export_users

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users/export")
def download_users(
    format: str = Query(default="csv", pattern="^(xlsx|csv)$"),
    admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    db: Session = Depends(get_db),
):
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin export is not configured")
    if not admin_key or not secrets.compare_digest(admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid admin key")
    content, media_type, extension = export_users(db, format)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="users.{extension}"'},
    )
