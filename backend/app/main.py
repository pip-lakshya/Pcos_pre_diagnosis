from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.contact import router as contact_router
from app.api.doctors import router as doctors_router
from app.api.predict import router as predict_router
from app.api.research import router as research_router
from app.api.screenings import router as screenings_router
from app.api.tts import router as tts_router
from app.config import settings
from app.db.database import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Narisaarthi — Empower Your PCOS Journey", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(predict_router)
app.include_router(chat_router)
app.include_router(contact_router)
app.include_router(doctors_router)
app.include_router(research_router)
app.include_router(screenings_router)
app.include_router(tts_router)


@app.get("/health")
def health():
    return {"status": "ok"}
