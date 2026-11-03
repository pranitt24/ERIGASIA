from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import Base, engine
from .routers import auth, jobs, student, teacher


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Ergasia API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

for r in (auth.router, student.router, teacher.router, jobs.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True, "pipeline": {c: settings.impl(c) for c in ("RETRIEVER", "GENERATOR", "GRADER", "KB")},
            "mastery": settings.MASTERY_BACKEND}


# Serve the frontend from the same server so `uvicorn app.main:app` runs the whole product.
if settings.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=settings.FRONTEND_DIR, html=True), name="frontend")
