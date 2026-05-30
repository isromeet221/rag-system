"""KRUTRIM Part B — FastAPI entry.

Run from repository root (parent of `parta/` and `partb/`):

  pip install -r partb/requirements.txt
  set RAG_PARTA_BASE_DIR=%CD%\\parta
  set PYTHONPATH=%CD%
  uvicorn partb.app:app --host 0.0.0.0 --port 9000
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from partb.config import MONGO_DB,PARTA_DIR
from partb.db import get_mongo
from partb.routers.auth_router import router as auth_router
from partb.routers.chats_router import router as chats_router
from partb.routers.meta_router import router as meta_router
from partb.logger import time_it, async_time_it

from partb.routers.pdf_router import router as pdf_router


@asynccontextmanager
@async_time_it
async def lifespan(app: FastAPI):
     
    if str(PARTA_DIR) not in sys.path:
        sys.path.insert(0, str(PARTA_DIR))

    print("[KRUTRIM] Warming models and DB connections…")
    if not PARTA_DIR.exists():
        print("[KRUTRIM] parta dir not found: {PARTA_DIR}")
    else:
        print("[KRUTRIM] parta dir: {PARTA_DIR}")

    try:
        db = get_mongo()[MONGO_DB]
        db["chats"].create_index([("user_id", 1), ("updated_at", -1)])
        db["chats"].create_index("chat_id", unique=True)
        db["messages"].create_index([("chat_id", 1), ("created_at", 1)])
        db["messages"].create_index("message_id", unique=True)
        db["users"].create_index("email", unique=True)
        print("[KRUTRIM] Part B ready.")
    except Exception as exc:
        print(f"[KRUTRIM] ERROR: {exc}")

    from partb.retrieval.pipeline import warm_models
    print("[KRUTRIM] Warming models")
    asyncio.create_task(asyncio.to_thread(warm_models))
    print("[KRUTRIM] ")
    yield
    print("[KRUTRIM] shutting down ")

app = FastAPI(title="KRUTRIM Part B", version="2.0.0", lifespan=lifespan, docs_url = "/api/docs", redoc_url = "/api/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_BASE = Path(__file__).resolve().parent
FRONTEND_DIR = _BASE / "frontend"
STATIC_DIR = _BASE / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(meta_router)
app.include_router(pdf_router)

@app.get("/", response_class=HTMLResponse)
@async_time_it
async def serve_frontend():
    html_path = FRONTEND_DIR / "chat.html"
    if html_path.is_file():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>frontend/chat.html not found</h2>", status_code=404)