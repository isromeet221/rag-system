"""
main_api.py
-----------
Central FastAPI application. Runs on port 8000 on the Master Node.

Endpoints:
  POST /auth/signup            → Create account
  POST /auth/login             → Get JWT token
  POST /upload_book            → Upload PDF + Book ID (protected)
  GET  /progress/{job_id}      → SSE stream of pipeline progress (protected)
  GET  /library                → List all completed books (protected)
  GET  /library/check          → Check if a book_id already exists (protected)
  GET  /pending_jobs           → Jobs resumable (extraction_done/ingestion_failed)
  POST /resume/{job_id}        → Re-queue Phase 2 for a resumable job (protected)
  GET  /jobs/{job_id}          → Simple JSON status snapshot (protected)
  GET  /health                 → Health check

FIXES vs original provided file:
  1. Resume endpoint reads "ready_path" + "prop_path" from job doc
     (was "chunks_path" — pipeline_controller saves ready_path/prop_path,
      not chunks_path, so the resume check always failed with "no path recorded")
  2. Resume validates BOTH checkpoint files exist before re-queuing
  3. TERMINAL_STATUSES updated to match pipeline_controller status strings
     ("completed" not "done", "extraction_failed"/"ingestion_failed" not "failed")

Run with: uvicorn main_api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt as pyjwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from parta.logger import async_time_it, logger, time_it

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# Load .env from partb directory
load_dotenv(BASE_DIR.parent / "partb" / ".env")

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URI = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = "rag_system"

JWT_SECRET = "ISRO_RAG_SECRET_CHANGE_IN_PROD"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

# Statuses that mean a pipeline is already active for this book
ACTIVE_STATUSES = ("queued", "extracting", "extraction_done", "ingesting")

# Statuses from which Phase 2 can be re-queued via /resume
RESUMABLE_STATUSES = ("extraction_done", "ingestion_failed")

# Statuses where the SSE stream stops sending events
# FIX: pipeline_controller uses "completed" not "done",
#      and "extraction_failed"/"ingestion_failed" not "failed"
TERMINAL_STATUSES = ("completed", "extraction_failed", "ingestion_failed")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
users_col = mongo_db["users"]
jobs_col = mongo_db["jobs"]
library_col = mongo_db["library"]

users_col.create_index("email", unique=True)
jobs_col.create_index("job_id", unique=True)
jobs_col.create_index([("uploaded_by", 1), ("status", 1)])
library_col.create_index("book_id", unique=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ISRO RAG System", version="2.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


@time_it
def create_token(user_id: str, name: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@time_it
def verify_token(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = auth[7:]
    try:
        return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired. Please login again.")
    except pyjwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token.")


# ---------------------------------------------------------------------------
# Background job queue
# ---------------------------------------------------------------------------
pipeline_queue: queue.Queue = queue.Queue()


@time_it
def queue_worker():
    """
    Background thread — picks jobs one at a time.
    phase2_only jobs call run_phase2 directly (resume path).
    """
    from pipeline_controller import run_phase2, run_pipeline

    while True:
        try:
            job = pipeline_queue.get(block=True, timeout=5)
            logger.info(
                "Queue job | id=%s | book=%s | phase=%s",
                job["job_id"],
                job["book_id"],
                job.get("phase", "full"),
            )
            try:
                if job.get("phase") == "phase2_only":
                    run_phase2(job, jobs_col, mongo_db)
                else:
                    run_pipeline(job, jobs_col, mongo_db)
            except Exception as e:
                logger.error("Unhandled queue error for %s: %s", job["job_id"], e)
            finally:
                pipeline_queue.task_done()
        except queue.Empty:
            continue


threading.Thread(target=queue_worker, daemon=True).start()
logger.info("Background queue worker started.")


# ---------------------------------------------------------------------------
# Routes: frontend
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
@time_it
def serve_frontend():
    html_file = BASE_DIR / "frontend" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Place index.html in /frontend/</h1>")


# ---------------------------------------------------------------------------
# Routes: auth
# ---------------------------------------------------------------------------


@app.post("/auth/signup")
@time_it
def signup(body: dict):
    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not name or not email or not password:
        raise HTTPException(400, "Name, email, and password are required.")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if "@" not in email:
        raise HTTPException(400, "Invalid email address.")

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id = str(uuid.uuid4())

    try:
        users_col.insert_one(
            {
                "user_id": user_id,
                "name": name,
                "email": email,
                "password": hashed,
                "role": "admin",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except DuplicateKeyError:
        raise HTTPException(409, "An account with this email already exists.")

    return {"message": "Account created successfully. Please login."}


@app.post("/auth/login")
@time_it
def login(body: dict):
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    user = users_col.find_one({"email": email}, {"_id": 0})
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(401, "Invalid email or password.")

    token = create_token(user["user_id"], user["name"], user["email"])
    return {
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
        },
    }


# ---------------------------------------------------------------------------
# Routes: book upload
# ---------------------------------------------------------------------------


@app.get("/library/check")
@time_it
def check_book_id(book_id: str, user=Depends(verify_token)):
    exists = library_col.find_one({"book_id": book_id}) is not None
    running = (
        jobs_col.find_one(
            {"book_id": book_id, "status": {"$in": list(ACTIVE_STATUSES)}}
        )
        is not None
    )
    return {"book_id": book_id, "exists": exists, "running": running}


@app.post("/upload_book")
@async_time_it
async def upload_book(
    book_id: str = Form(...),
    file: UploadFile = File(...),
    ocr_enabled: bool = Form(False),
    user: dict = Depends(verify_token),
):
    if not re.match(r"^[A-Za-z0-9_\-]+$", book_id):
        raise HTTPException(
            400, "Book ID can only contain letters, numbers, hyphens, and underscores."
        )

    if library_col.find_one({"book_id": book_id}):
        raise HTTPException(409, f"Book ID '{book_id}' already exists in the library.")

    if jobs_col.find_one(
        {"book_id": book_id, "status": {"$in": list(ACTIVE_STATUSES)}}
    ):
        raise HTTPException(409, f"A job for '{book_id}' is already in progress.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    pdf_path = DATA_RAW_DIR / f"{book_id}.pdf"
    content = await file.read()
    with open(pdf_path, "wb") as fh:
        fh.write(content)

    job_id = str(uuid.uuid4())
    queue_position = pipeline_queue.qsize() + 1

    job_doc = {
        "job_id": job_id,
        "book_id": book_id,
        "uploaded_by": user["user_id"],
        "uploaded_by_name": user.get("name", ""),
        "status": "queued",
        "percent": 0,
        "stage": "Queued",
        "message": f"Position {queue_position} in queue",
        "queue_position": queue_position,
        "ocr_enabled": ocr_enabled,
        # Checkpoint paths — written by Phase 1 before Phase 2 starts
        "ready_path": None,
        "prop_path": None,
        "qdrant_progress": {"status": "pending", "percent": 0},
        "neo4j_progress": {"status": "pending", "percent": 0},
        "confidence_report": None,
        "started_at": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }
    jobs_col.insert_one(job_doc)

    pipeline_queue.put(
        {
            "job_id": job_id,
            "book_id": book_id,
            "pdf_path": str(pdf_path),
            "user_id": user["user_id"],
            "ocr_enabled": ocr_enabled,
        }
    )

    logger.info(
        "Upload accepted | book=%s | job=%s | queue_pos=%d",
        book_id,
        job_id,
        queue_position,
    )
    return {
        "job_id": job_id,
        "book_id": book_id,
        "status": "queued",
        "queue_position": queue_position,
        "message": "Upload received. Pipeline will start shortly.",
    }


# ---------------------------------------------------------------------------
# Routes: SSE progress stream
# ---------------------------------------------------------------------------


@app.get("/progress/{job_id}")
@async_time_it
async def progress_stream(job_id: str, request: Request):
    """Server-Sent Events stream. Polls MongoDB every 2 s until terminal status."""

    async def event_generator():
        while True:
            try:
                if await request.is_disconnected():
                    break
            except Exception:
                pass

            doc = jobs_col.find_one({"job_id": job_id}, {"_id": 0, "password": 0})
            if not doc:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return

            payload = {
                "job_id": doc.get("job_id"),
                "book_id": doc.get("book_id"),
                "status": doc.get("status"),
                "percent": doc.get("percent", 0),
                "stage": doc.get("stage", ""),
                "message": doc.get("message", ""),
                "qdrant_progress": doc.get("qdrant_progress", {}),
                "neo4j_progress": doc.get("neo4j_progress", {}),
                "confidence_report": doc.get("confidence_report"),
                "error": doc.get("error"),
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if doc.get("status") in TERMINAL_STATUSES:
                return

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# Routes: pending jobs + resume
# ---------------------------------------------------------------------------


@app.get("/pending_jobs")
@time_it
def pending_jobs(user=Depends(verify_token)):
    """
    Returns jobs in resumable states (extraction_done or ingestion_failed).
    These can be re-queued via POST /resume/{job_id}.
    """
    docs = list(
        jobs_col.find(
            {"status": {"$in": list(RESUMABLE_STATUSES)}},
            {"_id": 0, "password": 0},
        )
    )
    for d in docs:
        d.pop("confidence_report", None)
    return {"jobs": docs, "count": len(docs)}


@app.post("/resume/{job_id}")
@time_it
def resume_job(job_id: str, user=Depends(verify_token)):
    """
    Re-queues Phase 2 for a job in a resumable state.

    WHAT HAPPENS:
      1. Validates job exists and is in a resumable status
      2. Reads ready_path + prop_path from the job doc (written by Phase 1)
      3. Verifies both checkpoint files exist on disk
      4. Resets qdrant/neo4j progress for stages not yet "done"
         (stages already "done" are preserved — they will be skipped in Phase 2)
      5. Re-queues with phase="phase2_only"

    FIX: was reading "chunks_path" — pipeline_controller saves "ready_path"
         and "prop_path". The old code always raised "no path recorded" because
         "chunks_path" was never written to the job doc.
    """
    doc = jobs_col.find_one({"job_id": job_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    if doc["status"] not in RESUMABLE_STATUSES:
        raise HTTPException(
            400,
            f"Job '{job_id}' is not resumable (status: {doc['status']}). "
            f"Only jobs with status {RESUMABLE_STATUSES} can be resumed.",
        )

    # FIX: read the correct field names
    ready_path = doc.get("ready_path", "")
    prop_path = doc.get("prop_path", "")

    if not ready_path or not prop_path:
        raise HTTPException(
            400,
            f"Job '{job_id}' has no checkpoint paths recorded. "
            "Phase 1 may not have completed successfully. "
            "Check that ready_path and prop_path are saved on the job document.",
        )

    # Verify both files exist on disk
    missing = []
    for label, path in [("ready_path", ready_path), ("prop_path", prop_path)]:
        if not Path(path).exists():
            missing.append(f"{label}: {Path(path).name}")

    if missing:
        raise HTTPException(
            400,
            "Checkpoint file(s) not found on disk: "
            + ", ".join(missing)
            + ". Phase 1 must be re-run before resuming.",
        )

    # Build the progress reset:
    # Stages already "done" keep their status — Phase 2 will skip them.
    # Stages that "failed" or "pending" are reset to pending for a fresh attempt.
    current_qdrant = doc.get("qdrant_progress", {})
    current_neo4j = doc.get("neo4j_progress", {})

    new_qdrant = (
        current_qdrant
        if current_qdrant.get("status") == "done"
        else {"status": "pending", "percent": 0}
    )
    new_neo4j = (
        current_neo4j
        if current_neo4j.get("status") == "done"
        else {"status": "pending", "percent": 0}
    )

    skipping = []
    resuming = []
    if current_qdrant.get("status") == "done":
        skipping.append("Qdrant (already done)")
    else:
        resuming.append("Qdrant")
    if current_neo4j.get("status") == "done":
        skipping.append("Neo4j (already done)")
    else:
        resuming.append("Neo4j")

    resume_msg = (
        f"Resuming Phase 2. "
        + (f"Re-running: {', '.join(resuming)}. " if resuming else "")
        + (f"Skipping: {', '.join(skipping)}." if skipping else "")
    )

    jobs_col.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "queued",
                "percent": 65,
                "stage": "Queued for ingestion",
                "message": resume_msg,
                "error": None,
                "qdrant_progress": new_qdrant,
                "neo4j_progress": new_neo4j,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

    pipeline_queue.put(
        {
            "job_id": job_id,
            "book_id": doc["book_id"],
            "pdf_path": str(DATA_RAW_DIR / f"{doc['book_id']}.pdf"),
            "user_id": doc.get("uploaded_by", ""),
            "ready_path": ready_path,
            "prop_path": prop_path,
            "phase": "phase2_only",
        }
    )

    logger.info(
        "Resume queued | book=%s | job=%s | resuming=%s | skipping=%s",
        doc["book_id"],
        job_id,
        resuming,
        skipping,
    )
    return {
        "job_id": job_id,
        "book_id": doc["book_id"],
        "status": "queued",
        "message": resume_msg,
        "resuming": resuming,
        "skipping": skipping,
    }


# ---------------------------------------------------------------------------
# Routes: library + job status
# ---------------------------------------------------------------------------


@app.get("/library")
@time_it
def get_library(user=Depends(verify_token)):
    books = list(library_col.find({"status": "ready"}, {"_id": 0}))
    return {"books": books, "count": len(books)}


@app.get("/jobs/{job_id}")
@time_it
def get_job_status(job_id: str, user=Depends(verify_token)):
    doc = jobs_col.find_one({"job_id": job_id}, {"_id": 0, "password": 0})
    if not doc:
        raise HTTPException(404, "Job not found")
    return doc


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
@time_it
def health():
    return {
        "status": "ok",
        "queue_size": pipeline_queue.qsize(),
        "mongo": "connected",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
