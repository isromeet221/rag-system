"""
extraction/extraction_server.py
--------------------------------
Unified pull-based job server for:
  - Extraction
  - Neo4j ingestion
  - Qdrant ingestion

Architecture:
  - Workers pull jobs
  - Each job has a lease_deadline
  - Expired lease returns job to PENDING
  - Idempotent submit_result
  - Per-job attempt_count caps retries

Run with:
    uvicorn extraction.extraction_server:app --host 0.0.0.0 --port 8004
"""

import sys
import uuid
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pypdf import PdfReader, PdfWriter

from parta.logger import async_time_it, logger, time_it

# ── Batch sizes for distributed workers ───────────────────────────────────────
# Each connected worker gets one batch. More workers = more parallelism.
NEO4J_BATCH_SIZE  = 30   # chunks per neo4j worker batch
QDRANT_BATCH_SIZE = 200  # items per qdrant worker batch

app = FastAPI(title="Unified Job Server")

# ── Modal Cloud Workers Config ────────────────────────────────────────────────
SPAWN_MODAL_WORKERS = True
MODAL_WORKERS_PER_JOB = 5
# Replace this with your Ngrok or public URL so the cloud workers can reach this local server (e.g., "https://xyz.ngrok-free.app")
PUBLIC_SERVER_URL = "https://jalisa-unreputed-cleta.ngrok-free.dev"
# ──────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 10
LEASE_SECONDS = 600
MAX_ATTEMPTS = 3
CLEANUP_DELAY_SEC = 300
# ponytail: hardcoded priority list; move to env/config only if it changes often.
WORKER_PRIORITY_IPS = [
    "192.168.1.10",
    "192.168.1.11",
    "192.168.1.12",
    "192.168.1.13",
    "192.168.1.14",
    "192.168.1.15",
    "192.168.1.16",
    "192.168.1.17",
    "192.168.1.18",
    "192.168.1.19",
]
PRIORITY_GRACE_SECONDS = 10

# ── In-memory state ───────────────────────────────────────────────────────────
# Each store maps book_id -> state dict
extractions: Dict[str, dict] = {}
neo4j_jobs: Dict[str, dict] = {}
qdrant_jobs: Dict[str, dict] = {}
job_lock = threading.Lock()
worker_last_seen: Dict[str, float] = {}
worker_ip_map: Dict[str, str] = {}  # worker_id → IP for busy/idle tracking


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _job_row(
    *,
    job_id: str,
    book_id: str,
    job_kind: str,
    status: str = "PENDING",
    chunk_idx: Optional[int] = None,
    start_offset: int = 0,
    page_count: int = 0,
    chunk_path: Optional[str] = None,
    assigned_to: Optional[str] = None,
    assigned_at: float = 0.0,
    lease_deadline: float = 0.0,
    attempt_count: int = 0,
    result: Optional[str] = None,
) -> dict:
    return {
        "job_id": job_id,
        "book_id": book_id,
        "job_kind": job_kind,
        "chunk_idx": chunk_idx,
        "start_offset": start_offset,
        "page_count": page_count,
        "chunk_path": chunk_path,
        "status": status,
        "assigned_to": assigned_to,
        "assigned_at": assigned_at,
        "lease_deadline": lease_deadline,
        "attempt_count": attempt_count,
        "result": result,
    }


def _new_book_state(*, total: int, meta: Optional[dict] = None) -> dict:
    return {
        "jobs": {},
        "queue": [],
        "total": total,
        "completed": 0,
        "failed": 0,
        "is_finished": False,
        "started_at": time.time(),
        "meta": meta or {},
    }


def _reject_if_active(store: Dict[str, dict], book_id: str, label: str):
    existing = store.get(book_id)
    if existing and not existing["is_finished"]:
        raise HTTPException(
            409,
            f"{label} for '{book_id}' is already running "
            f"({existing['completed']}/{existing['total']} done).",
        )


def _finish_check(store: Dict[str, dict], book_id: str, cleanup_fn=None):
    state = store.get(book_id)
    if not state:
        return

    done = state["completed"] + state["failed"]
    if done >= state["total"] and not state["is_finished"]:
        state["is_finished"] = True
        label = "FAILED" if state["failed"] > 0 else "COMPLETE"
        logger.info(
            "%s %s — '%s' (%d/%d)",
            label,
            state["meta"].get("label", "JOB"),
            book_id,
            state["completed"],
            state["total"],
        )
        if cleanup_fn:
            threading.Thread(
                target=cleanup_fn,
                args=(state["meta"],),
                daemon=True,
            ).start()


def _single_job_state(*, book_id: str, job_kind: str, meta: Optional[dict] = None) -> dict:
    jid = str(uuid.uuid4())
    state = _new_book_state(total=1, meta={"label": job_kind.upper(), **(meta or {})})
    state["jobs"][jid] = _job_row(job_id=jid, book_id=book_id, job_kind=job_kind)
    state["queue"] = [jid]
    return state


def _make_wait_or_shutdown() -> dict:
    # Keep workers alive for later jobs. No shutdown games.
    return {"action": "WAIT"}


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _assign_pending_job(
    *,
    store: Dict[str, dict],
    worker_id: str,
    worker_ip: str = "unknown",
    expected_kind: Optional[str] = None,
    extra_response: Optional[dict] = None,
) -> dict:
    now = time.time()
    if expected_kind:
        worker_last_seen[f"{expected_kind}:{worker_ip}"] = now

    if expected_kind and WORKER_PRIORITY_IPS:
        rank = WORKER_PRIORITY_IPS.index(worker_ip) if worker_ip in WORKER_PRIORITY_IPS else len(WORKER_PRIORITY_IPS)
        active_higher_priority = sum(
            now - worker_last_seen.get(f"{expected_kind}:{ip}", 0) <= PRIORITY_GRACE_SECONDS
            for ip in WORKER_PRIORITY_IPS[:rank]
        )
        # ── Don't count busy higher-priority workers as available ────────
        busy_ips: set = set()
        live_assigned = set()
        for state in store.values():
            if state["is_finished"]:
                continue
            for job in state["jobs"].values():
                if job["status"] == "PROCESSING" and job["assigned_to"]:
                    live_assigned.add(job["assigned_to"])
                    ip = worker_ip_map.get(job["assigned_to"])
                    if ip and ip in WORKER_PRIORITY_IPS[:rank]:
                        busy_ips.add(ip)
        # Drop stale worker_ip_map entries for workers no longer assigned.
        for wid in list(worker_ip_map.keys()):
            if wid not in live_assigned:
                del worker_ip_map[wid]
        idle_higher_priority = active_higher_priority - len(busy_ips)
        pending_jobs = sum(
            1
            for state in store.values()
            if not state["is_finished"]
            for job in state["jobs"].values()
            if job["status"] == "PENDING" and job["job_kind"] == expected_kind
        )
        if idle_higher_priority > 0 and pending_jobs <= idle_higher_priority:
            logger.info(
                "%s worker %s (%s) waiting for idle higher-priority worker",
                expected_kind.upper(),
                worker_id,
                worker_ip,
            )
            return _make_wait_or_shutdown()

    for book_id, state in store.items():
        if state["is_finished"]:
            continue

        jobs = state["jobs"]
        queue = state["queue"]

        # Rescue expired leases
        for jid, job in jobs.items():
            if job["status"] == "PROCESSING" and now > job["lease_deadline"]:
                job["attempt_count"] += 1
                logger.warning(
                    "%s lease expired for %s (held by %s, attempt %d/%d)",
                    job["job_kind"].upper(),
                    jid,
                    job["assigned_to"],
                    job["attempt_count"],
                    MAX_ATTEMPTS,
                )

                if job["attempt_count"] >= MAX_ATTEMPTS:
                    job["status"] = "FAILED"
                    state["failed"] += 1
                    logger.error(
                        "%s job %s permanently failed after %d attempts",
                        job["job_kind"].upper(),
                        jid,
                        MAX_ATTEMPTS,
                    )
                    _finish_check(store, book_id)
                else:
                    job["status"] = "PENDING"
                    job["assigned_to"] = None
                    job["assigned_at"] = 0
                    job["lease_deadline"] = 0

        for jid in queue:
            job = jobs[jid]
            if expected_kind and job["job_kind"] != expected_kind:
                continue
            if job["status"] != "PENDING":
                continue

            job["status"] = "PROCESSING"
            job["assigned_to"] = worker_id
            job["assigned_at"] = now
            job["lease_deadline"] = now + LEASE_SECONDS
            worker_ip_map[worker_id] = worker_ip

            logger.info(
                "%s job %s assigned to %s (%s)",
                job["job_kind"].upper(),
                jid,
                worker_id,
                worker_ip,
            )

            response = {
                "action": "PROCESS",
                "job_id": jid,
                "book_id": book_id,
                "worker_id": worker_id,
                "job_kind": job["job_kind"],
                "attempt_count": job["attempt_count"],
            }
            if job["chunk_idx"] is not None:
                response["chunk_idx"] = job["chunk_idx"]
                response["start_offset"] = job["start_offset"]
                response["page_count"] = job["page_count"]
            if job.get("chunk_path"):
                response["chunk_path"] = job["chunk_path"]
            response["ocr_enabled"] = state.get("meta", {}).get("ocr_enabled", False)
            response.update(extra_response or {})
            return response

    return _make_wait_or_shutdown()


def _submit_result(
    *,
    store: Dict[str, dict],
    payload: dict,
    require_content: bool = False,
    cleanup_fn=None,
) -> dict:
    jid = payload.get("job_id")
    worker_id = payload.get("worker_id", "unknown")
    success = bool(payload.get("success", False))
    content = payload.get("content", "")

    if not jid:
        raise HTTPException(400, "job_id is required")

    with job_lock:
        for book_id, state in store.items():
            if jid not in state["jobs"]:
                continue

            job = state["jobs"][jid]

            if job["status"] == "COMPLETED":
                return {"status": "ok", "note": "already completed, ignored"}

            if success and (content or not require_content):
                job["status"] = "COMPLETED"
                job["result"] = content
                state["completed"] += 1
                logger.info(
                    "%s job %s done by %s [%d/%d]",
                    job["job_kind"].upper(),
                    jid,
                    worker_id,
                    state["completed"],
                    state["total"],
                )
            else:
                job["attempt_count"] += 1
                error_msg = payload.get("error", "Unknown error")
                logger.warning(
                    "%s job %s failed by %s (attempt %d/%d) Error: %s",
                    job["job_kind"].upper(),
                    jid,
                    worker_id,
                    job["attempt_count"],
                    MAX_ATTEMPTS,
                    error_msg
                )

                if job["attempt_count"] >= MAX_ATTEMPTS:
                    job["status"] = "FAILED"
                    state["failed"] += 1
                    logger.error(
                        "%s job %s permanently failed after %d attempts",
                        job["job_kind"].upper(),
                        jid,
                        MAX_ATTEMPTS,
                    )
                else:
                    job["status"] = "PENDING"
                    job["assigned_to"] = None
                    job["assigned_at"] = 0
                    job["lease_deadline"] = 0

            _finish_check(store, book_id, cleanup_fn=cleanup_fn)
            return {"status": "ok"}

    raise HTTPException(404, f"Job {jid} not found")


def _status_response(state: Optional[dict], *, book_id: str, failed_label: str) -> dict:
    if not state:
        return {"status": "not_found", "book_id": book_id}

    failed_jobs = [
        jid for jid in state["queue"]
        if state["jobs"][jid]["status"] == "FAILED"
    ]

    overall = "running"
    if state["is_finished"]:
        overall = "failed" if state["failed"] > 0 else "completed"

    return {
        "book_id": book_id,
        "status": overall,
        "total": state["total"],
        "completed": state["completed"],
        "failed": state["failed"],
        "is_finished": state["is_finished"],
        "percent": int(state["completed"] / max(state["total"], 1) * 100),
        failed_label: failed_jobs,
    }


def _cleanup_after_delay(meta: dict):
    chunk_dir = meta.get("chunk_dir")
    if not chunk_dir:
        return
    time.sleep(CLEANUP_DELAY_SEC)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    logger.info("Cleaned temp dir: %s", chunk_dir)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_extraction")
@time_it
def start_extraction(payload: dict):
    book_id = payload.get("book_id")
    pdf_path = payload.get("pdf_path")
    base_dir = payload.get("base_dir", ".")
    ocr_enabled = bool(payload.get("ocr_enabled", False))

    if not book_id or not pdf_path:
        raise HTTPException(400, "book_id and pdf_path are required")
    if not Path(pdf_path).exists():
        raise HTTPException(404, f"PDF not found: {pdf_path}")

    with job_lock:
        _reject_if_active(extractions, book_id, "Extraction")

    chunk_dir = Path(base_dir) / f"temp_extract_{book_id}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    jobs = {}
    queue_order = []

    logger.info("Splitting %d pages for '%s'...", total_pages, book_id)

    for chunk_idx, start in enumerate(range(0, total_pages, CHUNK_SIZE)):
        end = min(start + CHUNK_SIZE, total_pages)

        writer = PdfWriter()
        for pg in range(start, end):
            writer.add_page(reader.pages[pg])

        chunk_path = chunk_dir / f"chunk_{chunk_idx}.pdf"
        with open(str(chunk_path), "wb") as f:
            writer.write(f)

        jid = str(uuid.uuid4())
        jobs[jid] = _job_row(
            job_id=jid,
            book_id=book_id,
            job_kind="extraction",
            chunk_idx=chunk_idx,
            start_offset=start,
            page_count=end - start,
            chunk_path=str(chunk_path),
        )
        queue_order.append(jid)

    state = _new_book_state(
        total=len(queue_order),
        meta={
            "label": "EXTRACTION",
            "chunk_dir": str(chunk_dir),
            "total_pages": total_pages,
            "ocr_enabled": ocr_enabled,
        },
    )
    state["jobs"] = jobs
    state["queue"] = queue_order

    with job_lock:
        extractions[book_id] = state

    logger.info("%d extraction chunks queued for '%s'", state["total"], book_id)

    if SPAWN_MODAL_WORKERS:
        try:
            import modal
            text_worker = modal.Function.from_name("rag-workers", "run_text_worker")
            logger.info(f"Spawning {MODAL_WORKERS_PER_JOB} Text workers on Modal...")
            for _ in range(MODAL_WORKERS_PER_JOB):
                text_worker.spawn(PUBLIC_SERVER_URL)
        except Exception as e:
            logger.error(f"Failed to spawn Modal workers: {e}")

    return {
        "status": "started",
        "book_id": book_id,
        "total_chunks": state["total"],
        "total_pages": total_pages,
        "ocr_enabled": ocr_enabled,
    }


@app.get("/get_job")
@time_it
def get_job(request: Request, worker_id: str = "unknown"):
    with job_lock:
        return _assign_pending_job(
            store=extractions,
            worker_id=worker_id,
            worker_ip=_client_ip(request),
            expected_kind="extraction",
        )


@app.get("/chunk/{job_id}")
@time_it
def get_chunk_binary(job_id: str):
    with job_lock:
        chunk_path = None
        for state in extractions.values():
            if job_id in state["jobs"]:
                chunk_path = state["jobs"][job_id]["chunk_path"]
                break

    if not chunk_path:
        raise HTTPException(404, f"Job {job_id} not found")
    if not Path(chunk_path).exists():
        raise HTTPException(404, f"Chunk file missing for job {job_id}")

    return FileResponse(
        path=chunk_path,
        media_type="application/octet-stream",
        filename=f"chunk_{job_id}.pdf",
    )


@app.post("/submit_result")
@time_it
def submit_result(payload: dict):
    return _submit_result(
        store=extractions,
        payload=payload,
        require_content=True,
        cleanup_fn=_cleanup_after_delay,
    )


@app.get("/extraction_status/{book_id}")
@time_it
def extraction_status(book_id: str):
    with job_lock:
        state = extractions.get(book_id)
    return _status_response(state, book_id=book_id, failed_label="failed_chunks")


@app.get("/get_result/{book_id}")
@time_it
def get_result(book_id: str):
    with job_lock:
        state = extractions.get(book_id)

    if not state:
        raise HTTPException(404, f"No extraction for '{book_id}'")
    if not state["is_finished"]:
        raise HTTPException(400, "Extraction not finished yet")

    failed = [
        state["jobs"][jid]["chunk_idx"]
        for jid in state["queue"]
        if state["jobs"][jid]["status"] == "FAILED"
    ]
    if failed:
        raise HTTPException(
            500,
            f"Chunks {failed} permanently failed after {MAX_ATTEMPTS} attempts.",
        )

    full_text = f"# Text Extraction: {book_id}\n\n"
    for jid in state["queue"]:
        full_text += state["jobs"][jid]["result"] or ""

    return {"book_id": book_id, "content": full_text}


@app.get("/download_ready/{book_id}")
def download_ready(book_id: str):
    with job_lock:
        state = neo4j_jobs.get(book_id) or qdrant_jobs.get(book_id)
    if not state:
        raise HTTPException(404, f"No job for {book_id}")
    path = state["meta"].get("ready_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, f"ready_path missing for {book_id}")
    return FileResponse(path=path, media_type="application/json", filename=f"{book_id}_ready.json")


@app.get("/download_prop/{book_id}")
def download_prop(book_id: str):
    with job_lock:
        state = qdrant_jobs.get(book_id)
    if not state:
        raise HTTPException(404, f"No qdrant job for {book_id}")
    path = state["meta"].get("prop_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, f"prop_path missing for {book_id}")
    return FileResponse(path=path, media_type="application/json", filename=f"{book_id}_prop.json")



# ─────────────────────────────────────────────────────────────────────────────
# NEO4J ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_neo4j")
@time_it
def start_neo4j(payload: dict):
    book_id = payload.get("book_id")
    ready_path = payload.get("ready_path")
    base_dir = payload.get("base_dir", ".")

    if not book_id or not ready_path:
        raise HTTPException(400, "book_id and ready_path are required")
    if not Path(ready_path).exists():
        raise HTTPException(404, f"Ready path not found: {ready_path}")

    with job_lock:
        _reject_if_active(neo4j_jobs, book_id, "Neo4j ingestion")

    # ── Read chunk count and split into batches for parallel workers ───────
    with open(ready_path, "r", encoding="utf-8") as f:
        total_chunks = len(json.load(f))

    batch_size = NEO4J_BATCH_SIZE
    jobs = {}
    queue_order = []

    for batch_idx, start in enumerate(range(0, total_chunks, batch_size)):
        end = min(start + batch_size, total_chunks)
        jid = str(uuid.uuid4())
        jobs[jid] = _job_row(
            job_id=jid, book_id=book_id, job_kind="neo4j",
            chunk_idx=batch_idx, start_offset=start, page_count=end - start,
        )
        queue_order.append(jid)

    state = _new_book_state(
        total=len(queue_order),
        meta={"label": "NEO4J", "ready_path": ready_path, "base_dir": base_dir,
              "total_chunks": total_chunks},
    )
    state["jobs"] = jobs
    state["queue"] = queue_order

    with job_lock:
        neo4j_jobs[book_id] = state

    logger.info(
        "Neo4j split into %d batches (%d chunks) for '%s'",
        len(queue_order), total_chunks, book_id,
    )

    if SPAWN_MODAL_WORKERS:
        try:
            import modal
            neo4j_worker = modal.Function.from_name("rag-workers", "run_neo4j_worker")
            logger.info(f"Spawning {MODAL_WORKERS_PER_JOB} Neo4j workers on Modal...")
            for _ in range(MODAL_WORKERS_PER_JOB):
                neo4j_worker.spawn(PUBLIC_SERVER_URL)
        except Exception as e:
            logger.error(f"Failed to spawn Modal workers: {e}")

    return {
        "status": "started",
        "book_id": book_id,
        "total_batches": len(queue_order),
        "total_chunks": total_chunks,
        "batch_size": batch_size,
        "ready_path": ready_path,
    }


@app.get("/get_neo4j_job")
@time_it
def get_neo4j_job(request: Request, worker_id: str = "unknown"):
    with job_lock:
        return _assign_pending_job(
            store=neo4j_jobs,
            worker_id=worker_id,
            worker_ip=_client_ip(request),
            expected_kind="neo4j",
        )


@app.post("/submit_neo4j_result")
@time_it
def submit_neo4j_result(payload: dict):
    return _submit_result(store=neo4j_jobs, payload=payload, require_content=True)


@app.get("/neo4j_status/{book_id}")
@time_it
def neo4j_status(book_id: str):
    with job_lock:
        state = neo4j_jobs.get(book_id)
    return _status_response(state, book_id=book_id, failed_label="failed_jobs")


@app.get("/get_neo4j_result/{book_id}")
@time_it
def get_neo4j_result(book_id: str):
    with job_lock:
        state = neo4j_jobs.get(book_id)
    if not state:
        raise HTTPException(404, f"No neo4j job for '{book_id}'")
    if not state["is_finished"]:
        raise HTTPException(400, "Neo4j job not finished yet")
    if state["failed"] > 0:
        raise HTTPException(500, "Neo4j job failed permanently")

    # ── Aggregate results from all batch workers ──────────────────────────
    totals = {
        "sections_written": 0, "entities_written": 0,
        "specs_written": 0, "tables_written": 0,
        "cooccurrence_edges": 0, "elapsed_seconds": 0,
    }
    for jid in state["queue"]:
        r = state["jobs"][jid].get("result")
        if r and isinstance(r, dict):
            for k in totals:
                totals[k] += r.get(k, 0)
    return totals


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_qdrant")
@time_it
def start_qdrant(payload: dict):
    book_id = payload.get("book_id")
    ready_path = payload.get("ready_path")
    prop_path = payload.get("prop_path")
    base_dir = payload.get("base_dir", ".")

    if not book_id or not ready_path or not prop_path:
        raise HTTPException(400, "book_id, ready_path and prop_path are required")
    if not Path(ready_path).exists():
        raise HTTPException(404, f"Ready path not found: {ready_path}")
    if not Path(prop_path).exists():
        raise HTTPException(404, f"Prop path not found: {prop_path}")

    with job_lock:
        _reject_if_active(qdrant_jobs, book_id, "Qdrant ingestion")

    # ── Read item counts and split into batches ───────────────────────────
    with open(prop_path, "r", encoding="utf-8") as f:
        total_props = len(json.load(f))
    with open(ready_path, "r", encoding="utf-8") as f:
        total_sections = len(json.load(f))

    batch_size = QDRANT_BATCH_SIZE
    jobs = {}
    queue_order = []

    # Proposition batches
    for batch_idx, start in enumerate(range(0, total_props, batch_size)):
        end = min(start + batch_size, total_props)
        jid = str(uuid.uuid4())
        jobs[jid] = _job_row(
            job_id=jid, book_id=book_id, job_kind="qdrant",
            chunk_idx=batch_idx, start_offset=start, page_count=end - start,
            # Encode batch_kind in chunk_path field (reuse existing field)
            chunk_path="propositions",
        )
        queue_order.append(jid)

    # Section batches
    sec_batch_offset = len(queue_order)
    for batch_idx, start in enumerate(range(0, total_sections, batch_size)):
        end = min(start + batch_size, total_sections)
        jid = str(uuid.uuid4())
        jobs[jid] = _job_row(
            job_id=jid, book_id=book_id, job_kind="qdrant",
            chunk_idx=sec_batch_offset + batch_idx,
            start_offset=start, page_count=end - start,
            chunk_path="sections",
        )
        queue_order.append(jid)

    state = _new_book_state(
        total=len(queue_order),
        meta={
            "label": "QDRANT", "ready_path": ready_path,
            "prop_path": prop_path, "base_dir": base_dir,
            "total_props": total_props, "total_sections": total_sections,
        },
    )
    state["jobs"] = jobs
    state["queue"] = queue_order

    with job_lock:
        qdrant_jobs[book_id] = state

    logger.info(
        "Qdrant split into %d batches (%d props + %d sections) for '%s'",
        len(queue_order), total_props, total_sections, book_id,
    )

    if SPAWN_MODAL_WORKERS:
        try:
            import modal
            qdrant_worker = modal.Function.from_name("rag-workers", "run_qdrant_worker")
            logger.info(f"Spawning {MODAL_WORKERS_PER_JOB} Qdrant workers on Modal...")
            for _ in range(MODAL_WORKERS_PER_JOB):
                qdrant_worker.spawn(PUBLIC_SERVER_URL)
        except Exception as e:
            logger.error(f"Failed to spawn Modal workers: {e}")

    return {
        "status": "started",
        "book_id": book_id,
        "total_batches": len(queue_order),
        "total_props": total_props,
        "total_sections": total_sections,
        "batch_size": batch_size,
        "ready_path": ready_path,
        "prop_path": prop_path,
    }


@app.get("/get_qdrant_job")
@time_it
def get_qdrant_job(request: Request, worker_id: str = "unknown"):
    with job_lock:
        return _assign_pending_job(
            store=qdrant_jobs,
            worker_id=worker_id,
            worker_ip=_client_ip(request),
            expected_kind="qdrant",
        )


@app.post("/submit_qdrant_result")
@time_it
def submit_qdrant_result(payload: dict):
    return _submit_result(store=qdrant_jobs, payload=payload, require_content=True)


@app.get("/qdrant_status/{book_id}")
@time_it
def qdrant_status(book_id: str):
    with job_lock:
        state = qdrant_jobs.get(book_id)
    return _status_response(state, book_id=book_id, failed_label="failed_jobs")


@app.get("/get_qdrant_result/{book_id}")
@time_it
def get_qdrant_result(book_id: str):
    with job_lock:
        state = qdrant_jobs.get(book_id)
    if not state:
        raise HTTPException(404, f"No qdrant job for '{book_id}'")
    if not state["is_finished"]:
        raise HTTPException(400, "Qdrant job not finished yet")
    if state["failed"] > 0:
        raise HTTPException(500, "Qdrant job failed permanently")

    # ── Aggregate stored counts from all batch workers ────────────────────
    total_stored = 0
    for jid in state["queue"]:
        r = state["jobs"][jid].get("result")
        if r and isinstance(r, dict):
            total_stored += r.get("chunks_stored", 0)
        elif r and isinstance(r, (int, float)):
            total_stored += int(r)
    return {
        "chunks_stored": total_stored,
        "chunks_json": [],  # batch mode — chunks.json written locally by workers
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
@time_it
def health():
    with job_lock:
        active = {
            "extraction": {bid: {"completed": s["completed"], "total": s["total"]} for bid, s in extractions.items() if not s["is_finished"]},
            "neo4j": {bid: {"completed": s["completed"], "total": s["total"]} for bid, s in neo4j_jobs.items() if not s["is_finished"]},
            "qdrant": {bid: {"completed": s["completed"], "total": s["total"]} for bid, s in qdrant_jobs.items() if not s["is_finished"]},
        }
    return {"status": "ok", "active": active}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)


