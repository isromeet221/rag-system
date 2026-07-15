"""
qdrant_workers.py — Batch Qdrant ingestion worker.
Polls the server for jobs, downloads data files, runs batch ingestion,
and submits results with retry + local crash-safe caching.
"""

import os
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.logger import log_process, setup_worker_logger
from workers.base_worker import (
    create_session,
    ResultCache,
    submit_with_retry,
)

from parta.processing.ingest_qdrant import run_qdrant_batch

# ─── Constants ────────────────────────────────────────────────────────────────
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8004")
WORKER_ID = f"qdrant-{uuid.uuid4().hex[:6]}"
BASE_DIR = Path(__file__).resolve().parent.parent / "parta"

# ─── Logger setup ─────────────────────────────────────────────────────────────
logger = setup_worker_logger("qdrant", WORKER_ID)

# ─── Persistent session — reuses TCP connection across all poll cycles ────────
_session = create_session()

# ─── Crash-safe result cache ──────────────────────────────────────────────────
_cache = ResultCache("qdrant")

# ─── Startup ──────────────────────────────────────────────────────────────────
logger.info("=" * 80)
logger.info("QDRANT WORKER STARTED (batch mode)")
logger.info("SERVER      : %s", SERVER_URL)
logger.info("BASE_DIR    : %s", BASE_DIR)
logger.info("CACHE_DIR   : %s", _cache.cache_dir)
logger.info("=" * 80)

# ── replay any results cached from a previous run before polling ──────────────
_cache.replay(_session, f"{SERVER_URL}/submit_qdrant_result")

is_connected = False


@log_process
def execute_qdrant_batch(book_id, ready_path, prop_path, batch_start, batch_count, batch_kind):
    return run_qdrant_batch(
        book_id=book_id,
        ready_path=ready_path,
        prop_path=prop_path,
        base_dir=str(BASE_DIR),
        batch_start=batch_start,
        batch_count=batch_count,
        batch_kind=batch_kind,
    )


# ─── Main polling loop ────────────────────────────────────────────────────────
while True:
    job_id = None
    local_ready_path = None
    local_prop_path = None

    try:
        r = _session.get(
            f"{SERVER_URL}/get_qdrant_job",
            params={"worker_id": WORKER_ID},
        )

        if not is_connected:
            logger.info("Connected to server")
            is_connected = True

        if r.status_code != 200:
            logger.error("Failed to get job (HTTP %d)", r.status_code)
            time.sleep(5)
            continue

        job = r.json()

        if job.get("action") != "PROCESS":
            time.sleep(2)
            continue

        job_id = job["job_id"]
        book_id = job["book_id"]
        batch_start = job.get("start_offset", 0)
        batch_count = job.get("page_count", 0)
        batch_idx = job.get("chunk_idx", 0)
        # batch_kind is encoded in chunk_path field by the server
        batch_kind = job.get("chunk_path", "propositions")

        logger.info("\n" + "=" * 80)
        logger.info("NEW QDRANT BATCH JOB")
        logger.info("JOB ID     : %s", job_id)
        logger.info("BOOK ID    : %s", book_id)
        logger.info("KIND       : %s", batch_kind)
        logger.info(
            "BATCH      : #%s (%s %d–%d)",
            batch_idx, batch_kind, batch_start, batch_start + batch_count - 1,
        )
        logger.info("=" * 80)

        # ── download ready.json from server ───────────────────────────────────
        logger.info("Downloading ready file for '%s'...", book_id)
        r2 = _session.get(f"{SERVER_URL}/download_ready/{book_id}")
        if r2.status_code != 200:
            raise RuntimeError(
                f"download_ready failed: HTTP {r2.status_code} — {r2.text[:200]}"
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix="_ready.json", delete=False
        ) as f:
            f.write(r2.content)
            local_ready_path = f.name

        # ── download prop.json from server ────────────────────────────────────
        logger.info("Downloading prop file for '%s'...", book_id)
        r3 = _session.get(f"{SERVER_URL}/download_prop/{book_id}")
        if r3.status_code != 200:
            raise RuntimeError(
                f"download_prop failed: HTTP {r3.status_code} — {r3.text[:200]}"
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix="_prop.json", delete=False
        ) as f:
            f.write(r3.content)
            local_prop_path = f.name

        logger.info(
            "Files ready — running batch ingestion (%s %d–%d)...",
            batch_kind, batch_start, batch_start + batch_count - 1,
        )

        # ── run batch ingestion ────────────────────────────────────────────────
        chunks_stored = execute_qdrant_batch(
            book_id, local_ready_path, local_prop_path, batch_start, batch_count, batch_kind
        )

        # ── build payload, cache locally before attempting submit ─────────────
        payload = {
            "job_id": job_id,
            "worker_id": WORKER_ID,
            "success": True,
            "content": {
                "chunks_stored": chunks_stored,
                "batch_kind": batch_kind,
                "batch_start": batch_start,
                "batch_count": batch_count,
            },
        }
        _cache.store(job_id, payload)
        submit_with_retry(_session, f"{SERVER_URL}/submit_qdrant_result", payload)
        _cache.clear(job_id)
        logger.info("Completion acknowledged for batch #%s (%s)", batch_idx, batch_kind)

    except requests.exceptions.ConnectionError:
        if is_connected:
            logger.error("Disconnected from server. Waiting to reconnect...")
            is_connected = False
        else:
            logger.error("Failed to connect to server at %s. Retrying...", SERVER_URL)
        time.sleep(5)

    except Exception as e:
        logger.error("Error: %s", e)
        logger.error(traceback.format_exc())

        if job_id:
            try:
                _session.post(
                    f"{SERVER_URL}/submit_qdrant_result",
                    json={
                        "job_id": job_id,
                        "worker_id": WORKER_ID,
                        "success": False,
                        "error": str(e),
                    },
                    timeout=30,
                )
            except Exception:
                pass

        time.sleep(5)

    finally:
        # ── always clean up temp files ─────────────────────────────────────────
        for p in (local_ready_path, local_prop_path):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
