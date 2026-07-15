"""
neo4j_workers.py — Batch Neo4j ingestion worker.
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

from parta.processing.ingest_neo4j import run_neo4j_batch

# ─── Constants ────────────────────────────────────────────────────────────────
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8004")
WORKER_ID = f"neo4j-{uuid.uuid4().hex[:6]}"
BASE_DIR = Path(__file__).resolve().parent.parent / "parta"

# ─── Logger setup ─────────────────────────────────────────────────────────────
logger = setup_worker_logger("neo4j", WORKER_ID)

# ─── Persistent session — reuses TCP connection across all poll cycles ────────
_session = create_session()

# ─── Crash-safe result cache ──────────────────────────────────────────────────
_cache = ResultCache("neo4j")

# ─── Startup ──────────────────────────────────────────────────────────────────
logger.info("=" * 80)
logger.info("NEO4J WORKER STARTED (batch mode)")
logger.info("SERVER      : %s", SERVER_URL)
logger.info("BASE_DIR    : %s", BASE_DIR)
logger.info("CACHE_DIR   : %s", _cache.cache_dir)
logger.info("=" * 80)

# ── replay any results cached from a previous run before polling ──────────────
_cache.replay(_session, f"{SERVER_URL}/submit_neo4j_result")

is_connected = False


@log_process
def execute_neo4j_batch(book_id, ready_path, batch_start, batch_count):
    return run_neo4j_batch(
        book_id=book_id,
        ready_path=ready_path,
        base_dir=str(BASE_DIR),
        batch_start=batch_start,
        batch_count=batch_count,
    )


# ─── Main polling loop ────────────────────────────────────────────────────────
while True:
    job_id = None
    book_id = None
    local_ready_path = None

    try:
        r = _session.get(
            f"{SERVER_URL}/get_neo4j_job",
            params={"worker_id": WORKER_ID},
        )

        if not is_connected:
            logger.info("Connected to server")
            is_connected = True

        if r.status_code != 200:
            logger.error("get_neo4j_job failed: HTTP %d", r.status_code)
            time.sleep(5)
            continue

        job = r.json()
        if job.get("action") != "PROCESS":
            time.sleep(2)
            continue

        job_id = job.get("job_id")
        book_id = job.get("book_id")
        batch_start = job.get("start_offset", 0)
        batch_count = job.get("page_count", 0)
        batch_idx = job.get("chunk_idx", 0)

        logger.info("\n" + "=" * 80)
        logger.info("NEW NEO4J BATCH JOB")
        logger.info("JOB ID     : %s", job_id)
        logger.info("BOOK ID    : %s", book_id)
        logger.info(
            "BATCH      : #%s (chunks %d–%d)",
            batch_idx, batch_start, batch_start + batch_count - 1,
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

        logger.info("Ready file saved to %s", local_ready_path)

        # ── run batch ingestion ────────────────────────────────────────────────
        result = execute_neo4j_batch(book_id, local_ready_path, batch_start, batch_count)

        logger.info(
            "Neo4j batch completed — entities=%d, specs=%d",
            result.get("entities_written", 0),
            result.get("specs_written", 0),
        )

        # ── build payload, cache locally before attempting submit ─────────────
        payload = {
            "job_id": job_id,
            "worker_id": WORKER_ID,
            "success": True,
            "content": result,
        }
        _cache.store(job_id, payload)
        submit_with_retry(_session, f"{SERVER_URL}/submit_neo4j_result", payload)
        _cache.clear(job_id)
        logger.info("Completion acknowledged for batch #%s of '%s'", batch_idx, book_id)

        logger.info("=" * 80)

    except requests.exceptions.ConnectionError:
        if is_connected:
            logger.error("Disconnected from server. Waiting to reconnect...")
            is_connected = False
        else:
            logger.error("Failed to connect to server at %s. Retrying...", SERVER_URL)
        time.sleep(5)

    except Exception as e:
        logger.error("Neo4j worker error: %s", e)
        logger.error(traceback.format_exc())

        if job_id:
            try:
                _session.post(
                    f"{SERVER_URL}/submit_neo4j_result",
                    json={
                        "job_id": job_id,
                        "worker_id": WORKER_ID,
                        "success": False,
                        "content": "",
                    },
                    timeout=30,
                )
            except Exception as e2:
                logger.error("Error submitting failure: %s", e2)
                logger.error(traceback.format_exc())

        time.sleep(5)

    finally:
        # ── always clean up the temp file ──────────────────────────────────────
        if local_ready_path:
            try:
                Path(local_ready_path).unlink(missing_ok=True)
            except Exception:
                pass
