import os
import json
import socket
import tempfile
import time
import traceback
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workers.logger import logger, worker_log_process, setup_worker_logger

import requests
from urllib3.connection import HTTPConnection

# ── TCP keepalive — prevents idle connections from being silently dropped ──────
# by load balancers / proxies / NAT gateways. Sends first probe after 60 s of
# idle, then every 10 s, giving up after 5 failed probes (50 s window).
HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
]

from parta.processing.ingest_qdrant import run_qdrant_batch

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8004")
WORKER_ID = f"qdrant-{uuid.uuid4().hex[:6]}"
setup_worker_logger("qdrant", WORKER_ID)
BASE_DIR = Path(__file__).resolve().parent.parent / "parta"

# ── persistent session — reuses TCP connection across all poll cycles ─────────
_session = requests.Session()
_session.headers.update({"ngrok-skip-browser-warning": "true"})

# ── local result cache — survives NAS disconnect ──────────────────────────────
RESULT_CACHE_DIR = Path(tempfile.gettempdir()) / "worker_result_cache" / "qdrant"
RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_result(job_id: str, payload: dict):
    path = RESULT_CACHE_DIR / f"{job_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _clear_cached_result(job_id: str):
    (RESULT_CACHE_DIR / f"{job_id}.json").unlink(missing_ok=True)


def _submit_with_retry(endpoint: str, payload: dict):
    delay = 5
    while True:
        try:
            r = _session.post(endpoint, json=payload, timeout=30)
            if r.status_code == 200:
                return
            logger.warning(f"[{WORKER_ID}] Submit returned HTTP {r.status_code}, retrying in {delay}s...")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{WORKER_ID}] Connection error on submit, retrying in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, 60)


def _replay_cached_results():
    cached = sorted(RESULT_CACHE_DIR.glob("*.json"))
    if not cached:
        return
    logger.info(f"[{WORKER_ID}] Replaying {len(cached)} cached result(s)...")
    for cache_file in cached:
        try:
            with open(cache_file, encoding="utf-8") as f:
                payload = json.load(f)
            r = _session.post(f"{SERVER_URL}/submit_qdrant_result", json=payload, timeout=30)
            if r.status_code == 200:
                cache_file.unlink()
                logger.info(f"[{WORKER_ID}] Replayed and cleared: {cache_file.name}")
            else:
                logger.warning(f"[{WORKER_ID}] Replay failed HTTP {r.status_code}: {cache_file.name}")
        except Exception as e:
            logger.warning(f"[{WORKER_ID}] Replay error for {cache_file.name}: {e}")


logger.info("=" * 80)
logger.info(f"[{WORKER_ID}] QDRANT WORKER STARTED (batch mode)")
logger.info(f"[{WORKER_ID}] SERVER      : {SERVER_URL}")
logger.info(f"[{WORKER_ID}] BASE_DIR    : {BASE_DIR}")
logger.info(f"[{WORKER_ID}] CACHE_DIR   : {RESULT_CACHE_DIR}")
logger.info("=" * 80)
is_connected = False


# ── replay any results cached from a previous run before polling ──────────────
_replay_cached_results()

@worker_log_process(WORKER_ID)
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
            logger.info(f"[{WORKER_ID}] Connected to server")
            is_connected = True

        if r.status_code != 200:
            logger.error(f"[{WORKER_ID}] Failed to get job ({r.status_code})")
            time.sleep(5)
            continue

        job = r.json()

        if job.get("action") != "PROCESS":
            time.sleep(2)
            continue

        job_id       = job["job_id"]
        book_id      = job["book_id"]
        batch_start  = job.get("start_offset", 0)
        batch_count  = job.get("page_count", 0)
        batch_idx    = job.get("chunk_idx", 0)
        # batch_kind is encoded in chunk_path field by the server
        batch_kind   = job.get("chunk_path", "propositions")
        # ready_path and prop_path from job response are server-side paths — do not use directly

        logger.info("\n" + "=" * 80)
        logger.info(f"[{WORKER_ID}] NEW QDRANT BATCH JOB")
        logger.info(f"[{WORKER_ID}] JOB ID     : {job_id}")
        logger.info(f"[{WORKER_ID}] BOOK ID    : {book_id}")
        logger.info(f"[{WORKER_ID}] KIND       : {batch_kind}")
        logger.info(f"[{WORKER_ID}] BATCH      : #{batch_idx} ({batch_kind} {batch_start}–{batch_start + batch_count - 1})")
        logger.info("=" * 80)

        # ── download ready.json from server ───────────────────────────────────
        logger.info(f"[{WORKER_ID}] Downloading ready file for '{book_id}'...")
        r2 = _session.get(
            f"{SERVER_URL}/download_ready/{book_id}",
        )
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
        logger.info(f"[{WORKER_ID}] Downloading prop file for '{book_id}'...")
        r3 = _session.get(
            f"{SERVER_URL}/download_prop/{book_id}",
        )
        if r3.status_code != 200:
            raise RuntimeError(
                f"download_prop failed: HTTP {r3.status_code} — {r3.text[:200]}"
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix="_prop.json", delete=False
        ) as f:
            f.write(r3.content)
            local_prop_path = f.name

        logger.info(f"[{WORKER_ID}] Files ready — running batch ingestion ({batch_kind} {batch_start}–{batch_start + batch_count - 1})...")

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
            }
        }
        _cache_result(job_id, payload)
        _submit_with_retry(f"{SERVER_URL}/submit_qdrant_result", payload)
        _clear_cached_result(job_id)
        logger.info(f"[{WORKER_ID}] Completion acknowledged for batch #{batch_idx} ({batch_kind})")

    except requests.exceptions.ConnectionError:
        if is_connected:
            logger.error(f"[{WORKER_ID}] Disconnected from server. Waiting to reconnect...")
            is_connected = False
        else:
            logger.error(f"[{WORKER_ID}] Failed to connect to server at {SERVER_URL}. Retrying...")
        time.sleep(5)

    except Exception as e:
        logger.error(f"[{WORKER_ID}] Error: {e}")
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
