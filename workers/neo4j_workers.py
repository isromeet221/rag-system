import os
import json
import socket
import time
import uuid
import tempfile
import traceback
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

from parta.processing.ingest_neo4j import run_neo4j_batch

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8004")
WORKER_ID  = f"neo4j-{uuid.uuid4().hex[:6]}"
setup_worker_logger("neo4j", WORKER_ID)
BASE_DIR   = Path(__file__).resolve().parent.parent / "parta"

# ── persistent session — reuses TCP connection across all poll cycles ─────────
_session = requests.Session()
_session.headers.update({"ngrok-skip-browser-warning": "true"})

# ── local result cache — survives NAS disconnect ──────────────────────────────
RESULT_CACHE_DIR = Path(tempfile.gettempdir()) / "worker_result_cache" / "neo4j"
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
            r = _session.post(f"{SERVER_URL}/submit_neo4j_result", json=payload, timeout=30)
            if r.status_code == 200:
                cache_file.unlink()
                logger.info(f"[{WORKER_ID}] Replayed and cleared: {cache_file.name}")
            else:
                logger.warning(f"[{WORKER_ID}] Replay failed HTTP {r.status_code}: {cache_file.name}")
        except Exception as e:
            logger.warning(f"[{WORKER_ID}] Replay error for {cache_file.name}: {e}")


logger.info("=" * 80)
logger.info(f"[{WORKER_ID}] NEO4J WORKER STARTED (batch mode)")
logger.info(f"[{WORKER_ID}] SERVER      : {SERVER_URL}")
logger.info(f"[{WORKER_ID}] BASE_DIR    : {BASE_DIR}")
logger.info(f"[{WORKER_ID}] CACHE_DIR   : {RESULT_CACHE_DIR}")
logger.info("=" * 80)
is_connected = False


# ── replay any results cached from a previous run before polling ──────────────
_replay_cached_results()

@worker_log_process(WORKER_ID)
def execute_neo4j_batch(book_id, ready_path, batch_start, batch_count):
    return run_neo4j_batch(
        book_id=book_id,
        ready_path=ready_path,
        base_dir=str(BASE_DIR),
        batch_start=batch_start,
        batch_count=batch_count,
    )

while True:
    job_id  = None
    book_id = None
    local_ready_path = None

    try:
        r = _session.get(
            f"{SERVER_URL}/get_neo4j_job",
            params={"worker_id": WORKER_ID},
        )

        if not is_connected:
            logger.info(f"[{WORKER_ID}] Connected to server")
            is_connected = True

        if r.status_code != 200:
            logger.error(f"[{WORKER_ID}] get_neo4j_job failed: HTTP {r.status_code}")
            time.sleep(5)
            continue

        job = r.json()
        if job.get("action") != "PROCESS":
            time.sleep(2)
            continue

        job_id       = job.get("job_id")
        book_id      = job.get("book_id")
        batch_start  = job.get("start_offset", 0)
        batch_count  = job.get("page_count", 0)
        batch_idx    = job.get("chunk_idx", 0)

        logger.info("\n" + "=" * 80)
        logger.info(f"[{WORKER_ID}] NEW NEO4J BATCH JOB")
        logger.info(f"[{WORKER_ID}] JOB ID     : {job_id}")
        logger.info(f"[{WORKER_ID}] BOOK ID    : {book_id}")
        logger.info(f"[{WORKER_ID}] BATCH      : #{batch_idx} (chunks {batch_start}–{batch_start + batch_count - 1})")
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

        logger.info(f"[{WORKER_ID}] Ready file saved to {local_ready_path}")

        # ── run batch ingestion ────────────────────────────────────────────────
        result = execute_neo4j_batch(book_id, local_ready_path, batch_start, batch_count)
        
        logger.info(f"[{WORKER_ID}] Neo4j batch completed — entities={result.get('entities_written', 0)}, specs={result.get('specs_written', 0)}")

        # ── build payload, cache locally before attempting submit ─────────────
        payload = {
            "job_id":    job_id,
            "worker_id": WORKER_ID,
            "success":   True,
            "content":   result,
        }
        _cache_result(job_id, payload)
        _submit_with_retry(f"{SERVER_URL}/submit_neo4j_result", payload)
        _clear_cached_result(job_id)
        logger.info(f"[{WORKER_ID}] Completion acknowledged for batch #{batch_idx} of '{book_id}'")

        logger.info("=" * 80)

    except requests.exceptions.ConnectionError:
        if is_connected:
            logger.error(f"[{WORKER_ID}] Disconnected from server. Waiting to reconnect...")
            is_connected = False
        else:
            logger.error(f"[{WORKER_ID}] Failed to connect to server at {SERVER_URL}. Retrying...")
        time.sleep(5)

    except Exception as e:
        logger.error(f"[{WORKER_ID}] Neo4j worker error: {e}")
        logger.error(traceback.format_exc())

        if job_id:
            try:
                _session.post(
                    f"{SERVER_URL}/submit_neo4j_result",
                    json={
                        "job_id":    job_id,
                        "worker_id": WORKER_ID,
                        "success":   False,
                        "content":   "",
                    },
                    timeout=30,
                )
            except Exception as e2:
                logger.error(f"[{WORKER_ID}] Error submitting failure: {e2}")
                logger.error(traceback.format_exc())

        time.sleep(5)

    finally:
        # ── always clean up the temp file ──────────────────────────────────────
        if local_ready_path:
            try:
                Path(local_ready_path).unlink(missing_ok=True)
            except Exception:
                pass
