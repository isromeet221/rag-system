import os
import time
import uuid
import tempfile
import traceback
from pathlib import Path

import requests

from processing.ingest_neo4j import run_neo4j_ingestion

SERVER_URL = os.environ.get("SERVER_URL", "http://192.168.X.X:8004")
WORKER_ID  = f"neo4j-{uuid.uuid4().hex[:6]}"
BASE_DIR   = Path(__file__).resolve().parent.parent

# ── persistent session — reuses TCP connection across all poll cycles ─────────
_session = requests.Session()

print("=" * 80)
print(f"[{WORKER_ID}] NEO4J WORKER STARTED")
print(f"[{WORKER_ID}] SERVER   : {SERVER_URL}")
print(f"[{WORKER_ID}] BASE_DIR : {BASE_DIR}")
print("=" * 80)

while True:
    job_id  = None
    book_id = None
    local_ready_path = None

    try:
        r = _session.get(
            f"{SERVER_URL}/get_neo4j_job",
            params={"worker_id": WORKER_ID},
            timeout=30,
        )

        if r.status_code != 200:
            print(f"[{WORKER_ID}] get_neo4j_job failed: HTTP {r.status_code}")
            time.sleep(5)
            continue

        job = r.json()
        if job.get("action") != "PROCESS":
            time.sleep(2)
            continue

        job_id  = job.get("job_id")
        book_id = job.get("book_id")
        # ready_path from job response is a server-side path — do not use it directly

        print("\n" + "=" * 80)
        print(f"[{WORKER_ID}] NEW NEO4J JOB")
        print(f"[{WORKER_ID}] JOB ID     : {job_id}")
        print(f"[{WORKER_ID}] BOOK ID    : {book_id}")
        print("=" * 80)

        # ── download ready.json from server ───────────────────────────────────
        print(f"[{WORKER_ID}] Downloading ready file for '{book_id}'...")
        r2 = _session.get(
            f"{SERVER_URL}/download_ready/{book_id}",
            timeout=60,
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

        print(f"[{WORKER_ID}] Ready file saved to {local_ready_path}")

        # ── run ingestion ──────────────────────────────────────────────────────
        print(f"[{WORKER_ID}] Starting Neo4j ingestion...")
        start_time = time.time()

        result = run_neo4j_ingestion(
            book_id=book_id,
            ready_path=local_ready_path,
            base_dir=str(BASE_DIR),
            progress_callback=None,
        )

        elapsed = round(time.time() - start_time, 2)
        print(f"[{WORKER_ID}] Neo4j ingestion completed in {elapsed}s")

        # ── submit success ─────────────────────────────────────────────────────
        response = _session.post(
            f"{SERVER_URL}/submit_neo4j_result",
            json={
                "job_id":    job_id,
                "worker_id": WORKER_ID,
                "success":   True,
                "content":   result,
            },
            timeout=30,
        )

        if response.status_code == 200:
            print(f"[{WORKER_ID}] Completion acknowledged for book '{book_id}'")
        else:
            print(f"[{WORKER_ID}] submit_neo4j_result failed: HTTP {response.status_code}")

        print("=" * 80)

    except Exception:
        print(f"[{WORKER_ID}] Neo4j worker error")
        traceback.print_exc()

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
            except Exception:
                traceback.print_exc()

        time.sleep(5)

    finally:
        # ── always clean up the temp file ──────────────────────────────────────
        if local_ready_path:
            try:
                Path(local_ready_path).unlink(missing_ok=True)
            except Exception:
                pass
