import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import requests
from processing.ingest_qdrant import run_qdrant_ingestion

SERVER_URL = os.environ.get("SERVER_URL", "http://192.168.X.X:8004")
WORKER_ID = f"qdrant-{uuid.uuid4().hex[:6]}"
BASE_DIR = Path(__file__).resolve().parent.parent

# ── persistent session — reuses TCP connection across all poll cycles ─────────
_session = requests.Session()

print(f"[{WORKER_ID}] Qdrant worker started. SERVER={SERVER_URL}")

while True:
    job_id = None
    local_ready_path = None
    local_prop_path = None

    try:
        r = _session.get(
            f"{SERVER_URL}/get_qdrant_job",
            params={"worker_id": WORKER_ID},
            timeout=30,
        )

        if r.status_code != 200:
            print(f"[{WORKER_ID}] Failed to get job ({r.status_code})")
            time.sleep(5)
            continue

        job = r.json()

        if job.get("action") != "PROCESS":
            time.sleep(2)
            continue

        job_id = job["job_id"]
        book_id = job["book_id"]
        # ready_path and prop_path from job response are server-side paths — do not use directly

        print(f"[{WORKER_ID}] Processing book={book_id} job={job_id}")

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

        # ── download prop.json from server ────────────────────────────────────
        print(f"[{WORKER_ID}] Downloading prop file for '{book_id}'...")
        r3 = _session.get(
            f"{SERVER_URL}/download_prop/{book_id}",
            timeout=60,
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

        print(f"[{WORKER_ID}] Files ready — running ingestion...")

        # ── run ingestion ──────────────────────────────────────────────────────
        chunks_stored = run_qdrant_ingestion(
            book_id=book_id,
            ready_path=local_ready_path,
            prop_path=local_prop_path,
            base_dir=str(BASE_DIR),
            progress_callback=None,
        )

        import json
        chunk_file = Path(BASE_DIR) / "data" / "qdrant" / f"{book_id}_chunks.json"
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks_json_data = json.load(f)

        # ── submit success ─────────────────────────────────────────────────────
        response = _session.post(
            f"{SERVER_URL}/submit_qdrant_result",
            json={
                "job_id": job_id,
                "worker_id": WORKER_ID,
                "success": True,
                "content": {
                    "chunks_stored": chunks_stored,
                    "chunks_json": chunks_json_data
                }
            },
            timeout=30,
        )

        print(f"[{WORKER_ID}] Completed job={job_id} status={response.status_code}")

    except Exception as e:
        print(f"[{WORKER_ID}] Error: {e}")
        traceback.print_exc()

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
                    timeout=10,
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
