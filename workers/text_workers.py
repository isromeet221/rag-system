"""
text_workers.py
Fixed version: model-load lock contention fix, shared model cache preserved,
failure reporting and connection-state logging restored, Windows file-lock
(WinError 32) hardened across every disk touch point.
"""

import sys
import os
import gc
import json
import tempfile
import time
import uuid
import random
import threading
from pathlib import Path
from typing import List

from logger import logger, worker_log_process, setup_worker_logger
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─────────────────────────────────────────────────────────────
# Model artifacts stay on the SHARED path. Do NOT give each worker
# its own HF_HOME / TRANSFORMERS_CACHE — that forces every worker to
# independently download/cache its own copy of the model, multiplying
# disk + NAS I/O instead of fixing contention.
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

os.environ.setdefault(
    "DOCLING_ARTIFACTS_PATH",
    str(BASE_DIR / "parta" / "portable" / "docling"),
)
# Stop huggingface_hub / transformers from doing remote lock-file checks
# against the shared cache on every cold start — this is the actual fix
# for "all workers stall waiting on the same lock file."
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat


SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8004")
WORKER_ID = f"text-{uuid.uuid4().hex[:6]}"

setup_worker_logger("text", WORKER_ID)

REQUEST_TIMEOUT = 30
WAIT_SLEEP = 2
ERROR_SLEEP = 5


# ─────────────────────────────────────────────────────────────
# Windows-safe file ops — retry with backoff on transient locks.
# WinError 32 ("file in use by another process") happens when a
# library (Docling/PyPdfium backend, antivirus, indexer, or our own
# trailing handle) hasn't released a handle yet. POSIX never hits
# this because multiple open handles to one inode are allowed there.
# ─────────────────────────────────────────────────────────────
def _win_safe_unlink(path: Path, attempts: int = 8, base_delay: float = 0.15):
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as e:
            if i == attempts - 1:
                logger.warning(f"[{WORKER_ID}] Giving up deleting {path} after {attempts} attempts: {e}")
                return
            time.sleep(base_delay * (2 ** i))


def _win_safe_write_bytes(path: Path, data: bytes, attempts: int = 5, base_delay: float = 0.15):
    for i in range(attempts):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


def _win_safe_write_text(path: Path, text: str, attempts: int = 5, base_delay: float = 0.15):
    for i in range(attempts):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


def _win_safe_read_text(path: Path, attempts: int = 5, base_delay: float = 0.15) -> str:
    for i in range(attempts):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


# ─────────────────────────────────────────────────────────────
# Model-init lock — this is what actually prevents concurrent
# model-load races WITHIN one process (multiple threads).
# NOTE: if your 5 workers are 5 separate `python text_workers.py`
# processes (not threads in one process), this lock does nothing
# across processes — each process gets its own lock object. Confirm
# how you're launching workers before relying on this alone.
# ─────────────────────────────────────────────────────────────
_model_init_lock = threading.Lock()
_docling_converter = None
_docling_ocr_converter = None


def _get_converter(ocr: bool = False):
    global _docling_converter, _docling_ocr_converter

    if not ocr and _docling_converter:
        return _docling_converter
    if ocr and _docling_ocr_converter:
        return _docling_ocr_converter

    with _model_init_lock:
        if not ocr:
            if _docling_converter is None:
                logger.info(f"[{WORKER_ID}] Loading Docling (standard)...")
                opts = PdfPipelineOptions(do_ocr=False)
                _docling_converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
                )
                logger.info(f"[{WORKER_ID}] Docling loaded (standard)")
            return _docling_converter

        if _docling_ocr_converter is None:
            model_dir = BASE_DIR / "parta" / "portable" / "docling" / "EasyOcr"

            logger.info(f"[{WORKER_ID}] Initialising OCR converter (models: {model_dir})")

            ocr_opts = EasyOcrOptions(
                lang=["en"],
                model_storage_directory=str(model_dir),
            )

            opts = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_opts)

            _docling_ocr_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )

            logger.info(f"[{WORKER_ID}] Docling loaded (OCR)")

        return _docling_ocr_converter


@worker_log_process(WORKER_ID)
def process_chunk(pdf_bytes: bytes, start_offset: int, ocr_enabled: bool = False) -> str:
    converter = _get_converter(ocr=ocr_enabled)

    tmp_path = Path(tempfile.gettempdir()) / f"docling_{uuid.uuid4().hex}_{start_offset}.pdf"
    try:
        tmp_path.write_bytes(pdf_bytes)
        # note: write_bytes opens and closes the handle in one shot — no
        # lingering Python handle for Docling to trip over on Windows.

        result = converter.convert(tmp_path)
        md_text = result.document.export_to_markdown()

        # Drop Docling's internal reference before we try to delete the file.
        # Some PDF backends (PyPdfium/PyMuPDF) keep a lazy/mmap-style handle
        # open until the result object is garbage collected, not the instant
        # export_to_markdown() returns — that trailing handle is what causes
        # WinError 32 on the unlink below if we don't release it first.
        del result
        gc.collect()  # force GC to release any cyclic refs (PyPdfium mmap)
    finally:
        # retry-with-backoff: antivirus / search indexer / a trailing Docling
        # handle commonly holds the file for tens-to-hundreds of ms after the
        # "owning" call returns, even when our own code closed everything
        _win_safe_unlink(tmp_path)

    return f"\n\n## Page {start_offset + 1}\n\n{md_text}".strip()


_session = requests.Session()

RESULT_CACHE_DIR = Path(tempfile.gettempdir()) / "worker_result_cache" / "text"
RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_result(job_id: str, payload: dict):
    path = RESULT_CACHE_DIR / f"{job_id}.json"
    _win_safe_write_text(path, json.dumps(payload, ensure_ascii=False))


def _clear_cached_result(job_id: str):
    _win_safe_unlink(RESULT_CACHE_DIR / f"{job_id}.json")


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
            payload = json.loads(_win_safe_read_text(cache_file))
            r = _session.post(f"{SERVER_URL}/submit_result", json=payload, timeout=30)
            if r.status_code == 200:
                _win_safe_unlink(cache_file)
                logger.info(f"[{WORKER_ID}] Replayed and cleared: {cache_file.name}")
            else:
                logger.warning(f"[{WORKER_ID}] Replay failed HTTP {r.status_code}: {cache_file.name}")
        except Exception as e:
            logger.warning(f"[{WORKER_ID}] Replay error: {cache_file.name} -> {e}")


is_connected = False


def start_worker():
    global is_connected

    logger.info("=" * 80)
    logger.info(f"[{WORKER_ID}] TEXT WORKER STARTED (extraction mode)")
    logger.info(f"[{WORKER_ID}] SERVER      : {SERVER_URL}")
    logger.info(f"[{WORKER_ID}] BASE_DIR    : {BASE_DIR}")
    logger.info(f"[{WORKER_ID}] CACHE_DIR   : {RESULT_CACHE_DIR}")
    logger.info("=" * 80)

    # stagger startup slightly — harmless even with the lock in place,
    # avoids 5 workers hitting the model-init lock at the exact same instant
    time.sleep(random.uniform(0.5, 4.0))

    _replay_cached_results()

    # warm up model before entering job loop — fail loud, not silent,
    # so a broken model path is caught at startup instead of mid-job
    try:
        _get_converter(ocr=False)
    except Exception as e:
        logger.error(f"[{WORKER_ID}] Model warm-up failed: {e}")

    while True:
        job_id = None
        try:
            resp = _session.get(
                f"{SERVER_URL}/get_job",
                params={"worker_id": WORKER_ID},
                timeout=REQUEST_TIMEOUT,
            )

            if not is_connected:
                logger.info(f"[{WORKER_ID}] Connected to server")
                is_connected = True

            if resp.status_code != 200:
                time.sleep(ERROR_SLEEP)
                continue

            data = resp.json()

            if data.get("action") == "WAIT":
                time.sleep(WAIT_SLEEP)
                continue

            if data.get("action") != "PROCESS":
                continue

            job_id = data["job_id"]
            book_id = data["book_id"]
            chunk_idx = data["chunk_idx"]
            start_offset = data.get("start_offset", 0)
            ocr_enabled = data.get("ocr_enabled", False)

            logger.info(f"[{WORKER_ID}] Processing chunk {chunk_idx} (book={book_id})")

            chunk_resp = _session.get(
                f"{SERVER_URL}/chunk/{job_id}",
                timeout=REQUEST_TIMEOUT,
            )

            if chunk_resp.status_code != 200:
                logger.error(f"[{WORKER_ID}] Failed to download chunk {job_id} (HTTP {chunk_resp.status_code})")
                continue

            try:
                content = process_chunk(chunk_resp.content, start_offset, ocr_enabled)
            except Exception as e:
                logger.error(f"[{WORKER_ID}] Extraction failed for chunk {chunk_idx}: {e}")
                # report failure immediately instead of letting the lease expire
                try:
                    _session.post(
                        f"{SERVER_URL}/submit_result",
                        json={
                            "job_id": job_id,
                            "worker_id": WORKER_ID,
                            "success": False,
                            "content": "",
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                except Exception as e2:
                    logger.error(f"[{WORKER_ID}] Error submitting failure: {e2}")
                continue

            payload = {
                "job_id": job_id,
                "worker_id": WORKER_ID,
                "success": True,
                "content": content,
            }

            _cache_result(job_id, payload)
            _submit_with_retry(f"{SERVER_URL}/submit_result", payload)
            _clear_cached_result(job_id)

            logger.info(f"[{WORKER_ID}] Completed chunk {chunk_idx}")

        except requests.exceptions.ConnectionError:
            if is_connected:
                logger.error(f"[{WORKER_ID}] Disconnected from server. Reconnecting...")
                is_connected = False
            time.sleep(ERROR_SLEEP)

        except Exception as e:
            logger.error(f"[{WORKER_ID}] Error: {e}")
            time.sleep(ERROR_SLEEP)


if __name__ == "__main__":
    start_worker()