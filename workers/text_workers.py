"""
text_workers.py — PDF extraction worker (Docling).
Fixed version:
- Local-disk model cache per machine (kills NAS/SMB file-lock contention
  that was producing WinError 32 and inconsistent transformers errors)
- Async delete queue for temp PDFs (unlink retries no longer block the job loop)
- Loud accelerate dependency check at startup
- Result cache + replay preserved, moved off system temp to LOCALAPPDATA
- Win-safe file ops provided by base_worker
"""

import sys
import os
import gc
import queue
import re
import shutil
import tempfile
import time
import uuid
import random
import threading
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.logger import log_process, setup_worker_logger
from workers.base_worker import (
    create_session,
    ResultCache,
    submit_with_retry,
    safe_unlink,
    safe_write_text,
    safe_read_text,
)

BASE_DIR = Path(__file__).resolve().parent.parent
WORKER_ID = f"text-{uuid.uuid4().hex[:6]}"

logger = setup_worker_logger("text", WORKER_ID)

# accelerate is a hard dependency for Docling's transformer-backed layout
# model when it loads with device_map. Missing it doesn't crash on import,
# it crashes mid-job with a confusing "device_map / tp_plan / torch.device
# context manager" error. Catch it loud at startup instead of mid-job.
import accelerate  # noqa: F401

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8004")

WAIT_SLEEP = 2
ERROR_SLEEP = 5


# ─────────────────────────────────────────────────────────────
# Async delete queue. safe_unlink retries for up to ~38s on a
# stubborn lock (8 attempts, exponential backoff). Doing that
# synchronously inside process_chunk blocks the job loop — one
# locked temp file stalls that worker for the entire retry window.
# Deletes now happen on a background thread instead.
# ─────────────────────────────────────────────────────────────
_delete_queue: "queue.Queue[Path]" = queue.Queue()


def _reaper_loop():
    while True:
        path = _delete_queue.get()
        safe_unlink(path)


threading.Thread(target=_reaper_loop, daemon=True, name="tmp-reaper").start()


def _queue_delete(path: Path):
    _delete_queue.put(path)


# ─────────────────────────────────────────────────────────────
# Model sync: NAS -> local disk, once per machine.
#
# This is the actual fix for the WinError 32 / weird transformers
# errors. SMB does not give the same locking guarantees as NTFS.
# Five machines independently reading the same model directory over
# the network is the root cause, not a Docling bug. Each worker now
# gets its own local copy and never touches the NAS for model files
# again after the first sync.
# ─────────────────────────────────────────────────────────────
NAS_DOCLING_DIR = BASE_DIR / "parta" / "portable" / "docling"
LOCAL_CACHE_ROOT = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "docling_worker_cache"


def _dir_signature(path: Path) -> str:
    """File count + total size + newest mtime. Cheap, not cryptographic,
    good enough to detect 'the NAS copy changed since last sync'."""
    if not path.exists():
        return "missing"
    count = 0
    total_size = 0
    newest = 0.0
    for f in path.rglob("*"):
        if f.is_file():
            st = f.stat()
            count += 1
            total_size += st.st_size
            newest = max(newest, st.st_mtime)
    return f"{count}-{total_size}-{int(newest)}"


def _sync_models_locally(nas_source: Path, local_dest: Path) -> Path:
    marker = local_dest / ".sync_signature"
    nas_sig = _dir_signature(nas_source)

    if marker.exists():
        try:
            if safe_read_text(marker).strip() == nas_sig and nas_sig != "missing":
                logger.info("Local model cache already up to date: %s", local_dest)
                return local_dest
        except Exception:
            pass

    if nas_sig == "missing":
        logger.error("NAS model source not found: %s", nas_source)
        return local_dest  # let docling fail loudly if it has to, don't crash here

    logger.info("Syncing models %s -> %s (one-time per machine)", nas_source, local_dest)

    if local_dest.exists():
        shutil.rmtree(local_dest, ignore_errors=True)

    shutil.copytree(nas_source, local_dest)
    safe_write_text(marker, nas_sig)

    logger.info("Model sync done")
    return local_dest


LOCAL_DOCLING_DIR = _sync_models_locally(NAS_DOCLING_DIR, LOCAL_CACHE_ROOT)

os.environ["DOCLING_ARTIFACTS_PATH"] = str(LOCAL_DOCLING_DIR)
# Offline mode now points at a local cache, not a NAS path. That's what
# actually kills lock-file contention. Pointing offline mode at a shared
# network path (the old setup) doesn't fix SMB locking, it just removes
# the network round-trip while leaving the lock contention intact.
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat


# ─────────────────────────────────────────────────────────────
# Model-init lock. Only protects against concurrent loads WITHIN
# one process (multiple threads). If the .bat spawns one
# `python text_workers.py` process per machine, each process owns
# its own lock object — this does nothing across machines, and
# doesn't need to anymore, since each machine loads from its own
# local copy now.
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
                logger.info("Loading Docling (standard)...")
                opts = PdfPipelineOptions(do_ocr=False)
                _docling_converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
                )
                logger.info("Docling loaded (standard)")
            return _docling_converter

        if _docling_ocr_converter is None:
            logger.info("Initialising OCR converter (default models)")

            ocr_opts = EasyOcrOptions(
                lang=["en"]
            )

            opts = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_opts)

            _docling_ocr_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )

            logger.info("Docling loaded (OCR)")

        return _docling_ocr_converter


import fitz


def _has_images(pdf_bytes: bytes) -> bool:
    try:
        doc = fitz.open("pdf", pdf_bytes)
        for page in doc:
            if page.get_images():
                return True
        return False
    except Exception as e:
        logger.warning("PyMuPDF image check failed, defaulting to OCR: %s", e)
        return True


@log_process
def process_chunk(pdf_bytes: bytes, start_offset: int, ocr_enabled: bool = False) -> str:
    actual_ocr = False
    if ocr_enabled:
        if _has_images(pdf_bytes):
            actual_ocr = True
            logger.info("Chunk %d contains images. Using OCR converter.", start_offset)
        else:
            logger.info("Chunk %d is pure text (0 images). Bypassing OCR for speed.", start_offset)

    converter = _get_converter(ocr=actual_ocr)

    tmp_path = Path(tempfile.gettempdir()) / f"docling_{uuid.uuid4().hex}_{start_offset}.pdf"
    try:
        tmp_path.write_bytes(pdf_bytes)

        result = converter.convert(tmp_path)
        doc = result.document

        page_nos = sorted(doc.pages.keys())
        parts = []
        for i, page_no in enumerate(page_nos):
            absolute_page = start_offset + i + 1
            page_md = doc.export_to_markdown(page_no=page_no)
            if page_md.strip():
                parts.append(f"## --- PAGE {absolute_page} ---\n\n{page_md.strip()}")

        del result
        gc.collect()
    finally:
        # was a blocking, synchronous retry-unlink. Now handed off to the
        # background reaper so a slow-to-release handle doesn't stall
        # the job loop for up to ~38s.
        _queue_delete(tmp_path)

    return "\n\n".join(parts) if parts else ""


# ─── Persistent session ───────────────────────────────────────────────────────
_session = create_session()

# Moved off system temp. Windows disk cleanup / some AV tools sweep
# %TEMP% on their own schedule — that defeats the entire point of a
# crash/power-cut recovery cache. LOCALAPPDATA persists.
_cache = ResultCache(
    "text",
    base_dir=Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "worker_result_cache",
)


is_connected = False


def start_worker():
    global is_connected

    logger.info("=" * 80)
    logger.info("TEXT WORKER STARTED (extraction mode)")
    logger.info("SERVER      : %s", SERVER_URL)
    logger.info("BASE_DIR    : %s", BASE_DIR)
    logger.info("MODEL_DIR   : ~/.cache/huggingface (local, built in image)")
    logger.info("CACHE_DIR   : %s", _cache.cache_dir)
    logger.info("=" * 80)

    time.sleep(random.uniform(0.5, 4.0))

    _cache.replay(_session, f"{SERVER_URL}/submit_result")

    try:
        _get_converter(ocr=False)
    except Exception as e:
        logger.error("Model warm-up failed: %s", e)

    while True:
        job_id = None
        try:
            resp = _session.get(
                f"{SERVER_URL}/get_job",
                params={"worker_id": WORKER_ID},
            )

            if not is_connected:
                logger.info("Connected to server")
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

            logger.info("Processing chunk %d (book=%s)", chunk_idx, book_id)

            chunk_resp = _session.get(
                f"{SERVER_URL}/chunk/{job_id}",
            )

            if chunk_resp.status_code != 200:
                logger.error("Failed to download chunk %s (HTTP %d)", job_id, chunk_resp.status_code)
                continue

            try:
                content = process_chunk(chunk_resp.content, start_offset, ocr_enabled)
            except Exception as e:
                logger.error("Extraction failed for chunk %d: %s", chunk_idx, e)
                try:
                    _session.post(
                        f"{SERVER_URL}/submit_result",
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
                continue

            payload = {
                "job_id": job_id,
                "worker_id": WORKER_ID,
                "success": True,
                "content": content,
            }

            _cache.store(job_id, payload)
            submit_with_retry(_session, f"{SERVER_URL}/submit_result", payload)
            _cache.clear(job_id)

            logger.info("Completed chunk %d", chunk_idx)

        except requests.exceptions.ConnectionError:
            if is_connected:
                logger.error("Disconnected from server. Reconnecting...")
                is_connected = False
            time.sleep(ERROR_SLEEP)

        except Exception as e:
            logger.error("Error: %s", e)
            time.sleep(ERROR_SLEEP)


if __name__ == "__main__":
    start_worker()
