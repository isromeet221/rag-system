"""
pipeline_controller.py
-----------------------
2-Phase ingestion orchestrator — GraphRAG Edition.

PHASE 1  (expensive, disk-checkpointed)
  Steps : Validate → Extract (Docling) → Chunk → Triple-Rep → Propositions
  Output: data/checkpoints/{book_id}_ready.json
          data/checkpoints/{book_id}_propositions.json
  Status: extracting → extraction_done  (or extraction_failed)
  Saved on job doc: ready_path, prop_path

PHASE 2  (cheap, fully resumable per-stage)
  Steps : Qdrant ingestion  ‖  Neo4j ingestion  (parallel threads)
  Input : ready_path + prop_path read from MongoDB job doc
  Status: ingesting → completed  (or ingestion_failed)

RESUMABILITY — each stage is individually skippable on resume:
  qdrant_progress.status == "done" → Qdrant skipped, stored result reused
  neo4j_progress.status  == "done" → Neo4j  skipped, stored result reused

  This means a file-naming error or DB timeout in one stage does NOT force
  you to re-run the other stage. The checkpoint files stay on disk.
  POST /resume/{job_id} re-queues Phase 2 only, reading paths from MongoDB.

FIXES vs original provided file:
  1. "processing.ingest_graph"   → "processing.ingest_neo4j"   (correct module)
  2. "run_graph_ingestion"       → "run_neo4j_ingestion"        (correct function)
  3. _run_neo4j_stage now receives ready_path (required by run_neo4j_ingestion)
  4. Neo4j skip-if-done guard added (mirrors existing Qdrant guard)
  5. Checkpoint paths stored as ready_path + prop_path (not chunks_path)
     main_api.py resume endpoint reads these same fields
"""

import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from parta.logger import async_time_it, log_process, logger, time_it

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
EXTRACTION_SERVER_URL = os.environ.get("EXTRACTION_SERVER_URL", "http://127.0.0.1:8004")


@time_it
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


@time_it
def update_progress(
    jobs_col,
    job_id,
    percent,
    stage,
    message,
    extra=None,
    qdrant_progress=None,
    neo4j_progress=None,
):
    update = {
        "percent": percent,
        "stage": stage,
        "message": message,
        "updated_at": _now_iso(),
    }
    if extra:
        update["extra"] = extra
    if qdrant_progress is not None:
        update["qdrant_progress"] = qdrant_progress
    if neo4j_progress is not None:
        update["neo4j_progress"] = neo4j_progress
    jobs_col.update_one({"job_id": job_id}, {"$set": update})


@time_it
def _make_callback(jobs_col, job_id):
    def callback(percent, stage, message, extra=None):
        update_progress(jobs_col, job_id, percent, stage, message, extra)

    return callback


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


@log_process
def _run_qdrant_stage(job_id, book_id, ready_path, prop_path, jobs_col):
    """
    Embeds propositions + sections into two Qdrant collections.
    Only called when qdrant_progress.status != "done".
    """
    try:
        import requests
        import time
        from pathlib import Path
        import json

        
        # 1. Start job
        resp = requests.post(
            f"{EXTRACTION_SERVER_URL}/start_qdrant",
            json={
                "book_id": book_id,
                "ready_path": ready_path,
                "prop_path": prop_path,
                "base_dir": str(BASE_DIR)
            },
            timeout=60
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Qdrant server rejected start: {resp.text}")

        jobs_col.update_one(
            {"job_id": job_id},
            {"$set": {"qdrant_progress": {"status": "running", "percent": 10, "message": "Queued on worker..."}, "updated_at": _now_iso()}}
        )

        # 2. Poll status
        while True:
            time.sleep(5)
            status_resp = requests.get(f"{EXTRACTION_SERVER_URL}/qdrant_status/{book_id}", timeout=15).json()
            is_finished = status_resp.get("is_finished", False)
            overall = status_resp.get("status", "running")

            if is_finished:
                if overall == "failed":
                    raise RuntimeError("Qdrant worker failed.")
                break

        # 3. Get result
        res = requests.get(f"{EXTRACTION_SERVER_URL}/get_qdrant_result/{book_id}", timeout=60).json()
        chunks_stored = res.get("chunks_stored", 0)

        # 4. Write local _chunks.json for confidence report using _write_chunks_log
        from processing.ingest_qdrant import _write_chunks_log
        with open(ready_path, "r", encoding="utf-8") as f:
            ready_chunks = json.load(f)
        with open(prop_path, "r", encoding="utf-8") as f:
            props = json.load(f)
        _write_chunks_log(ready_chunks, book_id, str(BASE_DIR), len(props), len(ready_chunks))

        jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "qdrant_progress": {
                        "status": "done",
                        "percent": 100,
                        "chunks_stored": chunks_stored,
                    }
                }
            },
        )
        return {"success": True, "chunks": chunks_stored}

    except Exception as e:
        jobs_col.update_one(
            {"job_id": job_id},
            {"$set": {"qdrant_progress": {"status": "failed", "error": str(e)}}},
        )
        raise


@log_process
def _run_neo4j_stage(job_id, book_id, ready_path, jobs_col):
    """
    5-layer GLiNER + Regex + Neo4j graph ingestion.
    Only called when neo4j_progress.status != "done".

    FIX 1: Module was "processing.ingest_graph" — does not exist.
            Correct module: "processing.ingest_neo4j"

    FIX 2: Function was "run_graph_ingestion" — does not exist.
            Correct function: "run_neo4j_ingestion"

    FIX 3: ready_path is now passed to this stage.
            run_neo4j_ingestion requires it. The original signature omitted it.
    """
    try:
        import requests
        import time

        
        # 1. Start job
        resp = requests.post(
            f"{EXTRACTION_SERVER_URL}/start_neo4j",
            json={
                "book_id": book_id,
                "ready_path": ready_path,
                "base_dir": str(BASE_DIR)
            },
            timeout=60
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Neo4j server rejected start: {resp.text}")

        jobs_col.update_one(
            {"job_id": job_id},
            {"$set": {"neo4j_progress": {"status": "running", "percent": 10, "message": "Queued on worker..."}, "updated_at": _now_iso()}}
        )

        # 2. Poll status
        while True:
            time.sleep(5)
            status_resp = requests.get(f"{EXTRACTION_SERVER_URL}/neo4j_status/{book_id}", timeout=15).json()
            is_finished = status_resp.get("is_finished", False)
            overall = status_resp.get("status", "running")

            if is_finished:
                if overall == "failed":
                    raise RuntimeError("Neo4j worker failed.")
                break

        # 3. Get result
        result = requests.get(f"{EXTRACTION_SERVER_URL}/get_neo4j_result/{book_id}", timeout=60).json()

        # Build graph_report that matches confidence_report + frontend expectations
        graph_report = {
            "mentions": result.get("entities_written", 0),
            "distinct_entities": result.get("entities_written", 0),
            "sections_written": result.get("sections_written", 0),
            "specs_written": result.get("specs_written", 0),
            "tables_written": result.get("tables_written", 0),
            "cooccurrence_edges": result.get("cooccurrence_edges", 0),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
            "entity_type_counts": {},
            "chunks_with_entity": result.get("sections_written", 0),
        }

        jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "neo4j_progress": {
                        "status": "done",
                        "percent": 100,
                        "entities_created": result.get("entities_written", 0),
                        "graph_report": graph_report,
                    }
                }
            },
        )
        return {
            "success": True,
            "entities_created": result.get("entities_written", 0),
            "jobs_completed": result.get("sections_written", 0),
            "graph_report": graph_report,
        }

    except Exception as e:
        jobs_col.update_one(
            {"job_id": job_id},
            {"$set": {"neo4j_progress": {"status": "failed", "error": str(e)}}},
        )
        raise


# ---------------------------------------------------------------------------
# Confidence report
# ---------------------------------------------------------------------------


@log_process
def _generate_confidence_report(
    book_id: str,
    ready_path: str,
    graph_report: dict = None,
) -> dict:
    """
    Assembles the confidence report shown on the frontend success screen.
    Reads _ready.json which contains the final chunks.
    Groups chunks by page_number to determine page-level statistics.
    """
    try:
        with open(ready_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        return {"error": "Could not read _ready.json"}

    page_word_counts = {}
    total_words = 0

    for chunk in chunks:
        text = chunk.get("content") or chunk.get("text", "")
        wc = len(text.split())
        total_words += wc

        page_range = chunk.get("page_range")
        if isinstance(page_range, dict):
            page_num = page_range.get("start", 0)
        elif isinstance(page_range, list) and page_range:
            page_num = page_range[0]
        else:
            page_num = chunk.get("page_number", 0)

        page_word_counts[page_num] = page_word_counts.get(page_num, 0) + wc

    good_pages = []
    short_pages = []
    blank_pages = []

    for page_num, wc in page_word_counts.items():
        if wc >= 50:
            good_pages.append(page_num)
        elif wc >= 10:
            short_pages.append(page_num)
        else:
            blank_pages.append(page_num)

    total_unique_pages = len(page_word_counts)
    
    # In the new pipeline, chunks are pre-computed in _ready.json.
    # Therefore, all text in _ready.json has produced chunks.
    zero_chunk_pages = [] 
    
    # Calculate multi-chunk pages
    chunks_by_page = {}
    for chunk in chunks:
        page_range = chunk.get("page_range")
        if isinstance(page_range, dict):
            pn = page_range.get("start", 0)
        elif isinstance(page_range, list) and page_range:
            pn = page_range[0]
        else:
            pn = chunk.get("page_number", 0)
        chunks_by_page.setdefault(pn, []).append(chunk)

    multi_chunk_pages = [pn for pn, cl in chunks_by_page.items() if len(cl) > 1]
    
    avg_words = round(total_words / max(len(chunks), 1), 1)
    coverage_pct = round(len(good_pages) / max(total_unique_pages, 1) * 100, 1)

    report = {
        "total_pages": total_unique_pages,
        "good_pages": len(good_pages),
        "short_pages": len(short_pages),
        "blank_image_pages": len(blank_pages),
        "total_chunks": len(chunks),
        "multi_chunk_pages": len(multi_chunk_pages),
        "zero_chunk_pages": zero_chunk_pages,
        "avg_words_per_chunk": avg_words,
        "coverage_percent": coverage_pct,
    }

    if graph_report:
        report["entity_mentions"] = graph_report.get("mentions", 0)
        report["distinct_entities"] = graph_report.get("distinct_entities", 0)
        report["chunks_with_entity"] = graph_report.get("chunks_with_entity", 0)
        report["entity_type_counts"] = graph_report.get("entity_type_counts", {})

    logger.info("Confidence Report — %s", book_id)
    logger.info(
        "  Pages    : %d total | %d good | %d short | %d sparse",
        total_unique_pages,
        len(good_pages),
        len(short_pages),
        len(blank_pages),
    )
    logger.info("  Vectors  : %d total | avg %.1f words/chunk", len(chunks), avg_words)
    logger.info("  Coverage : %.1f%%", coverage_pct)
    if graph_report:
        logger.info(
            "  Graph    : %d mentions | %d specs | %d tables",
            graph_report.get("mentions", 0),
            graph_report.get("specs_written", 0),
            graph_report.get("tables_written", 0),
        )
    if zero_chunk_pages:
        logger.warning("  Sections with no vectors: %s", zero_chunk_pages[:10])

    return report


# ---------------------------------------------------------------------------
# PHASE 1
# ---------------------------------------------------------------------------


@log_process
def run_phase1(job: dict, jobs_col) -> tuple:
    """
    Runs Steps 1-3.
    Saves ready_path and prop_path to the job doc in MongoDB on success.
    Returns (ready_path, prop_path).
    """
    job_id = job["job_id"]
    book_id = job["book_id"]
    pdf_path = job["pdf_path"]

    callback = _make_callback(jobs_col, job_id)
    logger.info("═══ Phase 1 | %s | Book: %s ═══", job_id, book_id)

    jobs_col.update_one(
        {"job_id": job_id}, {"$set": {"status": "extracting", "started_at": _now_iso()}}
    )

    try:
        callback(percent=2, stage="Preparing", message="Validating uploaded PDF...")
        pdf = Path(pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        callback(percent=5, stage="Preparing", message=f"File ready: {pdf.name}")

        from extraction.master import run_extraction

        callback(
            percent=7,
            stage="Text Extraction",
            message="Starting distributed extraction...",
        )
        ocr_enabled = job.get("ocr_enabled", False)
        run_extraction(book_id, pdf_path, str(BASE_DIR), callback, ocr_enabled=ocr_enabled)

        from processing.chunk import run_chunking

        callback(
            percent=51,
            stage="Chunking",
            message="Splitting document by section headers...",
        )
        ready_path = run_chunking(book_id, str(BASE_DIR), callback)

        from processing.triple_rep import run_triple_rep

        callback(
            percent=56,
            stage="Table Processing",
            message="Building triple representations for tables...",
        )
        run_triple_rep(book_id, ready_path, callback)

        from processing.build_metadata import run_build_metadata

        callback(
            percent=58,
            stage="Metadata Processing",
            message="Building page metadata and format tables...",
        )
        run_build_metadata(book_id, ready_path, str(BASE_DIR), callback)

        from processing.propositions import run_propositions

        callback(
            percent=60,
            stage="Proposition Extraction",
            message="Extracting atomic propositions from sections...",
        )
        prop_path = run_propositions(book_id, ready_path, str(BASE_DIR), callback)

        # ── Write checkpoint paths to MongoDB BEFORE returning ────────────────
        # These survive server restarts. /resume reads them to start Phase 2.
        jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "extraction_done",
                    "ready_path": ready_path,
                    "prop_path": prop_path,
                    "percent": 65,
                    "stage": "Extraction Complete",
                    "message": "Checkpoints saved. Starting ingestion...",
                    "updated_at": _now_iso(),
                }
            },
        )
        logger.info(
            "Phase 1 checkpoints saved — ready_path and prop_path written to MongoDB"
        )
        return ready_path, prop_path

    except Exception as e:
        error_msg = str(e)
        logger.error("Phase 1 FAILED: %s", error_msg)
        jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "extraction_failed",
                    "error": error_msg,
                    "updated_at": _now_iso(),
                }
            },
        )
        raise


# ---------------------------------------------------------------------------
# PHASE 2
# ---------------------------------------------------------------------------


@log_process
def run_phase2(job: dict, jobs_col, mongo_db):
    """
    Reads checkpoint paths from MongoDB.
    Runs Qdrant + Neo4j in parallel, skipping any stage already marked "done".
    """
    job_id = job["job_id"]
    book_id = job["book_id"]
    user_id = job.get("user_id") or job.get("uploaded_by", "")

    # Always read paths from MongoDB — survives server restarts
    job_doc = jobs_col.find_one({"job_id": job_id}) or {}
    ready_path = job_doc.get("ready_path") or job.get("ready_path", "")
    prop_path = job_doc.get("prop_path") or job.get("prop_path", "")

    logger.info("═══ Phase 2 | %s | Book: %s ═══", job_id, book_id)

    if not ready_path or not prop_path:
        _fail_phase2(
            jobs_col,
            job_id,
            "Checkpoint paths not found. Phase 1 must complete before Phase 2.",
        )
        return

    # Verify files still exist on disk
    missing = [p for p in [ready_path, prop_path] if not Path(p).exists()]
    if missing:
        _fail_phase2(
            jobs_col,
            job_id,
            "Checkpoint file(s) missing from disk: "
            + ", ".join(Path(p).name for p in missing)
            + ". Phase 1 must be re-run.",
        )
        return

    jobs_col.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "ingesting",
                "percent": 65,
                "stage": "Building Knowledge Base",
                "message": "Starting vector embedding and graph construction...",
                "updated_at": _now_iso(),
            }
        },
    )

    # ── Per-stage skip logic ──────────────────────────────────────────────────
    existing = jobs_col.find_one({"job_id": job_id}) or {}
    qdrant_status = existing.get("qdrant_progress", {}).get("status", "")
    neo4j_status = existing.get("neo4j_progress", {}).get("status", "")

    qdrant_done = qdrant_status == "done"
    neo4j_done = neo4j_status == "done"

    if qdrant_done:
        logger.info("Qdrant already done for %s — skipping.", book_id)
    if neo4j_done:
        logger.info("Neo4j already done for %s — skipping.", book_id)

    qdrant_stored = existing.get("qdrant_progress", {})
    neo4j_stored = existing.get("neo4j_progress", {})

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_qdrant = (
                None
                if qdrant_done
                else executor.submit(
                    _run_qdrant_stage,
                    job_id,
                    book_id,
                    ready_path,
                    prop_path,
                    jobs_col,
                )
            )

            future_neo4j = (
                None
                if neo4j_done
                else executor.submit(
                    _run_neo4j_stage,
                    job_id,
                    book_id,
                    ready_path,
                    jobs_col,
                )
            )

            # Poll progress
            while True:
                q_done = qdrant_done or (
                    future_qdrant is not None and future_qdrant.done()
                )
                n_done = neo4j_done or (
                    future_neo4j is not None and future_neo4j.done()
                )
                if q_done and n_done:
                    break

                time.sleep(3)
                doc = (
                    jobs_col.find_one(
                        {"job_id": job_id}, {"qdrant_progress": 1, "neo4j_progress": 1}
                    )
                    or {}
                )
                q_pct = (
                    100
                    if qdrant_done
                    else doc.get("qdrant_progress", {}).get("percent", 0)
                )
                n_pct = (
                    100
                    if neo4j_done
                    else doc.get("neo4j_progress", {}).get("percent", 0)
                )
                combined = 65 + int(((q_pct + n_pct) / 2) * 0.34)
                jobs_col.update_one(
                    {"job_id": job_id},
                    {
                        "$set": {
                            "percent": min(combined, 99),
                            "stage": "Building Knowledge Base",
                            "message": f"Vectors: {q_pct}% | Graph: {n_pct}%",
                            "updated_at": _now_iso(),
                        }
                    },
                )

            # Collect results — .result() re-raises any exception from the thread
            qdrant_result = (
                future_qdrant.result()
                if future_qdrant is not None
                else {"success": True, "chunks": qdrant_stored.get("chunks_stored", 0)}
            )
            neo4j_result = (
                future_neo4j.result()
                if future_neo4j is not None
                else {
                    "success": True,
                    "entities_created": neo4j_stored.get("entities_created", 0),
                    "jobs_completed": 0,
                    "graph_report": neo4j_stored.get("graph_report", {}),
                }
            )

        # Confidence report
        graph_report_dict = neo4j_result.get("graph_report") or {}
        confidence = _generate_confidence_report(book_id, ready_path, graph_report_dict)

        # Library save
        library_col = mongo_db["library"]
        library_col.update_one(
            {"book_id": book_id},
            {
                "$set": {
                    "book_id": book_id,
                    "book_title": book_id.replace("_", " ").replace("-", " "),
                    "uploaded_by": user_id,
                    "status": "ready",
                    "total_sections": confidence.get("total_pages", 0),
                    "qdrant_chunks_stored": qdrant_result.get("chunks", 0),
                    "neo4j_entities": neo4j_result.get("entities_created", 0),
                    "confidence_report": confidence,
                    "completed_at": _now_iso(),
                }
            },
            upsert=True,
        )

        jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "percent": 100,
                    "stage": "Complete",
                    "message": f"{book_id} is ready for queries.",
                    "confidence_report": confidence,
                    "completed_at": _now_iso(),
                }
            },
        )
        logger.info("Phase 2 complete for %s", book_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("Phase 2 FAILED for %s: %s", job_id, error_msg)
        _fail_phase2(jobs_col, job_id, error_msg)


@time_it
def _fail_phase2(jobs_col, job_id: str, error_msg: str):
    """Marks ingestion_failed — checkpoint files stay on disk for /resume."""
    jobs_col.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "ingestion_failed",
                "error": error_msg,
                "updated_at": _now_iso(),
            }
        },
    )


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------


@log_process
def run_pipeline(job: dict, jobs_col, mongo_db):
    """
    Full pipeline entry point called by main_api.py queue_worker.
    phase2_only → jumps directly to Phase 2 (resume path).
    """
    if job.get("phase") == "phase2_only":
        run_phase2(job, jobs_col, mongo_db)
        return

    job_id = job["job_id"]
    book_id = job["book_id"]
    logger.info("═══ Full pipeline | %s | Book: %s ═══", job_id, book_id)

    try:
        run_phase1(job, jobs_col)
    except Exception:
        return  # already marked extraction_failed

    run_phase2(job, jobs_col, mongo_db)
