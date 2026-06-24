"""
processing/ingest_qdrant.py
----------------------------
STEP 4 of the new processing pipeline.

Replaces the old processing/ingest_vectors.py entirely.

Reads  → data/checkpoints/{book_id}_ready.json        (sections)
         data/checkpoints/{book_id}_propositions.json  (atomic facts)
Uses   → portable/nomic/   (local Nomic embed-text-v1.5, offline)
Target → http://localhost:6333  (Qdrant, hardcoded)
Writes → data/qdrant/{book_id}_chunks.json
         ⚠️  FILENAME LOCKED — pipeline_controller._generate_confidence_report
             reads this exact path. Format is kept backward-compatible.

TWO-COLLECTION ARCHITECTURE
────────────────────────────
Collection "propositions"
  One vector per atomic sentence or table row proposition.
  Purpose: precise fact retrieval.
  A query "what is the thrust of Vikas engine" hits the exact
  proposition "Vikas Engine Thrust is 799 kN." with near-perfect
  cosine similarity instead of a diluted 600-word paragraph.

  Payload per point:
    book_id, parent_chunk_id, section_path, page, source_type
    (source_type: "text" | "table_row" | "table_full")

Collection "sections"
  One vector per full section chunk (text or table linearized_text).
  Purpose: full context retrieval — used AFTER propositions search.
  The parent_chunk_id on each proposition points to a section ID.
  Part B fetches the full section text to send as LLM context.

  Payload per point:
    book_id, section_path, page_range, chunk_type

RETRIEVAL FLOW (Part B, not implemented here)
  1. Embed user query
  2. Search "propositions" collection → top-K precise matches
  3. Collect unique parent_chunk_ids from matches
  4. Fetch those IDs from "sections" collection
  5. Send full section texts as LLM context
  This is "small-to-big retrieval" and produces dramatically better
  answers than searching a single flat collection of large chunks.

CONFIDENCE REPORT COMPATIBILITY
  pipeline_controller._generate_confidence_report() reads:
    _ready.json  → expects items with "text" and "page_number" fields
    _chunks.json → expects items with "text" and "page_number" fields
  Our new format uses "content" and "page_range".
  This file writes _chunks.json with the old compatible field names.
  NOTE: pipeline_controller._generate_confidence_report also needs
  a 2-line patch for _ready.json reading (handled in pipeline_controller).

Called by pipeline_controller.py:
    from processing.ingest_qdrant import run_qdrant_ingestion
    chunks = run_qdrant_ingestion(book_id, ready_path, prop_path,
                                  str(BASE_DIR), qdrant_cb)
"""

import json
from parta.logger import time_it, async_time_it, logger
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict

logging.getLogger("transformers").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED CONFIGURATION — no environment variables
# ─────────────────────────────────────────────────────────────────────────────
QDRANT_URL           = "https://9a5b0165-5dab-4d30-8b0c-95319c7c1191.us-east-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY       = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODVhODM3MDEtYWYyMS00MWQ2LTgzOTItN2FmYWExZTQyNTI2In0.FYpoWi_q1lwgOs58R_OqsboC32qWhl60LJXv3Rtg4OY"
COLLECTION_PROPS     = "RAG_PROPOSITIons"
COLLECTION_SECTIONS  = "RAG_sections"
EMBEDDING_DIM        = 768             # Nomic embed-text-v1.5 output size
BATCH_SIZE           = 64              # points per upsert call

# Nomic prefix — required for correct embedding behaviour
NOMIC_QUERY_PREFIX   = "search_document: "

# ─────────────────────────────────────────────────────────────────────────────
# LAZY-LOADED GLOBALS — initialised once per process
# ─────────────────────────────────────────────────────────────────────────────
_embed_model = None
_q_client    = None


@time_it
def _get_embed_model(base_dir: str):
    """Loads Nomic model once and caches it."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    from sentence_transformers import SentenceTransformer
    model_path = Path(base_dir) / "portable" / "nomic"

    if not model_path.exists():
        raise FileNotFoundError(
            f"[QDRANT] Nomic model not found at: {model_path}\n"
            f"         Place the offline nomic model in portable/nomic/"
        )

    logger.info(f"[QDRANT] Loading Nomic embedding model from {model_path}...")
    t0 = time.time()
    _embed_model = SentenceTransformer(
        str(model_path),
        trust_remote_code=True,
        device="cpu",
    )
    t1 = time.time()
    logger.info(f"[QDRANT] Model loaded: Nomic embed-text-v1.5 in {t1 - t0:.2f}s")
    return _embed_model


@time_it
def _get_qdrant_client():
    """Creates Qdrant client once and caches it."""
    global _q_client
    if _q_client is not None:
        return _q_client

    from qdrant_client import QdrantClient
    _q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info(f"[QDRANT] Model loaded: QdrantClient connected to {QDRANT_URL}")
    return _q_client


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION SETUP
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _ensure_collection(client, name: str):
    """
    Creates collection if it does not exist.
    Creates payload indexes for efficient filtered search.
    MERGE-safe: called on every run, safe if collection already exists.
    """
    from qdrant_client import models as qm

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=EMBEDDING_DIM,
                distance=qm.Distance.COSINE,
            ),
        )
        print(f"[QDRANT] Created collection: '{name}'")
    else:
        print(f"[QDRANT] Collection '{name}' already exists.")

    # Payload indexes — without these, filtered search scans entire collection
    _safe_create_index(client, name, "book_id",    "keyword")
    _safe_create_index(client, name, "source_type","keyword")
    _safe_create_index(client, name, "page",       "integer")


@time_it
def _safe_create_index(client, collection: str, field: str, schema: str):
    """Creates a payload index, silently ignores if it already exists."""
    from qdrant_client import models as qm

    schema_map = {
        "keyword": qm.PayloadSchemaType.KEYWORD,
        "integer": qm.PayloadSchemaType.INTEGER,
    }
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema_map[schema],
        )
    except Exception:
        pass  # Already exists — safe to ignore


# ─────────────────────────────────────────────────────────────────────────────
# STABLE POINT IDS
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _point_id(book_id: str, collection: str, item_id: str) -> str:
    """
    Deterministic UUID for a Qdrant point.
    Same inputs → same ID → upsert is idempotent.
    """
    key = f"{collection}::{book_id}::{item_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EMBEDDING HELPER
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _embed_batch(model, texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of texts with the Nomic prefix.
    Returns a list of float vectors.
    Nomic requires the "search_document: " prefix for correct embeddings.
    """
    prefixed = [NOMIC_QUERY_PREFIX + t for t in texts]
    vectors  = model.encode(prefixed, show_progress_bar=False)
    return [v.tolist() for v in vectors]


@time_it
def _embed_batches_with_workers(model, batches: List[List[str]]) -> List[List[List[float]]]:
    """Embeds prepared batches sequentially. Cross-machine parallelism is
    handled at the worker level via run_qdrant_batch()."""
    return [_embed_batch(model, batch) for batch in batches]


@time_it
def _upsert_prepared_batches(client, collection_name: str, prepared_batches: List[tuple], model) -> int:
    """Embeds prepared text/meta batches using workers, then upserts to Qdrant."""
    from qdrant_client import models as qm
    if not prepared_batches:
        return 0
    text_batches = [b[0] for b in prepared_batches]
    meta_batches = [b[1] for b in prepared_batches]
    vector_batches = _embed_batches_with_workers(model, text_batches)
    total = 0
    for vectors, metas in zip(vector_batches, meta_batches):
        points = [qm.PointStruct(id=m["point_id"], vector=v, payload=m["payload"]) for v, m in zip(vectors, metas)]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
    logger.info("[QDRANT] Upserted vectors | collection=%s | points=%s", collection_name, total)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION 1 — PROPOSITIONS
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _ingest_propositions(
    propositions: List[dict],
    model,
    client,
    book_id:  str,
    progress_callback,
    total_items: int,
    items_done:  int,
) -> tuple:
    """
    Embeds and upserts all propositions into the "propositions" collection.

    Returns:
        (points_upserted: int, items_done: int)
    """
    points_upserted = 0
    batch_texts  = []
    batch_meta   = []
    prepared_batches = []

    def _flush(texts, metas):
        if not texts:
            return
        prepared_batches.append((list(texts), list(metas)))

    for prop in propositions:
        text = prop.get("text", "").strip()
        if not text:
            continue

        point_id = _point_id(book_id, COLLECTION_PROPS, prop["proposition_id"])

        payload = {
            "book_id":         book_id,
            "parent_chunk_id": prop.get("parent_chunk_id"),
            "section_path":    prop.get("section_path", []),
            "page":            prop.get("page", 0),
            "source_type":     prop.get("source_type", "text"),
            "text":            text,   # stored for retrieval without re-fetch
        }

        batch_texts.append(text)
        batch_meta.append({"point_id": point_id, "payload": payload})
        items_done += 1

        if len(batch_texts) >= BATCH_SIZE:
            _flush(batch_texts, batch_meta)
            batch_texts = []
            batch_meta  = []

            if progress_callback:
                pct = 65 + int((items_done / max(total_items, 1)) * 10)
                progress_callback(
                    percent=min(pct, 74),
                    stage="Vector Ingestion",
                    message=(
                        f"Embedding propositions: "
                        f"{points_upserted} stored so far..."
                    ),
                    extra={
                        "chunks_done":  points_upserted,
                        "total_chunks": total_items,
                    },
                )

    # Flush remainder
    _flush(batch_texts, batch_meta)
    points_upserted = _upsert_prepared_batches(client, COLLECTION_PROPS, prepared_batches, model)
    return points_upserted, items_done


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION 2 — SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _ingest_sections(
    chunks:   List[dict],
    model,
    client,
    book_id:  str,
    progress_callback,
    total_items: int,
    items_done:  int,
) -> tuple:
    """
    Embeds and upserts all section chunks into the "sections" collection.

    For text chunks  → embeds chunk["content"]
    For table chunks → embeds chunk["linearized_text"] (richer than raw markdown)
                       falls back to chunk["content"] if linearized_text missing

    Returns:
        (points_upserted: int, items_done: int)
    """
    points_upserted = 0
    batch_texts = []
    batch_meta  = []
    prepared_batches = []

    def _flush(texts, metas):
        if not texts:
            return
        prepared_batches.append((list(texts), list(metas)))

    for chunk in chunks:
        chunk_type = chunk.get("type", "text")
        chunk_id   = chunk.get("chunk_id", "")

        if chunk_type == "text":
            text = chunk.get("content", "").strip()
        else:
            # Table: prefer linearized_text, fall back to raw markdown content
            text = (
                chunk.get("linearized_text", "").strip()
                or chunk.get("content", "").strip()
            )

        if not text or len(text) < 20:
            continue

        point_id = _point_id(book_id, COLLECTION_SECTIONS, chunk_id)
        page_range = chunk.get("page_range", {"start": 0, "end": 0})

        payload = {
            "book_id":      book_id,
            "chunk_id":     chunk_id,
            "section_path": chunk.get("section_path", []),
            "page_range":   page_range,
            "chunk_type":   chunk_type,
            "parent_id":    chunk.get("parent_id"),
            "text":         text,   # full section text stored in payload
        }

        batch_texts.append(text)
        batch_meta.append({"point_id": point_id, "payload": payload})
        items_done += 1

        if len(batch_texts) >= BATCH_SIZE:
            _flush(batch_texts, batch_meta)
            batch_texts = []
            batch_meta  = []

            if progress_callback:
                pct = 74 + int((items_done / max(total_items, 1)) * 6)
                progress_callback(
                    percent=min(pct, 79),
                    stage="Vector Ingestion",
                    message=(
                        f"Embedding sections: "
                        f"{points_upserted} stored so far..."
                    ),
                    extra={
                        "chunks_done":  points_upserted,
                        "total_chunks": total_items,
                    },
                )

    _flush(batch_texts, batch_meta)
    points_upserted = _upsert_prepared_batches(client, COLLECTION_SECTIONS, prepared_batches, model)
    return points_upserted, items_done


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE REPORT COMPATIBILITY OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _write_chunks_log(
    chunks:    List[dict],
    book_id:   str,
    base_dir:  str,
    prop_count: int,
    sect_count: int,
):
    """
    Writes data/qdrant/{book_id}_chunks.json in the format that
    pipeline_controller._generate_confidence_report() expects.

    That function reads:
      c.get("page_number")  → int
      c.get("text")         → str (for word count)

    Our new format has "page_range" and "content", so we map them.
    We write one entry per section chunk — same granularity as before.

    Also writes a summary header entry for total count visibility.
    """
    out_dir = Path(base_dir) / "data" / "qdrant"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{book_id}_chunks.json"

    compat_entries = []
    for chunk in chunks:
        page_range = chunk.get("page_range", {})
        page_num   = page_range.get("start", 0)

        # Use content for text, linearized_text for tables
        if chunk.get("type") == "text":
            text = chunk.get("content", "")
        else:
            text = (
                chunk.get("linearized_text", "")
                or chunk.get("content", "")
            )

        compat_entries.append({
            # Fields expected by confidence report
            "page_number": page_num,
            "text":        text,
            # Extended fields for debugging
            "chunk_id":    chunk.get("chunk_id"),
            "chunk_type":  chunk.get("type"),
            "section_path": chunk.get("section_path", []),
            "book_id":     book_id,
        })

    # Write log
    log = {
        "book_id":            book_id,
        "total_propositions": prop_count,
        "total_sections":     sect_count,
        "chunks":             compat_entries,   # ← confidence report reads this
    }

    # The confidence report does json.load() and iterates the result directly
    # as a list. So we must write the list, not a dict wrapper.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compat_entries, f, indent=2, ensure_ascii=False)

    print(f"[QDRANT] Chunk log written: {out_path.name}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — called by pipeline_controller.py
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def run_qdrant_ingestion(
    book_id:           str,
    ready_path:        str,
    prop_path:         str,
    base_dir:          str,
    progress_callback = None,
) -> int:
    """
    Main entry point called by pipeline_controller.py

    Args:
        book_id   : e.g. "PSLV-C50"
        ready_path: path to {book_id}_ready.json        (sections)
        prop_path : path to {book_id}_propositions.json (atomic facts)
        base_dir  : project root
        progress_callback : optional fn(percent, stage, message, extra=None)

    Returns:
        int — total number of vectors stored (propositions + sections)

    Raises:
        FileNotFoundError if either input file is missing
        RuntimeError if Qdrant is unreachable or Nomic model missing
    """
    ready_file = Path(ready_path)
    prop_file  = Path(prop_path)

    if not ready_file.exists():
        raise FileNotFoundError(
            f"[QDRANT] _ready.json not found: {ready_path}\n"
            f"         Run chunk.py and triple_rep.py first."
        )
    if not prop_file.exists():
        raise FileNotFoundError(
            f"[QDRANT] _propositions.json not found: {prop_path}\n"
            f"         Run propositions.py first."
        )

    if progress_callback:
        progress_callback(
            percent=65,
            stage="Vector Ingestion",
            message="Loading Nomic embedding model...",
        )

    # ── Load model and client ─────────────────────────────────────────────────
    model  = _get_embed_model(base_dir)
    client = _get_qdrant_client()

    # ── Ensure both collections exist with indexes ────────────────────────────
    _ensure_collection(client, COLLECTION_PROPS)
    _ensure_collection(client, COLLECTION_SECTIONS)

    # ── Load data ─────────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(
            percent=66,
            stage="Vector Ingestion",
            message="Reading propositions and sections...",
        )

    with open(prop_file, "r", encoding="utf-8") as f:
        propositions = json.load(f)

    with open(ready_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_items = len(propositions) + len(chunks)

    print(f"\n[QDRANT] Starting dual-collection ingestion for '{book_id}'")
    print(f"[QDRANT] {len(propositions)} propositions | {len(chunks)} sections")
    logger.info("[QDRANT] Starting vector ingestion | book=%s | propositions=%s | sections=%s", book_id, len(propositions), len(chunks))

    t_start    = time.perf_counter()
    items_done = 0

    # ── Ingest propositions ───────────────────────────────────────────────────
    if progress_callback:
        progress_callback(
            percent=67,
            stage="Vector Ingestion",
            message=(
                f"Embedding {len(propositions)} propositions into "
                f"'{COLLECTION_PROPS}' collection..."
            ),
            extra={"chunks_done": 0, "total_chunks": total_items},
        )

    print(f"[QDRANT] Embedding propositions → '{COLLECTION_PROPS}'...")
    prop_count, items_done = _ingest_propositions(
        propositions, model, client, book_id,
        progress_callback, total_items, items_done,
    )
    print(f"[QDRANT] {prop_count} propositions stored")

    # ── Ingest sections ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(
            percent=74,
            stage="Vector Ingestion",
            message=(
                f"Embedding {len(chunks)} sections into "
                f"'{COLLECTION_SECTIONS}' collection..."
            ),
            extra={"chunks_done": prop_count, "total_chunks": total_items},
        )

    print(f"[QDRANT] Embedding sections → '{COLLECTION_SECTIONS}'...")
    sect_count, items_done = _ingest_sections(
        chunks, model, client, book_id,
        progress_callback, total_items, items_done,
    )
    print(f"[QDRANT] {sect_count} sections stored")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed     = time.perf_counter() - t_start
    total_stored = prop_count + sect_count

    print(f"\n[QDRANT] ═══ Ingestion Complete ═══")
    print(f"         Propositions : {prop_count}")
    print(f"         Sections     : {sect_count}")
    print(f"         Total vectors: {total_stored}")
    print(f"         Time         : {elapsed:.1f}s")

    if progress_callback:
        progress_callback(
            percent=80,
            stage="Vector Ingestion",
            message=(
                f"All vectors stored. "
                f"{prop_count} propositions + {sect_count} sections "
                f"in {elapsed:.1f}s."
            ),
            extra={
                "chunks_done":  total_stored,
                "total_chunks": total_stored,
            },
        )

    # ── Write compatibility log for confidence report ─────────────────────────
    _write_chunks_log(chunks, book_id, base_dir, prop_count, sect_count)

    return total_stored


@time_it
def run_qdrant_batch(
    book_id:       str,
    ready_path:    str,
    prop_path:     str,
    base_dir:      str,
    batch_start:   int,
    batch_count:   int,
    batch_kind:    str = "propositions",
) -> int:
    """
    Distributed batch entry point — processes a slice of propositions OR sections.

    batch_kind:
        "propositions" → embeds propositions[batch_start : batch_start+batch_count]
        "sections"     → embeds chunks[batch_start : batch_start+batch_count]

    Each worker loads its own Nomic model, connects to Qdrant independently.
    Upserts use deterministic point IDs so concurrent workers are idempotent.

    Returns:
        int — number of vectors stored by this batch
    """
    from qdrant_client import models as qm

    model  = _get_embed_model(base_dir)
    client = _get_qdrant_client()

    _ensure_collection(client, COLLECTION_PROPS)
    _ensure_collection(client, COLLECTION_SECTIONS)

    t_start = time.perf_counter()

    if batch_kind == "propositions":
        with open(prop_path, "r", encoding="utf-8") as f:
            all_props = json.load(f)
        items = all_props[batch_start : batch_start + batch_count]
        logger.info(
            "[QDRANT-BATCH] Embedding propositions %d–%d of %d for '%s'",
            batch_start, batch_start + len(items) - 1, len(all_props), book_id,
        )

        batch_texts = []
        batch_meta  = []
        prepared_batches = []

        def _flush(texts, metas):
            if texts:
                prepared_batches.append((list(texts), list(metas)))

        for prop in items:
            text = prop.get("text", "").strip()
            if not text:
                continue
            point_id = _point_id(book_id, COLLECTION_PROPS, prop["proposition_id"])
            payload = {
                "book_id":         book_id,
                "parent_chunk_id": prop.get("parent_chunk_id"),
                "section_path":    prop.get("section_path", []),
                "page":            prop.get("page", 0),
                "source_type":     prop.get("source_type", "text"),
                "text":            text,
            }
            batch_texts.append(text)
            batch_meta.append({"point_id": point_id, "payload": payload})
            if len(batch_texts) >= BATCH_SIZE:
                _flush(batch_texts, batch_meta)
                batch_texts, batch_meta = [], []

        _flush(batch_texts, batch_meta)
        stored = _upsert_prepared_batches(client, COLLECTION_PROPS, prepared_batches, model)

    else:  # sections
        with open(ready_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        items = all_chunks[batch_start : batch_start + batch_count]
        logger.info(
            "[QDRANT-BATCH] Embedding sections %d–%d of %d for '%s'",
            batch_start, batch_start + len(items) - 1, len(all_chunks), book_id,
        )

        batch_texts = []
        batch_meta  = []
        prepared_batches = []

        def _flush(texts, metas):
            if texts:
                prepared_batches.append((list(texts), list(metas)))

        for chunk in items:
            chunk_type = chunk.get("type", "text")
            chunk_id   = chunk.get("chunk_id", "")
            if chunk_type == "text":
                text = chunk.get("content", "").strip()
            else:
                text = (chunk.get("linearized_text", "").strip()
                        or chunk.get("content", "").strip())
            if not text or len(text) < 20:
                continue
            point_id = _point_id(book_id, COLLECTION_SECTIONS, chunk_id)
            page_range = chunk.get("page_range", {"start": 0, "end": 0})
            payload = {
                "book_id":      book_id,
                "chunk_id":     chunk_id,
                "section_path": chunk.get("section_path", []),
                "page_range":   page_range,
                "chunk_type":   chunk_type,
                "parent_id":    chunk.get("parent_id"),
                "text":         text,
            }
            batch_texts.append(text)
            batch_meta.append({"point_id": point_id, "payload": payload})
            if len(batch_texts) >= BATCH_SIZE:
                _flush(batch_texts, batch_meta)
                batch_texts, batch_meta = [], []

        _flush(batch_texts, batch_meta)
        stored = _upsert_prepared_batches(client, COLLECTION_SECTIONS, prepared_batches, model)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[QDRANT-BATCH] Batch done | kind=%s | %d–%d | stored=%d | %.1fs",
        batch_kind, batch_start, batch_start + len(items) - 1, stored, elapsed,
    )
    return stored


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE MODE — run directly for debugging
# python processing/ingest_qdrant.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    BASE_DIR = Path(__file__).resolve().parent.parent

    def _print_callback(percent, stage, message, extra=None):
        print(f"  [{percent}%] {stage}: {message}")

    checkpoint_dir = BASE_DIR / "data" / "checkpoints"
    ready_files    = sorted(checkpoint_dir.glob("*_ready.json"))

    if not ready_files:
        print("[QDRANT] No *_ready.json files found.")
        print("         Run chunk.py, triple_rep.py, propositions.py first.")
        sys.exit(1)

    for rf in ready_files:
        book_id  = rf.stem.replace("_ready", "")
        prop_file = checkpoint_dir / f"{book_id}_propositions.json"

        if not prop_file.exists():
            print(f"[QDRANT] No propositions file for {book_id}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Ingesting: {book_id}")
        print(f"{'='*60}")

        try:
            total = run_qdrant_ingestion(
                book_id   = book_id,
                ready_path = str(rf),
                prop_path  = str(prop_file),
                base_dir   = str(BASE_DIR),
                progress_callback = _print_callback,
            )
            print(f"\n  Total vectors stored: {total}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
