"""
partb/retrieval/pipeline.py
----------------------------
Hybrid retrieval pipeline — now with page-level context expansion.

CHANGES IN THIS VERSION vs previous:

  NEW — Step 7: Page Expansion
  ─────────────────────────────
  After CrossEncoder reranking (Step 6), the pipeline now expands the
  top ranked chunks to full page content using {book_id}_metadata.json
  produced by Part A's build_metadata.py.

  For rank-1 chunk on page N:
    → Fetch page N-1  (intro context, section heading)
    → Fetch page N    (primary answer — what the ranker found)
    → Fetch page N+1  (table continuation, overflow rows)
    → Cap: if combined > PAGE_EXPAND_MAX_CHARS (9000), drop N-1 first,
           then trim N+1

  For rank-2 chunk:
    → Fetch its page only (1 page, no adjacent expansion)
    → Skip if already fetched as part of rank-1 expansion (deduplication)

  Ranks 3-8:
    → Use existing chunk text logic (unchanged)

  NEW FUNCTION: load_page_content(book_id, page_number)
    Reads data/metadata/{book_id}_metadata.json → returns full_content
    for the given page, or None if page/file not found.

  NEW FUNCTION: expand_to_pages(top_chunks)
    Orchestrates the N-1/N/N+1 expansion for rank-1,
    single-page expansion for rank-2, deduplication across both.

  MODIFIED: build_context()
    Now accepts page_blocks (list of full-page content strings).
    Page blocks are injected between the Spec block and the fallback
    chunk text, filling the context budget in order.

  MODIFIED: retrieve_bundle()
    Calls expand_to_pages() between rerank_candidates() and build_context().

  ALL OTHER FUNCTIONS UNCHANGED from previous bug-fixed version:
    format_table_for_llm, format_specs_block, search_propositions,
    extract_query_entities, neo4j_sections_for_entities,
    neo4j_specs_for_terms, _parse_payload, fetch_sections_by_chunk_ids,
    search_sections_direct, merge_candidates, rerank_candidates,
    _sentences, _history_block, build_user_message, run_rag_stream
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from typing import Any, AsyncIterator

from partb.config import (
    BOOST_BOTH,
    COLLECTION_PROPS,
    COLLECTION_SECTIONS,
    ENTITY_LABELS,
    GLINER_QUERY_THRESHOLD,
    LONG_CHUNK_WORDS,
    METADATA_DIR,
    MODE_CONFIG,
    NEO4J_ENTITY_LIMIT,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PAGE_EXPAND_MAX_CHARS,
    PARTA_DIR,
    PORTABLE_DIR,
    PROP_RETRIEVE_LIMIT,
    QDRANT_URL,
    RERANKER_DIR,
    SECT_RETRIEVE_LIMIT,
)
from partb.logger import time_it, async_time_it

from partb.retrieval.prompts import get_system_prompt

logging.getLogger("transformers").setLevel(logging.ERROR)

if str(PARTA_DIR) not in sys.path:
    sys.path.insert(0, str(PARTA_DIR))

_gliner      = None
_reranker    = None
_neo_driver  = None
_qdrant_cl   = None
_nomic_model = None

# In-process cache for metadata JSON — avoids re-reading disk on every query
_metadata_cache: dict[str, dict] = {}


@time_it
def get_gliner():
    global _gliner
    if _gliner is None:
        from gliner import GLiNER
        model_dir = PORTABLE_DIR / "gliner"
        if not model_dir.exists():
            raise FileNotFoundError(f"GLiNER not found at {model_dir}")
        _gliner = GLiNER.from_pretrained(str(model_dir), local_files_only=True).to("cpu")
    return _gliner


@time_it
def get_neo4j():
    global _neo_driver
    if _neo_driver is None:
        from neo4j import GraphDatabase
        _neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        _neo_driver.verify_connectivity()
    return _neo_driver


@time_it
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        cfg_path = RERANKER_DIR / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Reranker not found at {RERANKER_DIR}")
        _reranker = CrossEncoder(str(RERANKER_DIR))
    return _reranker


@time_it
def get_qdrant():
    global _qdrant_cl
    if _qdrant_cl is None:
        from qdrant_client import QdrantClient
        from partb.config import QDRANT_API_KEY
        _qdrant_cl = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_cl


@time_it
def get_nomic():
    global _nomic_model
    if _nomic_model is None:
        from sentence_transformers import SentenceTransformer
        model_dir = PORTABLE_DIR / "nomic"
        if not model_dir.exists():
            raise FileNotFoundError(f"Nomic not found at {model_dir}")
        _nomic_model = SentenceTransformer(str(model_dir), trust_remote_code=True, device="cpu")
    return _nomic_model


@time_it
def warm_models() -> None:
    get_neo4j(); get_gliner(); get_reranker(); get_nomic(); get_qdrant()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def format_table_for_llm(chunk: dict) -> str:
    """
    Converts a table chunk into LLM-readable bullet list format.
    Priority: structured_json → linearized_text → raw text.
    """
    section_path  = chunk.get("section_path") or []
    section_label = " > ".join(section_path) if section_path else "Unknown Section"
    pr            = chunk.get("page_range") or [0, 0]
    bid           = chunk.get("book_id") or "?"

    structured = chunk.get("structured_json")
    if structured and isinstance(structured, dict):
        headers = structured.get("headers") or []
        rows    = structured.get("rows") or []

        if headers and rows:
            lines = [f"Specification Data [{section_label}] [Book: {bid} | Page: {pr[0]}-{pr[1]}]:"]
            for row in rows:
                lowered = {k.lower().strip(): v for k, v in row.items()}
                param = (
                    lowered.get("parameter") or lowered.get("item") or
                    lowered.get("name") or lowered.get("description") or
                    lowered.get("property") or lowered.get("characteristic") or
                    next(iter(row.values()), "")
                )
                value = (
                    lowered.get("value") or lowered.get("values") or
                    lowered.get("data") or lowered.get("result") or
                    lowered.get("measurement") or ""
                )
                unit = lowered.get("unit") or lowered.get("units") or ""
                if param and value:
                    unit_str = f" {str(unit).strip()}" if str(unit).strip() else ""
                    lines.append(f"  - {param}: {value}{unit_str}")
                else:
                    parts = []
                    for h in headers:
                        v = (row.get(h) or "").strip()
                        if v:
                            parts.append(f"{h}: {v}")
                    if parts:
                        lines.append("  - " + " | ".join(parts))
            if len(lines) > 1:
                return "\n".join(lines)

    linearized = (chunk.get("linearized_text") or "").strip()
    if linearized and len(linearized) > 30:
        return (
            f"Specification Data [{section_label}] [Book: {bid} | Page: {pr[0]}-{pr[1]}]:\n"
            f"{linearized}"
        )

    raw = (chunk.get("text") or chunk.get("content") or "").strip()
    if raw:
        return (
            f"Data [{section_label}] [Book: {bid} | Page: {pr[0]}-{pr[1]}]:\n"
            f"{raw}"
        )
    return ""


@time_it
def format_specs_block(specs: list[dict]) -> str:
    """Formats Neo4j Spec nodes as a verified-facts block at the top of context."""
    if not specs:
        return ""
    lines = ["=== VERIFIED TECHNICAL SPECIFICATIONS (Knowledge Graph) ==="]
    seen  = set()
    for sp in specs[:30]:
        entity  = (sp.get("entity") or "").strip()
        raw     = (sp.get("raw") or "").strip()
        if not entity or not raw:
            continue
        key = (entity.lower(), raw.lower())
        if key in seen:
            continue
        seen.add(key)
        section  = (sp.get("section") or "").strip()
        sec_note = f" [{section}]" if section else ""
        lines.append(f"  - {entity.title()}: {raw}{sec_note}")
    if len(lines) == 1:
        return ""
    lines.append("=" * 56)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL STEPS 1-6
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def search_propositions(query: str, book_ids: list[str], limit: int) -> list[dict]:
    from qdrant_client import models as qm
    client    = get_qdrant()
    model     = get_nomic()
    query_vec = model.encode("search_query: " + query, show_progress_bar=False).tolist()
    filters = None
    if book_ids:
        filters = qm.Filter(must=[qm.FieldCondition(key="book_id", match=qm.MatchAny(any=book_ids))])
    try:
        hits = client.search(collection_name=COLLECTION_PROPS, query_vector=query_vec,
                             query_filter=filters, limit=limit, with_payload=True)
    except Exception as exc:
        logging.warning("Propositions search failed: %s", exc)
        return []
    results = []
    for h in hits:
        pl = h.payload or {}
        pr = pl.get("page_range")
        page_list = ([pr.get("start", 0), pr.get("end", 0)] if isinstance(pr, dict)
                     else pr if isinstance(pr, list) else [pl.get("page", 0)] * 2)
        results.append({
            "proposition_id":  str(h.id),
            "text":            pl.get("text") or "",
            "parent_chunk_id": pl.get("parent_chunk_id"),
            "section_path":    pl.get("section_path") or [],
            "page_range":      page_list,
            "source_type":     pl.get("source_type") or "text",
            "book_id":         pl.get("book_id"),
            "score":           h.score,
        })
    return results


@time_it
def extract_query_entities(query: str) -> list[str]:
    model = get_gliner()
    try:
        preds = model.predict_entities(
            query.replace("\n", " ").strip(), ENTITY_LABELS, threshold=GLINER_QUERY_THRESHOLD
        )
    except Exception as exc:
        logging.warning("GLiNER failed: %s", exc)
        return []
    seen = set()
    terms = []
    for p in preds:
        t = (p.get("text") or "").strip().lower()
        if t and t not in seen and len(t) >= 3:
            seen.add(t); terms.append(t)
    return terms


@time_it
def neo4j_sections_for_entities(book_ids: list[str], entity_terms: list[str]) -> list[str]:
    if not entity_terms or not book_ids:
        return []
    cypher = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(s)
    WHERE s.book_id IN $book_ids AND (s:Section OR s:Subsection)
      AND ANY(t IN $terms WHERE
          toLower(e.name) = toLower(t)
          OR (size(t) >= 4 AND toLower(e.name) CONTAINS toLower(t))
          OR (size(t) >= 4 AND toLower(t) CONTAINS toLower(e.name)))
    RETURN DISTINCT s.name AS section_name LIMIT $lim
    """
    try:
        with get_neo4j().session() as session:
            rows = session.run(cypher, book_ids=book_ids, terms=entity_terms, lim=NEO4J_ENTITY_LIMIT)
            return [
                r["section_name"] for r in rows
                if r.get("section_name") is not None
                and str(r["section_name"]).strip()
            ]
    except Exception as exc:
        logging.warning("Neo4j traversal failed: %s", exc)
        return []


@time_it
def neo4j_specs_for_terms(book_ids: list[str], entity_terms: list[str]) -> list[dict]:
    if not entity_terms or not book_ids:
        return []
    cypher = """
    MATCH (e:Entity)-[:HAS_SPECIFICATION]->(sp:Spec)
    WHERE e.book_id IN $book_ids
      AND ANY(t IN $terms WHERE toLower(e.name) CONTAINS toLower(t)
              OR toLower(t) CONTAINS toLower(e.name))
    RETURN e.name AS entity, sp.value AS value, sp.unit AS unit,
           sp.raw AS raw, sp.section AS section LIMIT 50
    """
    try:
        with get_neo4j().session() as session:
            rows = session.run(cypher, book_ids=book_ids, terms=entity_terms)
            return [
                {
                    "entity":  r["entity"]  or "",
                    "value":   r["value"],
                    "unit":    r["unit"]    or "",
                    "raw":     r["raw"]     or "",
                    "section": r["section"] or "",
                }
                for r in rows if r.get("raw")
            ]
    except Exception as exc:
        logging.warning("Neo4j spec lookup failed: %s", exc)
        return []


@time_it
def _parse_payload(pl: dict, pid: str) -> dict:
    """Parses a Qdrant section payload into a standard dict."""
    pr = pl.get("page_range")
    page_list = ([pr.get("start", 0), pr.get("end", 0)] if isinstance(pr, dict)
                 else pr if isinstance(pr, list) else [0, 0])
    return {
        "chunk_id":        pid,
        "text":            pl.get("text") or "",
        "book_id":         pl.get("book_id"),
        "section_path":    pl.get("section_path") or [],
        "page_range":      page_list,
        "chunk_type":      pl.get("chunk_type") or "text",
        "structured_json": pl.get("structured_json"),
        "linearized_text": pl.get("linearized_text"),
        "from_qdrant":     True,
        "from_neo4j":      False,
        "qdrant_score":    None,
    }


@time_it
def fetch_sections_by_chunk_ids(chunk_ids: list[str], book_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    client = get_qdrant()
    results = []
    for i in range(0, len(chunk_ids), 64):
        try:
            pts = client.retrieve(collection_name=COLLECTION_SECTIONS,
                                  ids=chunk_ids[i: i + 64], with_payload=True)
            for p in pts:
                results.append(_parse_payload(p.payload or {}, str(p.id)))
        except Exception as exc:
            logging.warning("Section fetch batch failed: %s", exc)
    return results


@time_it
def search_sections_direct(query: str, book_ids: list[str],
                            section_names: list[str], limit: int) -> list[dict]:
    from qdrant_client import models as qm
    client    = get_qdrant()
    query_vec = get_nomic().encode("search_query: " + query, show_progress_bar=False).tolist()
    book_filter = (qm.Filter(must=[qm.FieldCondition(key="book_id", match=qm.MatchAny(any=book_ids))])
                   if book_ids else None)
    try:
        hits = client.search(collection_name=COLLECTION_SECTIONS, query_vector=query_vec,
                             query_filter=book_filter, limit=limit, with_payload=True)
    except Exception as exc:
        logging.warning("Sections search failed: %s", exc)
        return []
    results = []
    seen = set()
    for h in hits:
        cid = str(h.id)
        if cid in seen:
            continue
        seen.add(cid)
        s = _parse_payload(h.payload or {}, cid)
        s["qdrant_score"] = h.score
        s["from_neo4j"]   = any(n in (s.get("section_path") or []) for n in section_names) if section_names else False
        results.append(s)
    return results


@time_it
def merge_candidates(parent_sections: list[dict], direct_sections: list[dict],
                     neo4j_names: list[str]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for s in parent_sections + direct_sections:
        cid = s.get("chunk_id", "")
        if not cid:
            continue
        if cid not in by_id:
            by_id[cid] = s.copy()
        else:
            by_id[cid]["from_qdrant"] = by_id[cid].get("from_qdrant") or s.get("from_qdrant")
            by_id[cid]["from_neo4j"]  = by_id[cid].get("from_neo4j")  or s.get("from_neo4j")
            for field in ("structured_json", "linearized_text"):
                if not by_id[cid].get(field) and s.get(field):
                    by_id[cid][field] = s[field]
            if (s.get("qdrant_score") or 0) > (by_id[cid].get("qdrant_score") or 0):
                by_id[cid]["qdrant_score"] = s["qdrant_score"]
    if neo4j_names:
        for s in by_id.values():
            if any(n in (s.get("section_path") or []) for n in neo4j_names if n):
                s["from_neo4j"] = True
    return [s for s in by_id.values() if (s.get("text") or "").strip()]


@time_it
def rerank_candidates(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    if not candidates:
        return []
    try:
        ce = get_reranker()
        scores = ce.predict([(query, c.get("text") or "") for c in candidates])
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
            if c.get("from_qdrant") and c.get("from_neo4j"):
                c["rerank_score"] += BOOST_BOTH
    except Exception as exc:
        logging.warning("Reranker failed (%s); using Qdrant scores.", exc)
        for c in candidates:
            base = c.get("qdrant_score")
            c["rerank_score"] = float(base) if isinstance(base, (int, float)) else 0.0
            if c.get("from_qdrant") and c.get("from_neo4j"):
                c["rerank_score"] += BOOST_BOTH
    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PAGE EXPANSION  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _load_metadata(book_id: str) -> dict | None:
    """
    Loads and caches {book_id}_metadata.json from METADATA_DIR.
    Returns None if file doesn't exist or fails to parse.
    Cache persists for server lifetime — avoids re-reading disk per query.
    """
    global _metadata_cache
    if book_id in _metadata_cache:
        return _metadata_cache[book_id]
    path = METADATA_DIR / f"{book_id}_metadata.json"
    if not path.is_file():
        logging.warning("[page_expand] metadata not found: %s", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _metadata_cache[book_id] = data
        return data
    except Exception as exc:
        logging.warning("[page_expand] failed to load metadata for %s: %s", book_id, exc)
        return None


@time_it
def load_page_content(book_id: str, page_number: int) -> str | None:
    """
    Returns full_content for a given page from the metadata JSON.
    Returns None if book metadata is missing or page is not indexed.
    """
    meta = _load_metadata(book_id)
    if not meta:
        return None
    entry = meta.get(str(page_number))
    if not entry:
        return None
    return (entry.get("full_content") or "").strip() or None


@time_it
def expand_to_pages(top_chunks: list[dict]) -> list[str]:
    """
    Step 7: Expands top ranked chunks to full page content from metadata.

    rank-1 chunk on page N:
      → fetch pages N-1, N, N+1
      → cap combined at PAGE_EXPAND_MAX_CHARS
      → if over cap: drop N-1 first, then trim N+1 tail

    rank-2 chunk:
      → fetch its single page only
      → skip if that page was already fetched for rank-1 (deduplication)

    ranks 3+:
      → not expanded here — handled by fallback chunk text in build_context()

    Returns list of non-empty page content strings in reading order.
    Falls back gracefully to [] if metadata file not found.
    """
    if not top_chunks:
        return []

    page_blocks: list[str] = []
    fetched_pages: set[tuple[str, int]] = set()   # (book_id, page_number) dedup

    def _fetch(book_id: str, page_num: int) -> str | None:
        if page_num < 0:
            return None
        key = (book_id, page_num)
        if key in fetched_pages:
            return None   # already included — skip
        fetched_pages.add(key)
        return load_page_content(book_id, page_num)

    # ── Rank-1: 3-page expansion (N-1, N, N+1) ───────────────────────────────
    rank1 = top_chunks[0]
    bid1  = rank1.get("book_id") or ""
    pr1   = rank1.get("page_range") or [0, 0]
    n1    = int(pr1[0]) if pr1 else 0

    if bid1 and n1 > 0:
        prev_content = _fetch(bid1, n1 - 1)   # N-1: intro/heading context
        main_content = _fetch(bid1, n1)        # N:   primary answer page
        next_content = _fetch(bid1, n1 + 1)   # N+1: table continuation

        # Enforce combined character budget
        combined_len = sum(len(c) for c in [prev_content, main_content, next_content] if c)

        if combined_len > PAGE_EXPAND_MAX_CHARS:
            # Drop N-1 first (least critical of the three)
            if prev_content:
                combined_len -= len(prev_content)
                prev_content = None
            # Trim N+1 tail if still over budget
            if combined_len > PAGE_EXPAND_MAX_CHARS and next_content:
                trim_to = PAGE_EXPAND_MAX_CHARS - len(main_content or "")
                if trim_to > 200:
                    next_content = next_content[:trim_to]
                else:
                    next_content = None

        for content in [prev_content, main_content, next_content]:
            if content:
                page_blocks.append(content)

    elif bid1:
        # page_range missing or zero — attempt to fetch page 0 as fallback
        fallback = _fetch(bid1, n1)
        if fallback:
            page_blocks.append(fallback)

    # ── Rank-2: single page only ──────────────────────────────────────────────
    if len(top_chunks) >= 2:
        rank2 = top_chunks[1]
        bid2  = rank2.get("book_id") or ""
        pr2   = rank2.get("page_range") or [0, 0]
        n2    = int(pr2[0]) if pr2 else 0

        if bid2 and n2 > 0:
            rank2_content = _fetch(bid2, n2)   # dedup: skips if page already fetched
            if rank2_content:
                page_blocks.append(rank2_content)

    return page_blocks


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDING  ← modified to accept page_blocks
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _sentences(text: str) -> list[str]:
    try:
        import nltk
        p = PORTABLE_DIR / "nltk_data"
        if p.exists() and str(p) not in nltk.data.path:
            nltk.data.path.insert(0, str(p))
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@time_it
def build_context(
    chunks: list[dict],
    query: str,
    specs: list[dict],
    max_chars: int,
    page_blocks: list[str] | None = None,
) -> str:
    """
    Builds the LLM context string.

    Context order (highest priority first):
      1. Spec nodes from Neo4j       — verified precise facts, always first
      2. Full page blocks            — N-1/N/N+1 of rank-1, rank-2 page
      3. Fallback chunk text         — ranks 3-8, tables formatted as bullet lists

    The page_blocks argument is injected between specs and fallback chunks.
    If page_blocks fills the budget, fallback chunks are skipped entirely.
    If metadata is not available (page_blocks=[]), falls back to original
    all-chunks behaviour — fully backward compatible.
    """
    parts: list[str] = []
    total = 0

    # ── Block 1: Spec nodes ───────────────────────────────────────────────────
    spec_block = format_specs_block(specs)
    if spec_block:
        parts.append(spec_block)
        total += len(spec_block)

    # ── Block 2+: Full page content (Step 7) ─────────────────────────────────
    if page_blocks:
        for pb in page_blocks:
            if not pb or not pb.strip():
                continue
            if total + len(pb) > max_chars:
                remaining = max_chars - total
                if remaining > 500:
                    pb = pb[:remaining]
                else:
                    break
            parts.append(pb)
            total += len(pb)
            if total >= max_chars:
                break

    if total >= max_chars:
        return "\n\n---\n\n".join(parts)

    # ── Block N+: Fallback chunk text for ranks 3-8 ───────────────────────────
    # Skip rank-1 and rank-2 if page blocks were provided for them
    fallback_chunks = chunks[2:] if (page_blocks and len(chunks) > 2) else chunks

    # Tables first — never cut off mid-table
    table_chunks = [c for c in fallback_chunks if c.get("chunk_type") == "table"]
    text_chunks  = [c for c in fallback_chunks if c.get("chunk_type") != "table"]
    ordered      = table_chunks + text_chunks

    ce = None

    for c in ordered:
        chunk_type = c.get("chunk_type", "text")
        pr         = c.get("page_range") or [0, 0]
        bid        = c.get("book_id") or "?"
        path_s     = " > ".join(c.get("section_path") or [])

        if chunk_type == "table":
            block = format_table_for_llm(c)
            if not block:
                raw = (c.get("text") or "").strip()
                if raw:
                    block = f"[Table | Book: {bid} | Page: {pr[0]}-{pr[1]}]\n{raw}"
            if not block:
                continue
            if total + len(block) > max_chars:
                continue   # skip partial tables — partial data is worse than none

        else:
            text = (c.get("text") or "").strip()
            if not text:
                continue
            label = f"[Book: {bid} | Page: {pr[0]}-{pr[1]}]"
            if path_s:
                label += f" | Section: {path_s}"
            label += "\n"

            if len(text.split()) <= LONG_CHUNK_WORDS:
                block = label + text
            else:
                sents = _sentences(text)
                segs: list[str] = []
                cur = ""
                for s in sents:
                    if len(cur.split()) + len(s.split()) <= 220:
                        cur = (cur + " " + s).strip()
                    else:
                        if cur:
                            segs.append(cur)
                        cur = s
                if cur:
                    segs.append(cur)
                if not segs:
                    block = label + text[:4000]
                else:
                    try:
                        if ce is None:
                            ce = get_reranker()
                        seg_scores = ce.predict([(query, seg) for seg in segs])
                        best_i = max(range(len(segs)), key=lambda i: seg_scores[i])
                        block = label + segs[best_i]
                    except Exception:
                        block = label + segs[0][:4000]

            if total + len(block) > max_chars:
                block = block[:max(0, max_chars - total)]

        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RETRIEVAL + STREAMING
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def retrieve_bundle(query: str, book_ids: list[str], mode: str) -> dict[str, Any]:
    cfg      = MODE_CONFIG.get(mode, MODE_CONFIG["balanced"])
    prop_lim = cfg.get("prop_retrieve_limit", PROP_RETRIEVE_LIMIT)
    sect_lim = cfg.get("sect_retrieve_limit", SECT_RETRIEVE_LIMIT)

    # Steps 1-6: unchanged
    prop_hits           = search_propositions(query, book_ids, prop_lim)
    entity_terms        = extract_query_entities(query)
    neo4j_section_names = neo4j_sections_for_entities(book_ids, entity_terms)
    specs               = neo4j_specs_for_terms(book_ids, entity_terms)

    parent_chunk_ids = list({p["parent_chunk_id"] for p in prop_hits if p.get("parent_chunk_id")})
    parent_sections  = fetch_sections_by_chunk_ids(parent_chunk_ids, book_ids)
    direct_sections  = search_sections_direct(query, book_ids, neo4j_section_names, sect_lim)
    candidates       = merge_candidates(parent_sections, direct_sections, neo4j_section_names)
    top              = rerank_candidates(query, candidates, cfg["final_top_n"])

    # Step 7: Page expansion — NEW
    # expand_to_pages() returns [] gracefully if metadata not found,
    # so this is fully backward compatible with books ingested before
    # build_metadata.py was added to the pipeline.
    page_blocks = expand_to_pages(top)

    # Step 8: Build context
    context       = build_context(top, query, specs, cfg["context_max_chars"], page_blocks)
    system_prompt = get_system_prompt(mode)

    sources = [{
        "chunk_id":     c.get("chunk_id"),
        "book_id":      c.get("book_id"),
        "page_range":   c.get("page_range"),
        "section_path": c.get("section_path"),
        "chunk_type":   c.get("chunk_type"),
        "from_qdrant":  c.get("from_qdrant"),
        "from_neo4j":   c.get("from_neo4j"),
        "rerank_score": round(c.get("rerank_score", 0.0), 4),
    } for c in top]

    return {"context": context, "sources": sources,
            "system_prompt": system_prompt, "mode": mode, "cfg": cfg}


@time_it
def _history_block(history: list[dict]) -> str:
    return "\n\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content') or ''}"
        for m in history
    )


@time_it
def build_user_message(query: str, context: str, history: list[dict]) -> str:
    """
    Adds an explicit per-message reminder to never reference tables.
    Redundancy with system prompt is intentional and effective.
    """
    query   = query   or ""
    context = context or ""
    reminder = (
        "\n\nIMPORTANT: Your answer must be COMPLETE and SELF-CONTAINED. "
        "DO NOT say 'see table', 'refer to table', 'as shown in table', "
        "'see page', or any similar phrase. "
        "If the context contains specification data or table values, "
        "EXTRACT and PRESENT all relevant values DIRECTLY in your answer. "
        "Never redirect the user to look at a table or page."
    )
    hb = (_history_block(history) or "").strip()
    if hb:
        return (f"Conversation so far:\n{hb}\n\n"
                f"Context from knowledge base:\n{context}\n\n"
                f"Current question: {query}{reminder}\nAnswer:")
    return (f"Context from knowledge base:\n{context}\n\n"
            f"Question: {query}{reminder}\nAnswer:")


@async_time_it
async def run_rag_stream(
    query: str, book_ids: list[str], mode: str, history: list[dict],
) -> AsyncIterator[dict]:
    yield {"type": "status", "message": "Searching knowledge base..."}
    loop = asyncio.get_running_loop()
    try:
        bundle = await loop.run_in_executor(None, lambda: retrieve_bundle(query, book_ids, mode))
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    yield {"type": "status", "message": "Generating answer..."}
    user_content = build_user_message(query, bundle["context"], history)
    messages = [
        {"role": "system", "content": bundle["system_prompt"]},
        {"role": "user",   "content": user_content},
    ]
    from partb.llm.stream_client import stream_llm
    full = ""
    had_error = False
    try:
        async for piece in stream_llm(messages, mode, bundle["cfg"]):
            if piece.get("type") == "error":
                had_error = True
            if piece.get("type") == "token":
                full += piece.get("content", "")
            yield piece
            if had_error:
                return
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return
    if not had_error:
        yield {"type": "done", "sources": bundle["sources"], "mode": mode, "full_text": full}