"""
partb/retrieval/pipeline.py
----------------------------
Hybrid retrieval pipeline — now with page-level context expansion.

CHANGES IN THIS VERSION vs previous:

  NEW — Step 7: Page Expansion
  ─────────────────────────────
  After Jina Reranker v3 listwise reranking (Step 6), the pipeline now expands the
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
import time
from typing import Any, AsyncIterator

from partb.config import (
    ADAPTIVE_DEPTH_EXPAND_MULTIPLIER,
    ADAPTIVE_DEPTH_INITIAL_FRACTION,
    ADAPTIVE_DEPTH_SCORE_THRESHOLD,
    BOOST_BOTH,
    COLLECTION_PROPS,
    COLLECTION_SECTIONS,
    CONTEXT_GREEDY,
    ENABLE_ADAPTIVE_DEPTH,
    ENABLE_HYBRID,
    ENABLE_MMR,
    ENABLE_QUERY_CLASSIFICATION,
    ENTITY_LABELS,
    GLINER_QUERY_THRESHOLD,
    HYBRID_POOL_MULTIPLIER,
    HYBRID_RRF_K,
    LONG_CHUNK_WORDS,
    METADATA_DIR,
    MMR_LAMBDA,
    MMR_POOL_MULTIPLIER,
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
    QUERY_TYPE_GENERAL,
    QUERY_TYPE_OVERRIDES,
    RERANKER_DIR,
    RERANK_FULL_CHUNK_WORDS,
    RERANK_LOG_DISTRIBUTION_EVERY,
    RERANK_MIN_SCORE,
    RERANK_SEGMENT_TOKENS,
    SECT_RETRIEVE_LIMIT,
)
from partb.logger import log_process, logger, time_it
from partb.retrieval.hybrid import build_and_fuse
from partb.retrieval.prompts import get_system_prompt

logging.getLogger("transformers").setLevel(logging.ERROR)

if str(PARTA_DIR) not in sys.path:
    sys.path.insert(0, str(PARTA_DIR))

_gliner = None
_reranker = None
_neo_driver = None
_qdrant_cl = None
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
        _gliner = GLiNER.from_pretrained(str(model_dir), local_files_only=True).to(
            "cpu"
        )
    return _gliner


@time_it
def get_neo4j():
    global _neo_driver
    if _neo_driver is None:
        from neo4j import GraphDatabase

        _neo_driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=200,
            keep_alive=True,
        )
        _neo_driver.verify_connectivity()
    return _neo_driver


@time_it
def get_reranker():
    global _reranker
    if _reranker is None:
        cfg_path = RERANKER_DIR / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Reranker not found at {RERANKER_DIR}")

        # Directly import JinaForRanking from the local modeling.py to avoid
        # AutoModel auto_map resolution issues (transformers 5.x vs 4.x
        # config compat). This is the same class that AutoModel would load
        # via trust_remote_code=True, but by importing it directly we
        # bypass the auto-class resolution entirely.
        import importlib.util

        reranker_str = str(RERANKER_DIR)
        modeling_path = str(RERANKER_DIR / "modeling.py")
        spec = importlib.util.spec_from_file_location(
            "jina_reranker_modeling", modeling_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot load modeling.py from {RERANKER_DIR} — "
                f"file may be missing or unreadable."
            )
        mod = importlib.util.module_from_spec(spec)
        # Suppress sys.path pollution: exec the module but keep its own
        # namespace clean.
        spec.loader.exec_module(mod)

        try:
            _reranker = mod.JinaForRanking.from_pretrained(reranker_str)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load JinaForRanking from {RERANKER_DIR}. "
                f"Error: {exc}"
            ) from exc
        _reranker.eval()
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
        _nomic_model = SentenceTransformer(
            str(model_dir), trust_remote_code=True, device="cpu"
        )
    return _nomic_model


@log_process
def warm_models() -> None:
    get_neo4j()
    get_qdrant()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE FORMATTING
# ─────────────────────────────────────────────────────────────────────────────


@time_it
def _format_pipe_table(headers: list[str], rows: list[dict]) -> str:
    """
    Builds a Markdown pipe table from headers and row dicts.

    Columns are padded to the widest value (or header length) for clean alignment.
    Special characters like newlines and pipe chars in cell values are escaped.

    Returns the table body (no label prefix) or empty string on failure.
    """
    if not headers or not rows:
        return ""

    # Compute column widths (header or data, whichever is wider)
    col_widths: list[int] = []
    for h in headers:
        h_str = str(h) if h else ""
        max_w = len(h_str)
        for row in rows:
            v = str(row.get(h, "") or "")
            # Replace escaped pipe for width calculation
            v_clean = v.replace("\\|", "|")
            max_w = max(max_w, len(v_clean))
        col_widths.append(max_w)

    # Sanitize a cell value: escape pipe chars, replace newlines with spaces
    def _sanitize(val: object) -> str:
        s = str(val) if val is not None else ""
        s = s.replace("|", "\\|")
        s = s.replace("\n", " ").replace("\r", "")
        return s

    lines: list[str] = []

    # Header row
    header_cells = [_sanitize(h).ljust(col_widths[i]) for i, h in enumerate(headers)]
    lines.append("| " + " | ".join(header_cells) + " |")

    # Separator row
    sep_cells = ["-" * w for w in col_widths]
    lines.append("| " + " | ".join(sep_cells) + " |")

    # Data rows
    for row in rows:
        data_cells = []
        for i, h in enumerate(headers):
            v = _sanitize(row.get(h, ""))
            data_cells.append(v.ljust(col_widths[i]))
        lines.append("| " + " | ".join(data_cells) + " |")

    return "\n".join(lines)


@time_it
def format_table_for_llm(chunk: dict) -> str:
    """
    Converts a table chunk into LLM-readable format.

    Priority:
      1. structured_json → Markdown pipe table (best for LLM comprehension)
      2. linearized_text → formatted as block text
      3. raw text → formatted as block text

    Returns the full formatted string including source label, or "" if empty.
    """
    section_path = chunk.get("section_path") or []
    section_label = " > ".join(section_path) if section_path else "Unknown Section"
    pr = chunk.get("page_range") or [0, 0]
    bid = chunk.get("book_id") or "?"
    # Enhanced citation with section name: [Book: X | § Section Name | Page: Y]
    citation = (
        f"[Book: {bid} | § {section_label} | Page: {pr[0]}-{pr[1]}]"
        if section_label != "Unknown Section"
        else f"[Book: {bid} | Page: {pr[0]}-{pr[1]}]"
    )
    source_label = f"Specification Data: {citation}:"

    # ── Path 1: structured_json → Markdown pipe table ─────────────────
    structured = chunk.get("structured_json")
    if structured and isinstance(structured, dict):
        headers = structured.get("headers") or []
        rows = structured.get("rows") or []
        if headers and rows:
            pipe_table = _format_pipe_table(headers, rows)
            if pipe_table:
                return f"{source_label}\n{pipe_table}"

    # ── Path 2: linearized_text ────────────────────────────────────────
    linearized = (chunk.get("linearized_text") or "").strip()
    if linearized and len(linearized) > 30:
        return f"{source_label}\n{linearized}"

    # ── Path 3: raw text ───────────────────────────────────────────────
    raw = (chunk.get("text") or chunk.get("content") or "").strip()
    if raw:
        citation = (
            f"[Book: {bid} | § {section_label} | Page: {pr[0]}-{pr[1]}]"
            if section_label != "Unknown Section"
            else f"[Book: {bid} | Page: {pr[0]}-{pr[1]}]"
        )
        return f"Data: {citation}:\n{raw}"

    return ""


@time_it
def format_specs_block(specs: list[dict]) -> str:
    """Formats Neo4j Spec nodes as a verified-facts block at the top of context."""
    if not specs:
        return ""
    lines = ["=== VERIFIED TECHNICAL SPECIFICATIONS (Knowledge Graph) ==="]
    seen = set()
    for sp in specs[:30]:
        entity = (sp.get("entity") or "").strip()
        raw = (sp.get("raw") or "").strip()
        if not entity or not raw:
            continue
        key = (entity.lower(), raw.lower())
        if key in seen:
            continue
        seen.add(key)
        section = (sp.get("section") or "").strip()
        sec_note = f" [{section}]" if section else ""
        lines.append(f"  - {entity.title()}: {raw}{sec_note}")
    if len(lines) == 1:
        return ""
    lines.append("=" * 56)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL STEPS 1-6
# ─────────────────────────────────────────────────────────────────────────────


@log_process
def search_propositions(query: str, book_ids: list[str], limit: int) -> list[dict]:
    from qdrant_client import models as qm

    client = get_qdrant()
    model = get_nomic()
    query_vec = model.encode("search_query: " + query, show_progress_bar=False).tolist()
    filters = None
    if book_ids:
        filters = qm.Filter(
            must=[qm.FieldCondition(key="book_id", match=qm.MatchAny(any=book_ids))]
        )
    try:
        response = client.query_points(
            collection_name=COLLECTION_PROPS,
            query=query_vec,
            query_filter=filters,
            limit=limit,
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        logging.warning("Propositions search failed: %s", exc)
        return []
    results = []
    for h in hits:
        pl = h.payload or {}
        pr = pl.get("page_range")
        page_list = (
            [pr.get("start", 0), pr.get("end", 0)]
            if isinstance(pr, dict)
            else pr
            if isinstance(pr, list)
            else [pl.get("page", 0)] * 2
        )
        results.append(
            {
                "proposition_id": str(h.id),
                "text": pl.get("text") or "",
                "parent_chunk_id": pl.get("parent_chunk_id"),
                "section_path": pl.get("section_path") or [],
                "page_range": page_list,
                "source_type": pl.get("source_type") or "text",
                "book_id": pl.get("book_id"),
                "score": h.score,
            }
        )
    return results


@log_process
def extract_query_entities(query: str) -> list[str]:
    model = get_gliner()
    try:
        preds = model.predict_entities(
            query.replace("\n", " ").strip(),
            ENTITY_LABELS,
            threshold=GLINER_QUERY_THRESHOLD,
        )
    except Exception as exc:
        logging.warning("GLiNER failed: %s", exc)
        return []
    seen = set()
    terms = []
    for p in preds:
        t = (p.get("text") or "").strip().lower()
        if t and t not in seen and len(t) >= 3:
            seen.add(t)
            terms.append(t)
    return terms


@log_process
def neo4j_sections_for_entities(
    book_ids: list[str], entity_terms: list[str]
) -> list[str]:
    if not entity_terms:
        return []
    if book_ids:
        cypher = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(s)
        WHERE s.book_id IN $book_ids AND (s:Section OR s:Subsection)
          AND ANY(t IN $terms WHERE
              toLower(e.name) = toLower(t)
              OR (size(t) >= 4 AND toLower(e.name) CONTAINS toLower(t))
              OR (size(t) >= 4 AND toLower(t) CONTAINS toLower(e.name)))
        RETURN DISTINCT s.name AS section_name LIMIT $lim
        """
        params = {"book_ids": book_ids, "terms": entity_terms, "lim": NEO4J_ENTITY_LIMIT}
    else:
        # No book filter — query across all books
        cypher = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(s)
        WHERE (s:Section OR s:Subsection)
          AND ANY(t IN $terms WHERE
              toLower(e.name) = toLower(t)
              OR (size(t) >= 4 AND toLower(e.name) CONTAINS toLower(t))
              OR (size(t) >= 4 AND toLower(t) CONTAINS toLower(e.name)))
        RETURN DISTINCT s.name AS section_name LIMIT $lim
        """
        params = {"terms": entity_terms, "lim": NEO4J_ENTITY_LIMIT}
    try:
        with get_neo4j().session() as session:
            rows = session.run(cypher, **params)
            return [
                r["section_name"]
                for r in rows
                if r.get("section_name") is not None and str(r["section_name"]).strip()
            ]
    except Exception as exc:
        logging.warning("Neo4j traversal failed: %s", exc)
        return []


@log_process
def neo4j_specs_for_terms(book_ids: list[str], entity_terms: list[str]) -> list[dict]:
    if not entity_terms:
        return []
    if book_ids:
        cypher = """
        MATCH (e:Entity)-[:HAS_SPECIFICATION]->(sp:Spec)
        WHERE e.book_id IN $book_ids
          AND ANY(t IN $terms WHERE toLower(e.name) CONTAINS toLower(t)
                  OR toLower(t) CONTAINS toLower(e.name))
        RETURN e.name AS entity, sp.value AS value, sp.unit AS unit,
               sp.raw AS raw, sp.section AS section LIMIT 50
        """
        params = {"book_ids": book_ids, "terms": entity_terms}
    else:
        cypher = """
        MATCH (e:Entity)-[:HAS_SPECIFICATION]->(sp:Spec)
        WHERE ANY(t IN $terms WHERE toLower(e.name) CONTAINS toLower(t)
                  OR toLower(t) CONTAINS toLower(e.name))
        RETURN e.name AS entity, sp.value AS value, sp.unit AS unit,
               sp.raw AS raw, sp.section AS section LIMIT 50
        """
        params = {"terms": entity_terms}
    try:
        with get_neo4j().session() as session:
            rows = session.run(cypher, **params)
            return [
                {
                    "entity": r["entity"] or "",
                    "value": r["value"],
                    "unit": r["unit"] or "",
                    "raw": r["raw"] or "",
                    "section": r["section"] or "",
                }
                for r in rows
                if r.get("raw")
            ]
    except Exception as exc:
        logging.warning("Neo4j spec lookup failed: %s", exc)
        return []



@time_it
def _parse_payload(pl: dict, pid: str, vector: list[float] | None = None) -> dict:
    """Parses a Qdrant section payload into a standard dict.

    In qdrant_client v1.12+, the .vector attribute on ScoredPoint may
    return a dict (keyed by vector name, e.g. {"": [...]} for unnamed)
    instead of a flat list. We normalize to flat list here so downstream
    consumers (_cosine_sim, mmr_select) always get a list.
    """
    # Normalize: Qdrant named-vector dict → flat list
    if isinstance(vector, dict):
        # Unnamed vector key is "" or the first key
        vector = next(iter(vector.values()), None)
    pr = pl.get("page_range")
    page_list = (
        [pr.get("start", 0), pr.get("end", 0)]
        if isinstance(pr, dict)
        else pr
        if isinstance(pr, list)
        else [0, 0]
    )
    return {
        "chunk_id": pl.get("chunk_id") or pid,
        "text": pl.get("text") or "",
        "book_id": pl.get("book_id"),
        "section_path": pl.get("section_path") or [],
        "page_range": page_list,
        "chunk_type": pl.get("chunk_type") or "text",
        "structured_json": pl.get("structured_json"),
        "linearized_text": pl.get("linearized_text"),
        "from_qdrant": True,
        "from_neo4j": False,
        "qdrant_score": None,
        "vector": vector,
    }


@time_it
def fetch_sections_by_chunk_ids(
    chunk_ids: list[str], book_ids: list[str]
) -> list[dict]:
    if not chunk_ids:
        return []
    client = get_qdrant()
    from qdrant_client import models as qm
    results = []
    for i in range(0, len(chunk_ids), 64):
        try:
            batch_ids = chunk_ids[i : i + 64]
            pts, _ = client.scroll(
                collection_name=COLLECTION_SECTIONS,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="chunk_id",
                            match=qm.MatchAny(any=batch_ids)
                        )
                    ]
                ),
                limit=len(batch_ids),
                with_payload=True,
                with_vectors=ENABLE_MMR,
            )
            for p in pts:
                vec = getattr(p, "vector", None) if ENABLE_MMR else None
                results.append(_parse_payload(p.payload or {}, str(p.id), vector=vec))
        except Exception as exc:
            logging.warning("Section fetch batch failed: %s", exc)
    return results


@time_it
def search_sections_direct(
    query: str, book_ids: list[str], section_names: list[str], limit: int
) -> list[dict]:
    from qdrant_client import models as qm

    client = get_qdrant()
    query_vec = (
        get_nomic().encode("search_query: " + query, show_progress_bar=False).tolist()
    )
    filters = (
        qm.Filter(
            must=[qm.FieldCondition(key="book_id", match=qm.MatchAny(any=book_ids))]
        )
        if book_ids
        else None
    )
    try:
        response = client.query_points(
            collection_name=COLLECTION_SECTIONS,
            query=query_vec,
            query_filter=filters,
            limit=limit,
            with_payload=True,
            with_vectors=ENABLE_MMR,
        )
        hits = response.points
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
        vec = getattr(h, "vector", None) if ENABLE_MMR else None
        s = _parse_payload(h.payload or {}, cid, vector=vec)
        s["qdrant_score"] = h.score
        s["from_neo4j"] = (
            any(n in (s.get("section_path") or []) for n in section_names)
            if section_names
            else False
        )
        results.append(s)
    return results


@time_it
def merge_candidates(
    parent_sections: list[dict], direct_sections: list[dict], neo4j_names: list[str]
) -> list[dict]:
    by_id: dict[str, dict] = {}
    for s in parent_sections + direct_sections:
        cid = s.get("chunk_id", "")
        if not cid:
            continue
        if cid not in by_id:
            by_id[cid] = s.copy()
        else:
            by_id[cid]["from_qdrant"] = by_id[cid].get("from_qdrant") or s.get(
                "from_qdrant"
            )
            by_id[cid]["from_neo4j"] = by_id[cid].get("from_neo4j") or s.get(
                "from_neo4j"
            )
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


@log_process
def rerank_candidates(query: str, candidates: list[dict], top_n: int, boost_mult: float = 1.0) -> list[dict]:
    """
    Rerank candidates using the Jina Reranker v3 listwise model.

    Args:
        query:     The user query.
        candidates: List of candidate dicts with 'text' key.
        top_n:     Number of top candidates to return.
        boost_mult: Multiplier for BOOST_BOTH when a candidate comes from
                    both Qdrant and Neo4j. Used by query classification to
                    boost spec_lookup candidates higher.
    """
    if not candidates:
        return []
    try:
        reranker = get_reranker()
        texts = [c.get("text", "") or "" for c in candidates]
        results = reranker.rerank(query, texts)
        for r in results:
            idx = r["index"]
            candidates[idx]["rerank_score"] = r["relevance_score"]
            if candidates[idx].get("from_qdrant") and candidates[idx].get("from_neo4j"):
                candidates[idx]["rerank_score"] += BOOST_BOTH * boost_mult
    except Exception as exc:
        logging.warning("Reranker failed (%s); using Qdrant scores.", exc)
        for c in candidates:
            base = c.get("qdrant_score")
            c["rerank_score"] = float(base) if isinstance(base, (int, float)) else 0.0
            if c.get("from_qdrant") and c.get("from_neo4j"):
                c["rerank_score"] += BOOST_BOTH * boost_mult

    # Score-distribution logging (sampled) so RERANK_MIN_SCORE / BOOST_BOTH can
    # be calibrated against real data rather than guessed.
    global _rerank_log_counter
    _rerank_log_counter += 1
    if RERANK_LOG_DISTRIBUTION_EVERY > 0 and (
        _rerank_log_counter % RERANK_LOG_DISTRIBUTION_EVERY == 0
    ):
        raw = sorted(c.get("rerank_score", 0.0) for c in candidates)
        if raw:
            n = len(raw)
            median = raw[n // 2] if n % 2 else (raw[n // 2 - 1] + raw[n // 2]) / 2
            boosted = sum(
                1 for c in candidates if c.get("from_qdrant") and c.get("from_neo4j")
            )
            logger.info(
                "[RERANK] dist | n=%d | min=%.4f | median=%.4f | max=%.4f | "
                "boosted=%d | top1_score=%.4f | floor=%.4f | boost_mult=%.1f",
                n,
                raw[0],
                median,
                raw[-1],
                boosted,
                raw[-1],
                RERANK_MIN_SCORE,
                boost_mult,
            )

    # Score floor — drop candidates that scored below the configured minimum.
    if RERANK_MIN_SCORE > 0:
        before = len(candidates)
        candidates = [c for c in candidates if c.get("rerank_score", 0.0) >= RERANK_MIN_SCORE]
        if before != len(candidates):
            logger.info(
                "[RERANK] floor filter | kept %d/%d (dropped %d below %.3f)",
                len(candidates), before, before - len(candidates), RERANK_MIN_SCORE,
            )

    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6b — MMR DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────


@time_it
def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = sum(ai * ai for ai in a) ** 0.5
    norm_b = sum(bi * bi for bi in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@time_it
def mmr_select(candidates: list[dict], top_n: int, lambda_: float = 0.7) -> list[dict]:
    """
    Maximum Marginal Relevance selection to balance relevance vs. diversity.

    MMR = λ * relevance(c) - (1-λ) * max_{j in selected} similarity(c, c_j)

    Uses rerank_score as the relevance term and cosine similarity between
    Nomic embedding vectors for the diversity penalty. Candidates without
    a stored vector fall back to pure relevance ordering.

    Args:
        candidates: List of candidate dicts sorted by rerank_score descending.
                    Each should have 'rerank_score' and optionally 'vector'.
        top_n:      Number of candidates to select.
        lambda_:    Trade-off parameter. 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        top_n candidates reordered by MMR score, maintaining the original
        dict structure so downstream code (page expansion, context building)
        is unaffected.
    """
    if not candidates or top_n <= 0:
        return []
    if top_n >= len(candidates):
        return list(candidates)

    selected: list[dict] = []
    remaining = list(candidates)

    # Seed with the highest-relevance candidate (already first after reranker sort)
    selected.append(remaining.pop(0))

    while len(selected) < top_n and remaining:
        mmr_scores = []
        for c in remaining:
            rel = c.get("rerank_score", 0.0)
            c_vec = c.get("vector")
            if c_vec is None:
                # No vector available — pure relevance
                mmr_scores.append(rel)
                continue

            # Diversity term: max similarity to any already-selected candidate
            max_sim = max(
                (_cosine_sim(c_vec, s.get("vector", [])) if s.get("vector") else 0.0)
                for s in selected
            )
            mmr_scores.append(lambda_ * rel - (1.0 - lambda_) * max_sim)

        best_idx = max(range(len(remaining)), key=lambda i: mmr_scores[i])
        selected.append(remaining.pop(best_idx))

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PAGE EXPANSION
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
        logging.warning(
            "[page_expand] failed to load metadata for %s: %s", book_id, exc
        )
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


@log_process
def expand_to_pages(
    top_chunks: list[dict],
    page_expand_range: int = 1,
) -> tuple[list[str], list[dict]]:
    """
    Step 7: Expands top ranked chunks to full page content from metadata.

    rank-1 chunk on page N with page_expand_range=R:
      → fetch pages N-R .. N .. N+R
      → cap combined at PAGE_EXPAND_MAX_CHARS
      → if over cap: drop outermost pages first, then trim the far edge

    rank-2 chunk:
      → fetch its single page only
      → skip if that page was already fetched for rank-1 (deduplication)

    ranks 3+:
      → not expanded here — handled by fallback chunk text in build_context()

    Args:
        top_chunks:       List of reranked chunks.
        page_expand_range: Number of pages to fetch before/after the main page.
                           0 = main page only (no adjacent expansion).
                           1 = N-1, N, N+1 (default, current behaviour).
                           2 = N-2, N-1, N, N+1, N+2.

    Returns:
        (page_blocks, expansions)
          page_blocks: list[str] — non-empty page content in reading order.
          expansions : list[dict] — one entry per fetched page, each:
              {
                "book_id":     str,
                "page":        int,
                "expanded_from_chunk_id": str,   # which ranked chunk triggered it
                "rank":        int,              # 1 = rank-1 expansion, 2 = rank-2
                "relation":    "prev"|"main"|"next"|"single"|"fallback",
              }
          This lets retrieve_bundle attribute the expanded content in sources.

    Falls back gracefully to ([], []) if metadata file not found.
    """
    if not top_chunks:
        return [], []

    page_blocks: list[str] = []
    expansions: list[dict] = []
    fetched_pages: set[tuple[str, int]] = set()  # (book_id, page_number) dedup

    def _fetch(book_id: str, page_num: int, src_chunk_id: str, rank: int,
               relation: str) -> str | None:
        if page_num < 0:
            return None
        key = (book_id, page_num)
        if key in fetched_pages:
            return None  # already included — skip
        fetched_pages.add(key)
        content = load_page_content(book_id, page_num)
        if content:
            expansions.append({
                "book_id": book_id,
                "page": page_num,
                "expanded_from_chunk_id": src_chunk_id,
                "rank": rank,
                "relation": relation,
            })
        return content

    # ── Rank-1: dynamic multi-page expansion (N-R .. N .. N+R) ────────────────
    rank1 = top_chunks[0]
    bid1 = rank1.get("book_id") or ""
    pr1 = rank1.get("page_range") or [0, 0]
    n1 = int(pr1[0]) if pr1 else 0
    cid1 = rank1.get("chunk_id") or ""

    if bid1 and n1 > 0 and page_expand_range > 0:
        # Fetch all pages in range [N-R, N+R]
        # Use parallel dicts keyed by offset so expansions stay aligned with
        # page_contents when budget trimming drops pages.
        page_contents: dict[int, str] = {}
        rank1_expansions: dict[int, dict] = {}
        for offset in range(-page_expand_range, page_expand_range + 1):
            page_num = n1 + offset
            if page_num < 0:
                continue
            key = (bid1, page_num)
            if key in fetched_pages:
                continue
            if offset < 0:
                relation = "prev"
            elif offset == 0:
                relation = "main"
            else:
                relation = "next"
            content = load_page_content(bid1, page_num)
            if content:
                fetched_pages.add(key)
                page_contents[offset] = content
                rank1_expansions[offset] = {
                    "book_id": bid1,
                    "page": page_num,
                    "expanded_from_chunk_id": cid1,
                    "rank": 1,
                    "relation": relation,
                }

        # Enforce combined character budget — drop outermost pages first
        combined_len = sum(len(c) for c in page_contents.values())
        if combined_len > PAGE_EXPAND_MAX_CHARS and len(page_contents) > 1:
            # Sort offsets by absolute distance from main (furthest first)
            offsets = sorted(page_contents.keys(), key=lambda o: abs(o), reverse=True)
            for offset in offsets:
                if combined_len <= PAGE_EXPAND_MAX_CHARS:
                    break
                if offset == 0:
                    continue  # never drop the main page
                combined_len -= len(page_contents[offset])
                del page_contents[offset]
                del rank1_expansions[offset]

        # Trim the farthest remaining non-main page if still over budget
        if combined_len > PAGE_EXPAND_MAX_CHARS and len(page_contents) > 1:
            offsets = sorted(page_contents.keys(), key=lambda o: abs(o), reverse=True)
            for offset in offsets:
                if offset == 0:
                    continue
                trim_to = len(page_contents[offset]) - (combined_len - PAGE_EXPAND_MAX_CHARS)
                if trim_to > 200:
                    page_contents[offset] = page_contents[offset][:trim_to]
                    combined_len = sum(len(c) for c in page_contents.values())
                break  # only trim one page

        # Append in reading order (ascending page number)
        for offset in sorted(page_contents.keys()):
            page_blocks.append(page_contents[offset])
            expansions.append(rank1_expansions[offset])

    elif bid1 and n1 > 0 and page_expand_range == 0:
        # page_expand_range=0: main page only, no adjacent expansion
        main_content = _fetch(bid1, n1, cid1, 1, "main")
        if main_content:
            page_blocks.append(main_content)

    elif bid1:
        # page_range missing or zero — attempt to fetch page 0 as fallback
        fallback = _fetch(bid1, n1, cid1, 1, "fallback")
        if fallback:
            page_blocks.append(fallback)

    # ── Rank-2: single page only ──────────────────────────────────────────────
    if len(top_chunks) >= 2:
        rank2 = top_chunks[1]
        bid2 = rank2.get("book_id") or ""
        pr2 = rank2.get("page_range") or [0, 0]
        n2 = int(pr2[0]) if pr2 else 0
        cid2 = rank2.get("chunk_id") or ""

        if bid2 and n2 > 0:
            rank2_content = _fetch(bid2, n2, cid2, 2, "single")  # dedup skips if already fetched
            if rank2_content:
                page_blocks.append(rank2_content)

    return page_blocks, expansions


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDING  ← modified to accept page_blocks
# ─────────────────────────────────────────────────────────────────────────────


@time_it
def _first_sentence(text: str) -> str:
    """Extract the first sentence from text, up to 200 chars."""
    text = (text or "").strip()
    if not text:
        return ""
    sents = _sentences(text)
    if sents:
        return sents[0][:200]
    return text[:200]


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
def _segment_windows(text: str, max_words: int) -> list[str]:
    """
    Splits long text into overlapping windows of roughly `max_words` words,
    aligned on sentence boundaries. Each window starts ~25% into the previous
    one so a relevant passage straddling a boundary is still scored intact.

    Safety net for extremely long text — Jina Reranker v3 handles 131K
    tokens so most chunks score in a single pass, but beyond that we
    segment-max pool.
    """
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    sents = _sentences(text)
    if not sents:
        return [text]

    step = max(max_words // 2, 50)  # 50% overlap
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        sw = len(s.split())
        if cur and cur_len + sw > max_words:
            windows.append(" ".join(cur).strip())
            # keep ~50% overlap: drop sentences until under step size
            while cur_len > step and len(cur) > 1:
                cur_len -= len(cur[0].split())
                cur.pop(0)
        cur.append(s)
        cur_len += sw
    if cur:
        windows.append(" ".join(cur).strip())
    return [w for w in windows if w]


# Monotonic query counter for sampling score-distribution log output.
_rerank_log_counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY CONTEXT ASSEMBLY (4.1a)
# ─────────────────────────────────────────────────────────────────────────────


@log_process
def _build_context_greedy(
    chunks: list[dict],
    query: str,
    specs: list[dict],
    max_chars: int,
    page_blocks: list[str] | None = None,
    expansions: list[dict] | None = None,
) -> tuple[str, dict]:
    """
    Greedy context assembly — selects items by value density (score/char)
    instead of fixed priority order (specs → pages → chunks).

    Algorithm:
      1. Always include spec block first (highest value density, small footprint).
      2. Build an item pool from page expansions + fallback chunks.
      3. Score each item by rerank_score / len(text) (value density).
      4. Sort by density descending; pick greedily until budget is full.
      5. Tables kept atomic: if a table doesn't fit the remaining budget, skip it.

    Usage tracking: page_blocks retain their expansion metadata so the source
    list in retrieve_bundle() reflects what the LLM actually saw.

    Returns (context, usage) with the same schema as build_context().
    """
    parts: list[str] = []
    total = 0
    usage: dict[str, Any] = {
        "used_chunk_ids": set(),
        "used_pages": [],
        "specs_included": False,
        "total_chars": 0,
    }

    # ── Block 1: Spec nodes (always included, highest priority) ──
    spec_block = format_specs_block(specs)
    if spec_block:
        parts.append(spec_block)
        total += len(spec_block)
        usage["specs_included"] = True

    if total >= max_chars:
        context = "\n\n---\n\n".join(parts)
        usage["total_chars"] = len(context)
        return context, usage

    # ── Build a chunk_id → rerank_score map for page-expansion scoring ──
    chunk_score_map: dict[str, float] = {
        c.get("chunk_id", ""): c.get("rerank_score", 0.0)
        for c in chunks
        if c.get("chunk_id")
    }

    # ── Helper to format a text chunk with label and long-chunk handling ──
    _greedy_reranker = None  # local ref for long-chunk segmentation

    def _format_text_chunk(c: dict) -> str | None:
        nonlocal _greedy_reranker
        text = (c.get("text") or "").strip()
        if not text:
            return None
        pr = c.get("page_range") or [0, 0]
        bid = c.get("book_id") or "?"
        path_s = " > ".join(c.get("section_path") or [])
        # Enhanced citation: [Book: X | § Section Name | Page: Y]
        citation = (
            f"[Book: {bid} | § {path_s} | Page: {pr[0]}-{pr[1]}]"
            if path_s
            else f"[Book: {bid} | Page: {pr[0]}-{pr[1]}]"
        )
        label = citation + "\n"

        if len(text.split()) <= LONG_CHUNK_WORDS:
            return label + text

        # Long chunk: segment on sentences, rerank segments, pick best
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
            return label + text[:4000]

        try:
            if _greedy_reranker is None:
                _greedy_reranker = get_reranker()
            seg_results = _greedy_reranker.rerank(query, segs)
            best_i = seg_results[0]["index"]
            return label + segs[best_i]
        except Exception:
            return label + segs[0][:4000]

    # ── Build item pool ────────────────────────────────────────────────
    # Each item: {text, score, density, type, chunk_id, expansion_idx}
    items: list[dict] = []

    # Add page blocks from expansion — inherit score from source chunk
    if page_blocks and expansions:
        exp_by_idx = expansions  # aligned 1:1 with page_blocks
        for i, pb in enumerate(page_blocks):
            if not pb or not pb.strip():
                continue
            score = 0.0
            if i < len(exp_by_idx):
                src_cid = exp_by_idx[i].get("expanded_from_chunk_id")
                if src_cid:
                    score = chunk_score_map.get(src_cid, 0.0)
            density = score / max(1, len(pb))
            items.append({
                "text": pb,
                "score": score,
                "density": density,
                "type": "page",
                "chunk_id": None,
                "expansion_idx": i,
            })

    # Add fallback chunks (skip chunks whose IDs were covered by page expansion)
    expanded_cids: set[str] = set()
    if expansions:
        for e in expansions:
            cid = e.get("expanded_from_chunk_id")
            if cid:
                expanded_cids.add(cid)
    if expanded_cids:
        fallback_chunks = [
            c for c in chunks if c.get("chunk_id") not in expanded_cids
        ]
    else:
        fallback_chunks = chunks
    for c in fallback_chunks:
        chunk_type = c.get("chunk_type", "text")
        cid = c.get("chunk_id", "")
        score = c.get("rerank_score", 0.0)

        if chunk_type == "table":
            block = format_table_for_llm(c)
            if not block:
                raw = (c.get("text") or "").strip()
                if raw:
                    pr = c.get("page_range") or [0, 0]
                    bid = c.get("book_id") or "?"
                    path_s = " > ".join(c.get("section_path") or [])
                    citation = (
                        f"[Table | Book: {bid} | § {path_s} | Page: {pr[0]}-{pr[1]}]"
                        if path_s
                        else f"[Table | Book: {bid} | Page: {pr[0]}-{pr[1]}]"
                    )
                    block = f"{citation}\n{raw}"
            if not block:
                continue
            density = score / max(1, len(block))
            items.append({
                "text": block,
                "score": score,
                "density": density,
                "type": "table",
                "chunk_id": cid,
                "expansion_idx": None,
            })
        else:
            block = _format_text_chunk(c)
            if not block:
                continue
            density = score / max(1, len(block))
            items.append({
                "text": block,
                "score": score,
                "density": density,
                "type": "chunk",
                "chunk_id": cid,
                "expansion_idx": None,
            })

    # ── Sort by density descending ──
    items.sort(key=lambda x: x["density"], reverse=True)

    # ── Greedily select items until budget is full ──
    for item in items:
        text = item["text"]
        text_len = len(text)

        # Tables: keep atomic — skip if doesn't fully fit
        if item["type"] == "table":
            if total + text_len > max_chars:
                continue
        elif item["type"] == "page":
            # Page blocks: trim if partial fits, skip if too small remnant
            if total + text_len > max_chars:
                remaining = max_chars - total
                if remaining > 500:
                    text = text[:remaining]
                    text_len = len(text)
                else:
                    continue
        else:
            # Chunks: trim if partial fits
            if total + text_len > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    text = text[:remaining]
                    text_len = len(text)
                else:
                    continue

        parts.append(text)
        total += text_len

        # Track usage
        if item["type"] == "page" and item["expansion_idx"] is not None:
            if expansions and item["expansion_idx"] < len(expansions):
                usage["used_pages"].append(expansions[item["expansion_idx"]])
        elif item.get("chunk_id"):
            usage["used_chunk_ids"].add(item["chunk_id"])

        if total >= max_chars:
            break

    context = "\n\n---\n\n".join(parts)
    usage["total_chars"] = len(context)
    return context, usage


@log_process
def build_context(
    chunks: list[dict],
    query: str,
    specs: list[dict],
    max_chars: int,
    page_blocks: list[str] | None = None,
    expansions: list[dict] | None = None,
) -> tuple[str, dict]:
    """
    Builds the LLM context string.

    When CONTEXT_GREEDY is enabled (env RAG_CONTEXT_GREEDY=1), delegates to
    _build_context_greedy() which selects items by value density (score/char)
    instead of fixed priority order.

    Default behaviour (CONTEXT_GREEDY=0):
      Context order (highest priority first):
        1. Spec nodes from Neo4j       — verified precise facts, always first
        2. Full page blocks            — N-1/N/N+1 of rank-1, rank-2 page
        3. Fallback chunk text         — ranks 3-8, tables formatted as bullet lists

      The page_blocks argument is injected between specs and fallback chunks.
      If page_blocks fills the budget, fallback chunks are skipped entirely.
      If metadata is not available (page_blocks=[]), falls back to original
      all-chunks behaviour — fully backward compatible.

    Returns:
        (context, usage)
          context : str — the assembled prompt context.
          usage   : dict tracking EXACTLY what reached the context:
              {
                "used_chunk_ids":   set[str]   — chunk_ids whose text was added,
                "used_pages":       list[dict] — copy of `expansions` entries whose
                                                 page content actually made it in
                                                 (filtered vs page_blocks budget),
                "specs_included":   bool       — whether the spec block was added,
                "total_chars":      int        — final context length,
              }
        This lets retrieve_bundle build source entries that reflect what the LLM
        actually saw, instead of every retrieved candidate.
    """
    if CONTEXT_GREEDY:
        return _build_context_greedy(
            chunks, query, specs, max_chars, page_blocks, expansions
        )
    parts: list[str] = []
    total = 0
    usage: dict[str, Any] = {
        "used_chunk_ids": set(),
        "used_pages": [],
        "specs_included": False,
        "total_chars": 0,
    }

    # ── Block 1: Spec nodes ───────────────────────────────────────────────────
    spec_block = format_specs_block(specs)
    if spec_block:
        parts.append(spec_block)
        total += len(spec_block)
        usage["specs_included"] = True

    # ── Block 2+: Full page content (Step 7) ─────────────────────────────────
    # We zip page_blocks against `expansions` metadata so we know which page
    # each block came from. page_blocks may contain None / empty strings that
    # were dropped before zip — filter and align index-by-index.
    if page_blocks:
        # Build a parallel list of expansion metadata aligned to page_blocks.
        # expand_to_pages appends one entry per non-empty fetched page, in the
        # same order it appends to page_blocks — so the lists are aligned.
        exp_by_idx = expansions or []
        for i, pb in enumerate(page_blocks):
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
            if i < len(exp_by_idx):
                usage["used_pages"].append(exp_by_idx[i])
            if total >= max_chars:
                break

    if total >= max_chars:
        context = "\n\n---\n\n".join(parts)
        usage["total_chars"] = len(context)
        return context, usage

    # ── Block N+: Fallback chunk text for ranks 3-8 ───────────────────────────
    # Skip chunks whose IDs were already covered by page expansion — only omit
    # those that actually have page content, rather than blanket-skipping by rank
    # offset (which can drop rank-1 when its page was not found but rank-2's was).
    expanded_cids: set[str] = set()
    if expansions:
        for e in expansions:
            cid = e.get("expanded_from_chunk_id")
            if cid:
                expanded_cids.add(cid)
    if expanded_cids:
        fallback_chunks = [
            c for c in chunks if c.get("chunk_id") not in expanded_cids
        ]
    else:
        fallback_chunks = chunks

    # Tables kept first so partial tables can be skipped atomically, but within
    # each type we now respect the reranker order (descending rerank_score).
    # Tables and text are sorted independently so a high-relevance table still
    # lands before any text chunk (preserving the no-partial-table guarantee),
    # while irrelevant tables no longer bury a top-ranked text chunk.
    table_chunks = sorted(
        [c for c in fallback_chunks if c.get("chunk_type") == "table"],
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )
    text_chunks = sorted(
        [c for c in fallback_chunks if c.get("chunk_type") != "table"],
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )
    ordered = table_chunks + text_chunks

    _reranker = None

    for c in ordered:
        chunk_type = c.get("chunk_type", "text")
        pr = c.get("page_range") or [0, 0]
        bid = c.get("book_id") or "?"
        path_s = " > ".join(c.get("section_path") or [])

        if chunk_type == "table":
            block = format_table_for_llm(c)
            if not block:
                raw = (c.get("text") or "").strip()
                if raw:
                    ts_path = " > ".join(c.get("section_path") or [])
                    citation = (
                        f"[Table | Book: {bid} | § {ts_path} | Page: {pr[0]}-{pr[1]}]"
                        if ts_path
                        else f"[Table | Book: {bid} | Page: {pr[0]}-{pr[1]}]"
                    )
                    block = f"{citation}\n{raw}"
            if not block:
                continue
            if total + len(block) > max_chars:
                continue  # skip partial tables — partial data is worse than none

        else:
            text = (c.get("text") or "").strip()
            if not text:
                continue
            # Enhanced citation: [Book: X | § Section Name | Page: Y]
            citation = (
                f"[Book: {bid} | § {path_s} | Page: {pr[0]}-{pr[1]}]"
                if path_s
                else f"[Book: {bid} | Page: {pr[0]}-{pr[1]}]"
            )
            label = citation + "\n"

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
                        if _reranker is None:
                            _reranker = get_reranker()
                        seg_results = _reranker.rerank(query, segs)
                        best_i = seg_results[0]["index"]
                        block = label + segs[best_i]
                    except Exception:
                        block = label + segs[0][:4000]

            if total + len(block) > max_chars:
                block = block[: max(0, max_chars - total)]

        parts.append(block)
        total += len(block)
        usage["used_chunk_ids"].add(c.get("chunk_id") or "")
        if total >= max_chars:
            break

    context = "\n\n---\n\n".join(parts)
    usage["total_chars"] = len(context)
    return context, usage


# ─────────────────────────────────────────────────────────────────────────────
# QUERY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────


# Pattern order matters: more specific patterns checked first.
# comparison > process > spec_lookup > overview
QUERY_PATTERNS: dict[str, re.Pattern] = {
    # 1. Comparison — strongest signal (vs, versus, compare, difference)
    "comparison": re.compile(
        r"\b(vs|versus|compare|difference)\b",
        re.I,
    ),
    # 2. Process — how/explain/describe/sequence/steps/procedure
    "process": re.compile(
        r"(how\s+(do|does|is|are|was)|explain|describe|sequence|steps|process|procedure)",
        re.I,
    ),
    # 3. Spec lookup
    #    a) "what is the X of/in/for Y" (flexible: allows multiple words between article and preposition)
    #    b) value + engineering unit: e.g. "58 bar", "799 kN", "120 s"
    #    c) model numbers: e.g. "S200", "PSLV-C50", "PS4"
    #    d) keywords: value, spec, parameter
    "spec_lookup": re.compile(
        r"(?:"
        r"what\s+(?:is|are)\s+(?:the\s+)+[\w\s]{3,60}?\s+(?:of|for|in)(?:\s+|$)|"
        r"\b\d+(?:\.\d+)?\s*(?:kn|mpa|bar|kg|s\b|km|n\b|m\b|mm|cm|kw|mw|%|k\b|psi|rpm|t\b|tonne|hz|sec|min|hr|mn)\b|"
        r"\b[A-Z]+-?\d+\b|"
        r"\b(value|spec|parameter)s?\b"
        r")",
        re.I,
    ),
    # 4. Overview — kept last so "what is" doesn't eat spec_lookup queries
    "overview": re.compile(
        r"^(what\s+is|tell\s+me\s+about|summarize|overview)",
        re.I,
    ),
}


@time_it
def classify_query(query: str) -> str:
    """
    Classifies a user query into a type for routing to specialized retrieval.

    Returns one of: "spec_lookup", "process", "comparison", "overview", "general".
    """
    if not query:
        return "general"
    for qtype, pattern in QUERY_PATTERNS.items():
        if pattern.search(query):
            return qtype
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RETRIEVAL + STREAMING
# ─────────────────────────────────────────────────────────────────────────────


@log_process
def retrieve_bundle(query: str, book_ids: list[str], mode: str) -> dict[str, Any]:
    cfg = dict(MODE_CONFIG.get(mode, MODE_CONFIG["balanced"]))  # copy so overrides don't mutate
    prop_lim = cfg.get("prop_retrieve_limit", PROP_RETRIEVE_LIMIT)
    sect_lim = cfg.get("sect_retrieve_limit", SECT_RETRIEVE_LIMIT)

    # ── Query classification & routing ────────────────────────────────────────
    query_type = classify_query(query) if ENABLE_QUERY_CLASSIFICATION else "general"
    type_overrides = (
        QUERY_TYPE_OVERRIDES.get(mode, {}).get(query_type, QUERY_TYPE_GENERAL)
        if ENABLE_QUERY_CLASSIFICATION else QUERY_TYPE_GENERAL
    )

    # Apply type overrides on top of the mode config
    effective_final_top_n = max(1, cfg["final_top_n"] + type_overrides.get("final_top_n_adjust", 0))
    effective_context_max_chars = max(2000, cfg["context_max_chars"] + type_overrides.get("context_max_chars_adjust", 0))
    effective_boost_mult = type_overrides.get("boost_both_mult", 1.0)
    effective_page_range = type_overrides.get("page_expand_range", 1)
    cross_book = type_overrides.get("cross_book", False)

    # cross_book: ignore user's book filter and search all books
    # Use empty list (not None) so downstream functions don't need `or []` guards
    effective_book_ids: list[str] = [] if cross_book else (book_ids or [])

    if query_type != "general":
        logger.info(
            "[QUERY] type=%s | final_top_n=%d | ctx_chars=%d | boost_mult=%.1f | "
            "page_range=%d | cross_book=%s",
            query_type, effective_final_top_n, effective_context_max_chars,
            effective_boost_mult, effective_page_range, cross_book,
        )

    # ── Adaptive depth: use conservative limits for first pass ──────────────
    if ENABLE_ADAPTIVE_DEPTH:
        first_prop_lim = max(5, int(prop_lim * ADAPTIVE_DEPTH_INITIAL_FRACTION))
        first_sect_lim = max(5, int(sect_lim * ADAPTIVE_DEPTH_INITIAL_FRACTION))
        logger.info(
            "[ADAPTIVE] first pass | prop_lim=%d (normal=%d) | sect_lim=%d (normal=%d)",
            first_prop_lim, prop_lim, first_sect_lim, sect_lim,
        )
    else:
        first_prop_lim = prop_lim
        first_sect_lim = sect_lim

    # Steps 1-6: use effective_book_ids instead of original book_ids
    prop_hits = search_propositions(query, effective_book_ids, first_prop_lim)
    entity_terms = extract_query_entities(query)
    neo4j_section_names = neo4j_sections_for_entities(effective_book_ids, entity_terms)
    specs = neo4j_specs_for_terms(effective_book_ids, entity_terms)

    parent_chunk_ids = list(
        {p["parent_chunk_id"] for p in prop_hits if p.get("parent_chunk_id")}
    )
    parent_sections = fetch_sections_by_chunk_ids(parent_chunk_ids, effective_book_ids)
    direct_sections = search_sections_direct(
        query, effective_book_ids, neo4j_section_names, first_sect_lim
    )

    # ── Step 4b: Hybrid Search Fusion (Vector + BM25) ────────────────────────
    # Fuse vector search results with BM25 keyword results using RRF to catch
    # exact-term matches that vector search may miss.
    if ENABLE_HYBRID:
        bm25_pool = first_sect_lim * HYBRID_POOL_MULTIPLIER
        fused = build_and_fuse(
            query,
            direct_sections,
            qdrant_client=get_qdrant(),
            book_ids=effective_book_ids or None,
            top_k=bm25_pool,
            rrf_k=HYBRID_RRF_K,
        )
        if fused:
            direct_sections = fused

    candidates = merge_candidates(parent_sections, direct_sections, neo4j_section_names)

    # Step 6: Rerank — fetch a larger pool when MMR is enabled so there
    # are enough candidates for diversity selection.
    pool_n = effective_final_top_n * MMR_POOL_MULTIPLIER if ENABLE_MMR else effective_final_top_n
    top_reranked = rerank_candidates(query, candidates, pool_n, boost_mult=effective_boost_mult)

    # ── Adaptive depth: second pass (expanded) if top score is too low ───────
    if (
        ENABLE_ADAPTIVE_DEPTH
        and top_reranked
        and top_reranked[0].get("rerank_score", 0.0) < ADAPTIVE_DEPTH_SCORE_THRESHOLD
    ):
        expanded_prop_lim = max(first_prop_lim + 1, int(first_prop_lim * ADAPTIVE_DEPTH_EXPAND_MULTIPLIER))
        expanded_sect_lim = max(first_sect_lim + 1, int(first_sect_lim * ADAPTIVE_DEPTH_EXPAND_MULTIPLIER))
        logger.info(
            "[ADAPTIVE] second pass | top_score=%.4f < %.2f | "
            "expanding prop: %d->%d | sect: %d->%d",
            top_reranked[0].get("rerank_score", 0.0),
            ADAPTIVE_DEPTH_SCORE_THRESHOLD,
            first_prop_lim, expanded_prop_lim,
            first_sect_lim, expanded_sect_lim,
        )

        # Re-run proposition search with expanded limits
        expanded_prop_hits = search_propositions(query, effective_book_ids, expanded_prop_lim)
        # Re-run section search with expanded limits
        expanded_direct = search_sections_direct(
            query, effective_book_ids, neo4j_section_names, expanded_sect_lim,
        )
        # Fetch parent sections for any new proposition IDs
        new_parent_ids = list(
            {p["parent_chunk_id"] for p in expanded_prop_hits if p.get("parent_chunk_id")}
        )
        new_ids_to_fetch = [cid for cid in new_parent_ids if cid not in set(parent_chunk_ids)]
        expanded_parent_sections = fetch_sections_by_chunk_ids(new_ids_to_fetch, effective_book_ids)

        # Hybrid fusion on expanded results if enabled
        if ENABLE_HYBRID:
            bm25_pool_expanded = expanded_sect_lim * HYBRID_POOL_MULTIPLIER
            fused = build_and_fuse(
                query,
                expanded_direct,
                qdrant_client=get_qdrant(),
                book_ids=effective_book_ids or None,
                top_k=bm25_pool_expanded,
                rrf_k=HYBRID_RRF_K,
            )
            if fused:
                expanded_direct = fused

        # Merge expanded results with existing candidates and re-rerank
        expanded_candidates = merge_candidates(
            parent_sections + expanded_parent_sections,
            direct_sections + expanded_direct,
            neo4j_section_names,
        )
        expanded_pool_n = max(pool_n, effective_final_top_n * ADAPTIVE_DEPTH_EXPAND_MULTIPLIER)
        top_reranked = rerank_candidates(
            query, expanded_candidates, expanded_pool_n, boost_mult=effective_boost_mult,
        )
        logger.info(
            "[ADAPTIVE] after expand | candidates: %d->%d | after rerank: %d | top_score=%.4f",
            len(candidates), len(expanded_candidates), len(top_reranked),
            top_reranked[0].get("rerank_score", 0.0) if top_reranked else 0.0,
        )

    # Step 6b: MMR diversity
    if ENABLE_MMR and len(top_reranked) > effective_final_top_n:
        top = mmr_select(top_reranked, effective_final_top_n, MMR_LAMBDA)
        logger.info(
            "[MMR] applied | pool=%d | selected=%d | lambda=%.2f",
            len(top_reranked), len(top), MMR_LAMBDA,
        )
    else:
        top = top_reranked[:effective_final_top_n]

    # Step 7: Page expansion with dynamic range
    page_blocks, expansions = expand_to_pages(top, page_expand_range=effective_page_range)

    # Step 8: Build context with adjusted budget
    # For overview/process queries, reserve budget for a section hierarchy summary
    hierarchy_block = (
        extract_section_hierarchy(effective_book_ids, expansions)
        if query_type in ("overview", "process") and expansions
        else None
    )
    hierarchy_chars = len(hierarchy_block) + 200 if hierarchy_block else 0  # +200 for "\n\n---\n\n" separator

    # Ensure detail context gets at least 1200 chars after reserving for hierarchy
    detail_budget = max(1200, effective_context_max_chars - hierarchy_chars)

    context, usage = build_context(
        top, query, specs,
        detail_budget,
        page_blocks, expansions,
    )

    # Prepend hierarchy block before the detail context
    if hierarchy_block and context:
        context = hierarchy_block + "\n\n---\n\n" + context
        logger.info(
            "[HIERARCHY] prepended for query_type=%s | block=%d chars | detail=%d chars | total=%d",
            query_type, len(hierarchy_block), len(context) - len(hierarchy_block) - 6, len(context),
        )

    # Build system prompt with query-type-specific instructions
    system_prompt = get_system_prompt(mode, query_type)

    # ── Sources: reflect what the LLM ACTUALLY saw ────────────────────────────
    # Map rank-1 / rank-2 chunk -> did its page make it into context? If so, we
    # attribute that chunk as "represented by its page expansion" even though
    # its chunk text was skipped by build_context (chunks[2:] / early return).
    page_expanded_chunk_ids: set[str] = {
        e["expanded_from_chunk_id"]
        for e in usage.get("used_pages", [])
        if e.get("expanded_from_chunk_id")
    }

    sources: list[dict] = []
    for c in top:
        cid = c.get("chunk_id") or ""
        chunk_text_in = cid in usage.get("used_chunk_ids", set())
        page_in = cid in page_expanded_chunk_ids
        if chunk_text_in:
            included = "chunk"
        elif page_in:
            included = "page_expansion"
        else:
            included = None  # retrieved but NOT shown to the LLM

        sources.append({
            "chunk_id": cid,
            "book_id": c.get("book_id"),
            "page_range": c.get("page_range"),
            "section_path": c.get("section_path"),
            "chunk_type": c.get("chunk_type"),
            "from_qdrant": c.get("from_qdrant"),
            "from_neo4j": c.get("from_neo4j"),
            "rerank_score": round(c.get("rerank_score", 0.0), 4),
            # included_in_context: True only if the LLM actually saw this chunk's
            # text or its page-expansion content. False = retrieved, then dropped
            # by the context budget / page-expansion path.
            "included_in_context": included is not None,
            "included_via": included,  # "chunk" | "page_expansion" | None
        })

    # Page-expansion-only entries: pages N-1/N+1 around rank-1 (and the rank-2
    # page if it wasn't already a chunk source) are real context the LLM saw but
    # had no source row. Emit them so the UI can show every page that informed
    # the answer.
    for e in usage.get("used_pages", []):
        key = (e["book_id"], e["page"])
        already = any(
            (s.get("book_id"), (s.get("page_range") or [0, 0])[0]) == key
            for s in sources
        )
        if already:
            continue
        sources.append({
            "chunk_id": None,
            "book_id": e["book_id"],
            "page_range": [e["page"], e["page"]],
            "section_path": [],
            "chunk_type": "page_expansion",
            "from_qdrant": False,
            "from_neo4j": False,
            "rerank_score": None,
            "included_in_context": True,
            "included_via": "page_expansion",
            "expanded_from_chunk_id": e.get("expanded_from_chunk_id"),
            "relation": e.get("relation"),  # prev | main | next | single | fallback
        })

    # Log a compact source summary for debugging "wrong source" complaints.
    shown = sum(1 for s in sources if s["included_in_context"])
    logger.info(
        "[CONTEXT] top_n=%d | included=%d | dropped=%d | page_expansions=%d | "
        "specs=%s | ctx_chars=%d/%d",
        len(top),
        shown,
        len(top) - shown,
        len([s for s in sources if s.get("chunk_type") == "page_expansion"]),
        usage.get("specs_included", False),
        usage.get("total_chars", 0),
        cfg["context_max_chars"],
    )

    return {
        "context": context,
        "sources": sources,
        "system_prompt": system_prompt,
        "mode": mode,
        "cfg": cfg,
    }


@time_it
def _history_block(history: list[dict]) -> str:
    return "\n\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content') or ''}"
        for m in history
    )


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CONTEXT (4.4)
# ─────────────────────────────────────────────────────────────────────────────


@log_process
def extract_section_hierarchy(
    book_ids: list[str],
    expansions: list[dict],
) -> str:
    """
    Builds a section hierarchy summary from the pages that are in context.

    Reads metadata for each book that has expanded pages, extracts section
    paths, builds a tree, and for each leaf section includes the first
    sentence as a preview.

    Returns a formatted Markdown string like:

    ## Document Structure

    **PSLV-C50**
    - 4.3 Propulsion System
      - **4.3.1 Vikas Engine**
        → The Vikas engine is a liquid-fueled rocket engine...
      - **4.3.2 S200 Booster**
        → The S200 is a solid rocket booster...
    - 5.0 Payload Fairing
      → The payload fairing protects the satellite...

    Returns empty string if no expansions or metadata is unavailable.
    """
    if not expansions:
        return ""

    # Collect unique (book_id, page) tuples from expansions
    seen: set[tuple[str, int]] = set()
    book_pages: dict[str, set[int]] = {}
    for e in expansions:
        bid = e.get("book_id", "")
        page = e.get("page", 0)
        if not bid:
            continue
        key = (bid, page)
        if key in seen:
            continue
        seen.add(key)
        if bid not in book_pages:
            book_pages[bid] = set()
        book_pages[bid].add(page)

    if not book_pages:
        return ""

    lines: list[str] = ["## Document Structure", ""]

    for bid in sorted(book_pages.keys()):
        meta = _load_metadata(bid)
        if not meta:
            continue

        lines.append(f"**{bid}**")

        # Collect all (section_path, first_sentence) tuples for this book
        section_entries: list[tuple[list[str], str]] = []
        for page_num in sorted(book_pages[bid]):
            entry = meta.get(str(page_num))
            if not entry:
                continue
            sections = entry.get("sections") or []
            if not sections:
                continue
            content = (entry.get("full_content") or "").strip()
            first = _first_sentence(content) if content else f"[Page {page_num}]"
            if not first:
                first = f"[Page {page_num}]"
            section_entries.append((sections, first))

        # Build indented hierarchy — deduplicate paths so each section
        # appears only once even across multiple pages.
        seen_paths: set[str] = set()
        for sections, first in section_entries:
            for depth, sec in enumerate(sections):
                indent = "  " * (depth + 1)
                path_key = " > ".join(sections[: depth + 1])
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                if depth == len(sections) - 1:
                    # Leaf section — bold + first sentence preview
                    lines.append(f"{indent}- **{sec}**")
                    if first:
                        lines.append(f"{indent}  → {first}")
                else:
                    # Parent section — just the name
                    lines.append(f"{indent}- {sec}")

        lines.append("")

    result = "\n".join(lines).strip()
    # If the result only has the header and optional book names but no actual
    # section entries (i.e., no lines with "- "), return empty.
    if result == "## Document Structure" or not any(" - " in l for l in lines):
        return ""  # no actual content added
    return result


@time_it
def build_user_message(query: str, context: str, history: list[dict]) -> str:
    """
    Adds an explicit per-message reminder to never reference tables.
    Redundancy with system prompt is intentional and effective.
    """
    query = query or ""
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
        return (
            f"Conversation so far:\n{hb}\n\n"
            f"Context from knowledge base:\n{context}\n\n"
            f"Current question: {query}{reminder}\nAnswer:"
        )
    return (
        f"Context from knowledge base:\n{context}\n\n"
        f"Question: {query}{reminder}\nAnswer:"
    )


async def run_rag_stream(
    query: str,
    book_ids: list[str],
    mode: str,
    history: list[dict],
) -> AsyncIterator[dict]:
    yield {"type": "status", "message": "Searching knowledge base..."}
    logger.info("RAG query | mode=%s | books=%s | query=%.80s", mode, book_ids, query)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        bundle = await loop.run_in_executor(
            None, lambda: retrieve_bundle(query, book_ids, mode)
        )
    except Exception as e:
        logger.error("retrieve_bundle failed: %s", e)
        yield {"type": "error", "message": str(e)}
        return
    logger.info("retrieve_bundle complete in %.3fs", time.perf_counter() - t0)

    yield {"type": "status", "message": "Generating answer..."}
    user_content = build_user_message(query, bundle["context"], history)
    messages = [
        {"role": "system", "content": bundle["system_prompt"]},
        {"role": "user", "content": user_content},
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
        yield {
            "type": "done",
            "sources": bundle["sources"],
            "mode": mode,
            "full_text": full,
        }
