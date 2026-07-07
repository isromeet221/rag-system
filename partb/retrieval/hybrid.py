"""
partb/retrieval/hybrid.py
-------------------------
Hybrid search implementation: vector (Qdrant) + keyword (BM25) fused via
Reciprocal Rank Fusion (RRF).

Provides:
  BM25Index     — Lazy-built BM25 index over section texts from Qdrant.
  rrf_fuse      — Reciprocal Rank Fusion over two ranked result lists.
  build_and_fuse — One-shot helper: BM25 search + vector results → fused list.

Usage in pipeline.py:
    from partb.retrieval.hybrid import get_bm25_index, rrf_fuse

    # In retrieve_bundle():
    if ENABLE_HYBRID:
        bm25_results = get_bm25_index().search(query, top_k=sect_lim * 2)
        direct_sections = rrf_fuse(
            direct_sections,   # from search_sections_direct
            bm25_results,      # from BM25Index.search
            k=HYBRID_RRF_K,
        )
"""

from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from partb.config import ENABLE_HYBRID, HYBRID_RRF_K, HYBRID_POOL_MULTIPLIER

logger = logging.getLogger("RAG.partb")

# ── Global singleton ─────────────────────────────────────────────────────────
_bm25_index: "BM25Index | None" = None


def get_bm25_index(qdrant_client=None) -> "BM25Index":
    """Return the global BM25Index singleton, building it on first call.

    Args:
        qdrant_client: Required on first call (when index needs to be built).
                       Pass ``None`` on subsequent calls.
    """
    global _bm25_index
    if _bm25_index is None:
        if qdrant_client is None:
            raise RuntimeError(
                "First call to get_bm25_index() requires a qdrant_client argument."
            )
        _bm25_index = BM25Index(qdrant_client)
    return _bm25_index


def reset_bm25_index() -> None:
    """Drop the cached index so it is rebuilt on next use.  Useful for tests."""
    global _bm25_index
    _bm25_index = None


# ── Tokenizer ────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25.

    Keeps alphanumeric tokens, strips punctuation (except internal hyphens
    which are meaningful in ISRO terms like "SSLV-D1", "PSLV-C50").
    """
    import re
    # Normalise whitespace, then split on non-alphanumeric, non-hyphen chars
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = text.split()
    # Drop very short tokens (< 2 chars) since they are noise for BM25
    return [t for t in tokens if len(t) >= 2]


# ── BM25 Index ───────────────────────────────────────────────────────────────


class BM25Index:
    """Lazy-built BM25 index over section texts from Qdrant.

    Builds on first call to :meth:`search` by scrolling all points from the
    Qdrant sections collection.  Once built, it is read-only and thread-safe
    for concurrent scoring.

    To avoid circular imports, the Qdrant client is passed in rather than
    imported from pipeline.py.
    """

    def __init__(self, qdrant_client) -> None:
        self._bm25: BM25Okapi | None = None
        self._client = qdrant_client
        # Parallel list of section metadata dicts, one per indexed document.
        self._documents: list[dict] = []
        self._built = False

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def built(self) -> bool:
        return self._built

    def search(
        self,
        query: str,
        top_k: int = 40,
        book_ids: list[str] | None = None,
    ) -> list[dict]:
        """Run BM25 search and return the top-k results.

        Results are dicts with the same schema as Qdrant section payloads
        (see :func:`partb.retrieval.pipeline._parse_payload`), plus a
        ``bm25_score`` key.

        If *book_ids* is provided, results are filtered to those books before
        scoring — more efficient when the index is large and only a subset
        of books matters.
        """
        if not self._ensure_built():
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)  # type: ignore[union-attr]

        # Build filtered + scored results
        results: list[dict] = []
        set_book_ids = set(book_ids) if book_ids else None
        for i, doc in enumerate(self._documents):
            if set_book_ids and doc.get("book_id") not in set_book_ids:
                continue
            score = float(scores[i])
            if score <= 0.0:
                continue  # no keyword match — skip
            entry = dict(doc)  # shallow copy
            entry["bm25_score"] = score
            results.append(entry)

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    def search_all_books(
        self,
        query: str,
        top_k: int = 40,
    ) -> list[dict]:
        """Convenience: BM25 search across all books (no book_id filter)."""
        return self.search(query, top_k=top_k, book_ids=None)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _ensure_built(self) -> bool:
        """Build the index on first call by scrolling Qdrant sections.

        Returns True if the index is available after the attempt.
        """
        if self._built:
            return True
        logger.info("[HYBRID] Building BM25 index from Qdrant sections …")
        try:
            self._build_from_qdrant()
            self._built = True
            logger.info(
                "[HYBRID] BM25 index built — %d documents, %d tokens",
                len(self._documents),
                sum(len(d.get("_tokens", [])) for d in self._documents[:1]),
            )
            return True
        except Exception as exc:
            logger.error("[HYBRID] Failed to build BM25 index: %s", exc)
            return False

    def _build_from_qdrant(self) -> None:
        """Scroll all points from the Qdrant sections collection and index.

        Idempotent-safe: clears any previously stored documents before
        rebuilding so the index stays in sync with the scores array.
        """
        from partb.config import COLLECTION_SECTIONS

        # Clear any stale documents from a previous incomplete build attempt
        self._documents = []

        tokenized_corpus: list[list[str]] = []

        # Scroll in batches
        next_offset = None
        batch_count = 0
        total_tokens = 0
        while True:
            pts, next_offset = self._client.scroll(
                collection_name=COLLECTION_SECTIONS,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not pts:
                break
            batch_count += 1
            for p in pts:
                pl = p.payload or {}
                text = (pl.get("text") or "").strip()
                if not text:
                    continue
                tokens = _tokenize(text)
                if not tokens:
                    continue
                total_tokens += len(tokens)
                # Store metadata for later retrieval
                pr = pl.get("page_range")
                page_list = (
                    [pr.get("start", 0), pr.get("end", 0)]
                    if isinstance(pr, dict)
                    else pr
                    if isinstance(pr, list)
                    else [0, 0]
                )
                self._documents.append({
                    "chunk_id": pl.get("chunk_id") or str(p.id),
                    "text": text,
                    "book_id": pl.get("book_id"),
                    "section_path": pl.get("section_path") or [],
                    "page_range": page_list,
                    "chunk_type": pl.get("chunk_type") or "text",
                    "structured_json": pl.get("structured_json"),
                    "linearized_text": pl.get("linearized_text"),
                    "from_qdrant": True,
                    "from_neo4j": False,
                    "_tokens": tokens,
                })
                tokenized_corpus.append(tokens)

            if next_offset is None:
                break

        logger.info(
            "[HYBRID] Scrolled %d batches from %s — %d documents, %d total tokens",
            batch_count, COLLECTION_SECTIONS, len(tokenized_corpus), total_tokens,
        )

        if not tokenized_corpus:
            logger.warning("[HYBRID] No documents to index!")
            self._bm25 = BM25Okapi([])
            return

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._built = True

        # Free token memory — no longer needed
        for d in self._documents:
            d.pop("_tokens", None)


# ── RRF Fusion ───────────────────────────────────────────────────────────────


def rrf_fuse(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion over two ranked result lists.

    Each result dict must have a ``chunk_id`` key for deduplication.

    Args:
        vector_results: Ranked section list from Qdrant vector search
                        (descending ``qdrant_score``).
        bm25_results:   Ranked section list from BM25 search
                        (descending ``bm25_score``).
        k:              RRF constant (default 60, standard value).

    Returns:
        A single merged list ordered by descending RRF score, with an
        additional ``rrf_score`` key on each result dict.
    """
    if not vector_results and not bm25_results:
        return []
    if not bm25_results:
        return vector_results
    if not vector_results:
        return bm25_results

    # Build rank maps: chunk_id → 1-based rank
    vector_rank: dict[str, int] = {
        r.get("chunk_id", ""): i + 1
        for i, r in enumerate(vector_results)
        if r.get("chunk_id")
    }
    bm25_rank: dict[str, int] = {
        r.get("chunk_id", ""): i + 1
        for i, r in enumerate(bm25_results)
        if r.get("chunk_id")
    }

    # Collect all unique chunk_ids from both lists
    all_ids: set[str] = set(vector_rank.keys()) | set(bm25_rank.keys())
    # Also include results without chunk_id (keep them as-is at the end)
    fallback_items: list[dict] = [
        r for r in vector_results + bm25_results
        if not r.get("chunk_id")
    ]

    # Build id → result dict map (prefer vector result for richer metadata)
    result_map: dict[str, dict] = {}
    for r in vector_results:
        cid = r.get("chunk_id")
        if cid:
            result_map[cid] = r
    for r in bm25_results:
        cid = r.get("chunk_id")
        if cid and cid not in result_map:
            result_map[cid] = r

    ranked: list[dict] = []
    for cid in all_ids:
        if cid not in result_map:
            continue
        rv = vector_rank.get(cid, 1_000_000)  # large number if absent
        rb = bm25_rank.get(cid, 1_000_000)
        rrf = (1.0 / (k + rv)) + (1.0 / (k + rb))
        entry = dict(result_map[cid])  # shallow copy
        entry["rrf_score"] = round(rrf, 6)
        # Carry the original scores for logging/debugging
        entry["hybrid_vector_rank"] = rv if rv < 1_000_000 else None
        entry["hybrid_bm25_rank"] = rb if rb < 1_000_000 else None
        # Ensure qdrant_score is set for BM25-only entries so merge_candidates
        # can use it as a tiebreak value. Otherwise BM25-only results default
        # to 0 in merge and never win the score tiebreak for a duplicate chunk_id.
        if entry.get("qdrant_score") is None and entry.get("bm25_score") is not None:
            entry["qdrant_score"] = entry["bm25_score"]
        ranked.append(entry)

    ranked.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Append any fallback items that lacked chunk_id
    ranked.extend(fallback_items)

    return ranked


# ── Convenience: build + fuse in one call ────────────────────────────────────


def build_and_fuse(
    query: str,
    vector_results: list[dict],
    qdrant_client=None,
    book_ids: list[str] | None = None,
    top_k: int = 40,
    rrf_k: int = 60,
) -> list[dict]:
    """One-shot helper: run BM25 search and fuse with vector results.

    Args:
        query:           User query text.
        vector_results:  Section list from Qdrant vector search.
        qdrant_client:   Qdrant client (passed to BM25 index for lazy build).
        book_ids:        Optional book filter for BM25.
        top_k:           BM25 top-k to retrieve.
        rrf_k:           RRF constant.

    Returns:
        Fused result list ordered by RRF score descending.
    """
    if not ENABLE_HYBRID:
        return vector_results

    bm25_results = get_bm25_index(qdrant_client).search(
        query, top_k=top_k, book_ids=book_ids,
    )
    if not bm25_results:
        return vector_results

    fused = rrf_fuse(vector_results, bm25_results, k=rrf_k)
    logger.info(
        "[HYBRID] fused | vector=%d | bm25=%d | after_rrf=%d | k=%d",
        len(vector_results), len(bm25_results), len(fused), rrf_k,
    )
    return fused
