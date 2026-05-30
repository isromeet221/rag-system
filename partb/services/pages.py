"""
partb/services/pages.py
------------------------
Page text service for the book viewer panel in the chat UI.

CHANGES IN THIS VERSION:
  Previously read from _ready.json (chunks) → showed one chunk per page.

  Now reads from _metadata.json (built by build_metadata.py):
    - get_page_text()        → full_content for the page
    - get_page_info()        → page entry with sections list
    - get_total_pages()      → total pages from __meta__
    - get_sorted_page_numbers() → all indexed page numbers sorted
    - count_pages()          → kept for backward compat with meta_router.py

  Falls back to _ready.json for books ingested before build_metadata.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from partb.logger import time_it, async_time_it

from partb.config import CHECKPOINTS_DIR, METADATA_DIR, PARTA_DATA_DIR


@time_it
def _metadata_path(book_id: str) -> Path:
    return METADATA_DIR / f"{book_id}_metadata.json"


@time_it
def _ready_path(book_id: str) -> Path:
    return CHECKPOINTS_DIR / f"{book_id}_ready.json"


@time_it
def _legacy_ready_path(book_id: str) -> Path:
    return PARTA_DATA_DIR / "metadata" / f"{book_id}_ready.json"


@time_it
def _load_metadata(book_id: str) -> dict | None:
    path = _metadata_path(book_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@time_it
def _load_ready(book_id: str) -> list | None:
    for path in [_ready_path(book_id), _legacy_ready_path(book_id)]:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else None
            except Exception:
                continue
    return None


@time_it
def get_page_text(book_id: str, page_number: int) -> str | None:
    """
    Returns full readable content for a given page.
    Reads from _metadata.json (preferred), falls back to _ready.json.
    """
    meta = _load_metadata(book_id)
    if meta is not None:
        entry = meta.get(str(page_number))
        if not entry:
            return None
        return (entry.get("full_content") or "").strip() or None

    # Fallback: join all chunks on this page from _ready.json
    chunks = _load_ready(book_id)
    if chunks is None:
        return None

    blocks: list[str] = []
    for rec in chunks:
        page_range = rec.get("page_range")
        if page_range:
            if isinstance(page_range, dict):
                start = int(page_range.get("start", 0))
                end   = int(page_range.get("end", start))
            elif isinstance(page_range, list) and page_range:
                start = int(page_range[0])
                end   = int(page_range[1]) if len(page_range) > 1 else start
            else:
                continue
            if not (start <= page_number <= end):
                continue
        elif rec.get("page_number") is not None:
            if int(rec.get("page_number", -1)) != page_number:
                continue
        else:
            continue

        text = (rec.get("content") or rec.get("text") or "").strip()
        if text:
            section_path = rec.get("section_path") or []
            if section_path:
                blocks.append(f"**{' > '.join(section_path)}**\n\n{text}")
            else:
                blocks.append(text)

    return "\n\n---\n\n".join(blocks) if blocks else None


@time_it
def get_page_info(book_id: str, page_number: int) -> dict | None:
    """
    Returns full metadata entry for a page: page_number, sections, full_content.
    Used by API to return section labels alongside content.
    """
    meta = _load_metadata(book_id)
    if meta is None:
        return None
    entry = meta.get(str(page_number))
    if not entry:
        return None
    return {
        "page_number":  entry.get("page_number", page_number),
        "sections":     entry.get("sections") or [],
        "full_content": (entry.get("full_content") or "").strip(),
    }


@time_it
def get_total_pages(book_id: str) -> int:
    """Total indexed page count. O(1) from __meta__ entry."""
    meta = _load_metadata(book_id)
    if meta is not None:
        total = (meta.get("__meta__") or {}).get("total_pages")
        if total and isinstance(total, int):
            return total
        return len([k for k in meta if k != "__meta__"])
    return count_pages(book_id)


@time_it
def get_sorted_page_numbers(book_id: str) -> list[int]:
    """
    All indexed page numbers in sorted order.
    Used by Prev/Next navigation to step only through pages with content.
    """
    meta = _load_metadata(book_id)
    if meta is not None:
        nums = []
        for k in meta:
            if k == "__meta__":
                continue
            try:
                nums.append(int(k))
            except ValueError:
                continue
        return sorted(nums)

    chunks = _load_ready(book_id)
    if not chunks:
        return []
    pages: set[int] = set()
    for rec in chunks:
        pr = rec.get("page_range")
        if pr:
            if isinstance(pr, dict):
                s, e = int(pr.get("start", 0)), int(pr.get("end", 0))
            elif isinstance(pr, list) and pr:
                s = int(pr[0]); e = int(pr[1]) if len(pr) > 1 else s
            else:
                continue
            for p in range(s, e + 1):
                pages.add(p)
        elif rec.get("page_number") is not None:
            pages.add(int(rec["page_number"]))
    return sorted(pages)


@time_it
def count_pages(book_id: str) -> int:
    """Backward-compatible page count for meta_router.py."""
    meta = _load_metadata(book_id)
    if meta is not None:
        total = (meta.get("__meta__") or {}).get("total_pages")
        if total and isinstance(total, int):
            return total
        return len([k for k in meta if k != "__meta__"])

    chunks = _load_ready(book_id)
    if not chunks:
        return 0
    pages: set[int] = set()
    for rec in chunks:
        pr = rec.get("page_range")
        if pr:
            if isinstance(pr, dict):
                s, e = int(pr.get("start", 0)), int(pr.get("end", 0))
            elif isinstance(pr, list) and pr:
                s = int(pr[0]); e = int(pr[1]) if len(pr) > 1 else s
            else:
                continue
            for p in range(s, e + 1):
                pages.add(p)
        elif rec.get("page_number") is not None:
            pages.add(int(rec["page_number"]))
    return len(pages) if pages else len(chunks)