"""
pdf_router.py  —  Part B
Serves raw PDF files from parta/data/raw/ to the iframe viewer in chat.html.

Endpoints:
  GET /pdf/{book_id}        → streams the PDF binary  (JWT protected)
  GET /pdf/{book_id}/info   → returns {"total_pages": N}  (JWT protected)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pypdf
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from partb.auth_jwt import verify_token
from partb.logger import time_it, async_time_it

from partb.config import MONGO_DB, PARTA_DATA_DIR
from partb.db import get_mongo

router = APIRouter(prefix="/pdf", tags=["pdf"])

PDF_DIR = PARTA_DATA_DIR / "raw"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip()).strip("-").lower()


@time_it
def _pdf_path(book_id: str) -> Path:
    """
    Resolve the original PDF for a book.

    Normal upload path is parta/data/raw/{book_id}.pdf.  In practice, older
    ingests or moved deployments can leave the PDF under a title/original
    filename, or under a different data directory.  Try those fallbacks before
    returning 404 so source-reference clicks keep working.
    """
    checked: list[Path] = []

    def add_candidate(path: Path | str | None) -> Path | None:
        if not path:
            return None
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = PARTA_DATA_DIR.parent / p
        checked.append(p)
        return p if p.is_file() else None

    names = {book_id, _slug(book_id), book_id.replace("-", "_"), book_id.replace("_", "-")}

    # Metadata stored by Part A can contain the human title or source path.
    try:
        doc = get_mongo()[MONGO_DB]["library"].find_one({"book_id": book_id}) or {}
        for key in ("pdf_path", "raw_pdf_path", "source_path", "file_path", "original_path"):
            found = add_candidate(doc.get(key))
            if found:
                return found
        for key in ("book_title", "title", "original_filename", "filename"):
            val = doc.get(key)
            if val:
                names.update({str(val), Path(str(val)).stem, _slug(str(val))})
    except Exception:
        # Do not fail PDF serving just because Mongo metadata is unavailable.
        pass

    # Standard and filename-variant locations.
    search_dirs = [PDF_DIR]
    extra_dirs = os.environ.get("RAG_PDF_DIRS", "")
    search_dirs.extend(Path(p) for p in extra_dirs.split(os.pathsep) if p.strip())
    for directory in search_dirs:
        for name in names:
            for candidate in (directory / f"{name}.pdf", directory / f"{name}"):
                found = add_candidate(candidate)
                if found:
                    return found

    # Last fallback: recursively scan the Part A data folder for matching PDF stem.
    roots = [PARTA_DATA_DIR, PARTA_DATA_DIR.parent]
    wanted = {_slug(n) for n in names if n}
    for root in roots:
        if not root.exists():
            checked.append(root)
            continue
        for pdf in root.rglob("*.pdf"):
            if _slug(pdf.stem) in wanted:
                return pdf

    checked_preview = ", ".join(str(p) for p in checked[:8])
    raise HTTPException(
        404,
        f"PDF not found for book '{book_id}'. Checked: {checked_preview}. "
        "Put the PDF at parta/data/raw/{book_id}.pdf or set RAG_PDF_DIRS to the folder containing PDFs.",
    )


# ── Allow token via query param so <iframe src="..."> works ──────
# Browsers cannot send Authorization headers for iframe src URLs.
@async_time_it
async def verify_token_or_query(
    token: str = Query(None),
    user=Depends(verify_token),
):
    return user


@router.get("/{book_id}/info")
@async_time_it
async def pdf_info(book_id: str, user=Depends(verify_token)):
    """Return total page count. Called once when opening the panel."""
    path = _pdf_path(book_id)
    try:
        reader = pypdf.PdfReader(str(path))
        return {"book_id": book_id, "total_pages": len(reader.pages)}
    except Exception as e:
        raise HTTPException(500, f"Could not read PDF: {e}")


@router.get("/{book_id}")
@async_time_it
async def serve_pdf(book_id: str, token: str = Query(None)):
    """
    Stream the raw PDF binary to the browser iframe.
    Token is accepted as a query param because iframe src
    cannot carry Authorization headers.
    """
    # Manual token check since we accept it as query param
    if not token:
        raise HTTPException(401, "Missing token")
    try:
        from partb.auth_jwt import decode_token
        decode_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")

    path = _pdf_path(book_id)
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=f"{book_id}.pdf",
        headers={"Cache-Control": "private, max-age=3600" , "content-Disposition": f'inline; filename="{book_id}.pdf"'},
    )
