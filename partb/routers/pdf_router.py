"""
pdf_router.py  —  Part B
Serves raw PDF files from parta/data/raw/ to the iframe viewer in chat.html.

Endpoints:
  GET /pdf/{book_id}        → streams the PDF binary  (JWT protected)
  GET /pdf/{book_id}/info   → returns {"total_pages": N}  (JWT protected)
"""
from __future__ import annotations

from pathlib import Path

import pypdf
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from partb.auth_jwt import verify_token
from partb.config import PARTA_DATA_DIR

router = APIRouter(prefix="/pdf", tags=["pdf"])

PDF_DIR = PARTA_DATA_DIR / "raw"


def _pdf_path(book_id: str) -> Path:
    path = PDF_DIR / f"{book_id}.pdf"
    if not path.is_file():
        raise HTTPException(404, f"PDF not found for book '{book_id}'")
    return path


# ── Allow token via query param so <iframe src="..."> works ──────
# Browsers cannot send Authorization headers for iframe src URLs.
async def verify_token_or_query(
    token: str = Query(None),
    user=Depends(verify_token),
):
    return user


@router.get("/{book_id}/info")
async def pdf_info(book_id: str, user=Depends(verify_token)):
    """Return total page count. Called once when opening the panel."""
    path = _pdf_path(book_id)
    try:
        reader = pypdf.PdfReader(str(path))
        return {"book_id": book_id, "total_pages": len(reader.pages)}
    except Exception as e:
        raise HTTPException(500, f"Could not read PDF: {e}")


@router.get("/{book_id}")
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
