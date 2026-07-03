"""Library, book page viewer, health, model list. partb/router/meta_router"""
from __future__ import annotations

import json
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException

from partb.auth_jwt import verify_token
from partb.config import  (
    COLLECTION_PROPS,
    COLLECTION_SECTIONS,
    LITELLM_API_KEY,
    LITELLM_BASE_URL, 
    MONGO_DB, 
    OLLAMA_URL, 
    USE_OLLAMA_DIRECT
)
from partb.db import get_mongo
from partb.logger import time_it, async_time_it

from partb.services.pages import count_pages, get_page_text

router = APIRouter(tags=["meta"])


@time_it
def library_col():
    return get_mongo()[MONGO_DB]["library"]


@router.get("/library")
@async_time_it
async def list_library(user: dict = Depends(verify_token)):
    books = list(
        library_col()
        .find({"status": "ready"}, {"_id": 0, "book_id": 1, "book_title": 1, "completed_at": 1, "confidence_report": 1})
        .sort("completed_at", -1)
    )
    out = []
    for b in books:
        bid = b["book_id"]
        confidence = b.get("confidence_report", {})
        total_chunks = confidence.get("total_chunks", 0) if isinstance(confidence, dict) else 0
        total_pages = count_pages(bid)
        if not total_pages and isinstance(confidence, dict):
            total_pages = confidence.get("total_pages", 0)
        out.append(
            {
                "book_id": bid,
                "title": b.get("book_title", bid),
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "created_at": (
                    b["completed_at"].isoformat()
                    if isinstance(b.get("completed_at"), datetime)
                    else b.get("completed_at", "")
                ),
            }
        )
    return {"books": out}


@router.get("/books/{book_id}/page/{page_number}")
@async_time_it
async def get_page(book_id: str, page_number: int, user: dict = Depends(verify_token)):
    book = library_col().find_one({"book_id": book_id, "status": "ready"})
    if not book:
        raise HTTPException(404, f"Book '{book_id}' not found.")

    text = get_page_text(book_id, page_number)
    if text is None:
        raise HTTPException(404, f"Page {page_number} not found.")
    return {"book_id": book_id, "page_number": page_number, "text": text}


@router.get("/health")
@async_time_it
async def health():
    status: dict = {}

    try:
        from partb.retrieval.pipeline import get_neo4j

        with get_neo4j().session() as s:
            s.run("RETURN 1")
        status["neo4j"] = {"status": "ok"}
    except Exception as e:
        status["neo4j"] = {"status": "error", "detail": str(e)}

    try:
        from partb.retrieval.pipeline import get_qdrant

        qc = get_qdrant()
        info = qc.get_collections()
        collection_names = [c.name for c in info.collections]
       
        col_status={}
        for col in [COLLECTION_PROPS, COLLECTION_SECTIONS]:
            if col in collection_names :
                col_info = qc.get_collection(col)
                col_status[col] = {
                    "status" : "ok",
                    "points" : col_info.points_count,
                }
            else:
                col_status[col] = {
                    "status" : "missing",
                    "detail" : f"Collection '{col}' not found.Run ingest_qdrant.py.",
                }
        status["qdrant"] = {
            "status": "ok",
            "collections": [c.name for c in info.collections],
        }
    except Exception as e:
        status["qdrant"] = {"status": "error", "detail": str(e)}

#________________MongoDB__________________

    try:
        get_mongo().admin.command("ping")
        status["mongodb"] = {"status": "ok"}
    except Exception as e:
        status["mongodb"] = {"status": "error", "detail": str(e)}

    if USE_OLLAMA_DIRECT:
        try:
            r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            r.raise_for_status()
            status["ollama"] = {"status": "ok", "models": [m["name"] for m in r.json().get("models", [])]}
        except Exception as e:
            status["ollama"] = {"status": "error", "detail": str(e)}
        status["litellm"] = {"status": "skipped", "note": "USE_OLLAMA_DIRECT=1"}
    else:
        try:
            h = {"Content-Type": "application/json"}
            if LITELLM_API_KEY:
                h["Authorization"] = f"Bearer {LITELLM_API_KEY}"
            r = httpx.get(f"{LITELLM_BASE_URL}/models", headers=h, timeout=5.0)
            r.raise_for_status()
            data = r.json()
            status["litellm"] = {"status": "ok", "data": data.get("data", data)}
        except Exception as e:
            status["litellm"] = {"status": "error", "detail": str(e)}

    return status


@router.get("/models")
@async_time_it
async def list_models(_: dict = Depends(verify_token)):
    if USE_OLLAMA_DIRECT:
        try:
            r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            r.raise_for_status()
            return {"models": r.json().get("models", [])}
        except Exception as e:
            raise HTTPException(503, f"Ollama unreachable: {e}") from e
    h = {}
    if LITELLM_API_KEY:
        h["Authorization"] = f"Bearer {LITELLM_API_KEY}"
    try:
        r = httpx.get(f"{LITELLM_BASE_URL}/models", headers=h, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(503, f"LiteLLM unreachable: {e}") from e

@router.delete("/books/{book_id}")
@async_time_it
async def delete_book(book_id: str, user: dict = Depends(verify_token)):
    from partb.retrieval.pipeline import get_neo4j, get_qdrant
    from qdrant_client.http import models

    try:
        with get_neo4j().session() as s:
            s.run("MATCH (n) WHERE n.book_id = $book_id DETACH DELETE n", book_id=book_id)

        qc = get_qdrant()
        filter_query = models.Filter(
            must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))]
        )
        for col in [COLLECTION_PROPS, COLLECTION_SECTIONS]:
            try:
                qc.delete(
                    collection_name=col,
                    points_selector=models.FilterSelector(filter=filter_query)
                )
            except Exception:
                pass

        library_col().delete_one({"book_id": book_id})

        return {"status": "success", "message": f"Book {book_id} deleted"}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")

