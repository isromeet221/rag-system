"""Chats + SSE ask endpoint.
partb/router/chats_router.py"""

from __future__ import annotations

import json
import time
import uuid
import asyncio
from datetime import datetime,timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from partb.auth_jwt import verify_token
from partb.config import MODE_CONFIG, MONGO_DB, MODE_ORDER
from partb.db import get_mongo
from partb.retrieval.pipeline import run_rag_stream
from partb.logger import time_it, async_time_it, logger

from partb.services.messages import get_all_messages,get_prior_messages,save_message
router = APIRouter(prefix="/chats", tags=["chats"])

class ChatCreate(BaseModel):
    title: str | None = None
    book_ids: list[Any]
    default_mode: str = "balanced"

    def normalized_book_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self.book_ids:
            value = _coerce_book_id(item)
            value = value.strip()
            if value:
                ids.append(value)
        return ids


def _coerce_book_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = (
            value.get("book_id")
            or value.get("id")
            or value.get("title")
            or value.get("book_title")
            or value.get("$oid")
        )
        return _coerce_book_id(nested)
    return str(value)

class ChatPatch(BaseModel):
    title: str | None = None
    default_mode: str | None = None

class AskBody(BaseModel):
    question: str
    mode: str = "balanced"

@time_it
def chats_col():
    return get_mongo()[MONGO_DB]["chats"]


@time_it
def _now_iso()-> str:
    return datetime.now(timezone.utc).isoformat()

@time_it
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

@time_it
def _assert_owner(chat_id: str, user_id: str) -> dict:
    chat = chats_col().find_one({"chat_id": chat_id},{"_id":0})
    if not chat:
        raise HTTPException(404, f"Chat '{chat_id}' not found.")
    if chat["user_id"] != user_id:
        raise HTTPException(403, "You do not own this chat.")
    return chat


@router.post("")
@async_time_it
async def create_chat(body: ChatCreate, user: dict = Depends(verify_token)):
    book_ids = body.normalized_book_ids()
    if not book_ids:
        raise HTTPException(400, "At least one book must be selected.")
    if body.default_mode not in MODE_ORDER:
        raise HTTPException(400, f"mode must be one of {MODE_ORDER}")

    now = _now_iso()

    chat = {
        "chat_id": str(uuid.uuid4()),
        "user_id": user["user_id"],
        "title": (body.title or "New Chat").strip(),
        "book_ids": book_ids,
        "default_mode": body.default_mode,
        "message_count": 0,
        "created_at": now,
        "updated_at": now,
  
    }
    chats_col().insert_one(chat)
    chat.pop("_id", None)
    return chat


@router.get("")
@time_it
def list_chats(user: dict = Depends(verify_token)):
    cursor = (
        chats_col() 
        .find({"user_id": user["user_id"]}, {"_id": 0})
        .sort("updated_at", -1))
    return list(cursor)


@router.get("/{chat_id}")
@time_it
def get_chat(chat_id: str, user: dict = Depends(verify_token)):
    return _assert_owner(chat_id, user["user_id"])

@router.patch("/{chat_id}")
@time_it
def patch_chat(chat_id: str, body: ChatPatch, user: dict = Depends(verify_token)):
    _assert_owner(chat_id, user["user_id"])
    updates = {"updated_at": _now_iso()}
    if body.title is not None:
        updates["title"] = body.title.strip() or "NewChat"
    if body.default_mode is not None:
        if body.default_mode not in MODE_ORDER:
            raise HTTPException(400, f"mode must be one of{MODE_ORDER}")
        updates["default_mode"] = body.default_mode
    chats_col().update_one({"chat_id": chat_id}, {"$set": updates})
    return chats_col().find_one({"chat_id" : chat_id},{"_id" : 0})


@router.delete("/{chat_id}")
@time_it
def delete_chat(chat_id: str, user: dict = Depends(verify_token)):
    _assert_owner(chat_id, user["user_id"])
    chats_col().delete_one({"chat_id": chat_id})
    get_mongo()[MONGO_DB]["messages"].delete_many({"chat_id": chat_id})
    return {"message": "Chat deleted"}

@router.get("/{chat_id}/messages")
@time_it
def list_messages(chat_id: str, user: dict = Depends(verify_token)):
    _assert_owner(chat_id, user["user_id"])
    return get_all_messages(chat_id)

@router.post("/{chat_id}/ask")
@async_time_it
async def ask(chat_id: str, body: AskBody, user: dict = Depends(verify_token)):
    chat = _assert_owner(chat_id, user["user_id"])
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    mode = body.mode if body.mode in MODE_ORDER else chat.get("default_mode", "balanced")
    cfg = MODE_CONFIG[mode]
    history = get_prior_messages(chat_id, cfg["history_pairs"])
    logger.info("[CHAT] Ask received | chat_id=%s | user_id=%s | mode=%s | books=%s | question_chars=%s | history=%s", chat_id, user.get("user_id"), mode, chat.get("book_ids"), len(question), len(history))

    save_message(
        chat_id, "user", question, mode, [])

    ccol = chats_col()
    if chat.get("message_count", 0) == 0:
        auto_title = question[:60] or "New Chat"
        ccol.update_one(
            {"chat_id": chat_id},
            {"$set": {"title": auto_title}},
        )
    ccol.update_one(
        {"chat_id": chat_id},
        {"$inc": {"message_count": 1}, "$set": {"updated_at": datetime.utcnow()}},
    )

    async def event_stream():
        full_answer = ""
        sources_out: list = []
        token_events = 0
        t0 = time.perf_counter()
        try:
            async for event in run_rag_stream(
                query=question,
                book_ids=chat["book_ids"],
                mode=mode,
                history=history,
            ):
                yield _sse(event)
                if event.get("type") == "status":
                    logger.info("[CHAT] RAG status | chat_id=%s | message=%s", chat_id, event.get("message"))
                if event.get("type") == "token":
                    token_events += 1
                    full_answer += event.get("content", "")
                    if token_events == 1:
                        logger.info("[CHAT] First answer token | chat_id=%s | elapsed=%.2fs", chat_id, time.perf_counter() - t0)
                if event.get("type") == "done":
                    sources_out = event.get("sources") or []
                    final_text = (event.get("full_text") or "").strip() or full_answer
                    if final_text:
                        save_message(
                            chat_id,"assistant",final_text,mode,sources_out,
                        )
                    logger.info("[CHAT] Answer complete | chat_id=%s | mode=%s | answer_chars=%s | token_events=%s | sources=%s | elapsed=%.2fs", chat_id, mode, len(final_text), token_events, len(sources_out), time.perf_counter() - t0)
                    ccol.update_one(
                        {"chat_id": chat_id},
                        {"$inc": {"message_count": 1}, "$set": {"updated_at": datetime.utcnow()}},
                    )
        except asyncio.CancelledError:
            logger.info("[CHAT] Request cancelled by client (Stopped) | chat_id=%s | tokens=%s", chat_id, token_events)
            if full_answer:
                save_message(chat_id, "assistant", full_answer + " (Stopped)", mode, sources_out)
                ccol.update_one(
                    {"chat_id": chat_id},
                    {"$inc": {"message_count": 1}, "$set": {"updated_at": datetime.utcnow()}},
                )
            raise
        except Exception as e:
            logger.exception("[CHAT] Ask stream failed | chat_id=%s", chat_id)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Connection": "keep-alive",
        },
    )
