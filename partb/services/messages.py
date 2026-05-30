"""MongoDB messages + prior history for prompts."""
from __future__ import annotations

import uuid
from datetime import datetime

from partb.config import MONGO_DB
from partb.logger import time_it, async_time_it

from partb.db import get_mongo


@time_it
def messages_col():
    return get_mongo()[MONGO_DB]["messages"]


@time_it
def save_message(
    chat_id: str,
    role: str,
    content: str,
    mode: str,
    sources: list,
) -> dict:
    doc = {
        "message_id": str(uuid.uuid4()),
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "mode": mode,
        "sources": sources,
        "created_at": datetime.utcnow(),
    }
    messages_col().insert_one(doc)
    return doc


@time_it
def get_prior_messages(chat_id: str, n_pairs: int) -> list[dict]:
    """
    Last n_pairs × 2 messages before the current send (oldest first).
    Call **before** inserting the new user message.
    """
    col = messages_col()
    msgs = list(
        col.find({"chat_id": chat_id}, {"role": 1, "content": 1, "_id": 0})
        .sort("created_at", -1)
        .limit(n_pairs * 2)
    )
    msgs.reverse()
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


@time_it
def get_all_messages(chat_id: str) -> list[dict]:
   
    col = messages_col()
    msgs = list(
        col.find({"chat_id": chat_id}, { "_id": 0})
        .sort("created_at", 1)
 
    )
    for m in msgs:
        if isinstance(m.get("created_at"), datetime):
            m["created_at"] =  m["created_at"].isoformat()
    return msgs