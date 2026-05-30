"""JWT helpers — include role; secret must match Part A (`RAG_JWT_SECRET`)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from fastapi import HTTPException, Request

from partb.logger import time_it, async_time_it

from partb.config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, JWT_SECRET


@time_it
def create_token(user_id: str, name: str, email: str, role: str = "user") -> str:
    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "role": role or "user",
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@time_it
def decode_token(token: str) -> dict:
    try:
        return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired. Please login again.")
    except pyjwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token.")




@time_it
def verify_token(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = auth[7:]
    return decode_token(token)