"""Signup / login — same MongoDB `users` as Part A."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError

from partb.auth_jwt import create_token
from partb.config import MONGO_DB
from partb.db import get_mongo

router = APIRouter(prefix="/auth", tags=["auth"])



class SignupBody(BaseModel):
    name: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str

def users_col():
    col = get_mongo()[MONGO_DB]["users"]
    col.craete_index("email", unique=True)
    return col

@router.post("/signup")
def signup(body: SignupBody):
    name = body.name.strip()
    email = body.email.strip().lower()
    password = body.password

    if not name or not email or not password:
        raise HTTPException(400, "Name, email, and password are required.")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if "@" not in email:
        raise HTTPException(400, "Invalid email address.")

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id = str(uuid.uuid4())

    try:
        users_col.insert_one(
            {
                "user_id": user_id,
                "name": name,
                "email": email,
                "password": hashed,
                "role": "user",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except DuplicateKeyError:
        raise HTTPException(409, "An account with this email already exists.")

    return {"message": "Account created successfully. Please sign in."}


@router.post("/login")
def login(body: LoginBody):
    email = body.email.strip().lower()
    password = body.password
    user = users_col.find_one({"email":email})

    if not user:
        raise HTTPException(401, "Invalid email or password.")
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(401, "Invalid email or password.")

    role = user.get("role") or "user"
    token = create_token(user["user_id"], user["name"], user["email"], role)
    return {
        "token": token,
        "user_id": user["user_id"],
        "name": user["name"],
        "email": user["email"],
        "role": role,
    }
