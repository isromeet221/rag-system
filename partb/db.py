"""MongoDB singleton."""
from __future__ import annotations

from pymongo import MongoClient

from partb.logger import time_it, async_time_it

from partb.config import MONGO_DB, MONGO_URI

_client: MongoClient | None = None


@time_it
def get_mongo() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client


@time_it
def db():
    return get_mongo()[MONGO_DB]