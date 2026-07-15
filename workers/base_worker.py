"""
base_worker.py — Workers
========================
Shared infrastructure for all worker types (text, qdrant, neo4j).

Provides:
  • configure_tcp_keepalive() — prevents idle connections from being dropped
  • ResultCache — crash-safe result cache with replay-on-startup
  • submit_with_retry() — exponential backoff for server submissions
  • create_session() — persistent requests.Session with keepalive headers
  • Safe file operations — retry-on-lock for Windows / network filesystems
"""

import json
import logging
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from urllib3.connection import HTTPConnection

_log = logging.getLogger(__name__)

# ─── TCP keepalive ────────────────────────────────────────────────────────────
#
# Windows uses WSAIoctl (SIO_KEEPALIVE_VALS) to configure keepalive timing
# because setsockopt only exposes SO_KEEPALIVE (on/off).  The default idle
# time is 2 hours — useless for workers behind NAT/proxies that drop idle
# connections in 60-300 s.  We monkey-patch HTTPConnection._new_conn so
# every socket gets Windows keepalive applied at creation time.

_keepalive_applied = False

# (onoff, idle_ms, interval_ms) — Windows socket.ioctl expects a 3-int seq.
_keepalive_vals = (1, 60_000, 10_000)  # on, 60 s idle, 10 s interval

_real_new_conn = HTTPConnection._new_conn


def _new_conn(self: HTTPConnection):
    """Create socket, set SO_KEEPALIVE, then apply Windows keepalive timing."""
    conn = _real_new_conn(self)
    try:
        conn.ioctl(socket.SIO_KEEPALIVE_VALS, _keepalive_vals)
    except OSError:
        _log.warning("WSAIoctl failed; keepalive timing may use system defaults")
    return conn


def configure_tcp_keepalive():
    """Apply TCP keepalive to all urllib3 HTTPConnections.

    Idempotent — safe to call multiple times or from multiple worker modules.
    Prevents idle connections from being silently dropped by load balancers,
    proxies, and NAT gateways.

    - Enables SO_KEEPALIVE on every socket.
    - On Windows: additionally configures timing via WSAIoctl
      (first probe after 60 s idle, then every 10 s).
    """
    global _keepalive_applied
    if _keepalive_applied:
        return

    HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    HTTPConnection._new_conn = _new_conn
    _keepalive_applied = True


# ─── Safe file operations (retry on transient locks) ─────────────────────────

def safe_unlink(path: Path, attempts: int = 8, base_delay: float = 0.15) -> None:
    """Delete a file, retrying on PermissionError with exponential backoff."""
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as e:
            if i == attempts - 1:
                _log.warning(
                    "Giving up deleting %s after %d attempts: %s",
                    path, attempts, e,
                )
                return
            time.sleep(base_delay * (2 ** i))


def safe_write_bytes(
    path: Path, data: bytes, attempts: int = 5, base_delay: float = 0.15
) -> None:
    """Write bytes to a file, retrying on PermissionError."""
    for i in range(attempts):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


def safe_write_text(
    path: Path, text: str, attempts: int = 5, base_delay: float = 0.15
) -> None:
    """Write text to a file, retrying on PermissionError."""
    for i in range(attempts):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


def safe_read_text(
    path: Path, attempts: int = 5, base_delay: float = 0.15
) -> str:
    """Read text from a file, retrying on PermissionError."""
    for i in range(attempts):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


# ─── Session ──────────────────────────────────────────────────────────────────

def create_session() -> requests.Session:
    """Create a persistent requests.Session with ngrok header and TCP keepalive."""
    configure_tcp_keepalive()
    session = requests.Session()
    session.headers.update({"ngrok-skip-browser-warning": "true"})
    return session


# ─── ResultCache ──────────────────────────────────────────────────────────────

class ResultCache:
    """Crash-safe local result cache with startup replay capability.

    Stores job results on local disk so they survive worker restarts, network
    blips, and server downtime.  On next startup, cached results are replayed
    to the server before polling for new jobs.

    Usage
    -----
        cache = ResultCache("qdrant", base_dir=RESULT_CACHE_DIR)
        cache.store(job_id, payload)
        cache.submit_with_retry(session, f"{SERVER_URL}/submit_qdrant_result", payload)
        cache.clear(job_id)

        # On startup:
        cache.replay(session, f"{SERVER_URL}/submit_qdrant_result")
    """

    def __init__(self, worker_type: str, base_dir: Path | None = None):
        if base_dir is None:
            base_dir = Path(tempfile.gettempdir()) / "worker_result_cache"
        self.cache_dir = base_dir / worker_type
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.cache_dir / f"{job_id}.json"

    def store(self, job_id: str, payload: dict) -> None:
        """Persist a result payload to the local cache."""
        safe_write_text(self._path(job_id), json.dumps(payload, ensure_ascii=False))

    def clear(self, job_id: str) -> None:
        """Remove a successfully-submitted result from the cache."""
        safe_unlink(self._path(job_id))

    def replay(self, session: requests.Session, submit_url: str) -> None:
        """Replay all cached results to the server, clearing on success."""
        cached = sorted(self.cache_dir.glob("*.json"))
        if not cached:
            return
        _log.info("Replaying %d cached result(s)...", len(cached))
        for cache_file in cached:
            try:
                payload = json.loads(safe_read_text(cache_file))
                r = session.post(submit_url, json=payload, timeout=30)
                if r.status_code == 200:
                    safe_unlink(cache_file)
                    _log.info("Replayed and cleared: %s", cache_file.name)
                else:
                    _log.warning(
                        "Replay failed HTTP %d: %s", r.status_code, cache_file.name
                    )
            except Exception as e:
                _log.warning("Replay error for %s: %s", cache_file.name, e)


# ─── submit_with_retry ────────────────────────────────────────────────────────

def submit_with_retry(
    session: requests.Session,
    endpoint: str,
    payload: dict,
    *,
    initial_delay: float = 5.0,
    max_delay: float = 60.0,
    timeout: float = 30.0,
) -> None:
    """POST a payload to the server with exponential backoff on failure.

    Blocks until successful.  Handles both ConnectionError and non-200 responses.
    """
    delay = initial_delay
    while True:
        try:
            r = session.post(endpoint, json=payload, timeout=timeout)
            if r.status_code == 200:
                return
            _log.warning(
                    "Submit returned HTTP %d, retrying in %.1fs...",
                    r.status_code, delay,
                )
        except requests.exceptions.ConnectionError:
            _log.warning(
                    "Connection error on submit, retrying in %.1fs...", delay,
                )
        time.sleep(delay)
        delay = min(delay * 2, max_delay)
