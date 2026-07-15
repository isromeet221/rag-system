"""
logger.py — Workers
===================
Provides timing decorators, log rotation, auto-prefixed worker IDs,
and a structured logger for the distributed worker system.

Exports
-------
logger           : WorkerAdapter — auto-prefixes with [worker_id]; use for manual logging
setup_worker_logger : function — configures rotating file handler + console; returns logger
time_it          : decorator — sync functions; aggregates _-prefix & fast calls
async_time_it    : decorator — async functions; same rules as time_it
log_process      : decorator — sync functions; ALWAYS logs START + DONE + total time
async_log_process: decorator — async version of log_process
TimeItContext    : context manager — times named code blocks

Rules
-----
• Functions whose name starts with '_' → silently aggregated (never logged live).
• Public functions with elapsed < _THRESHOLD (0.05 s) → aggregated.
• Public functions with elapsed >= _THRESHOLD → logged live as INFO.
• @log_process / @async_log_process → always logs START + DONE/FAILED with total
  time, regardless of function name or duration.  Use on major job stages.
• At program exit, a SESSION SUMMARY table is printed for all aggregated calls.
• Log files rotate at 5 MB (keeps 3 backups).
• Worker ID is auto-prefixed by the WorkerAdapter — no need to manually insert it.
"""

import atexit
import functools
import logging
import logging.handlers
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ─── Threshold for live-logging public functions ──────────────────────────────
# Public functions that complete faster than this are silently aggregated.
_THRESHOLD: float = 0.05  # seconds

# ─── Session-level aggregated stats ───────────────────────────────────────────
_agg: dict[str, dict] = {}


def _record_agg(name: str, elapsed: float) -> None:
    s = _agg.setdefault(name, {"count": 0, "total": 0.0})
    s["count"] += 1
    s["total"] += elapsed


# ─── WorkerAdapter ────────────────────────────────────────────────────────────

class WorkerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that automatically prefixes every message with [worker_id].

    Usage:
        adapter = WorkerAdapter(raw_logger, {"worker_id": "qdrant-a1b2c3"})
        adapter.info("Connected to server")
        # → [qdrant-a1b2c3] Connected to server
    """

    def process(self, msg, kwargs):
        return f"[{self.extra['worker_id']}] {msg}", kwargs


# ─── Module-level logger (set by setup_worker_logger) ─────────────────────────
logger: WorkerAdapter | None = None


@atexit.register
def _print_session_summary() -> None:
    """Prints a summary table of all aggregated (internal/fast) function calls."""
    if not _agg:
        return
    out = logger if logger is not None else logging.getLogger("worker")
    out.info("=" * 72)
    out.info("SESSION SUMMARY — aggregated (internal / fast functions)")
    out.info("%-58s  %6s  %9s  %9s", "Function", "Calls", "Total(s)", "Avg(s)")
    out.info("-" * 72)
    for name, s in sorted(_agg.items(), key=lambda kv: -kv[1]["total"]):
        avg = s["total"] / max(s["count"], 1)
        out.info(
            "  %-56s  %6d  %9.3f  %9.4f",
            name,
            s["count"],
            s["total"],
            avg,
        )
    out.info("=" * 72)


# ─── Label helper ─────────────────────────────────────────────────────────────

def _label(func: Callable) -> str:
    """Return a compact 'module.qualname' label."""
    mod = func.__module__ or ""
    for prefix in ("workers.",):
        if mod.startswith(prefix):
            mod = mod[len(prefix):]
            break
    return f"{mod}.{func.__qualname__}" if mod else func.__qualname__


# ─── setup_worker_logger ─────────────────────────────────────────────────────

def setup_worker_logger(
    worker_type: str,
    worker_id: str,
    *,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> WorkerAdapter:
    """Configure logging for a worker process.

    Creates:
      • A RotatingFileHandler → logs/{worker_type}/{worker_id}.log (rotates at max_bytes)
      • A StreamHandler → stdout (for Modal / Docker log capture)

    Returns a WorkerAdapter that auto-prefixes messages with [worker_id].
    Also sets the module-level ``logger`` variable.

    Idempotent — second call with same worker_type is a no-op.
    """
    global logger

    try:
        ip_addr = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip_addr = "127.0.0.1"

    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / "logs" / worker_type
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{worker_id}.log"

    raw = logging.getLogger(f"worker.{worker_type}")
    raw.setLevel(logging.INFO)
    raw.propagate = False

    if not raw.handlers:
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-5s  [IP: %(ip)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            defaults={"ip": ip_addr},
        )

        # Rotating file handler — prevents unbounded log growth
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        raw.addHandler(fh)

        # Console handler for Docker / Modal / direct execution
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        raw.addHandler(sh)

    adapter = WorkerAdapter(raw, {"worker_id": worker_id})
    logger = adapter
    return adapter


# ─── time_it — sync ──────────────────────────────────────────────────────────

def time_it(func: Callable) -> Callable:
    """Timing decorator for synchronous functions.

    • '_'-prefixed functions  → always aggregated, never logged live.
    • Public, elapsed >= 0.05 s → logged as INFO.
    • Public, elapsed <  0.05 s → aggregated.
    """
    _internal = func.__name__.startswith("_")
    _name = _label(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if _internal or elapsed < _THRESHOLD:
            _record_agg(_name, elapsed)
        elif logger is not None:
            logger.info("%-55s  %.3fs", _name, elapsed)
        return result

    return wrapper


# ─── async_time_it — async ───────────────────────────────────────────────────

def async_time_it(func: Callable) -> Callable:
    """Async version of time_it.  Same aggregation rules."""
    _internal = func.__name__.startswith("_")
    _name = _label(func)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        t0 = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if _internal or elapsed < _THRESHOLD:
            _record_agg(_name, elapsed)
        elif logger is not None:
            logger.info("%-55s  %.3fs", _name, elapsed)
        return result

    return wrapper


# ─── log_process — always logs START + DONE/FAILED ───────────────────────────

def log_process(func: Callable) -> Callable:
    """Decorator for major job-processing stages.

    Always emits:
      • INFO  ┌─ START  <name>           when the function is called
      • INFO  └─ DONE   <name>  [total]  on success
      • ERROR └─ FAILED <name>  [total]  on exception (then re-raises)

    Worker ID prefix is applied automatically by the WorkerAdapter.
    """
    _name = _label(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if logger is not None:
            logger.info("┌─ START  %s", _name)
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            if logger is not None:
                logger.info("└─ DONE   %-50s  [total: %.3fs]", _name, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            if logger is not None:
                logger.error("└─ FAILED %-50s  [%.3fs]  %s", _name, elapsed, exc)
            raise

    return wrapper


# ─── async_log_process ────────────────────────────────────────────────────────

def async_log_process(func: Callable) -> Callable:
    """Async version of log_process."""
    _name = _label(func)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        if logger is not None:
            logger.info("┌─ START  %s", _name)
        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            if logger is not None:
                logger.info("└─ DONE   %-50s  [total: %.3fs]", _name, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            if logger is not None:
                logger.error("└─ FAILED %-50s  [%.3fs]  %s", _name, elapsed, exc)
            raise

    return wrapper


# ─── TimeItContext ────────────────────────────────────────────────────────────

class TimeItContext:
    """Context manager for timing named code blocks inside a function.

    Usage
    -----
        with TimeItContext("Embedding batch 3/10"):
            ...

    Set always_log=True to force logging even for sub-threshold durations.
    """

    def __init__(self, block_name: str, always_log: bool = False):
        self.block_name = block_name
        self.always_log = always_log
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        if exc_type:
            if logger is not None:
                logger.warning("  [block] %-48s  FAILED  %.3fs", self.block_name, elapsed)
        elif self.always_log or elapsed >= _THRESHOLD:
            if logger is not None:
                logger.info("  [block] %-48s  %.3fs", self.block_name, elapsed)
        else:
            _record_agg(f"[block] {self.block_name}", elapsed)


# ─── Legacy compatibility — worker_log_process ────────────────────────────────
# Existing decorator kept for backward compat; new code should use log_process.

def worker_log_process(worker_id: str) -> Callable:
    """Legacy decorator.  Prefer @log_process — worker_id is auto-prefixed."""
    return log_process
