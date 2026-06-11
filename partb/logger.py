"""
logger.py — Part B
==================
Provides timing decorators and a structured logger for the RAG query pipeline.

Exports
-------
logger           : logging.Logger  — use for manual logger.info() calls
time_it          : decorator — sync functions; aggregates `_`-prefix & fast calls
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
  time, regardless of function name or duration.  Use on major retrieval stages.
• At program exit, a SESSION SUMMARY table is printed for all aggregated calls.
"""

import atexit
import functools
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ─── Log directory & file ────────────────────────────────────────────────────
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = _DIR / ".logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "rag.log"

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("RAG.partb")
logger.setLevel(logging.INFO)
logger.propagate = False  # prevent duplicate output through root logger

if not logger.handlers:
    _FMT = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a")
    _fh.setFormatter(_FMT)
    logger.addHandler(_fh)

    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_FMT)
    logger.addHandler(_sh)

# ─── Aggregated stats for internal / fast functions ──────────────────────────
_agg: dict[str, dict] = {}


def _record_agg(name: str, elapsed: float) -> None:
    s = _agg.setdefault(name, {"count": 0, "total": 0.0})
    s["count"] += 1
    s["total"] += elapsed


@atexit.register
def _print_session_summary() -> None:
    """Prints a summary table of all aggregated (internal/fast) function calls."""
    if not _agg:
        return
    logger.info("=" * 72)
    logger.info("SESSION SUMMARY — aggregated (internal / fast functions)")
    logger.info("%-58s  %6s  %9s  %9s", "Function", "Calls", "Total(s)", "Avg(s)")
    logger.info("-" * 72)
    for name, s in sorted(_agg.items(), key=lambda kv: -kv[1]["total"]):
        avg = s["total"] / max(s["count"], 1)
        logger.info(
            "  %-56s  %6d  %9.3f  %9.4f",
            name,
            s["count"],
            s["total"],
            avg,
        )
    logger.info("=" * 72)


# ─── Threshold for live-logging public functions ──────────────────────────────
# Public functions that complete faster than this are silently aggregated.
_THRESHOLD: float = 0.05  # seconds


def _label(func: Callable) -> str:
    """Return a compact 'module.qualname' label, stripping the top-level package."""
    mod = func.__module__ or ""
    for prefix in ("parta.", "partb."):
        if mod.startswith(prefix):
            mod = mod[len(prefix) :]
            break
    return f"{mod}.{func.__qualname__}" if mod else func.__qualname__


# ─── time_it — sync ──────────────────────────────────────────────────────────
def time_it(func: Callable) -> Callable:
    """
    Timing decorator for synchronous functions.

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
        else:
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
        else:
            logger.info("%-55s  %.3fs", _name, elapsed)
        return result

    return wrapper


# ─── log_process — always logs START + DONE/FAILED ───────────────────────────
def log_process(func: Callable) -> Callable:
    """
    Decorator for major retrieval / pipeline stages.

    Always emits:
      • INFO  ┌─ START  <name>           when the function is called
      • INFO  └─ DONE   <name>  [total]  on success
      • ERROR └─ FAILED <name>  [total]  on exception (then re-raises)

    Apply this instead of @time_it on top-level stage functions where you
    need real-time visibility into start and total elapsed time, regardless
    of function name or expected duration.
    """
    _name = _label(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info("|- START  %s", _name)
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info("|- DONE   %-50s  [total: %.3fs]", _name, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("|- FAILED %-50s  [%.3fs]  %s", _name, elapsed, exc)
            raise

    return wrapper


# ─── async_log_process ────────────────────────────────────────────────────────
def async_log_process(func: Callable) -> Callable:
    """Async version of log_process."""
    _name = _label(func)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        logger.info("|- START  %s", _name)
        t0 = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info("|- DONE   %-50s  [total: %.3fs]", _name, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("|- FAILED %-50s  [%.3fs]  %s", _name, elapsed, exc)
            raise

    return wrapper


# ─── TimeItContext ────────────────────────────────────────────────────────────
class TimeItContext:
    """
    Context manager for timing named code blocks inside a function.

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
            logger.warning("  [block] %-48s  FAILED  %.3fs", self.block_name, elapsed)
        elif self.always_log or elapsed >= _THRESHOLD:
            logger.info("  [block] %-48s  %.3fs", self.block_name, elapsed)
        else:
            _record_agg(f"[block] {self.block_name}", elapsed)
