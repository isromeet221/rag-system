"""
test_logger.py — Comprehensive tests for workers/logger.py

Tests all exports with proper handler capture and predictable function labels.
"""

import asyncio
import logging
import logging.handlers
import re
import time
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import workers.logger as log_mod


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class CaptureHandler(logging.Handler):
    """Handler that captures LogRecords for assertion."""
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]

    @property
    def infos(self):
        return [r for r in self.records if r.levelno == logging.INFO]

    @property
    def warnings(self):
        return [r for r in self.records if r.levelno == logging.WARNING]

    @property
    def errors(self):
        return [r for r in self.records if r.levelno >= logging.ERROR]


def _capture(logger_name: str = "worker.test") -> CaptureHandler:
    """Attach a CaptureHandler to the given logger and return it."""
    raw = logging.getLogger(logger_name)
    handler = CaptureHandler()
    raw.addHandler(handler)
    return handler


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests."""
    log_mod.logger = None
    log_mod._agg.clear()
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("worker."):
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.propagate = True
    yield
    log_mod.logger = None
    log_mod._agg.clear()


def _setup(worker_id: str = "test-abc123") -> log_mod.WorkerAdapter:
    """Set up a worker logger and return the adapter."""
    return log_mod.setup_worker_logger("test", worker_id)


# ── Module-level functions for decorator tests (clean __qualname__) ──────────

def _public_fast():
    return 42

def public_slow():
    time.sleep(0.06)
    return "done"

def _internal_func():
    time.sleep(0.06)
    return 99

def _stage_ok():
    return "ok"

def _stage_fail():
    raise ValueError("boom")

def _stage_returns_dict():
    return {"a": 1, "b": [2, 3]}

async def _async_fast():
    return 42

async def _async_stage_ok():
    await asyncio.sleep(0.01)
    return "async ok"

async def _async_stage_fail():
    raise RuntimeError("async boom")


# ──────────────────────────────────────────────────────────────────────────────
# WorkerAdapter
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkerAdapter:
    def test_prefixes_messages(self):
        raw = logging.getLogger("worker.test")
        adapter = log_mod.WorkerAdapter(raw, {"worker_id": "qd-a1b2c3"})
        msg, kwargs = adapter.process("hello world", {})
        assert msg == "[qd-a1b2c3] hello world"

    def test_format_args_passthrough(self):
        raw = logging.getLogger("worker.test")
        raw.setLevel(logging.INFO)
        cap = _capture()
        log_mod.logger = log_mod.WorkerAdapter(raw, {"worker_id": "w1"})

        log_mod.logger.info("value=%d name=%s", 42, "foo")
        assert len(cap.records) == 1
        assert "[w1] value=42 name=foo" in cap.records[0].getMessage()

    def test_adapter_from_setup_worker_logger(self):
        adapter = _setup("qd-a1b2c3")
        assert isinstance(adapter, log_mod.WorkerAdapter)
        assert adapter.extra["worker_id"] == "qd-a1b2c3"
        assert log_mod.logger is adapter


# ──────────────────────────────────────────────────────────────────────────────
# setup_worker_logger
# ──────────────────────────────────────────────────────────────────────────────

class TestSetupWorkerLogger:
    def test_returns_worker_adapter(self):
        adapter = _setup()
        assert isinstance(adapter, log_mod.WorkerAdapter)

    def test_sets_module_logger(self):
        adapter = _setup()
        assert log_mod.logger is adapter

    def test_creates_file_handler(self):
        _setup()
        raw = logging.getLogger("worker.test")
        file_handlers = [h for h in raw.handlers
                         if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 5 * 1024 * 1024
        assert file_handlers[0].backupCount == 3

    def test_creates_stream_handler(self):
        _setup()
        raw = logging.getLogger("worker.test")
        stream_handlers = [h for h in raw.handlers
                           if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(stream_handlers) >= 1

    def test_custom_max_bytes_and_backup_count(self):
        log_mod.setup_worker_logger("custom", "x-1", max_bytes=1000, backup_count=1)
        raw = logging.getLogger("worker.custom")
        fh = [h for h in raw.handlers
              if isinstance(h, logging.handlers.RotatingFileHandler)][0]
        assert fh.maxBytes == 1000
        assert fh.backupCount == 1
        raw.handlers.clear()

    def test_logger_level_is_info(self):
        _setup()
        raw = logging.getLogger("worker.test")
        assert raw.level == logging.INFO

    def test_logger_does_not_propagate(self):
        _setup()
        raw = logging.getLogger("worker.test")
        assert raw.propagate is False


# ──────────────────────────────────────────────────────────────────────────────
# time_it (sync decorator)
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeIt:
    def test_fast_public_function_aggregated(self):
        _setup()
        cap = _capture()

        decorated = log_mod.time_it(_public_fast)
        result = decorated()
        assert result == 42
        assert len(cap.records) == 0
        # Check aggregation (label is based on __module__ + __qualname__)
        assert any("_public_fast" in k for k in log_mod._agg)

    def test_slow_public_function_logged(self):
        _setup()
        cap = _capture()

        decorated = log_mod.time_it(public_slow)
        result = decorated()
        assert result == "done"
        assert len(cap.records) == 1
        assert "public_slow" in cap.records[0].getMessage()

    def test_internal_function_always_aggregated(self):
        _setup()
        cap = _capture()

        decorated = log_mod.time_it(_internal_func)
        result = decorated()
        assert result == 99
        assert len(cap.records) == 0  # _-prefix never logged
        assert any("_internal_func" in k for k in log_mod._agg)

    def test_preserves_function_metadata(self):
        @log_mod.time_it
        def my_func(a, b):
            """Docstring"""
            return a + b

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Docstring"
        assert my_func(1, 2) == 3

    def test_aggregates_multiple_calls(self):
        _setup()

        @log_mod.time_it
        def multi():
            pass

        for _ in range(5):
            multi()

        key = [k for k in log_mod._agg if "multi" in k][0]
        assert log_mod._agg[key]["count"] == 5


# ──────────────────────────────────────────────────────────────────────────────
# async_time_it
# ──────────────────────────────────────────────────────────────────────────────

class TestAsyncTimeIt:
    def test_fast_async_aggregated(self):
        _setup()
        cap = _capture()

        decorated = log_mod.async_time_it(_async_fast)
        result = asyncio.run(decorated())
        assert result == 42
        assert len(cap.records) == 0
        assert any("_async_fast" in k for k in log_mod._agg)

    def test_preserves_coroutine_metadata(self):
        @log_mod.async_time_it
        async def coro(x):
            """Async docstring"""
            return x * 2

        assert coro.__name__ == "coro"
        assert coro.__doc__ == "Async docstring"


# ──────────────────────────────────────────────────────────────────────────────
# log_process (sync decorator)
# ──────────────────────────────────────────────────────────────────────────────

class TestLogProcess:
    def test_logs_start_and_done(self):
        _setup()
        cap = _capture()

        decorated = log_mod.log_process(_stage_ok)
        result = decorated()
        assert result == "ok"

        assert len(cap.records) >= 2
        msgs = cap.messages
        assert any("START" in m and "_stage_ok" in m for m in msgs)
        assert any("DONE" in m and "_stage_ok" in m and "total:" in m for m in msgs)

    def test_logs_failed_and_re_raises(self):
        _setup()
        cap = _capture()

        decorated = log_mod.log_process(_stage_fail)
        with pytest.raises(ValueError, match="boom"):
            decorated()

        assert len(cap.errors) >= 1
        error_msg = cap.errors[0].getMessage()
        assert "FAILED" in error_msg
        assert "_stage_fail" in error_msg
        assert "boom" in error_msg

    def test_return_value_preserved(self):
        _setup()
        decorated = log_mod.log_process(_stage_returns_dict)
        result = decorated()
        assert result == {"a": 1, "b": [2, 3]}

    def test_preserves_metadata(self):
        @log_mod.log_process
        def stage_func():
            """Stage doc"""
            pass

        assert stage_func.__name__ == "stage_func"
        assert stage_func.__doc__ == "Stage doc"


# ──────────────────────────────────────────────────────────────────────────────
# async_log_process
# ──────────────────────────────────────────────────────────────────────────────

class TestAsyncLogProcess:
    def test_logs_start_and_done_async(self):
        _setup()
        cap = _capture()

        decorated = log_mod.async_log_process(_async_stage_ok)
        result = asyncio.run(decorated())
        assert result == "async ok"

        assert len(cap.records) >= 2
        msgs = cap.messages
        assert any("START" in m for m in msgs)
        assert any("DONE" in m for m in msgs)

    def test_logs_failed_and_re_raises_async(self):
        _setup()
        cap = _capture()

        decorated = log_mod.async_log_process(_async_stage_fail)
        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(decorated())

        assert len(cap.errors) >= 1
        assert "FAILED" in cap.errors[0].getMessage()


# ──────────────────────────────────────────────────────────────────────────────
# TimeItContext
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeItContext:
    def test_logs_when_above_threshold(self):
        _setup()
        cap = _capture()

        with log_mod.TimeItContext("test-block"):
            time.sleep(0.06)

        assert len(cap.records) == 1
        assert "[block] test-block" in cap.records[0].getMessage()

    def test_aggregates_when_below_threshold(self):
        _setup()
        cap = _capture()

        with log_mod.TimeItContext("fast-block"):
            pass

        assert len(cap.records) == 0
        assert any("fast-block" in k for k in log_mod._agg)

    def test_always_log_forces_logging(self):
        _setup()
        cap = _capture()

        with log_mod.TimeItContext("fast-block", always_log=True):
            pass

        assert len(cap.records) == 1
        assert "[block] fast-block" in cap.records[0].getMessage()

    def test_logs_failed_with_warning(self):
        _setup()
        cap = _capture()

        with pytest.raises(ValueError, match="block error"):
            with log_mod.TimeItContext("fail-block"):
                raise ValueError("block error")

        assert len(cap.warnings) >= 1
        wmsg = cap.warnings[0].getMessage()
        assert "FAILED" in wmsg
        assert "fail-block" in wmsg


# ──────────────────────────────────────────────────────────────────────────────
# Session summary
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionSummary:
    def test_empty_agg_produces_no_output(self):
        _setup()
        cap = _capture()

        log_mod._agg.clear()
        log_mod._print_session_summary()
        assert len(cap.records) == 0

    def test_nonempty_agg_produces_table(self):
        _setup()
        cap = _capture()

        log_mod._agg.clear()
        log_mod._record_agg("foo.bar", 0.123)
        log_mod._record_agg("foo.bar", 0.456)
        log_mod._record_agg("baz.qux", 0.050)

        log_mod._print_session_summary()

        assert len(cap.records) >= 4  # header ÷r divider ÷r rows ÷r footer
        msgs = cap.messages
        assert any("SESSION SUMMARY" in m for m in msgs)
        assert any("foo.bar" in m for m in msgs)
        assert any("baz.qux" in m for m in msgs)


# ──────────────────────────────────────────────────────────────────────────────
# worker_log_process — backward compat alias
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkerLogProcessBackwardCompat:
    def test_acts_as_log_process(self):
        _setup()
        cap = _capture()

        @log_mod.worker_log_process("old-worker-id")
        def old_style():
            return "legacy"

        result = old_style()
        assert result == "legacy"
        assert len(cap.records) >= 2
        msgs = cap.messages
        assert any("START" in m for m in msgs)
        assert any("DONE" in m for m in msgs)

    def test_preserves_function_name(self):
        @log_mod.worker_log_process("any-id")
        def named():
            pass

        assert named.__name__ == "named"


# ──────────────────────────────────────────────────────────────────────────────
# Integration — WorkerAdapter prefix appears in decorator output
# ──────────────────────────────────────────────────────────────────────────────

class TestAdapterPrefixInDecorators:
    def test_log_process_includes_worker_id_prefix(self):
        _setup("worker-xyz")
        cap = _capture()

        @log_mod.log_process
        def my_job():
            return "done"

        my_job()
        assert "[worker-xyz]" in cap.records[0].getMessage()

    def test_time_it_includes_worker_id_for_slow_calls(self):
        _setup("slow-worker")
        cap = _capture()

        @log_mod.time_it
        def slow_job():
            time.sleep(0.06)

        slow_job()
        assert "[slow-worker]" in cap.records[0].getMessage()

    def test_time_it_context_includes_worker_id(self):
        _setup("ctx-worker")
        cap = _capture()

        with log_mod.TimeItContext("block-x", always_log=True):
            time.sleep(0.01)

        assert "[ctx-worker]" in cap.records[0].getMessage()

    def test_multiple_workers_get_distinct_prefixes(self):
        _setup("alpha")
        cap1 = _capture()

        @log_mod.log_process
        def job_a():
            pass

        job_a()
        assert "[alpha]" in cap1.records[0].getMessage()

        # Reset for a second worker
        log_mod.logger = None
        log_mod._agg.clear()
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("worker."):
                logging.getLogger(name).handlers.clear()
                logging.getLogger(name).propagate = True

        log_mod.setup_worker_logger("test2", "beta")
        cap2 = _capture("worker.test2")

        @log_mod.log_process
        def job_b():
            pass

        job_b()
        assert "[beta]" in cap2.records[0].getMessage()


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_time_it_with_args_and_kwargs(self):
        _setup()

        @log_mod.time_it
        def with_args(a, b, *, c=3):
            return a + b + c

        assert with_args(1, 2, c=10) == 13

    def test_log_process_nested_exception_type(self):
        _setup()
        cap = _capture()

        @log_mod.log_process
        def raises_type_error():
            raise TypeError("type issue")

        with pytest.raises(TypeError, match="type issue"):
            raises_type_error()

        assert len(cap.errors) == 1
        assert "FAILED" in cap.errors[0].getMessage()

    def test_time_it_context_exception_propagates(self):
        _setup()

        with pytest.raises(ZeroDivisionError):
            with log_mod.TimeItContext("divide"):
                1 / 0
