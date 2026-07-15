"""
test_base_worker.py — Comprehensive tests for workers/base_worker.py

Tests:
  • configure_tcp_keepalive — idempotency, socket options applied
  • safe_unlink / safe_write_text / safe_read_text / safe_write_bytes — success + retry
  • create_session — session headers, keepalive triggered
  • ResultCache — store, clear, replay (success + failure paths)
  • submit_with_retry — success, non-200 retry, connection error retry, backoff
"""

import json
import logging
import socket
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest
import requests
from urllib3.connection import HTTPConnection

import workers.base_worker as bw


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_keepalive():
    """Ensure keepalive state is fresh between tests."""
    bw._keepalive_applied = False
    # Save original socket options
    original = list(HTTPConnection.default_socket_options)
    yield
    bw._keepalive_applied = False
    HTTPConnection.default_socket_options = original


# ──────────────────────────────────────────────────────────────────────────────
# configure_tcp_keepalive
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigureTcpKeepalive:
    def test_applies_socket_options(self):
        before = len(HTTPConnection.default_socket_options)
        bw.configure_tcp_keepalive()
        after = len(HTTPConnection.default_socket_options)
        assert after > before
        # Verify keepalive options are present (each is a 3-tuple: level, optname, value)
        keepalive_opts = [t for t in HTTPConnection.default_socket_options
                          if t[0] == socket.SOL_SOCKET and t[1] == socket.SO_KEEPALIVE]
        assert len(keepalive_opts) >= 1
        assert keepalive_opts[-1][2] == 1

    def test_is_idempotent(self):
        bw.configure_tcp_keepalive()
        count_after_first = len(HTTPConnection.default_socket_options)
        bw.configure_tcp_keepalive()
        count_after_second = len(HTTPConnection.default_socket_options)
        assert count_after_first == count_after_second


# ──────────────────────────────────────────────────────────────────────────────
# Safe file operations
# ──────────────────────────────────────────────────────────────────────────────

class TestSafeUnlink:
    def test_unlinks_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        bw.safe_unlink(f)
        assert not f.exists()

    def test_succeeds_on_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        bw.safe_unlink(f)  # should not raise

    def test_retries_on_permission_error(self, tmp_path):
        f = tmp_path / "locked.txt"
        f.write_text("data")

        call_count = [0]
        original_unlink = Path.unlink

        def mock_unlink(self_obj, missing_ok=False):
            call_count[0] += 1
            if call_count[0] < 3:
                raise PermissionError("locked")
            original_unlink(self_obj, missing_ok=missing_ok)

        with mock.patch.object(Path, "unlink", mock_unlink):
            bw.safe_unlink(f)

        assert call_count[0] == 3


class TestSafeWriteText:
    def test_writes_file(self, tmp_path):
        f = tmp_path / "write.txt"
        bw.safe_write_text(f, "hello world")
        assert f.read_text() == "hello world"

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "overwrite.txt"
        f.write_text("old")
        bw.safe_write_text(f, "new")
        assert f.read_text() == "new"


class TestSafeReadText:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "read.txt"
        f.write_text("content")
        assert bw.safe_read_text(f) == "content"

    def test_raises_on_missing(self, tmp_path):
        f = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            bw.safe_read_text(f)


class TestSafeWriteBytes:
    def test_writes_bytes(self, tmp_path):
        f = tmp_path / "binary.bin"
        bw.safe_write_bytes(f, b"\x00\x01\x02")
        assert f.read_bytes() == b"\x00\x01\x02"


# ──────────────────────────────────────────────────────────────────────────────
# create_session
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateSession:
    def test_returns_session_with_ngrok_header(self):
        session = bw.create_session()
        assert isinstance(session, requests.Session)
        assert session.headers.get("ngrok-skip-browser-warning") == "true"

    def test_triggers_keepalive(self):
        bw._keepalive_applied = False
        bw.create_session()
        assert bw._keepalive_applied is True


# ──────────────────────────────────────────────────────────────────────────────
# ResultCache
# ──────────────────────────────────────────────────────────────────────────────

class TestResultCache:
    @pytest.fixture
    def cache(self, tmp_path):
        """Create a ResultCache pointing at a temp directory."""
        base = tmp_path / "worker_result_cache"
        return bw.ResultCache("test-worker", base_dir=base)

    def test_store_writes_json_file(self, cache):
        cache.store("job-001", {"status": "ok", "chunks": 5})
        path = cache.cache_dir / "job-001.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"status": "ok", "chunks": 5}

    def test_store_handles_unicode(self, cache):
        cache.store("job-002", {"text": "héllo wörld 🚀"})
        path = cache.cache_dir / "job-002.json"
        data = json.loads(path.read_text())
        assert data["text"] == "héllo wörld 🚀"

    def test_clear_removes_file(self, cache):
        cache.store("job-003", {"x": 1})
        cache.clear("job-003")
        assert not (cache.cache_dir / "job-003.json").exists()

    def test_clear_missing_job_does_not_raise(self, cache):
        cache.clear("nonexistent")  # should not raise

    def test_replay_with_no_cached_files_produces_no_requests(self, cache):
        mock_session = mock.MagicMock(spec=requests.Session)
        cache.replay(mock_session, "http://example.com/submit")
        mock_session.post.assert_not_called()

    def test_replay_submits_and_clears_on_success(self, cache):
        cache.store("job-010", {"result": "ok"})
        cache.store("job-020", {"result": "also ok"})

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_session = mock.MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_resp

        cache.replay(mock_session, "http://server/submit")

        assert mock_session.post.call_count == 2
        assert not (cache.cache_dir / "job-010.json").exists()
        assert not (cache.cache_dir / "job-020.json").exists()

    def test_replay_keeps_file_on_failure(self, cache):
        cache.store("job-fail", {"result": "nope"})

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 500
        mock_session = mock.MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_resp

        cache.replay(mock_session, "http://server/submit")

        # File should still exist since server returned non-200
        assert (cache.cache_dir / "job-fail.json").exists()

    def test_replay_handles_corrupt_json(self, cache):
        # Write a corrupt file manually
        corrupt = cache.cache_dir / "corrupt.json"
        corrupt.write_text("not valid json{{{")

        mock_session = mock.MagicMock(spec=requests.Session)
        # Should not raise
        cache.replay(mock_session, "http://server/submit")
        mock_session.post.assert_not_called()

    def test_cache_dir_created_automatically(self, tmp_path):
        base = tmp_path / "new_cache"
        cache = bw.ResultCache("auto-worker", base_dir=base)
        assert cache.cache_dir.exists()
        assert cache.cache_dir.is_dir()


# ──────────────────────────────────────────────────────────────────────────────
# submit_with_retry
# ──────────────────────────────────────────────────────────────────────────────

class TestSubmitWithRetry:
    def test_returns_on_200(self):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_session = mock.MagicMock(spec=requests.Session)
        mock_session.post.return_value = mock_resp

        bw.submit_with_retry(mock_session, "http://ok/submit", {"x": 1},
                             initial_delay=0.001, max_delay=0.01)

        mock_session.post.assert_called_once()

    def test_retries_on_non_200(self):
        fail_resp = mock.MagicMock()
        fail_resp.status_code = 503
        ok_resp = mock.MagicMock()
        ok_resp.status_code = 200

        mock_session = mock.MagicMock(spec=requests.Session)
        mock_session.post.side_effect = [fail_resp, fail_resp, ok_resp]

        bw.submit_with_retry(mock_session, "http://retry/submit", {"x": 1},
                             initial_delay=0.001, max_delay=0.01)

        assert mock_session.post.call_count == 3

    def test_retries_on_connection_error(self):
        ok_resp = mock.MagicMock()
        ok_resp.status_code = 200

        mock_session = mock.MagicMock(spec=requests.Session)
        mock_session.post.side_effect = [
            requests.exceptions.ConnectionError("no route"),
            ok_resp,
        ]

        bw.submit_with_retry(mock_session, "http://retry/submit", {"x": 1},
                             initial_delay=0.001, max_delay=0.01)

        assert mock_session.post.call_count == 2

    def test_backoff_increases_delay(self):
        """Verify that the delay doubles each retry, up to max_delay."""
        import itertools
        mock_session = mock.MagicMock(spec=requests.Session)
        # Infinite chain of ConnectionError so we don't hit StopIteration
        mock_session.post.side_effect = itertools.cycle([
            requests.exceptions.ConnectionError("fail"),
        ])

        with mock.patch("time.sleep") as mock_sleep:
            # Break the infinite loop after 4 sleeps
            mock_sleep.side_effect = [None, None, None, RuntimeError("stop loop")]
            try:
                bw.submit_with_retry(mock_session, "http://x/submit", {"x": 1},
                                     initial_delay=1.0, max_delay=10.0)
            except RuntimeError:
                pass

            # Delays should be: 1.0, 2.0, 4.0, 8.0
            sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
            assert sleep_calls[0] == 1.0
            assert sleep_calls[1] == 2.0
            assert sleep_calls[2] == 4.0
