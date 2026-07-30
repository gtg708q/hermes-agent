"""Cross-process request-dump retention regression tests."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_dump_worker(logs_dir_raw: str, index: int) -> None:
    from agent.agent_runtime_helpers import _write_bounded_request_dump

    logs_dir = Path(logs_dir_raw)
    payload = {
        "timestamp": str(index),
        "fingerprint": f"fp-{index}",
        "request": {"body": {"index": index}},
    }
    _write_bounded_request_dump(
        logs_dir=logs_dir,
        dump_file=logs_dir / f"request_dump_worker_{index}.json",
        payload=payload,
        max_files=2,
        max_bytes=4096,
        ttl_days=7,
        dedup_seconds=0,
    )


def _hold_request_dump_lock(logs_dir_raw: str, ready, release) -> None:
    from agent.agent_runtime_helpers import _request_dump_lock

    with _request_dump_lock(Path(logs_dir_raw), timeout_seconds=5) as acquired:
        if not acquired:
            raise RuntimeError("worker could not acquire request-dump lock")
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("timed out waiting to release request-dump lock")


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock contention test")
def test_request_dump_skips_quickly_on_process_lock_contention(tmp_path):
    from agent import agent_runtime_helpers as helpers

    _write_bounded_request_dump = helpers._write_bounded_request_dump

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    process = ctx.Process(
        target=_hold_request_dump_lock,
        args=(str(tmp_path), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10)
        dump_file = tmp_path / "request_dump_contended.json"

        started = time.monotonic()
        wrote = _write_bounded_request_dump(
            logs_dir=tmp_path,
            dump_file=dump_file,
            payload={"fingerprint": "contended"},
            max_files=2,
            max_bytes=4096,
            ttl_days=7,
            dedup_seconds=0,
        )
        elapsed = time.monotonic() - started

        assert wrote is False
        assert elapsed < 1.0
        assert not dump_file.exists()
        assert str(tmp_path.resolve()) not in helpers._REQUEST_DUMP_SCAN_STATES

        release.set()
        process.join(timeout=10)
        assert process.exitcode == 0

        assert _write_bounded_request_dump(
            logs_dir=tmp_path,
            dump_file=dump_file,
            payload={"fingerprint": "released"},
            max_files=2,
            max_bytes=4096,
            ttl_days=7,
            dedup_seconds=0,
        ) is True
        assert dump_file.exists()
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


def test_request_dump_windows_lock_timeout_does_not_unlock_unacquired_lock(
    monkeypatch, tmp_path
):
    from agent import agent_runtime_helpers as helpers

    calls = []
    opened_handles = []
    contended = True

    original_open = type(tmp_path).open

    class _TrackedHandle:
        def __init__(self, inner):
            self.inner = inner
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            self.closed = True
            self.inner.close()

    def tracked_open(path, *args, **kwargs):
        handle = _TrackedHandle(original_open(path, *args, **kwargs))
        opened_handles.append(handle)
        return handle

    def locking(_fd, mode, _size):
        calls.append(mode)
        if mode == fake_msvcrt.LK_NBLCK and contended:
            raise OSError("lock is held")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(type(tmp_path), "open", tracked_open)
    monkeypatch.setattr(helpers.os, "name", "nt")

    started = time.monotonic()
    with helpers._request_dump_lock(
        tmp_path, timeout_seconds=0.05, poll_seconds=0.005
    ) as acquired:
        assert acquired is False
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert fake_msvcrt.LK_UNLCK not in calls
    assert opened_handles and all(handle.closed for handle in opened_handles)

    contended = False
    with helpers._request_dump_lock(
        tmp_path, timeout_seconds=0.05, poll_seconds=0.005
    ) as acquired:
        assert acquired is True
    assert calls[-1] == fake_msvcrt.LK_UNLCK
    assert all(handle.closed for handle in opened_handles)


def test_request_dump_count_cap_is_cross_process_safe(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_write_dump_worker, args=(str(tmp_path), index))
        for index in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    dumps = list(tmp_path.glob("request_dump_*.json"))
    assert len(dumps) == 2
    assert all(json.loads(path.read_text(encoding="utf-8")) for path in dumps)


def test_large_request_dump_sweep_is_globally_bounded_per_call(monkeypatch, tmp_path):
    from agent import agent_runtime_helpers as helpers

    for index in range(2048):
        path = tmp_path / f"request_dump_legacy_{index:05d}.json"
        path.write_text(json.dumps({"fingerprint": f"legacy-{index}"}))

    original_scandir = helpers.os.scandir
    original_unlink = Path.unlink
    scanners = []
    counters = {"next": 0, "stat": 0, "unlink": 0}

    class _CountingEntry:
        def __init__(self, entry):
            self._entry = entry
            self.name = entry.name
            self.path = entry.path

        def is_symlink(self):
            return self._entry.is_symlink()

        def is_file(self, *, follow_symlinks=True):
            return self._entry.is_file(follow_symlinks=follow_symlinks)

        def stat(self, *, follow_symlinks=True):
            counters["stat"] += 1
            return self._entry.stat(follow_symlinks=follow_symlinks)

    class _CountingScanner:
        def __init__(self, path):
            self._scanner = original_scandir(path)
            self.closed = False
            scanners.append(self)

        def __iter__(self):
            return self

        def __next__(self):
            counters["next"] += 1
            return _CountingEntry(next(self._scanner))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            self.closed = True
            self._scanner.close()

    def _counted_unlink(path, *args, **kwargs):
        counters["unlink"] += 1
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(helpers.os, "scandir", _CountingScanner)
    monkeypatch.setattr(Path, "unlink", _counted_unlink)

    scan_key = str(tmp_path.resolve())
    for invocation in range(100):
        counters.update(next=0, stat=0, unlink=0)
        helpers._write_bounded_request_dump(
            logs_dir=tmp_path,
            dump_file=tmp_path / f"request_dump_new_{invocation:05d}.json",
            payload={"fingerprint": f"new-{invocation}"},
            max_files=7,
            max_bytes=4096,
            ttl_days=7,
            dedup_seconds=0,
            scan_limit=32,
        )
        assert counters["next"] <= 33
        assert counters["stat"] <= 32
        assert counters["unlink"] <= 33
        state = helpers._REQUEST_DUMP_SCAN_STATES.get(scan_key)
        if state is not None:
            assert len(state["survivors"]) <= 7
        if scan_key not in helpers._REQUEST_DUMP_SCAN_STATES:
            break
    else:
        pytest.fail("bounded request-dump sweep did not finish")

    assert len(list(tmp_path.glob("request_dump_*.json"))) <= 7
    assert scanners and all(scanner.closed for scanner in scanners)
    assert scan_key not in helpers._REQUEST_DUMP_SCAN_STATES


def test_request_dump_deduplication_crosses_scan_pages(monkeypatch, tmp_path):
    from agent import agent_runtime_helpers as helpers

    now = 2_000_000_000.0
    duplicate_paths = []
    for index in range(40):
        fingerprint = "cross-page" if index in {0, 39} else f"unique-{index}"
        path = tmp_path / f"request_dump_legacy_{index:05d}.json"
        path.write_text(json.dumps({"fingerprint": fingerprint}))
        os.utime(path, (now - 100 - index, now - 100 - index))
        if fingerprint == "cross-page":
            duplicate_paths.append(path)

    original_scandir = helpers.os.scandir

    class _OrderedScanner:
        def __init__(self, path):
            with original_scandir(path) as scanner:
                self._entries = iter(sorted(scanner, key=lambda entry: entry.name))
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._entries)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            self.closed = True

    monkeypatch.setattr(helpers.os, "scandir", _OrderedScanner)
    monkeypatch.setattr(helpers.time, "time", lambda: now)

    scan_key = str(tmp_path.resolve())
    for invocation in range(20):
        helpers._write_bounded_request_dump(
            logs_dir=tmp_path,
            dump_file=tmp_path / f"request_dump_new_{invocation:05d}.json",
            payload={"fingerprint": "cross-page"},
            max_files=5,
            max_bytes=4096,
            ttl_days=7,
            dedup_seconds=3600,
            scan_limit=8,
        )
        if scan_key not in helpers._REQUEST_DUMP_SCAN_STATES:
            break
    else:
        pytest.fail("bounded request-dump sweep did not finish")

    matching = [
        path for path in tmp_path.glob("request_dump_*.json")
        if json.loads(path.read_text())["fingerprint"] == "cross-page"
    ]
    assert matching == [duplicate_paths[0]]
    assert len(list(tmp_path.glob("request_dump_*.json"))) <= 5
