"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _install_mock_server(monkeypatch, script: str = "errors", server_id: str = "pyright"):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,  # always use workspace root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


def test_service_returns_empty_when_disabled(tmp_path):
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="auto",
    )
    assert not svc.is_active()
    f = tmp_path / "x.py"
    f.write_text("")
    assert svc.get_diagnostics_sync(str(f)) == []
    svc.shutdown()


def test_service_skips_files_outside_workspace(tmp_path):
    """Files outside any git worktree must not trigger LSP."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    f = tmp_path / "x.py"
    f.write_text("")
    # No .git anywhere — service should report not enabled for this file.
    assert not svc.enabled_for(str(f))
    svc.shutdown()


def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = svc.get_diagnostics_sync(str(f), line_shift=lambda L: L)
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_status_includes_clients(mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.get_diagnostics_sync(str(f))
        info = svc.get_status()
        assert info["enabled"] is True
        assert any(c["server_id"] == "pyright" for c in info["clients"])
    finally:
        svc.shutdown()


def test_idle_reaper_shuts_down_and_removes_stale_client():
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0.01,
        max_clients=4,
    )
    client = MagicMock()
    client.shutdown = AsyncMock()
    key = ("pyright", "/tmp/stale-workspace")
    svc._clients[key] = client
    svc._last_used[key] = time.time() - 60
    try:
        reaped = svc._loop.run(svc._reap_idle_clients(), timeout=1.0)
        assert reaped == 1
        assert key not in svc._clients
        client.shutdown.assert_awaited_once()
    finally:
        svc.shutdown()


def test_client_cap_refuses_new_workspace_when_every_client_is_recent(monkeypatch):
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=600,
        max_clients=1,
    )
    existing = MagicMock()
    existing.is_running = True
    key = ("pyright", "/tmp/recent-workspace")
    svc._clients[key] = existing
    svc._last_used[key] = time.time()
    server = MagicMock()
    server.server_id = "pyright"
    server.resolve_root.return_value = "/tmp/new-workspace"
    monkeypatch.setattr("agent.lsp.manager.find_server_for_file", lambda _path: server)
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: ("/tmp/new-workspace", True),
    )
    try:
        assert svc.available_for("/tmp/new-workspace/new.py") is False
        client = svc._loop.run(svc._get_or_spawn("/tmp/new-workspace/new.py"), timeout=1.0)
        assert client is None
        assert len(svc._clients) == 1
    finally:
        svc.shutdown()


def test_idle_reaper_never_shuts_down_a_leased_client():
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0.01,
        max_clients=1,
    )
    client = MagicMock()
    client.shutdown = AsyncMock()
    key = ("pyright", "/tmp/busy-workspace")
    svc._clients[key] = client
    svc._last_used[key] = time.time() - 60
    svc._client_leases[key] = 1
    try:
        reaped = svc._loop.run(svc._reap_idle_clients(), timeout=1.0)
        assert reaped == 0
        assert svc._clients[key] is client
        client.shutdown.assert_not_awaited()
    finally:
        svc.shutdown()


def test_idle_reaper_removes_recent_dead_client():
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=600,
        max_clients=1,
    )
    client = MagicMock()
    client.is_running = False
    client.shutdown = AsyncMock()
    key = ("pyright", "/tmp/dead-workspace")
    svc._clients[key] = client
    svc._last_used[key] = time.time()
    try:
        reaped = svc._loop.run(svc._reap_idle_clients(), timeout=1.0)
        assert reaped == 1
        assert key not in svc._clients
        client.shutdown.assert_awaited_once()
    finally:
        svc.shutdown()
