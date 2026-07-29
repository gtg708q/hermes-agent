"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import hashlib
import json
import threading
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient as _AiohttpTestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _api_request_profile,
    _approval_event_choices,
    _derive_idempotent_run_id,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


class TestClient(_AiohttpTestClient):
    """Give legacy run tests a unique key unless they explicitly test headers."""

    def post(self, path, *, headers=None, **kwargs):
        if headers is None:
            headers = _idempotency_headers()
        elif "Idempotency-Key" not in headers and headers:
            headers = {**headers, **_idempotency_headers()}
        return super().post(path, headers=headers, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _create_profiled_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create a minimal multiplex app that drives the real profile context."""
    @web.middleware
    async def _profile_context(request, handler):
        token = _api_request_profile.set(request.match_info["profile"])
        try:
            return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[_profile_context])
    app["api_server_adapter"] = adapter
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
    app.router.add_get("/p/{profile}/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _idempotency_headers(key: str | None = None) -> dict[str, str]:
    return {"Idempotency-Key": key if key is not None else uuid.uuid4().hex}


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_idempotent_run_fails_closed_after_response_store_disk_fallback(
        self, adapter
    ):
        import sqlite3

        real_connect = sqlite3.connect
        calls = 0

        def _fail_disk_once(path, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("disk unavailable")
            return real_connect(path, *args, **kwargs)

        adapter._response_store.close()
        with patch("sqlite3.connect", side_effect=_fail_disk_once):
            adapter._response_store = ResponseStore(db_path="/unavailable/response_store.db")

        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent") as create_agent:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                payload = await response.json()

        assert response.status == 503
        assert payload["error"]["code"] == "idempotency_store_unavailable"
        create_agent.assert_not_called()

    def test_idempotent_run_id_scopes_key_by_canonical_profile(self):
        key = "same-client-key"

        unprefixed = _derive_idempotent_run_id(key, None)
        explicit_default = _derive_idempotent_run_id(key, "default")
        secondary = _derive_idempotent_run_id(key, "research")

        assert unprefixed == explicit_default
        assert secondary != unprefixed

    @pytest.mark.asyncio
    async def test_start_preserves_headerless_contract_and_validates_supplied_key(self, adapter):
        app = _create_runs_app(adapter)
        with patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                missing = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers={}
                )
                empty = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": ""},
                )
                oversized = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Idempotency-Key": "x" * 256},
                )

        assert missing.status == 202
        assert empty.status == 400
        assert oversized.status == 400
        assert len(adapter._run_statuses) == 1

    @pytest.mark.asyncio
    async def test_idempotency_key_derives_exact_run_id_and_replay_returns_status(self, adapter):
        key = "sc15:conversation:42:turn:7"
        expected = _derive_idempotent_run_id(key, None)
        app = _create_runs_app(adapter)
        release = threading.Event()
        started = threading.Event()

        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()

            def _run(**_kwargs):
                started.set()
                release.wait(timeout=5)
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = _run
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent

            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                assert first.status == 202
                first_body = await first.json()
                assert first_body["run_id"] == expected
                assert await asyncio.to_thread(started.wait, 2)

                replay = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                replay_body = await replay.json()
                assert replay.status == 200
                assert replay_body["run_id"] == expected
                assert replay_body["status"] in {"queued", "running"}
                assert mock_create.call_count == 1
                release.set()

    @pytest.mark.asyncio
    async def test_idempotency_key_conflicting_payload_is_rejected(self, adapter):
        key = "same-key-different-payload"
        app = _create_runs_app(adapter)
        release = threading.Event()
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = lambda **_kwargs: (
                release.wait(timeout=5) or {"final_response": "done"}
            )
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs", json={"input": "first"}, headers=_idempotency_headers(key)
                )
                conflict = await cli.post(
                    "/v1/runs", json={"input": "second"}, headers=_idempotency_headers(key)
                )
                payload = await conflict.json()
                release.set()

        assert first.status == 202
        assert conflict.status == 409
        assert payload["error"]["code"] == "idempotency_key_conflict"
        assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_idempotency_ownership_replays_across_adapter_restart(self, adapter):
        key = "durable-owner-across-restart"
        app = _create_runs_app(adapter)
        release = threading.Event()
        with patch.object(adapter, "_create_agent") as first_create:
            first_agent = MagicMock()
            first_agent.run_conversation.side_effect = lambda **_kwargs: (
                release.wait(timeout=5) or {"final_response": "done"}
            )
            first_agent.session_prompt_tokens = 0
            first_agent.session_completion_tokens = 0
            first_agent.session_total_tokens = 0
            first_create.return_value = first_agent
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                assert first.status == 202

                restarted = _make_adapter()
                restarted_app = _create_runs_app(restarted)
                with patch.object(restarted, "_create_agent") as restarted_create:
                    async with TestClient(TestServer(restarted_app)) as restarted_cli:
                        replay = await restarted_cli.post(
                            "/v1/runs",
                            json={"input": "hello"},
                            headers=_idempotency_headers(key),
                        )
                        replay_payload = await replay.json()
                    restarted_create.assert_not_called()
                release.set()

        assert replay.status == 200
        assert replay_payload["status"] in {"queued", "running"}

    def test_expired_queued_claim_is_reclaimed_across_stores(self, tmp_path):
        db_path = tmp_path / "response-store.db"
        first = ResponseStore(db_path=str(db_path))
        second = ResponseStore(db_path=str(db_path))
        initial = {"run_id": "run_queued_reclaim", "status": "queued"}

        assert first.claim_run(
            "run_queued_reclaim",
            "fingerprint",
            initial,
            owner_id="owner-a",
            lease_seconds=30,
            now=100,
        )[0] is True
        still_owned = second.claim_run(
            "run_queued_reclaim",
            "fingerprint",
            initial,
            owner_id="owner-b",
            lease_seconds=30,
            now=129,
        )
        reclaimed = second.claim_run(
            "run_queued_reclaim",
            "fingerprint",
            initial,
            owner_id="owner-b",
            lease_seconds=30,
            now=131,
        )

        assert still_owned[0] is False
        assert still_owned[1] is False
        assert reclaimed[0] is True
        assert reclaimed[1] is False
        first.close()
        second.close()

    @pytest.mark.parametrize(
        "active_status", ["running", "waiting_for_approval", "stopping"]
    )
    def test_active_post_start_claim_is_replay_only_across_stores(
        self, tmp_path, active_status
    ):
        db_path = tmp_path / "response-store.db"
        first = ResponseStore(db_path=str(db_path))
        second = ResponseStore(db_path=str(db_path))
        active = {"run_id": "run_active", "status": active_status}

        assert first.claim_run(
            "run_active",
            "fingerprint",
            active,
            owner_id="owner-a",
            lease_seconds=30,
            now=100,
        )[0] is True

        owned, conflict, replay = second.claim_run(
            "run_active",
            "fingerprint",
            {**active, "status": "queued"},
            owner_id="owner-b",
            lease_seconds=30,
            now=129,
        )

        assert owned is False
        assert conflict is False
        assert replay == active
        first.close()
        second.close()

    @pytest.mark.parametrize(
        "expired_status", ["running", "waiting_for_approval", "stopping"]
    )
    def test_expired_post_start_claim_fails_closed_fences_owner_and_is_gc_eligible(
        self, tmp_path, expired_status
    ):
        db_path = tmp_path / "response-store.db"
        first = ResponseStore(db_path=str(db_path))
        second = ResponseStore(db_path=str(db_path))
        active = {
            "object": "hermes.run",
            "run_id": "run_expired",
            "status": expired_status,
            "created_at": 90,
            "updated_at": 100,
        }
        assert first.claim_run(
            "run_expired",
            "fingerprint",
            active,
            owner_id="owner-a",
            lease_seconds=30,
            now=100,
        )[0] is True

        owned, conflict, failed = second.claim_run(
            "run_expired",
            "fingerprint",
            {**active, "status": "queued"},
            owner_id="owner-b",
            lease_seconds=30,
            now=131,
        )

        assert owned is False
        assert conflict is False
        assert failed["status"] == "failed"
        assert failed["updated_at"] == 131
        assert "owner lease expired" in failed["error"].lower()
        assert failed["last_event"] == "run.failed"
        terminal_at = second._conn.execute(
            "SELECT terminal_at FROM run_idempotency WHERE run_id = ?",
            ("run_expired",),
        ).fetchone()[0]
        assert terminal_at == 131

        assert first.put_run_status(
            "run_expired",
            {**active, "status": "completed", "output": "stale"},
            owner_id="owner-a",
            lease_seconds=30,
            now=132,
        ) is False
        assert second.get_run_status("run_expired") == failed
        assert second.delete_expired_terminal_runs(132) == 1
        assert second.get_run_status("run_expired") is None
        first.close()
        second.close()

    def test_terminal_failed_claim_replay_is_stable_across_stores(self, tmp_path):
        db_path = tmp_path / "response-store.db"
        first = ResponseStore(db_path=str(db_path))
        second = ResponseStore(db_path=str(db_path))
        running = {"run_id": "run_stable_failed", "status": "running"}
        first.claim_run(
            "run_stable_failed",
            "fingerprint",
            running,
            owner_id="owner-a",
            lease_seconds=30,
            now=100,
        )
        failed = second.claim_run(
            "run_stable_failed",
            "fingerprint",
            {**running, "status": "queued"},
            owner_id="owner-b",
            lease_seconds=30,
            now=131,
        )[2]

        replay = first.claim_run(
            "run_stable_failed",
            "fingerprint",
            {**running, "status": "queued"},
            owner_id="owner-c",
            lease_seconds=30,
            now=500,
        )

        assert replay == (False, False, failed)
        terminal_at = first._conn.execute(
            "SELECT terminal_at FROM run_idempotency WHERE run_id = ?",
            ("run_stable_failed",),
        ).fetchone()[0]
        assert terminal_at == 131
        first.close()
        second.close()

    def test_reclaimed_claim_fences_stale_owner_status_writes(self, tmp_path):
        store = ResponseStore(db_path=str(tmp_path / "response-store.db"))
        initial = {"run_id": "run_fenced", "status": "queued"}
        store.claim_run(
            "run_fenced",
            "fingerprint",
            initial,
            owner_id="owner-a",
            lease_seconds=30,
            now=100,
        )
        assert store.claim_run(
            "run_fenced",
            "fingerprint",
            initial,
            owner_id="owner-b",
            lease_seconds=30,
            now=131,
        )[0] is True

        assert store.put_run_status(
            "run_fenced",
            {**initial, "status": "completed", "output": "stale"},
            owner_id="owner-a",
            lease_seconds=30,
            now=132,
        ) is False
        assert store.get_run_status("run_fenced")["status"] == "queued"
        store.close()

    def test_heartbeat_keeps_active_owner_from_being_reclaimed(self, tmp_path):
        store = ResponseStore(db_path=str(tmp_path / "response-store.db"))
        initial = {"run_id": "run_heartbeat", "status": "running"}
        store.claim_run(
            "run_heartbeat",
            "fingerprint",
            initial,
            owner_id="owner-a",
            lease_seconds=30,
            now=100,
        )
        assert store.renew_run_lease(
            "run_heartbeat", "owner-a", 30, now=125
        ) is True

        owned, conflict, existing = store.claim_run(
            "run_heartbeat",
            "fingerprint",
            initial,
            owner_id="owner-b",
            lease_seconds=30,
            now=131,
        )

        assert owned is False
        assert conflict is False
        assert existing["status"] == "running"
        store.close()

    @pytest.mark.asyncio
    async def test_stale_queued_claim_is_restarted_and_executes(self, tmp_path):
        key = "replay-after-owner-crash"
        body = {"input": "hello"}
        run_id = _derive_idempotent_run_id(key, None)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "body": body,
                    "gateway_session_key": None,
                    "profile": "default",
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        store = ResponseStore(db_path=str(tmp_path / "response-store.db"))
        store.claim_run(
            run_id,
            fingerprint,
            {"run_id": run_id, "status": "queued"},
            owner_id="crashed-owner",
            lease_seconds=30,
            now=time.time() - 31,
        )
        adapter = _make_adapter()
        adapter._response_store.close()
        adapter._response_store = store
        app = _create_runs_app(adapter)

        with patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "recovered"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs", json=body, headers=_idempotency_headers(key)
                )
                payload = await response.json()
                for _ in range(20):
                    if create_agent.called:
                        break
                    await asyncio.sleep(0.05)

        assert response.status == 202
        assert payload["run_id"] == run_id
        create_agent.assert_called_once()
        store.close()

    @pytest.mark.asyncio
    async def test_stale_queued_recovery_reserves_capacity_before_body_parse(
        self, tmp_path
    ):
        """A reclaimable queued row is execution work, not a replay-only drain."""
        stale_key = "stale-queued-capacity"
        new_key = "new-while-stale-parses"
        stale_body = {"input": "recover me"}
        stale_run_id = _derive_idempotent_run_id(stale_key, None)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"body": stale_body, "gateway_session_key": None, "profile": "default"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        store = ResponseStore(db_path=str(tmp_path / "response-store.db"))
        store.claim_run(
            stale_run_id,
            fingerprint,
            {"run_id": stale_run_id, "status": "queued"},
            owner_id="crashed-owner",
            lease_seconds=30,
            now=time.time() - 31,
        )
        metadata = store.get_run_claim_metadata(stale_run_id, profile="default")
        assert metadata is not None
        assert metadata["status"]["status"] == "queued"
        assert metadata["lease_expires_at"] <= time.time()

        adapter = _make_adapter()
        adapter._response_store.close()
        adapter._response_store = store
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)
        parse_started = asyncio.Event()
        release_parse = asyncio.Event()
        original_read = adapter._read_json_body

        async def _blocked_stale_read(request):
            if request.headers.get("Idempotency-Key") == stale_key:
                parse_started.set()
                await release_parse.wait()
            return await original_read(request)

        with patch.object(adapter, "_read_json_body", side_effect=_blocked_stale_read), \
             patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "recovered"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                stale_task = asyncio.create_task(cli.post(
                    "/v1/runs", json=stale_body, headers=_idempotency_headers(stale_key)
                ))
                await asyncio.wait_for(parse_started.wait(), timeout=1)
                new_response = await cli.post(
                    "/v1/runs",
                    json={"input": "new work"},
                    headers=_idempotency_headers(new_key),
                )
                release_parse.set()
                stale_response = await stale_task
                for _ in range(20):
                    if create_agent.called:
                        break
                    await asyncio.sleep(0.02)

        assert new_response.status == 429
        assert stale_response.status == 202
        create_agent.assert_called_once()
        durable = store.get_run_status(stale_run_id)
        assert durable is not None
        assert durable["status"] in {"queued", "running", "completed"}
        store.close()

    @pytest.mark.asyncio
    async def test_expired_running_claim_resolves_failed_without_execution(
        self, tmp_path
    ):
        key = "expired-running-fails-closed"
        body = {"input": "hello"}
        run_id = _derive_idempotent_run_id(key, None)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "body": body,
                    "gateway_session_key": None,
                    "profile": "default",
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        db_path = tmp_path / "response-store.db"
        crashed_store = ResponseStore(db_path=str(db_path))
        replay_store = ResponseStore(db_path=str(db_path))
        crashed_store.claim_run(
            run_id,
            fingerprint,
            {"object": "hermes.run", "run_id": run_id, "status": "running"},
            owner_id="crashed-owner",
            lease_seconds=30,
            now=time.time() - 31,
        )
        adapter = _make_adapter()
        adapter._response_store.close()
        adapter._response_store = replay_store
        app = _create_runs_app(adapter)

        with patch.object(adapter, "_create_agent") as create_agent:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs", json=body, headers=_idempotency_headers(key)
                )
                payload = await response.json()

        assert response.status == 200
        assert payload["run_id"] == run_id
        assert payload["status"] == "failed"
        assert "owner lease expired" in payload["error"].lower()
        create_agent.assert_not_called()
        crashed_store.close()
        replay_store.close()

    @pytest.mark.asyncio
    async def test_concurrent_same_key_launches_exactly_one_run(self, adapter):
        key = "same-key-after-client-response-loss"
        app = _create_runs_app(adapter)
        release = threading.Event()

        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = lambda **_kwargs: (
                release.wait(timeout=5) or {"final_response": "done"}
            )
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent

            async with TestClient(TestServer(app)) as cli:
                responses = await asyncio.gather(*[
                    cli.post(
                        "/v1/runs",
                        json={"input": "hello"},
                        headers=_idempotency_headers(key),
                    )
                    for _ in range(8)
                ])
                bodies = [await response.json() for response in responses]
                assert {body["run_id"] for body in bodies} == {
                    _derive_idempotent_run_id(key, None)
                }
                assert sum(response.status == 202 for response in responses) == 1
                assert sum(response.status == 200 for response in responses) == 7
                assert len(adapter._active_run_tasks) == 1
                release.set()

    def test_ttl_sweep_retains_active_idempotent_run_status(self, adapter):
        run_id = "run_active"
        adapter._run_statuses[run_id] = {
            "object": "hermes.run",
            "run_id": run_id,
            "status": "running",
            "updated_at": 1.0,
        }

        adapter._sweep_orphaned_runs_once(now=1.0 + adapter._RUN_STATUS_TTL + 1)

        assert adapter._run_statuses[run_id]["status"] == "running"

    def test_ttl_sweep_deletes_expired_terminal_rows_after_restart(self, tmp_path):
        db_path = tmp_path / "response-store.db"
        before_restart = ResponseStore(db_path=str(db_path))
        now = time.time()
        initial = {
            "object": "hermes.run",
            "run_id": "run_expired_after_restart",
            "status": "queued",
            "created_at": now - 100_000,
            "updated_at": now - 100_000,
        }
        before_restart.claim_run("run_expired_after_restart", "fingerprint", initial)
        before_restart.put_run_status(
            "run_expired_after_restart", {**initial, "status": "completed"}
        )
        before_restart._conn.execute(
            "UPDATE run_idempotency SET terminal_at = ? WHERE run_id = ?",
            (now - APIServerAdapter._RUN_STATUS_TTL - 1, "run_expired_after_restart"),
        )
        before_restart._conn.commit()
        before_restart.close()

        restarted = _make_adapter()
        restarted._response_store.close()
        restarted._response_store = ResponseStore(db_path=str(db_path))
        assert restarted._run_statuses == {}

        restarted._sweep_orphaned_runs_once(now=now)

        assert restarted._response_store.get_run_status(
            "run_expired_after_restart"
        ) is None
        restarted._response_store.close()

    def test_expired_terminal_sql_gc_honors_batch_limit(self, tmp_path):
        store = ResponseStore(db_path=str(tmp_path / "bounded-response-store.db"))
        now = time.time()
        for index in range(3):
            run_id = f"run_expired_{index}"
            status = {"run_id": run_id, "status": "queued"}
            store.claim_run(run_id, f"fingerprint-{index}", status)
            store.put_run_status(run_id, {**status, "status": "completed"})
        store._conn.execute(
            "UPDATE run_idempotency SET terminal_at = ?",
            (now - 100,),
        )
        store._conn.commit()

        assert store.delete_expired_terminal_runs(now, limit=2) == 2
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM run_idempotency"
        ).fetchone()[0]

        assert remaining == 1
        store.close()

    def test_one_sweep_drains_more_than_one_expired_sql_batch(self, tmp_path):
        store = ResponseStore(db_path=str(tmp_path / "catch-up-response-store.db"))
        now = time.time()
        for index in range(250):
            run_id = f"run_backlog_{index}"
            status = {"run_id": run_id, "status": "queued"}
            store.claim_run(run_id, f"fingerprint-{index}", status)
            store.put_run_status(run_id, {**status, "status": "completed"})
        with store._lock:
            store._conn.execute(
                "UPDATE run_idempotency SET terminal_at = ?", (now - 100,)
            )
            store._conn.commit()

        restarted = _make_adapter()
        restarted._response_store.close()
        restarted._response_store = store
        restarted._sweep_orphaned_runs_once(now=now + restarted._RUN_STATUS_TTL)

        with store._lock:
            remaining = store._conn.execute(
                "SELECT COUNT(*) FROM run_idempotency"
            ).fetchone()[0]
        assert remaining == 0
        store.close()

    @pytest.mark.asyncio
    async def test_terminal_status_write_retries_before_run_releases_ownership(
        self, adapter
    ):
        app = _create_runs_app(adapter)
        original_put = adapter._response_store.put_run_status
        terminal_attempts = 0
        durable = None

        def _flaky_put(run_id, status, **kwargs):
            nonlocal terminal_attempts
            if status.get("status") == "completed":
                terminal_attempts += 1
                if terminal_attempts == 1:
                    raise OSError("transient terminal write failure")
            return original_put(run_id, status, **kwargs)

        with (
            patch.object(adapter._response_store, "put_run_status", side_effect=_flaky_put),
            patch.object(adapter, "_create_agent") as create_agent,
        ):
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                for _ in range(50):
                    durable = adapter._response_store.get_run_status(run_id)
                    if durable and durable.get("status") == "completed":
                        break
                    await asyncio.sleep(0.02)

        assert terminal_attempts >= 2
        assert durable is not None
        assert durable["status"] == "completed"

    @pytest.mark.asyncio
    async def test_terminal_status_retry_stops_and_cleans_up_after_lease_loss(
        self, adapter
    ):
        app = _create_runs_app(adapter)
        original_put = adapter._response_store.put_run_status
        terminal_attempts = 0

        def _lose_terminal_write(run_id, status, **kwargs):
            nonlocal terminal_attempts
            if status.get("status") == "completed":
                terminal_attempts += 1
                return False
            return original_put(run_id, status, **kwargs)

        with (
            patch.object(
                adapter._response_store,
                "put_run_status",
                side_effect=_lose_terminal_write,
            ),
            patch.object(
                adapter._response_store,
                "renew_run_lease",
                return_value=False,
            ) as renew_lease,
            patch.object(adapter, "_create_agent") as create_agent,
        ):
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent

            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                task = adapter._active_run_tasks[run_id]
                await asyncio.wait_for(task, timeout=1)

        assert terminal_attempts == 1
        renew_lease.assert_called_once()
        assert run_id not in adapter._active_run_tasks
        assert run_id not in adapter._active_run_agents
        assert run_id not in adapter._run_approval_sessions

    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_start_empty_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": ""})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_history_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "conversation_history": {"role": "user"}},
            )
        assert resp.status == 400
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_with_valid_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202

    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


    @pytest.mark.asyncio
    async def test_overlapping_capacity_admission_selects_one_before_body_await(
        self, adapter
    ):
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)
        first_parse_entered = asyncio.Event()
        release_parse = asyncio.Event()
        original_read = adapter._read_json_body
        parse_calls = 0

        async def _blocked_read(request):
            nonlocal parse_calls
            parse_calls += 1
            first_parse_entered.set()
            await release_parse.wait()
            return await original_read(request)

        first_key = "capacity-first"
        second_key = "capacity-second"
        with patch.object(adapter, "_read_json_body", side_effect=_blocked_read), \
             patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                first_task = asyncio.create_task(
                    cli.post(
                        "/v1/runs",
                        json={"input": "first"},
                        headers=_idempotency_headers(first_key),
                    )
                )
                await asyncio.wait_for(first_parse_entered.wait(), timeout=1)
                second_task = asyncio.create_task(
                    cli.post(
                        "/v1/runs",
                        json={"input": "second"},
                        headers=_idempotency_headers(second_key),
                    )
                )
                await asyncio.sleep(0.05)
                second_finished_before_release = second_task.done()
                release_parse.set()
                first, second = await asyncio.gather(first_task, second_task)

        assert second_finished_before_release is True
        assert parse_calls == 1
        assert first.status == 202
        assert second.status == 429
        assert adapter._response_store.get_run_status(
            _derive_idempotent_run_id(second_key, None)
        ) is None

    @pytest.mark.asyncio
    async def test_idempotent_replay_bypasses_full_execution_capacity(self, adapter):
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)
        key = "capacity-replay"
        release = threading.Event()
        started = threading.Event()

        with patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()

            def _run(**_kwargs):
                started.set()
                release.wait(timeout=5)
                return {"final_response": "done"}

            agent.run_conversation.side_effect = _run
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                assert first.status == 202
                assert await asyncio.to_thread(started.wait, 1)

                replay = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                replay_body = await replay.json()
                release.set()

        assert replay.status == 200
        assert replay_body["run_id"] == _derive_idempotent_run_id(key, None)
        assert replay_body["status"] == "running"
        assert create_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_slow_replay_parse_is_drain_visible_but_does_not_consume_capacity(
        self, adapter
    ):
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)
        replay_key = "slow-replay-reservation"
        replay_body = {"input": "replay"}

        with patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent

            async with TestClient(TestServer(app)) as cli:
                original = await cli.post(
                    "/v1/runs",
                    json=replay_body,
                    headers=_idempotency_headers(replay_key),
                )
                original_run_id = (await original.json())["run_id"]
                status = None
                for _ in range(50):
                    status = adapter._response_store.get_run_status(original_run_id)
                    if status and status.get("status") == "completed":
                        break
                    await asyncio.sleep(0.02)
                assert status is not None
                assert status["status"] == "completed"

                replay_parse_started = asyncio.Event()
                release_replay_parse = asyncio.Event()
                original_read = adapter._read_json_body

                async def _slow_replay_read(request):
                    if request.headers.get("Idempotency-Key") == replay_key:
                        replay_parse_started.set()
                        await release_replay_parse.wait()
                    return await original_read(request)

                with patch.object(
                    adapter, "_read_json_body", side_effect=_slow_replay_read
                ):
                    replay_task = asyncio.create_task(
                        cli.post(
                            "/v1/runs",
                            json=replay_body,
                            headers=_idempotency_headers(replay_key),
                        )
                    )
                    await asyncio.wait_for(replay_parse_started.wait(), timeout=1)
                    assert adapter.active_agent_work_count() == 1

                    new_response = await cli.post(
                        "/v1/runs",
                        json={"input": "new execution"},
                        headers=_idempotency_headers("new-execution-during-replay"),
                    )
                    release_replay_parse.set()
                    replay_response = await replay_task

        assert new_response.status == 202
        assert replay_response.status == 200
        assert create_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_replay_preflight_store_failure_returns_controlled_503(self, adapter):
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)

        with patch.object(
            adapter._response_store,
            "get_run_claim_metadata",
            side_effect=OSError("response store read failed"),
        ), patch.object(adapter, "_read_json_body", new_callable=AsyncMock) as read_body:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers=_idempotency_headers("lookup-io-failure"),
                )
                payload = await response.json()

        assert response.status == 503
        assert payload["error"]["code"] == "idempotency_store_unavailable"
        assert "persist idempotent run ownership" in payload["error"]["message"].lower()
        read_body.assert_not_awaited()
        assert adapter.active_agent_work_count() == 0

    @pytest.mark.asyncio
    async def test_expired_running_owner_terminalizes_while_executor_stays_tracked(
        self, tmp_path
    ):
        db_path = tmp_path / "response-store.db"
        first_store = ResponseStore(db_path=str(db_path))
        second_store = ResponseStore(db_path=str(db_path))
        first = _make_adapter()
        second = _make_adapter()
        first._response_store.close()
        second._response_store.close()
        first._response_store = first_store
        second._response_store = second_store
        first._RUN_OWNER_LEASE_SECONDS = 0.05
        first._RUN_OWNER_HEARTBEAT_SECONDS = 10.0
        second._RUN_OWNER_LEASE_SECONDS = 0.05
        release = threading.Event()
        started = threading.Event()
        key = "running-owner-must-not-be-taken-over"
        first_app = _create_runs_app(first)
        second_app = _create_runs_app(second)

        with patch.object(first, "_create_agent") as first_create, patch.object(
            second, "_create_agent"
        ) as second_create:
            first_agent = MagicMock()

            def _noncooperative_run(**_kwargs):
                started.set()
                release.wait(timeout=5)
                return {"final_response": "old-owner-finished"}

            first_agent.run_conversation.side_effect = _noncooperative_run
            first_agent.session_prompt_tokens = 0
            first_agent.session_completion_tokens = 0
            first_agent.session_total_tokens = 0
            first_create.return_value = first_agent

            async with TestClient(TestServer(first_app)) as first_cli, TestClient(
                TestServer(second_app)
            ) as second_cli:
                accepted = await first_cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                run_id = (await accepted.json())["run_id"]
                assert await asyncio.to_thread(started.wait, 1)
                with second_store._lock:
                    second_store._conn.execute(
                        "UPDATE run_idempotency SET lease_expires_at = 0 WHERE run_id = ?",
                        (run_id,),
                    )
                    second_store._conn.commit()

                replay = await second_cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=_idempotency_headers(key)
                )
                replay_body = await replay.json()

                assert replay.status == 200
                assert replay_body["status"] == "failed"
                assert "owner lease expired" in replay_body["error"].lower()
                second_create.assert_not_called()
                assert run_id in first._active_run_tasks
                assert not first._active_run_tasks[run_id].done()

                release.set()
                await asyncio.wait_for(first._active_run_tasks[run_id], timeout=1)

        durable = first_store.get_run_status(run_id)
        assert durable is not None
        assert durable["status"] == "failed"
        assert "output" not in durable
        first_store.close()
        second_store.close()

    @pytest.mark.asyncio
    async def test_real_lease_loss_tracks_noncooperative_executor_and_fences_terminal(
        self, tmp_path
    ):
        db_path = tmp_path / "response-store.db"
        owner_store = ResponseStore(db_path=str(db_path))
        successor_store = ResponseStore(db_path=str(db_path))
        adapter = _make_adapter()
        adapter._response_store.close()
        adapter._response_store = owner_store
        adapter._RUN_OWNER_HEARTBEAT_SECONDS = 0.01
        adapter._RUN_OWNER_LEASE_SECONDS = 1.0
        app = _create_runs_app(adapter)
        release = threading.Event()
        started = threading.Event()
        interrupted = threading.Event()

        agent = MagicMock()

        def _interrupt(_message=None):
            interrupted.set()

        def _noncooperative_run(**_kwargs):
            started.set()
            release.wait(timeout=5)
            return {"final_response": "stale-owner-output"}

        agent.interrupt.side_effect = _interrupt
        agent.run_conversation.side_effect = _noncooperative_run
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                assert await asyncio.to_thread(started.wait, 1)

                with successor_store._lock:
                    successor_store._conn.execute(
                        "UPDATE run_idempotency SET owner_id = ? WHERE run_id = ?",
                        ("successor-owner", run_id),
                    )
                    successor_store._conn.commit()

                assert await asyncio.to_thread(interrupted.wait, 1)
                assert run_id in adapter._active_run_tasks
                assert not adapter._active_run_tasks[run_id].done()
                assert run_id in adapter._active_run_agents

                current_status = successor_store.get_run_status(run_id)
                assert current_status is not None
                successor_status = {
                    **current_status,
                    "status": "completed",
                    "output": "successor-output",
                }
                assert successor_store.put_run_status(
                    run_id,
                    successor_status,
                    owner_id="successor-owner",
                    lease_seconds=1,
                ) is True

                release.set()
                await asyncio.wait_for(adapter._active_run_tasks[run_id], timeout=1)

        durable = successor_store.get_run_status(run_id)
        assert durable is not None
        assert durable["status"] == "completed"
        assert durable["output"] == "successor-output"
        assert run_id not in adapter._active_run_tasks
        assert run_id not in adapter._active_run_agents
        owner_store.close()
        successor_store.close()

    @pytest.mark.asyncio
    async def test_lease_heartbeat_loss_interrupts_agent_without_terminal_write(
        self, adapter
    ):
        adapter._RUN_OWNER_HEARTBEAT_SECONDS = 0.01
        app = _create_runs_app(adapter)
        agent, ready, interrupted = _make_slow_agent()

        with patch.object(adapter, "_create_agent", return_value=agent), \
             patch.object(
                 adapter._response_store, "renew_run_lease", return_value=False
             ) as renew, \
             patch.object(
                 adapter, "_set_terminal_run_status", new_callable=AsyncMock
             ) as terminal_write:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await response.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 1)
                for _ in range(50):
                    if interrupted.is_set() and run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.01)

        renew.assert_called()
        agent.interrupt.assert_called_once_with(
            "Durable run lease ownership was lost"
        )
        terminal_write.assert_not_awaited()
        assert run_id not in adapter._active_run_tasks
        assert run_id not in adapter._active_run_agents


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_durable_status_is_not_visible_across_profiles(self, adapter):
        app = _create_profiled_runs_app(adapter)
        owner_response = None
        owner_payload = {}
        with patch.object(adapter, "_create_agent") as create_agent:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "private-b-output"}
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            create_agent.return_value = agent
            async with TestClient(TestServer(app)) as cli:
                started = await cli.post(
                    "/p/profile-b/v1/runs",
                    json={"input": "hello"},
                    headers=_idempotency_headers("profile-private-run"),
                )
                run_id = (await started.json())["run_id"]
                for _ in range(50):
                    owner_response = await cli.get(
                        f"/p/profile-b/v1/runs/{run_id}"
                    )
                    owner_payload = await owner_response.json()
                    if owner_payload.get("status") == "completed":
                        break
                    await asyncio.sleep(0.02)

                # Simulate a gateway restart: force both requests through the
                # adapter-wide durable fallback rather than the in-memory map.
                adapter._run_statuses.clear()
                adapter._run_profiles.clear()
                foreign_response = await cli.get(
                    f"/p/profile-a/v1/runs/{run_id}"
                )
                foreign_payload = await foreign_response.json()
                owner_response = await cli.get(
                    f"/p/profile-b/v1/runs/{run_id}"
                )
                owner_payload = await owner_response.json()

        assert owner_response is not None
        assert owner_response.status == 200
        assert owner_payload["output"] == "private-b-output"
        assert foreign_response.status == 404
        assert "private-b-output" not in json.dumps(foreign_payload)

    @pytest.mark.asyncio
    async def test_status_completed_run_includes_output_and_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                assert status["output"] == "done"
                assert status["usage"]["total_tokens"] == 6
                assert status["last_event"] == "run.completed"

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body



    @pytest.mark.asyncio
    async def test_approval_response_without_pending_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 409
                approval_data = await approval_resp.json()
                assert approval_data["error"]["code"] in {
                    "approval_not_active",
                    "approval_not_pending",
                }

    @pytest.mark.asyncio
    async def test_approval_string_false_does_not_resolve_all(self, adapter):
        """Quoted false must not fan out approval resolution across the queue."""
        app = _create_runs_app(adapter)
        run_id = "run_bool_parse"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_approval_sessions[run_id] = "session-123"

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "all": "false"},
                )

        assert approval_resp.status == 200
        mock_resolve.assert_called_once_with(
            "session-123",
            "once",
            resolve_all=False,
        )

    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


    @pytest.mark.asyncio
    async def test_events_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:
    def test_sweep_keeps_transport_with_active_subscriber(self, adapter):
        run_id = "run_subscribed"
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_streams_created[run_id] = 0
        adapter._run_stream_subscribers.add(run_id)

        adapter._sweep_orphaned_runs_once(time.time())

        assert adapter._run_streams[run_id] is queue
        assert run_id in adapter._run_streams_created

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

    @pytest.mark.asyncio
    async def test_expired_transport_stops_buffering_new_deltas(self, adapter):
        """An unconsumed expired queue must not grow for the rest of a live run."""
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)
                expired_queue = adapter._run_streams[run_id]
                stream_delta = mock_create.call_args.kwargs["stream_delta_callback"]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                adapter._sweep_orphaned_runs_once(time.time())
                before = expired_queue.qsize()
                stream_delta("must-not-buffer")
                mock_agent.interrupt("finish test")
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert expired_queue.qsize() == before

    @pytest.mark.asyncio
    async def test_expired_orphan_run_state_is_reaped(self, adapter):
        run_id = "run_expired_orphan"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0
        adapter._run_approval_sessions[run_id] = run_id

        pending = approval_mod._ApprovalEntry({
            "command": "bash -c orphaned",
            "description": "orphaned approval",
            "pattern_keys": ["shell-c"],
        })
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]

        with patch(
            "gateway.platforms.api_server.asyncio.sleep",
            side_effect=[None, asyncio.CancelledError()],
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert run_id not in adapter._run_streams
        assert run_id not in adapter._run_streams_created
        assert run_id not in adapter._run_approval_sessions
        assert pending.event.is_set()
        with approval_mod._lock:
            assert run_id not in approval_mod._gateway_queues


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_before_agent_creation_prevents_run_start(self, adapter):
        """A stop accepted while queued must prevent agent construction."""
        app = _create_runs_app(adapter)
        original_create_task = asyncio.create_task
        task_started = asyncio.Event()
        allow_task = asyncio.Event()

        def _delayed_create_task(coro):
            async def _delayed():
                task_started.set()
                await allow_task.wait()
                return await coro

            return original_create_task(_delayed())

        with patch("gateway.platforms.api_server.asyncio.create_task", side_effect=_delayed_create_task), \
             patch.object(adapter, "_create_agent") as mock_create:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                await task_started.wait()

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                allow_task.set()

                for _ in range(20):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                mock_create.assert_not_called()
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.5)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/stop")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_stop_already_completed_run_returns_404(self, adapter):
        """Stopping a run that already finished should return 404 (refs cleaned up)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                # Start and wait for completion
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)

                # Run should be done, refs cleaned up
                assert run_id not in adapter._active_run_agents

                # Stop should return 404
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_interrupt_exception_does_not_crash(self, adapter):
        """If agent.interrupt() raises, stop should still succeed."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()

                # Override the interrupt side_effect to raise. Still trip
                # ``interrupted`` so the slow_run thread unblocks at teardown
                # — without this the agent thread blocks the full 10s
                # timeout and the test teardown waits the same amount.
                def _raising_interrupt(message=None):
                    interrupted.set()
                    raise RuntimeError("interrupt failed")

                mock_agent.interrupt = MagicMock(side_effect=_raising_interrupt)
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
