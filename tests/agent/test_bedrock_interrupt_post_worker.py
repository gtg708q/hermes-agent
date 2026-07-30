"""Regression: /stop must not be swallowed on the Bedrock streaming path.

Companion to the OpenAI/Anthropic streaming post-worker guard. The Bedrock
Converse stream callback (bedrock_adapter.stream_converse_with_callbacks) breaks
out of its event loop on interrupt and returns a PARTIAL response WITHOUT
raising. The worker thread then sets result["response"] and exits cleanly with
agent._interrupt_requested still True. Without a post-worker re-check in the
poll loop, interruptible_streaming_api_call would return that partial response
and silently swallow the /stop signal.
"""
from types import SimpleNamespace
import threading
import time
from unittest.mock import patch

import pytest

from agent import chat_completion_helpers as cch
from agent.stream_single_writer import claim_stream_writer as real_claim_stream_writer


class _FakeAgent:
    api_mode = "bedrock_converse"
    _interrupt_requested = False  # not interrupted at entry (passes pre-flight)
    _disable_streaming = False
    reasoning_callback = None
    stream_delta_callback = None
    # Real AIAgent always carries these; the streaming stale-timeout derivation
    # (chat_completion_helpers._derive_stream_stale_timeout) reads them.
    provider = "bedrock"
    model = "anthropic.claude-3-sonnet-20240229-v1:0"
    _consecutive_stale_streams = 0

    def _has_stream_consumers(self):
        return False

    def _buffer_status(self, *a, **k):
        pass

    def _claim_stream_writer(self):
        return 1

    def _fire_stream_delta(self, text):
        pass

    def _fire_tool_gen_started(self, name):
        pass

    def _fire_reasoning_delta(self, text):
        pass

    def _safe_print(self, *a, **k):
        pass


def test_bedrock_stream_interrupt_not_swallowed_post_worker():
    """A /stop arriving MID-stream: the pre-flight check (top of function) has
    already passed, the worker's stream callback breaks and returns a partial
    response WITHOUT raising, leaving _interrupt_requested True. The post-worker
    re-check must raise InterruptedError instead of returning the partial."""
    agent = _FakeAgent()

    partial = SimpleNamespace(choices=[], usage=None, stop_reason="interrupted")

    # Simulate the real adapter: on interrupt it breaks out and returns a
    # partial response WITHOUT raising. Flip the interrupt flag here to model
    # /stop arriving mid-stream (after the pre-flight check, during the worker).
    def _fake_stream(*args, **kwargs):
        agent._interrupt_requested = True
        return partial

    fake_client = SimpleNamespace(converse_stream=lambda **kw: {"stream": []})

    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", side_effect=_fake_stream), \
         patch("agent.bedrock_adapter.normalize_converse_response", side_effect=lambda r: r), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None):
        api_kwargs = {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True}
        with pytest.raises(InterruptedError):
            cch.interruptible_streaming_api_call(agent, api_kwargs)


def test_bedrock_stream_returns_normally_when_not_interrupted():
    """Sanity: with no interrupt, the same path returns the response (guard
    must not fire spuriously)."""
    agent = _FakeAgent()
    agent._interrupt_requested = False

    resp = SimpleNamespace(choices=[], usage=None, stop_reason="end_turn")
    fake_client = SimpleNamespace(converse_stream=lambda **kw: {"stream": []})

    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", return_value=resp), \
         patch("agent.bedrock_adapter.normalize_converse_response", side_effect=lambda r: r), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None):
        api_kwargs = {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True}
        out = cch.interruptible_streaming_api_call(agent, api_kwargs)
        assert out is resp


def test_bedrock_deadline_fences_worker_callbacks_after_bounded_return():
    agent = _FakeAgent()
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    deltas = []
    agent.stream_delta_callback = deltas.append
    agent._has_stream_consumers = lambda: True
    agent._fire_stream_delta = deltas.append
    release = threading.Event()

    def _blocking_stream(*_args, **kwargs):
        release.wait(timeout=2)
        kwargs["on_text_delta"]("late-bedrock-delta")
        return SimpleNamespace(choices=[], usage=None, stop_reason="end_turn")

    fake_client = SimpleNamespace(converse_stream=lambda **kw: {"stream": []})
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", side_effect=_blocking_stream), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None):
        started = time.monotonic()
        with pytest.raises(Exception, match="wall_clock_budget_reached"):
            cch.interruptible_streaming_api_call(
                agent,
                {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True},
            )
        elapsed = time.monotonic() - started
        # A cached gateway agent immediately begins another turn and replaces
        # its mutable deadline. The abandoned worker still belongs to the old
        # turn and must remain fenced.
        agent._turn_deadline_monotonic = time.monotonic() + 30
        release.set()
        time.sleep(0.1)

    assert elapsed < 0.3
    assert deltas == []


def test_bedrock_deadline_fences_worker_before_late_stream_writer_claim():
    agent = _FakeAgent()
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    release = threading.Event()
    claims = []

    def _blocking_converse_stream(**_kwargs):
        release.wait(timeout=2)
        return {"stream": []}

    fake_client = SimpleNamespace(converse_stream=_blocking_converse_stream)
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks") as stream_callbacks, \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None), \
         patch("agent.chat_completion_helpers.claim_stream_writer", side_effect=lambda _agent: claims.append("old")):
        with pytest.raises(Exception, match="wall_clock_budget_reached"):
            cch.interruptible_streaming_api_call(
                agent,
                {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True},
            )
        agent._turn_deadline_monotonic = time.monotonic() + 30
        release.set()
        time.sleep(0.1)

    assert claims == []
    stream_callbacks.assert_not_called()


def test_bedrock_old_turn_cannot_reclaim_writer_after_new_turn_streams():
    agent = _FakeAgent()
    agent._current_turn_id = "old-turn"
    agent._stream_writer_lock = threading.Lock()
    agent._stream_writer_token = 0
    agent._stream_writer_tls = threading.local()
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent._has_stream_consumers = lambda: True

    def _claim(*, expected_generation=None, is_valid=None):
        with agent._stream_writer_lock:
            if (
                expected_generation is not None
                and agent._stream_writer_token != expected_generation
            ):
                return 0
            if is_valid is not None and not is_valid():
                return 0
            agent._stream_writer_token += 1
            token = agent._stream_writer_token
            agent._stream_writer_tls.token = token
            return token

    def _fire_stream_delta(text):
        if agent._stream_writer_tls.token == agent._stream_writer_token:
            delivered.append(text)

    agent._claim_stream_writer = _claim
    agent._fire_stream_delta = _fire_stream_delta

    old_at_claim_boundary = threading.Event()
    resume_old = threading.Event()
    old_error = []

    def _claim_with_old_pause(claim_agent, **kwargs):
        if threading.current_thread().name == "old-bedrock-provider-worker":
            old_at_claim_boundary.set()
            assert resume_old.wait(timeout=2)
        return real_claim_stream_writer(claim_agent, **kwargs)

    def _stream_response(raw_response, **kwargs):
        kwargs["on_text_delta"](raw_response["label"])
        return SimpleNamespace(choices=[], usage=None, stop_reason="end_turn")

    def _run_old_turn():
        try:
            cch.interruptible_streaming_api_call(
                agent,
                {
                    "modelId": "old",
                    "__bedrock_region__": "us-east-1",
                    "__bedrock_converse__": True,
                },
            )
        except BaseException as exc:
            old_error.append(exc)

    def _converse_stream(**kwargs):
        if kwargs["modelId"] == "old":
            threading.current_thread().name = "old-bedrock-provider-worker"
        return {"label": kwargs["modelId"]}

    fake_client = SimpleNamespace(converse_stream=_converse_stream)
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", side_effect=_stream_response), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None), \
         patch("agent.chat_completion_helpers.claim_stream_writer", side_effect=_claim_with_old_pause):
        # Start the deadline only after importing/patching the optional Bedrock
        # adapter, so a first-run lazy dependency install cannot consume it.
        agent._turn_deadline_monotonic = time.monotonic() + 0.05
        old_turn = threading.Thread(target=_run_old_turn)
        old_turn.start()
        assert old_at_claim_boundary.wait(timeout=2)

        # Let the immutable old deadline expire, then roll the cached agent to a
        # new turn and let that turn claim and stream before the old worker runs.
        time.sleep(0.06)
        old_turn.join(timeout=1)
        assert not old_turn.is_alive()
        agent._current_turn_id = "new-turn"
        agent._turn_deadline_monotonic = time.monotonic() + 30
        new_response = cch.interruptible_streaming_api_call(
            agent,
            {
                "modelId": "new",
                "__bedrock_region__": "us-east-1",
                "__bedrock_converse__": True,
            },
        )
        new_writer_token = agent._stream_writer_token

        resume_old.set()
        time.sleep(0.1)

    assert old_error and "wall_clock_budget_reached" in str(old_error[0])
    assert new_response.stop_reason == "end_turn"
    assert delivered == ["new"]
    assert agent._stream_writer_token == new_writer_token
