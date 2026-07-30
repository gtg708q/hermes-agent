import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import conversation_loop
from agent.tool_executor import (
    _apply_tool_transport_deadline,
    _tool_execution_signature,
)


def test_terminal_deadline_preserves_omitted_default_until_it_is_stricter():
    agent = SimpleNamespace(_turn_deadline_monotonic=time.monotonic() + 900)
    assert _apply_tool_transport_deadline(agent, "terminal", {"command": "true"}) == {
        "command": "true"
    }

    agent._turn_deadline_monotonic = time.monotonic() + 0.5
    bounded = _apply_tool_transport_deadline(agent, "terminal", {"command": "true"})
    assert 0 < bounded["timeout"] <= 0.5


def test_terminal_deadline_preserves_explicit_timeout_above_stock_maximum():
    agent = SimpleNamespace(_turn_deadline_monotonic=time.monotonic() + 900)
    bounded = _apply_tool_transport_deadline(
        agent, "terminal", {"command": "true", "timeout": 800}
    )
    assert bounded["timeout"] == 800


def test_repeated_error_signature_includes_tool_identity_and_normalized_args():
    same_result = "Error: unavailable"
    first = _tool_execution_signature("web_search", {"query": "a", "limit": 5}, same_result)
    reordered = _tool_execution_signature("web_search", {"limit": 5, "query": "a"}, same_result)
    other_args = _tool_execution_signature("web_search", {"query": "b", "limit": 5}, same_result)
    other_tool = _tool_execution_signature("web_extract", {"query": "a", "limit": 5}, same_result)

    assert first == reordered
    assert first != other_args
    assert first != other_tool


def test_repeated_error_guard_does_not_merge_different_tool_calls():
    agent = SimpleNamespace(
        max_wall_clock_seconds=0,
        repeated_tool_error_limit=2,
        no_progress_tool_limit=0,
    )
    content = "Error: unavailable"
    messages = [
        {"role": "user", "content": "try"},
        {
            "role": "tool",
            "content": content,
            "_tool_execution_status": "error",
            "_tool_execution_signature": _tool_execution_signature(
                "web_search", {"query": "a"}, content
            ),
        },
        {
            "role": "tool",
            "content": content,
            "_tool_execution_status": "error",
            "_tool_execution_signature": _tool_execution_signature(
                "web_extract", {"url": "https://example.com"}, content
            ),
        },
    ]
    assert conversation_loop._loop_guard_reason(agent, messages, time.monotonic()) is None


def test_whole_turn_deadline_bounds_blocked_finalization_and_fences_next_turn(monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def _blocked_inner(agent, *args, **kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"final_response": "late", "messages": []}

    monkeypatch.setattr(conversation_loop, "_run_conversation_inner", _blocked_inner)
    agent = MagicMock()
    agent.max_wall_clock_seconds = 0.05
    agent._interrupt_requested = False

    started = time.monotonic()
    result = conversation_loop.run_conversation(agent, "hello")
    elapsed = time.monotonic() - started

    assert entered.is_set()
    assert elapsed < 0.5
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "wall_clock_budget_reached"
    assert "wall_clock_budget_reached" in result["final_response"]

    second = conversation_loop.run_conversation(agent, "must not overlap")
    assert second["failed"] is True
    assert "previous timed-out turn is still finalizing" in second["final_response"]

    release.set()
    worker = agent._deadline_turn_worker
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_whole_turn_deadline_rejects_result_finished_during_handoff(monkeypatch):
    def _late_inner(agent, *args, **kwargs):
        time.sleep(0.03)
        return {"final_response": "late success", "messages": []}

    monkeypatch.setattr(conversation_loop, "_run_conversation_inner", _late_inner)
    agent = MagicMock()
    agent.max_wall_clock_seconds = 0.01
    agent._interrupt_requested = False

    result = conversation_loop.run_conversation(agent, "hello")

    assert result["failed"] is True
    assert result["turn_exit_reason"] == "wall_clock_budget_reached"
    assert result["final_response"] != "late success"
