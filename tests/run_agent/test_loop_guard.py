from types import SimpleNamespace

from agent.conversation_loop import _loop_guard_reason


def test_loop_guard_stops_after_wall_clock_budget(monkeypatch):
    agent = SimpleNamespace(
        max_wall_clock_seconds=30,
        repeated_tool_error_limit=0,
        no_progress_tool_limit=0,
    )
    monkeypatch.setattr("agent.conversation_loop.time.monotonic", lambda: 31)

    assert _loop_guard_reason(agent, [], 0) == "wall_clock_budget_reached"


def test_loop_guard_stops_repeated_identical_tool_errors():
    agent = SimpleNamespace(
        max_wall_clock_seconds=0,
        repeated_tool_error_limit=3,
        no_progress_tool_limit=0,
    )
    messages = []
    for index in range(3):
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call-{index}",
                    "function": {"name": "terminal", "arguments": "{}"},
                }],
            },
            {"role": "tool", "content": "ERROR: same deterministic failure"},
        ])

    assert (
        _loop_guard_reason(agent, messages, 0)
        == "repeated_tool_error_limit_reached"
    )


def test_loop_guard_stops_repeated_tool_calls_with_new_provider_ids():
    agent = SimpleNamespace(
        max_wall_clock_seconds=0,
        repeated_tool_error_limit=0,
        no_progress_tool_limit=3,
    )
    messages = []
    for index in range(3):
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call-{index}",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"retry unchanged"}',
                    },
                }],
            },
            {"role": "tool", "content": f"attempt {index}"},
        ])

    assert _loop_guard_reason(agent, messages, 0) == "no_progress_tool_limit_reached"


def test_loop_guard_defaults_to_disabled():
    assert _loop_guard_reason(SimpleNamespace(), [], 0) is None
