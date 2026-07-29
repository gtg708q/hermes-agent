"""Runtime tests for tool-call loop guardrails."""

import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.errors import TurnWallClockExceeded
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_repeated_tool_error_limit_is_an_explicit_failed_turn():
    agent = _make_agent(
        "web_search",
        config={"agent": {"repeated_tool_error_limit": 2}},
    )
    tool_call_1 = _mock_tool_call("web_search", '{"query":"same"}', "c-error-1")
    tool_call_2 = _mock_tool_call("web_search", '{"query":"same"}', "c-error-2")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call_1]),
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call_2]),
    ]

    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"success": False, "error": "same failure"}),
    ):
        result = agent.run_conversation("keep trying")

    assert result["turn_exit_reason"] == "repeated_tool_error_limit_reached"
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["api_calls"] == 2


def test_api_deadline_is_an_explicit_failed_turn():
    agent = _make_agent(config={"agent": {"max_wall_clock_seconds": 30}})

    with patch(
        "hermes_cli.middleware.run_llm_execution_middleware",
        side_effect=TurnWallClockExceeded("wall_clock_budget_reached"),
    ):
        result = agent.run_conversation("wait for the provider")

    assert result["turn_exit_reason"] == "wall_clock_budget_reached"
    assert result["completed"] is False
    assert result["failed"] is True


def test_preflight_deadline_closes_and_persists_the_assistant_turn(monkeypatch):
    agent = _make_agent(config={"agent": {"max_wall_clock_seconds": 0.5}})
    persisted = []
    agent._persist_session = lambda messages, history: persisted.append(
        [dict(message) for message in messages]
    )
    monotonic_values = iter([0.0])

    def elapsed_after_turn_start():
        return next(monotonic_values, 1.0)

    monkeypatch.setattr("agent.conversation_loop.time.monotonic", elapsed_after_turn_start)

    result = agent.run_conversation("do not replay me")

    assert result["turn_exit_reason"] == "wall_clock_budget_reached"
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": result["final_response"],
    }
    assert persisted[-1][-1] == result["messages"][-1]
    agent.client.chat.completions.create.assert_not_called()


def test_sequential_blocking_tool_returns_at_whole_turn_deadline_without_late_output():
    agent = _make_agent("web_search")
    release = threading.Event()
    completed = []
    agent._turn_deadline_monotonic = time.monotonic() + 0.1
    agent.tool_progress_callback = lambda event, *_a, **_k: completed.append(event)
    tc = _mock_tool_call("web_search", '{"query":"slow"}', "c-deadline")
    messages = []

    def _blocking_call(*_args, **_kwargs):
        release.wait(timeout=2)
        return "late-result-must-not-land"

    started = time.monotonic()
    with patch("run_agent.handle_function_call", side_effect=_blocking_call):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[tc]), messages, "task-1"
        )
        elapsed = time.monotonic() - started
        snapshot = list(messages)
        release.set()
        time.sleep(0.1)

    assert elapsed < 0.5
    assert agent._wall_clock_expired is True
    assert messages == snapshot
    assert len(messages) == 1
    assert "wall-clock budget" in messages[0]["content"]
    assert "late-result-must-not-land" not in str(messages)
    assert "tool.completed" not in completed


def test_sequential_deadline_worker_receives_interrupt_and_clears_targeted_bit():
    from tools.interrupt import _interrupted_threads, _lock, is_interrupted

    agent = _make_agent("web_search")
    started = threading.Event()
    agent._turn_deadline_monotonic = time.monotonic() + 2.0
    tc = _mock_tool_call("web_search", '{"query":"interruptible"}', "c-interrupt")
    messages = []

    def _cooperative_call(*_args, **_kwargs):
        started.set()
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not is_interrupted():
            time.sleep(0.01)
        return "interrupted" if is_interrupted() else "missed-interrupt"

    def _interrupt():
        assert started.wait(timeout=1)
        agent.interrupt("hard stop")

    interrupter = threading.Thread(target=_interrupt)
    interrupter.start()
    began = time.monotonic()
    with patch("run_agent.handle_function_call", side_effect=_cooperative_call):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[tc]), messages, "task-1"
        )
    elapsed = time.monotonic() - began
    interrupter.join(timeout=1)

    assert elapsed < 0.5
    assert agent._tool_worker_threads == set()
    with _lock:
        assert _interrupted_threads == set()


def test_sequential_deadline_hard_stop_does_not_wait_for_noncooperative_tool():
    from tools.interrupt import _interrupted_threads, _lock

    agent = _make_agent("web_search")
    started = threading.Event()
    release = threading.Event()
    worker_exited = threading.Event()
    agent._turn_deadline_monotonic = time.monotonic() + 2.0
    tc = _mock_tool_call("web_search", '{"query":"noncooperative"}', "c-stop")
    messages = []

    def _blocking_call(*_args, **_kwargs):
        started.set()
        release.wait(timeout=3)
        worker_exited.set()
        return "late-result"

    def _interrupt():
        assert started.wait(timeout=1)
        agent.interrupt("hard stop")

    interrupter = threading.Thread(target=_interrupt)
    interrupter.start()
    began = time.monotonic()
    try:
        with patch("run_agent.handle_function_call", side_effect=_blocking_call):
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[tc]), messages, "task-1"
            )
        elapsed = time.monotonic() - began
        assert elapsed < 0.5
        assert messages[0]["effect_disposition"] == "unknown"
        assert "interrupt" in messages[0]["content"].lower()
    finally:
        release.set()
        interrupter.join(timeout=1)
        assert worker_exited.wait(timeout=1)

    deadline = time.monotonic() + 1
    while agent._tool_worker_threads and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent._tool_worker_threads == set()
    with _lock:
        assert _interrupted_threads == set()


def test_sequential_deadline_fence_does_not_wait_for_blocked_persistence():
    agent = _make_agent("web_search")
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    tc = _mock_tool_call("web_search", '{"query":"fast"}', "c-persist")
    messages = []

    def _blocking_flush(_messages):
        persistence_started.set()
        release_persistence.wait(timeout=2)
        return True

    agent._flush_messages_to_session_db = _blocking_flush
    timer = threading.Timer(0.6, release_persistence.set)
    timer.start()
    began = time.monotonic()
    try:
        with patch("run_agent.handle_function_call", return_value="fast-result"):
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[tc]), messages, "task-1"
            )
        elapsed = time.monotonic() - began
    finally:
        release_persistence.set()
        timer.cancel()

    assert persistence_started.is_set()
    assert elapsed < 0.3
    assert agent._wall_clock_expired is True


def test_sequential_deadline_preserves_and_persists_completed_prefix_in_order():
    agent = _make_agent("web_search")
    release = threading.Event()
    persisted = []
    agent._turn_deadline_monotonic = time.monotonic() + 0.1
    agent._flush_messages_to_session_db = lambda current: (
        persisted.append([dict(item) for item in current]) or True
    )
    calls = [
        _mock_tool_call("web_search", '{"query":"first"}', "c-first"),
        _mock_tool_call("web_search", '{"query":"blocking"}', "c-blocking"),
        _mock_tool_call("web_search", '{"query":"never"}', "c-never"),
    ]
    messages = []

    def _invoke(_name, args, *_rest, **_kwargs):
        if args["query"] == "first":
            return "first-success"
        release.wait(timeout=2)
        return "late-result-must-not-land"

    try:
        with patch("run_agent.handle_function_call", side_effect=_invoke):
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
            )

        assert [item["tool_call_id"] for item in messages] == [
            "c-first", "c-blocking", "c-never"
        ]
        assert messages[0]["content"] == "first-success"
        assert "effect_disposition" not in messages[0]
        assert messages[1]["effect_disposition"] == "unknown"
        assert messages[2]["effect_disposition"] == "none"
        persist_deadline = time.monotonic() + 1
        while (not persisted or persisted[-1] != messages) and time.monotonic() < persist_deadline:
            time.sleep(0.01)
        assert persisted[-1] == messages
    finally:
        release.set()


def test_sequential_deadline_marks_calls_not_started_during_inter_tool_delay_none():
    agent = _make_agent("web_search")
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    agent.tool_delay = 1
    calls = [
        _mock_tool_call("web_search", '{"query":"first"}', "c-delay-first"),
        _mock_tool_call("web_search", '{"query":"second"}', "c-delay-second"),
    ]
    messages = []

    with patch("run_agent.handle_function_call", return_value="completed-before-delay"):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert messages[0]["tool_call_id"] == "c-delay-first"
    assert messages[1]["tool_call_id"] == "c-delay-second"
    assert messages[1]["effect_disposition"] == "none"
    assert "was not started" in messages[1]["content"]


def test_sequential_tool_start_callback_cannot_dispatch_after_deadline():
    agent = _make_agent("web_search")
    callback_entered = threading.Event()
    release_callback = threading.Event()
    callback_finished = threading.Event()
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    call = _mock_tool_call("web_search", '{"query":"callback-race"}', "c-callback")
    messages = []

    def _blocking_start_callback(*_args, **_kwargs):
        callback_entered.set()
        release_callback.wait(timeout=2)
        callback_finished.set()

    agent.tool_start_callback = _blocking_start_callback
    with patch("run_agent.handle_function_call", return_value="must-not-run") as invoke:
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[call]), messages, "task-1"
        )
        assert callback_entered.is_set()
        assert messages[0]["effect_disposition"] == "none"
        release_callback.set()
        assert callback_finished.wait(timeout=1)
        time.sleep(0.05)

    invoke.assert_not_called()
    assert "was not started" in messages[0]["content"]


def test_concurrent_blocking_tools_do_not_publish_late_results_after_deadline():
    agent = _make_agent("web_search")
    release = threading.Event()
    agent._turn_deadline_monotonic = time.monotonic() + 0.1
    calls = [
        _mock_tool_call("web_search", '{"query":"one"}', "c-one"),
        _mock_tool_call("web_search", '{"query":"two"}', "c-two"),
    ]
    messages = []

    def _blocking_invoke(*_args, **_kwargs):
        release.wait(timeout=2)
        return "late-concurrent-result"

    started = time.monotonic()
    with patch.object(agent, "_invoke_tool", side_effect=_blocking_invoke):
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )
        elapsed = time.monotonic() - started
        snapshot = list(messages)
        release.set()
        time.sleep(0.1)

    assert elapsed < 0.5
    assert messages == snapshot
    assert len(messages) == 2
    assert all("timed out" in item["content"] for item in messages)
    assert "late-concurrent-result" not in str(messages)


def test_concurrent_tool_start_callback_cannot_submit_after_deadline():
    agent = _make_agent("web_search")
    agent._turn_deadline_monotonic = time.monotonic() + 0.03
    call = _mock_tool_call("web_search", '{"query":"callback-race"}', "c-callback")
    messages = []

    def _blocking_start_callback(*_args, **_kwargs):
        time.sleep(0.08)

    agent.tool_start_callback = _blocking_start_callback
    with patch.object(agent, "_invoke_tool", return_value="must-not-run") as invoke:
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=[call]), messages, "task-1"
        )

    invoke.assert_not_called()
    assert len(messages) == 1
    assert messages[0]["effect_disposition"] == "none"
    assert "was not started" in messages[0]["content"]


def test_concurrent_preflight_middleware_is_bounded_by_turn_deadline():
    agent = _make_agent("web_search")
    entered = threading.Event()
    release = threading.Event()
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    calls = [
        _mock_tool_call("web_search", '{"query":"one"}', "c-preflight-one"),
        _mock_tool_call("web_search", '{"query":"two"}', "c-preflight-two"),
    ]
    messages = []

    def _blocking_middleware(_agent, **kwargs):
        entered.set()
        release.wait(timeout=2)
        return kwargs["function_args"], []

    timer = threading.Timer(0.6, release.set)
    timer.start()
    began = time.monotonic()
    try:
        with (
            patch(
                "agent.tool_executor._apply_tool_request_middleware_for_agent",
                side_effect=_blocking_middleware,
            ),
            patch.object(agent, "_invoke_tool", return_value="must-not-run") as invoke,
        ):
            agent._execute_tool_calls_concurrent(
                SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
            )
        elapsed = time.monotonic() - began
    finally:
        release.set()
        timer.cancel()

    assert entered.is_set()
    assert elapsed < 0.3
    invoke.assert_not_called()
    assert [message["tool_call_id"] for message in messages] == [
        "c-preflight-one",
        "c-preflight-two",
    ]
    assert all(message["effect_disposition"] == "none" for message in messages)


def test_concurrent_result_finishing_during_deadline_fence_stays_unknown():
    agent = _make_agent("web_search")
    release = threading.Event()
    worker_finished = threading.Event()
    completed = []
    agent._turn_deadline_monotonic = time.monotonic() + 0.05
    agent.tool_complete_callback = lambda *_a, **_k: completed.append("complete")
    call = _mock_tool_call("web_search", '{"query":"race"}', "c-race")
    messages = []

    def _invoke(*_args, **_kwargs):
        release.wait(timeout=2)
        worker_finished.set()
        return "late-race-result"

    def _release_at_fence(*_args, **_kwargs):
        release.set()
        worker_finished.wait(timeout=1)

    with (
        patch.object(agent, "_invoke_tool", side_effect=_invoke),
        patch("run_agent._set_interrupt", side_effect=_release_at_fence),
    ):
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=[call]), messages, "task-1"
        )

    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "c-race"
    assert messages[0]["effect_disposition"] == "unknown"
    assert "timed out" in messages[0]["content"]
    assert "late-race-result" not in messages[0]["content"]
    assert completed == []


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)


def test_config_enabled_hard_stop_run_conversation_returns_controlled_guardrail_halt_without_top_level_error():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 2
    assert result["api_calls"] == 3
    assert result["api_calls"] < agent.max_iterations
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "error" not in result
    assert result["completed"] is True
    assert "stopped retrying" in result["final_response"]
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"
    assert result["guardrail"]["tool_name"] == "web_search"

    assistant_tool_calls = [m for m in result["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    for assistant_msg in assistant_tool_calls:
        call_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        following_results = [m for m in result["messages"] if m.get("role") == "tool" and m.get("tool_call_id") in call_ids]
        assert len(following_results) == len(call_ids)


def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )
