"""Malformed model tool arguments are rejected at the dispatch boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _loop_guard_reason
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.tool_delay = 0
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _tool_call(call_id: str, arguments: str, name: str = "web_search"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _execute_malformed_calls(dispatch_mode: str, calls: list) -> list[dict]:
    agent = _make_agent()
    messages = []
    with patch(
        "agent.tool_executor.maybe_persist_tool_result",
        side_effect=lambda **kwargs: kwargs["content"],
    ):
        execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
        execute(SimpleNamespace(content="", tool_calls=calls), messages, "task-1")
    return messages


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
@pytest.mark.parametrize(
    "bad_arguments",
    [
        pytest.param("not-json", id="malformed-json"),
        pytest.param('"scalar"', id="scalar"),
        pytest.param("[]", id="list"),
        pytest.param("", id="empty"),
        pytest.param('{"query": "cut off', id="truncated"),
    ],
)
def test_malformed_arguments_are_rejected_without_blocking_valid_sibling(
    dispatch_mode: str,
    bad_arguments: str,
):
    agent = _make_agent()
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("call-bad", bad_arguments),
            _tool_call("call-good", '{"query": "valid"}'),
        ],
    )
    messages = []
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        call_id = kwargs.get("tool_call_id") or (positional[0] if positional else None)
        executed.append((name, args, call_id))
        return json.dumps({"ok": args["query"]})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_dispatch),
        patch.object(agent, "_invoke_tool", side_effect=fake_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
        execute(assistant_message, messages, "task-1")

    assert executed == [("web_search", {"query": "valid"}, "call-good")]
    assert [message["tool_call_id"] for message in messages] == ["call-bad", "call-good"]
    assert len([message for message in messages if message["tool_call_id"] == "call-bad"]) == 1

    assert '"error": "Invalid tool arguments"' in messages[0]["content"]
    assert "JSON object" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "valid"}


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_distinct_malformed_calls_have_distinct_loop_guard_signatures(dispatch_mode: str):
    messages = _execute_malformed_calls(
        dispatch_mode,
        [
            _tool_call("call-a", '{"query":"alpha"', "web_search"),
            _tool_call("call-b", '{"query":"beta"', "web_extract"),
        ],
    )
    agent = SimpleNamespace(
        max_wall_clock_seconds=0,
        repeated_tool_error_limit=2,
        no_progress_tool_limit=0,
    )

    assert messages[0]["_tool_execution_signature"] != messages[1]["_tool_execution_signature"]
    assert _loop_guard_reason(agent, messages, 0) is None


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_repeated_malformed_call_triggers_repeated_error_guard(dispatch_mode: str):
    raw_arguments = '{"query":"same"'
    messages = _execute_malformed_calls(
        dispatch_mode,
        [
            _tool_call("call-a", raw_arguments),
            _tool_call("call-b", raw_arguments),
        ],
    )
    agent = SimpleNamespace(
        max_wall_clock_seconds=0,
        repeated_tool_error_limit=2,
        no_progress_tool_limit=0,
    )

    assert messages[0]["_tool_execution_signature"] == messages[1]["_tool_execution_signature"]
    assert _loop_guard_reason(agent, messages, 0) == "repeated_tool_error_limit_reached"


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_malformed_argument_secret_is_only_used_in_opaque_signature(
    dispatch_mode: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    secret = "sk-malformed-secret-should-not-leak"
    first = _execute_malformed_calls(
        dispatch_mode,
        [_tool_call("call-secret", f'{{"api_key":"{secret}"')],
    )[0]
    second = _execute_malformed_calls(
        dispatch_mode,
        [_tool_call("call-other", '{"api_key":"different-secret"')],
    )[0]

    captured = capsys.readouterr()
    assert first["_tool_execution_signature"] != second["_tool_execution_signature"]
    assert secret not in json.dumps(first)
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
