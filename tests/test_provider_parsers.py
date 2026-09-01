from collections.abc import AsyncIterator
from typing import Any

import pytest

from anthropic_bridge.providers.copilot.client import CopilotProvider
from anthropic_bridge.providers.openai.client import OpenAIProvider
from anthropic_bridge.providers.openrouter.client import OpenRouterProvider
from anthropic_bridge.providers.responses_api import stream_responses_api
from anthropic_bridge.providers.utils import classify_upstream_error, is_transient_error

from .conftest import collect_events, fake_client_factory


@pytest.mark.asyncio
async def test_openrouter_parser_skips_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        ": OPENROUTER PROCESSING\n\n",
        'data: {"choices":[]}\n\n',
        'data: {"usage":{"prompt_tokens":3}}\n\n',
        (
            'data: {"choices":[{"index":0,"delta":{"content":"Hi","role":"assistant"},'
            '"finish_reason":null}]}\n\n'
        ),
        (
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
        ),
        "data: [DONE]\n\n",
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    provider = OpenRouterProvider("openrouter/google/gemini-3-pro-preview", "token")
    events = await collect_events(
        provider._stream_openrouter(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
        )
    )

    assert ("message_stop", {"type": "message_stop"}) in events
    assert not any(event == "error" for event, _ in events)
    assert any(
        event == "content_block_delta"
        and data["delta"].get("type") == "text_delta"
        and data["delta"].get("text") == "Hi"
        for event, data in events
    )


@pytest.mark.asyncio
async def test_copilot_parser_skips_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        "event: message\n",
        'data: {"choices":[]}\n',
        'data: {"usage":{"prompt_tokens":2}}\n',
        (
            'data: {"choices":[{"index":0,"delta":{"content":"Hi","role":"assistant"},'
            '"finish_reason":null}]}\n'
        ),
        (
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n'
        ),
        "data: [DONE]\n",
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.copilot.client.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    provider = CopilotProvider("copilot/claude-opus-4.6", token="token")
    events = await collect_events(
        provider._stream_chat(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
            "token",
        )
    )

    assert ("message_stop", {"type": "message_stop"}) in events
    assert not any(event == "error" for event, _ in events)
    assert any(
        event == "content_block_delta"
        and data["delta"].get("type") == "text_delta"
        and data["delta"].get("text") == "Hi"
        for event, data in events
    )


@pytest.mark.asyncio
async def test_openai_provider_omits_instructions_without_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_get_auth(*args: Any) -> tuple[str, None, float]:
        return "token", None, 0

    async def fake_stream(
        endpoint: str,
        headers: dict[str, str],
        request_body: dict[str, Any],
        target_model: str,
    ) -> AsyncIterator[str]:
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        captured["body"] = request_body
        captured["target_model"] = target_model
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    monkeypatch.setattr(
        "anthropic_bridge.providers.openai.client.get_auth",
        fake_get_auth,
    )
    monkeypatch.setattr(
        "anthropic_bridge.providers.openai.client.stream_responses_api",
        fake_stream,
    )

    provider = OpenAIProvider("openai/gpt-5.2")
    events = await collect_events(
        provider.handle({"messages": [{"role": "user", "content": "Hi"}]})
    )

    assert events == [("message_stop", {"type": "message_stop"})]
    assert "instructions" not in captured["body"]
    assert captured["body"]["input"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_openrouter_empty_stream_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        fake_client_factory([": OPENROUTER PROCESSING\n\n", "data: [DONE]\n\n"]),
    )

    provider = OpenRouterProvider("openrouter/google/gemini-3-pro-preview", "token")
    events = await collect_events(
        provider._stream_openrouter(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
        )
    )

    assert any(event == "error" for event, _ in events)


@pytest.mark.asyncio
async def test_responses_api_ignores_duplicate_reasoning_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        "event: response.reasoning_summary_text.delta",
        'data: {"delta":"Plan"}',
        "event: response.reasoning.delta",
        'data: {"delta":"Plan"}',
        "event: response.output_text.delta",
        'data: {"delta":"Answer"}',
        "event: response.completed",
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}',
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.responses_api.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    events = await collect_events(
        stream_responses_api(
            "https://example.test/responses",
            {},
            {
                "input": [{"role": "user", "content": "Hi"}],
                "reasoning": {"effort": "low", "summary": "auto"},
            },
            "gpt-5.2",
        )
    )

    thinking_deltas = [
        data["delta"]["thinking"]
        for event, data in events
        if event == "content_block_delta"
        and data["delta"].get("type") == "thinking_delta"
    ]

    assert thinking_deltas == ["Plan"]


@pytest.mark.asyncio
async def test_responses_api_does_not_repeat_tool_args_on_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        "event: response.output_item.added",
        'data: {"item":{"type":"function_call","call_id":"call_1","name":"calc"}}',
        "event: response.function_call_arguments.delta",
        'data: {"call_id":"call_1","delta":"{\\"x\\":1}"}',
        "event: response.output_item.done",
        'data: {"item":{"type":"function_call","call_id":"call_1","arguments":"{\\"x\\":1}"}}',
        "event: response.completed",
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}',
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.responses_api.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    events = await collect_events(
        stream_responses_api(
            "https://example.test/responses",
            {},
            {"input": [{"role": "user", "content": "Hi"}]},
            "gpt-5.2",
        )
    )

    arg_deltas = [
        data["delta"]["partial_json"]
        for event, data in events
        if event == "content_block_delta"
        and data["delta"].get("type") == "input_json_delta"
    ]

    assert arg_deltas == ['{"x":1}']


def test_openrouter_inject_gemini_reasoning_deduplicates_cached_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCache:
        def get(self, tool_call_id: str) -> list[dict[str, Any]] | None:
            mapping = {
                "tool_a": [{"id": "r1", "type": "reasoning"}],
                "tool_b": [{"id": "r1", "type": "reasoning"}],
            }
            return mapping.get(tool_call_id)

    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.get_reasoning_cache",
        lambda: FakeCache(),
    )

    provider = OpenRouterProvider("openrouter/google/gemini-3-pro-preview", "token")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "tool_a", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tool_b", "function": {"name": "b", "arguments": "{}"}},
            ],
        }
    ]

    provider._inject_gemini_reasoning(messages)

    assert messages[0]["reasoning_details"] == [{"id": "r1", "type": "reasoning"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("anthropic_payload", "expected_reasoning"),
    [
        (
            {"output_config": {"effort": "max"}},
            {"effort": "max"},
        ),
        (
            {"output_config": {"effort": "xhigh"}},
            {"effort": "xhigh"},
        ),
        (
            {"output_config": {"effort": "high"}},
            {"effort": "high"},
        ),
        (
            {"output_config": {"effort": "medium"}},
            {"effort": "medium"},
        ),
        (
            {"output_config": {"effort": "low"}},
            {"effort": "low"},
        ),
        (
            {"output_config": {"effort": "minimal"}},
            {"effort": "minimal"},
        ),
        (
            {"output_config": {"effort": "none"}},
            {"effort": "none"},
        ),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 5000}},
            {"max_tokens": 5000},
        ),
        (
            {"thinking": {"type": "adaptive"}},
            {"enabled": True},
        ),
        (
            {},
            {"enabled": False},
        ),
    ],
)
async def test_openrouter_reasoning_mapping(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_payload: dict[str, Any],
    expected_reasoning: dict[str, Any],
) -> None:
    captured_body: dict[str, Any] = {}

    async def fake_stream(body: dict[str, Any]) -> AsyncIterator[str]:
        nonlocal captured_body
        captured_body = body
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    provider = OpenRouterProvider("openrouter/openai/gpt-5.2", "token")
    monkeypatch.setattr(provider, "_stream_openrouter", fake_stream)

    payload = {
        "model": "openrouter/openai/gpt-5.2",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        **anthropic_payload,
    }
    await collect_events(provider.handle(payload))

    assert captured_body.get("reasoning") == expected_reasoning


def test_classify_upstream_error_and_is_transient() -> None:
    assert is_transient_error(429, "Rate limit exceeded")
    assert is_transient_error(503, "Service unavailable")
    assert is_transient_error(200, "qwen/qwen3.8-flash is temporarily rate-limited upstream")
    assert not is_transient_error(400, "Bad Request")
    assert not is_transient_error(401, "Unauthorized")

    err_type, msg = classify_upstream_error(429, '{"error":{"message":"Rate limit","code":429}}')
    assert err_type == "rate_limit_error"
    assert msg == "Rate limit"

    err_type, msg = classify_upstream_error(
        429,
        '{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"temporarily rate-limited"}}}',
    )
    assert err_type == "rate_limit_error"
    assert msg == "temporarily rate-limited"

    err_type, msg = classify_upstream_error(401, '{"error":{"message":"Invalid token"}}')
    assert err_type == "authentication_error"


@pytest.mark.asyncio
async def test_openrouter_recovers_from_transient_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class FakeResponse:
        def __init__(self, status_code: int, chunks: list[str], error_body: bytes = b""):
            self.status_code = status_code
            self._chunks = chunks
            self._error_body = error_body

        async def aread(self) -> bytes:
            return self._error_body

        async def aiter_text(self) -> AsyncIterator[str]:
            for chunk in self._chunks:
                yield chunk

        async def __aenter__(self) -> "FakeResponse":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def stream(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return FakeResponse(
                    429,
                    [],
                    b'{"error":{"message":"Temporarily rate-limited upstream","code":429}}',
                )
            return FakeResponse(
                200,
                [
                    'data: {"choices":[{"index":0,"delta":{"content":"Success after retry","role":"assistant"},"finish_reason":"stop"}]}\n\n',
                    "data: [DONE]\n\n",
                ],
            )

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.asyncio.sleep",
        noop_sleep,
    )

    provider = OpenRouterProvider("openrouter/qwen/qwen3.8-flash", "token")
    events = await collect_events(
        provider.handle(
            {
                "model": "openrouter/qwen/qwen3.8-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
    )

    assert attempts == 2
    assert ("message_stop", {"type": "message_stop"}) in events
    assert any(
        event == "content_block_delta"
        and data["delta"].get("text") == "Success after retry"
        for event, data in events
    )


@pytest.mark.asyncio
async def test_openrouter_exhausts_transient_429_returns_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class FakeResponse:
        def __init__(self, status_code: int, chunks: list[str], error_body: bytes = b""):
            self.status_code = status_code
            self._chunks = chunks
            self._error_body = error_body

        async def aread(self) -> bytes:
            return self._error_body

        async def aiter_text(self) -> AsyncIterator[str]:
            for chunk in self._chunks:
                yield chunk

        async def __aenter__(self) -> "FakeResponse":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def stream(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            nonlocal attempts
            attempts += 1
            return FakeResponse(
                429,
                [],
                b'{"error":{"message":"Rate limit exceeded","code":429}}',
            )

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.asyncio.sleep",
        noop_sleep,
    )

    provider = OpenRouterProvider("openrouter/qwen/qwen3.8-flash", "token")
    events = await collect_events(
        provider.handle(
            {
                "model": "openrouter/qwen/qwen3.8-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
    )

    # 1 initial + 2 transient retries = 3 attempts total
    assert attempts == 3
    error_events = [data for event, data in events if event == "error"]
    assert len(error_events) == 1
    assert error_events[0]["error"]["type"] == "rate_limit_error"
    assert error_events[0]["error"]["message"] == "Rate limit exceeded"
