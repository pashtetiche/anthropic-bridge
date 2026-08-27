import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from anthropic_bridge.logging import (
    BridgeAccessFormatter,
    extract_client_reasoning_summary,
    pop_request_log,
    record_request_log,
    update_last_request_log,
)
from anthropic_bridge.providers.openrouter.client import OpenRouterProvider
from anthropic_bridge.providers.openrouter.mandatory_reasoning import (
    clear_mandatory_reasoning_models,
    is_mandatory_reasoning_error,
    is_mandatory_reasoning_model,
    register_mandatory_reasoning_model,
)

from .conftest import collect_events


@pytest.fixture(autouse=True)
def cleanup_mandatory_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_mandatory_reasoning_models()
    yield
    clear_mandatory_reasoning_models()


def test_mandatory_reasoning_env_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OPENROUTER_MANDATORY_REASONING_MODELS",
        "@preset/glm-5-3-flash, custom/model-x",
    )

    assert is_mandatory_reasoning_model("@preset/glm-5-3-flash")
    assert is_mandatory_reasoning_model("glm-5-3-flash")
    assert is_mandatory_reasoning_model("openrouter/@preset/glm-5-3-flash")
    assert is_mandatory_reasoning_model("openrouter/glm-5-3-flash")
    assert is_mandatory_reasoning_model("custom/model-x")
    assert not is_mandatory_reasoning_model("openai/gpt-5.2")


def test_mandatory_reasoning_json_env_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OPENROUTER_MANDATORY_REASONING_MODELS",
        json.dumps(["zhipu/glm-5-3-flash"]),
    )

    assert is_mandatory_reasoning_model("zhipu/glm-5-3-flash")
    assert is_mandatory_reasoning_model("glm-5-3-flash")
    assert is_mandatory_reasoning_model("@preset/glm-5-3-flash")


def test_register_mandatory_reasoning_model(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not is_mandatory_reasoning_model("@preset/glm-5-3-flash")

    register_mandatory_reasoning_model("@preset/glm-5-3-flash")

    assert is_mandatory_reasoning_model("@preset/glm-5-3-flash")
    assert is_mandatory_reasoning_model("glm-5-3-flash")

    import os
    env_val = os.environ.get("OPENROUTER_MANDATORY_REASONING_MODELS", "")
    assert "glm-5-3-flash" in env_val


def test_is_mandatory_reasoning_error() -> None:
    raw_error = (
        '{"error":{"message":"Reasoning is mandatory for this endpoint '
        'and cannot be disabled.","code":400,"metadata":{"provider_name":null}}}'
    )
    assert is_mandatory_reasoning_error(raw_error)
    assert is_mandatory_reasoning_error({"message": "Reasoning is mandatory for this endpoint and cannot be disabled."})
    assert is_mandatory_reasoning_error("reasoning is mandatory")
    assert not is_mandatory_reasoning_error("Rate limit exceeded")
    assert not is_mandatory_reasoning_error({"message": "Model not found"})


@pytest.mark.asyncio
async def test_mandatory_reasoning_forces_enabled_when_client_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_mandatory_reasoning_model("@preset/glm-5-3-flash")

    captured_payload: dict[str, Any] = {}

    async def fake_stream(body: dict[str, Any]) -> AsyncIterator[str]:
        nonlocal captured_payload
        captured_payload = body
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    provider = OpenRouterProvider("openrouter/@preset/glm-5-3-flash", "token")
    monkeypatch.setattr(provider, "_stream_openrouter", fake_stream)

    # 1. Empty payload (normally reasoning: {enabled: False})
    await collect_events(
        provider.handle(
            {
                "model": "openrouter/@preset/glm-5-3-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
    )
    assert captured_payload.get("reasoning") == {"effort": "low"}

    # 2. Explicit thinking disabled
    await collect_events(
        provider.handle(
            {
                "model": "openrouter/@preset/glm-5-3-flash",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "disabled"},
            }
        )
    )
    assert captured_payload.get("reasoning") == {"effort": "low"}

    # 3. Explicit effort none
    await collect_events(
        provider.handle(
            {
                "model": "openrouter/@preset/glm-5-3-flash",
                "messages": [{"role": "user", "content": "Hi"}],
                "output_config": {"effort": "none"},
            }
        )
    )
    assert captured_payload.get("reasoning") == {"effort": "low"}

    # 4. Explicit effort high should be preserved
    await collect_events(
        provider.handle(
            {
                "model": "openrouter/@preset/glm-5-3-flash",
                "messages": [{"role": "user", "content": "Hi"}],
                "output_config": {"effort": "high"},
            }
        )
    )
    assert captured_payload.get("reasoning") == {"effort": "high"}


@pytest.mark.asyncio
async def test_stream_openrouter_auto_recovers_from_400_mandatory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    captured_payloads: list[dict[str, Any]] = []

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
            import copy
            captured_payloads.append(copy.deepcopy(kwargs.get("json", {})))
            if attempts == 1:
                return FakeResponse(
                    400,
                    [],
                    b'{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400}}',
                )
            return FakeResponse(
                200,
                [
                    'data: {"choices":[{"index":0,"delta":{"content":"Success","role":"assistant"},"finish_reason":"stop"}]}\n\n',
                    "data: [DONE]\n\n",
                ],
            )

    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        FakeClient,
    )

    record_request_log(
        client_addr="127.0.0.1:58560",
        model="openrouter/@preset/glm-5-3-flash",
        client_reasoning="none",
        bridge_reasoning="enabled=False",
    )

    provider = OpenRouterProvider("openrouter/@preset/glm-5-3-flash", "token")
    events = await collect_events(
        provider.handle(
            {
                "model": "openrouter/@preset/glm-5-3-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
    )

    # Auto-recovery succeeded
    assert attempts == 2
    assert is_mandatory_reasoning_model("@preset/glm-5-3-flash")
    assert captured_payloads[0].get("reasoning") == {"enabled": False}
    assert captured_payloads[1].get("reasoning") == {"effort": "low"}

    assert any(
        event == "content_block_delta"
        and data["delta"].get("type") == "text_delta"
        and data["delta"].get("text") == "Success"
        for event, data in events
    )

    # Verify log entry was updated to effort=low (forced)
    log_details = pop_request_log("127.0.0.1:58560")
    assert "bridge reasoning: effort=low (forced)" in log_details


def test_access_logger_formats_model_and_reasoning() -> None:
    formatter = BridgeAccessFormatter(
        '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )

    client_addr = "127.0.0.1:40332"
    record_request_log(
        client_addr=client_addr,
        model="@preset/glm-5-3-flash",
        client_reasoning="none",
        bridge_reasoning="effort=low (forced)",
    )

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        (client_addr, "POST", "/v1/messages?beta=true", "1.1", 200),
        None,
    )

    formatted = formatter.format(record)
    assert (
        formatted
        == 'INFO:     127.0.0.1:40332 - "POST /v1/messages?beta=true HTTP/1.1" 200 OK | '
        "model: @preset/glm-5-3-flash | client reasoning: none | bridge reasoning: effort=low (forced)"
    )


def test_update_last_request_log() -> None:
    record_request_log(
        client_addr="1.2.3.4:1234",
        model="test-model",
        client_reasoning="none",
        bridge_reasoning="enabled=False",
    )
    update_last_request_log("effort=low (forced)", client_addr="1.2.3.4:1234")
    log = pop_request_log("1.2.3.4:1234")
    assert log == "model: test-model | client reasoning: none | bridge reasoning: effort=low (forced)"


def test_extract_client_reasoning_summary() -> None:
    assert extract_client_reasoning_summary({}) == "none"
    assert extract_client_reasoning_summary({"thinking": {"type": "disabled"}}) == "disabled"
    assert extract_client_reasoning_summary({"thinking": {"type": "enabled", "budget_tokens": 5000}}) == "budget=5000"
    assert extract_client_reasoning_summary({"thinking": {"type": "adaptive"}}) == "adaptive"
    assert extract_client_reasoning_summary({"output_config": {"effort": "high"}}) == "effort=high"
