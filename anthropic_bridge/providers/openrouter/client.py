import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...cache import get_reasoning_cache
from ...transform import (
    convert_anthropic_messages_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_anthropic_tools_to_openai,
)
from ..utils import (
    AnthropicSSEEmitter,
    estimate_input_tokens,
    first_choice,
    sse,
    yield_error_events,
)
from .registry import ProviderRegistry

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Mng-dev-ai/claudex",
    "X-Title": "Claudex",
}
PRESET_PREFIX = "@preset/"


class OpenRouterProvider:
    def __init__(self, target_model: str, api_key: str):
        self.target_model = target_model.removeprefix("openrouter/")
        self.capability_model = self.target_model.removeprefix(PRESET_PREFIX)
        self.api_key = api_key
        self.provider_registry = ProviderRegistry(self.capability_model)
        self._is_gemini = (
            "gemini" in self.capability_model.lower()
            or "google/" in self.capability_model.lower()
        )

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        try:
            self.provider_registry.reset()

            messages = self._convert_messages(payload)
            # print(
            #     f"[probe] MID={'ponytail' in json.dumps(messages, ensure_ascii=False).lower()}",
            #     file=sys.stderr, flush=True,
            # )
            tools = convert_anthropic_tools_to_openai(payload.get("tools"))

            openrouter_payload: dict[str, Any] = {
                "model": self.target_model,
                "messages": messages,
                "temperature": payload.get("temperature", 1),
                "stream": True,
                "max_tokens": payload.get("max_tokens", 16000),
                "stream_options": {"include_usage": True},
                "usage": {"include": True},
            }

            if tools:
                openrouter_payload["tools"] = tools
                tool_choice = convert_anthropic_tool_choice_to_openai(
                    payload.get("tool_choice")
                )
                if tool_choice:
                    openrouter_payload["tool_choice"] = tool_choice

            EFFORT_MAP = {
                "max": "max",
                "xhigh": "xhigh",
                "high": "high",
                "medium": "medium",
                "low": "low",
                "minimal": "minimal",
                "none": "none",
            }
            output_config = payload.get("output_config") or {}
            has_format = bool(output_config.get("format"))
            effort = output_config.get("effort")

            if has_format:
                pass  # structured-output: reasoning не навязываем, апстрим сам разберётся
            elif effort in EFFORT_MAP:
                openrouter_payload["reasoning"] = {"effort": EFFORT_MAP[effort]}
            elif payload.get("thinking"):
                thinking = payload["thinking"]
                if isinstance(thinking, dict) and thinking.get("budget_tokens"):
                    openrouter_payload["reasoning"] = {
                        "max_tokens": thinking["budget_tokens"]
                    }
                else:
                    openrouter_payload["reasoning"] = {"enabled": True}
            else:
                openrouter_payload["reasoning"] = {"enabled": False}

            self.provider_registry.prepare_request(openrouter_payload, payload)

            # _out = json.dumps(openrouter_payload, ensure_ascii=False).lower()
            # print(f"[probe] OUT={'ponytail' in _out}", file=sys.stderr, flush=True)

        except Exception as e:
            async for event in yield_error_events(str(e), self.target_model):
                yield event
            return

        async for event in self._stream_openrouter(openrouter_payload):
            yield event

    def _convert_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = convert_anthropic_messages_to_openai(
            payload.get("messages", []), payload.get("system")
        )

        if self._is_gemini:
            self._inject_gemini_reasoning(messages)

        if (
            "grok" in self.capability_model.lower()
            or "x-ai" in self.capability_model.lower()
        ):
            instruction = (
                "IMPORTANT: When calling tools, you MUST use the OpenAI tool_calls format with JSON. "
                "NEVER use XML format like <xai:function_call>."
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += "\n\n" + instruction
            else:
                messages.insert(0, {"role": "system", "content": instruction})

        return messages

    def _inject_gemini_reasoning(
        self,
        openai_messages: list[dict[str, Any]],
    ) -> None:
        cache = get_reasoning_cache()
        for msg in openai_messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue

            for tc in msg.get("tool_calls", []):
                tool_id = tc.get("id", "")
                cached = cache.get(tool_id)
                if cached:
                    if "reasoning_details" not in msg:
                        msg["reasoning_details"] = []
                    self._append_unique_reasoning_details(
                        msg["reasoning_details"], cached
                    )

    async def _stream_openrouter(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        estimated_input = await asyncio.to_thread(
            estimate_input_tokens,
            payload.get("messages", []),
            payload.get("tools"),
        )

        emitter = AnthropicSSEEmitter(self.target_model, estimated_input)
        for _e in emitter.message_start():
            yield _e

        usage: dict[str, Any] | None = None
        current_reasoning_details: list[dict[str, Any]] = []
        had_error = False

        t0 = time.monotonic()
        ttft: float | None = None
        actual_model = ""
        actual_provider = ""

        async with (
            httpx.AsyncClient(timeout=300.0) as client,
            client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    **OPENROUTER_HEADERS,
                },
                json=payload,
            ) as response,
        ):
            if response.status_code != 200:
                error_text = await response.aread()
                decoded = error_text.decode(errors="replace")
                print(
                    f"[{self.target_model}] HTTP {response.status_code} | "
                    f"{decoded[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
                for _e in emitter.error_and_finish(decoded):
                    yield _e
                return

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if not actual_model and data.get("model"):
                        actual_model = data["model"]
                        actual_provider = data.get("provider") or "?"

                    if data.get("error"):
                        had_error = True
                        error = data["error"]
                        if isinstance(error, dict):
                            message = error.get("message", "OpenRouter API error")
                        else:
                            message = str(error)
                        print(
                            f"[{self.target_model}] stream error | {message}",
                            file=sys.stderr,
                            flush=True,
                        )
                        yield sse(
                            "error",
                            {
                                "type": "error",
                                "error": {"type": "api_error", "message": message},
                            },
                        )
                        continue

                    if data.get("usage"):
                        usage = data["usage"]

                    choice = first_choice(data)
                    if choice is None:
                        continue

                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        delta = {}

                    if self._is_gemini and delta.get("reasoning_details"):
                        self._append_unique_reasoning_details(
                            current_reasoning_details, delta["reasoning_details"]
                        )

                    reasoning = delta.get("reasoning") or ""
                    content = delta.get("content") or ""

                    if ttft is None and (reasoning or content):
                        ttft = time.monotonic() - t0

                    if reasoning:
                        for _e in emitter.thinking_delta(reasoning):
                            yield _e
                    else:
                        # summary из reasoning_details только если плоского reasoning нет
                        # (openai дублирует контент в оба поля — избегаем двойного эмита)
                        for rd in (delta.get("reasoning_details") or []):
                            if rd.get("type") == "reasoning.summary":
                                summary_delta = rd.get("summary") or ""
                                if summary_delta:
                                    for _e in emitter.thinking_delta(summary_delta):
                                        yield _e

                    if content:
                        for _e in emitter.close_thinking():
                            yield _e

                        result = self.provider_registry.process_text_content(content, "")
                        clean_text = result.cleaned_text

                        if clean_text:
                            for _e in emitter.text_delta(clean_text):
                                yield _e

                        for tc in result.extracted_tool_calls:
                            for _e in emitter.close_text():
                                yield _e
                            for _e in emitter.add_tool(tc.id, tc.id, tc.name):
                                yield _e
                            for _e in emitter.tool_delta(tc.id, json.dumps(tc.arguments)):
                                yield _e
                            for _e in emitter.close_tool(tc.id):
                                yield _e
                            if self._is_gemini and current_reasoning_details:
                                get_reasoning_cache().set(
                                    tc.id, current_reasoning_details.copy()
                                )

                    tool_calls = delta.get("tool_calls", [])
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        if emitter.get_tool(idx) is None:
                            tool_id = tc.get("id") or f"tool_{idx}"
                            for _e in emitter.register_tool(idx, tool_id):
                                yield _e

                        fn = tc.get("function", {})
                        if fn.get("name"):
                            for _e in emitter.start_tool(idx, fn["name"]):
                                yield _e

                        tool_entry = emitter.get_tool(idx)
                        if fn.get("arguments") and tool_entry and tool_entry["started"]:
                            for _e in emitter.tool_delta(idx, fn["arguments"]):
                                yield _e

                    finish = choice.get("finish_reason")
                    if finish == "tool_calls":
                        for key in emitter.tool_keys:
                            for _e in emitter.close_tool(key):
                                yield _e
                            t = emitter.get_tool(key)
                            if t and self._is_gemini and current_reasoning_details:
                                get_reasoning_cache().set(
                                    t["id"], current_reasoning_details.copy()
                                )

        # Cache reasoning details for tools closed at stream end
        if self._is_gemini and current_reasoning_details:
            for key in emitter.tool_keys:
                t = emitter.get_tool(key)
                if t and not t["closed"]:
                    get_reasoning_cache().set(t["id"], current_reasoning_details.copy())

        if not emitter.had_content and not emitter.has_tools and not had_error:
            yield sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": (
                            f"No content received from OpenRouter API for model "
                            f"'{self.target_model}'. The provider may be rate-limited "
                            f"or the model may not be available."
                        ),
                    },
                },
            )

        # --- usage: OpenRouter отдаёт в OpenAI-форме ---
        usage_summary = usage or {}
        prompt_details = usage_summary.get("prompt_tokens_details") or {}
        completion_details = usage_summary.get("completion_tokens_details") or {}

        prompt_t = usage_summary.get("prompt_tokens", 0) or 0
        cached_t = prompt_details.get("cached_tokens", 0) or 0
        completion_t = usage_summary.get("completion_tokens", 0) or 0
        reasoning_t = completion_details.get("reasoning_tokens", 0) or 0
        cost = usage_summary.get("cost")

        hit_rate = f"{cached_t / prompt_t:.0%}" if prompt_t else "—"
        log_parts = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"in {prompt_t}",
            f"cached {cached_t} ({hit_rate})",
            f"out {completion_t}",
            f"reason {reasoning_t}",
            f"ttft {ttft:.1f}s" if ttft is not None else "ttft —",
            f"total {time.monotonic() - t0:.1f}s",
        ]
        if cost is not None:
            log_parts.append(f"${cost:.5f}")

        print(
            f"[{self.target_model}] -> {actual_model or '?'} @ "
            f"{actual_provider or '?'} | " + " | ".join(log_parts),
            file=sys.stderr,
            flush=True,
        )

        for _e in emitter.finish({
            # Anthropic считает input и кеш раздельно, OpenAI включает cached в prompt
            "input_tokens": max(prompt_t - cached_t, 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached_t,
            "output_tokens": completion_t,
        }):
            yield _e

    @staticmethod
    def _append_unique_reasoning_details(
        target: list[dict[str, Any]], details: list[dict[str, Any]]
    ) -> None:
        seen = {json.dumps(item, sort_keys=True) for item in target}
        for item in details:
            encoded = json.dumps(item, sort_keys=True)
            if encoded in seen:
                continue
            target.append(item)
            seen.add(encoded)
