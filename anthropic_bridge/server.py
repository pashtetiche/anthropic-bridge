from dataclasses import dataclass
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import get_reasoning_cache
from .logging import (
    extract_client_reasoning_summary,
    is_debug_enabled,
    log_timestamp,
    record_request_log,
    setup_access_logging,
)
from .protocol import collect_anthropic_response, estimate_anthropic_input_tokens
from .providers import CopilotProvider, OpenAIProvider, OpenRouterProvider
from .providers.openai.auth import auth_file_exists

ProviderType = Literal["openrouter", "copilot", "openai"]


def _extract_session_id(request: Request, body: dict[str, Any]) -> str | None:
    for header in (
        "x-claude-code-session-id",
        "x-session-id",
        "session-id",
        "x-claude-session-id",
        "claude-session-id",
        "claude-code-session-id",
        "anthropic-session-id",
    ):
        if val := request.headers.get(header):
            return str(val).strip()[:256]

    if body_sid := body.get("session_id"):
        return str(body_sid).strip()[:256]

    if metadata := body.get("metadata"):
        if isinstance(metadata, dict):
            for key in ("session_id", "session", "conversation_id"):
                if val := metadata.get(key):
                    return str(val).strip()[:256]

            if user_id := metadata.get("user_id"):
                if isinstance(user_id, str):
                    if user_id.startswith("{") and "session_id" in user_id:
                        try:
                            import json

                            parsed = json.loads(user_id)
                            if isinstance(parsed, dict) and (sid := parsed.get("session_id")):
                                return str(sid).strip()[:256]
                        except Exception:
                            pass
                    return user_id.strip()[:256]

    return None


@dataclass
class ProxyConfig:
    openrouter_api_key: str | None = None
    copilot_token: str | None = None


class AnthropicBridge:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = FastAPI(title="Anthropic Bridge")
        self._openrouter_clients: dict[str, OpenRouterProvider] = {}
        self._openai_clients: dict[str, OpenAIProvider] = {}
        self._copilot_clients: dict[str, CopilotProvider] = {}
        self._setup_routes()
        self._setup_cors()
        setup_access_logging()
        get_reasoning_cache()

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        @self.app.api_route("/", methods=["GET", "HEAD"])
        async def root() -> dict[str, str]:
            return {"status": "ok", "message": "Anthropic Bridge"}

        @self.app.api_route("/health", methods=["GET", "HEAD"])
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.api_route("/api/hello", methods=["GET", "HEAD"])
        async def api_hello() -> JSONResponse:
            return JSONResponse(
                {
                    "status": "ok",
                    "service": "anthropic-bridge",
                    "message": "Hello! This is an Anthropic-to-OpenRouter/Copilot bridge.",
                    "endpoints": {
                        "messages": "/v1/messages",
                        "count_tokens": "/v1/messages/count_tokens",
                        "health": "/health",
                    },
                },
                headers={"X-Service": "anthropic-bridge"},
            )

        @self.app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request) -> JSONResponse:
            body = await request.json()
            session_id = _extract_session_id(request, body)
            model = body.get("model", "")
            client_reasoning = extract_client_reasoning_summary(body)
            client_addr = (
                f"{request.client.host}:{request.client.port}"
                if request.client
                else ""
            )
            record_request_log(
                client_addr=client_addr,
                model=model,
                client_reasoning=client_reasoning,
                bridge_reasoning="count_tokens",
                session_id=session_id,
            )
            return JSONResponse({"input_tokens": estimate_anthropic_input_tokens(body)})

        @self.app.post("/v1/messages", response_model=None)
        async def messages(request: Request) -> StreamingResponse | JSONResponse:
            body = await request.json()
            session_id = _extract_session_id(request, body)
            if session_id and "session_id" not in body:
                body["session_id"] = session_id

            if is_debug_enabled():
                import sys

                thinking_blocks = []
                for m_idx, msg in enumerate(body.get("messages", [])):
                    content = msg.get("content")
                    if isinstance(content, list):
                        for _b_idx, block in enumerate(content):
                            if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                                sig = block.get("signature", "")
                                thinking_blocks.append({
                                    "msg_idx": m_idx,
                                    "role": msg.get("role"),
                                    "type": block.get("type"),
                                    "text_len": len(block.get("thinking", "")),
                                    "preview": block.get("thinking", "")[:60],
                                    "signature": (sig[:25] + "...") if len(sig) > 25 else sig,
                                    "sig_len": len(sig),
                                })
                if thinking_blocks:
                    print(
                        f"{log_timestamp()} >>> [PROBE Client -> Bridge] Incoming request has {len(thinking_blocks)} thinking block(s): {thinking_blocks}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    has_tool_history = any(
                        isinstance(m.get("content"), list)
                        and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m["content"])
                        for m in body.get("messages", [])
                    )
                    if has_tool_history:
                        print(
                            f"{log_timestamp()} >>> [PROBE Client -> Bridge] Incoming request has tool_use/tool_result, but NO thinking blocks in history!",
                            file=sys.stderr,
                            flush=True,
                        )

                # Анализ входящих меток cache_control
                system_val = body.get("system")
                sys_cc = False
                if isinstance(system_val, list):
                    sys_cc = any(isinstance(b, dict) and "cache_control" in b for b in system_val)
                elif isinstance(system_val, dict) and "cache_control" in system_val:
                    sys_cc = True

                tools_val = body.get("tools") or []
                tools_cc = any(isinstance(t, dict) and "cache_control" in t for t in tools_val)

                msg_cc_indices = []
                for m_idx, msg in enumerate(body.get("messages", [])):
                    c = msg.get("content")
                    if isinstance(c, list):
                        if any(isinstance(b, dict) and "cache_control" in b for b in c):
                            msg_cc_indices.append((m_idx, msg.get("role")))
                    elif isinstance(msg, dict) and "cache_control" in msg:
                        msg_cc_indices.append((m_idx, msg.get("role")))

                print(
                    f"{log_timestamp()} >>> [PROBE Client -> Bridge] CacheControl incoming: "
                    f"system_cc={sys_cc}, tools_cc={tools_cc} (of {len(tools_val)}), "
                    f"msg_breakpoints={msg_cc_indices} (total msgs: {len(body.get('messages', []))})",
                    file=sys.stderr,
                    flush=True,
                )

                if session_id:
                    print(f"{log_timestamp()} [debug] session_id={session_id}", file=sys.stderr, flush=True)

            model = body.get("model", "")
            client_reasoning = extract_client_reasoning_summary(body)
            client_addr = (
                f"{request.client.host}:{request.client.port}"
                if request.client
                else ""
            )

            provider = self._get_provider(model)
            if provider is None:
                record_request_log(
                    client_addr=client_addr,
                    model=model,
                    client_reasoning=client_reasoning,
                    bridge_reasoning="none (unauthorized)",
                    session_id=session_id,
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": self._get_provider_error_message(model),
                        }
                    },
                )

            bridge_reasoning = (
                provider.get_reasoning_summary(body)
                if hasattr(provider, "get_reasoning_summary")
                else "none"
            )
            record_request_log(
                client_addr=client_addr,
                model=model,
                client_reasoning=client_reasoning,
                bridge_reasoning=bridge_reasoning,
                session_id=session_id,
            )

            if body.get("stream") is not True:
                message, error = await collect_anthropic_response(provider.handle(body))
                if error:
                    err_obj = error.get("error", {})
                    err_type = (
                        err_obj.get("type", "api_error")
                        if isinstance(err_obj, dict)
                        else "api_error"
                    )
                    status_code = 502
                    if err_type == "rate_limit_error":
                        status_code = 429
                    elif err_type == "overloaded_error":
                        status_code = 529
                    elif err_type == "invalid_request_error":
                        status_code = 400
                    elif err_type == "authentication_error":
                        status_code = 401
                    return JSONResponse(status_code=status_code, content=error)
                if message is None:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Provider returned no message.",
                            },
                        },
                    )
                return JSONResponse(message)

            return StreamingResponse(
                provider.handle(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _get_provider(
        self, model: str
    ) -> CopilotProvider | OpenAIProvider | OpenRouterProvider | None:
        requested_provider = self._get_requested_provider(model)

        if requested_provider:
            requested_model = self._model_for_provider(model, requested_provider)
            return self._make_provider(requested_model, requested_provider)

        for provider_type in ("openrouter", "copilot", "openai"):
            provider_model = self._model_for_provider(model, provider_type)
            provider = self._make_provider(provider_model, provider_type)
            if provider:
                return provider

        return None

    def _get_provider_error_message(self, model: str) -> str:
        requested_provider = self._get_requested_provider(model)
        if requested_provider is None:
            return (
                "No provider configured. Set OPENROUTER_API_KEY, "
                "GITHUB_COPILOT_TOKEN, or configure OpenAI auth."
            )

        messages = {
            "openai": "OpenAI auth not configured. Run 'codex login' to use openai/* models.",
            "copilot": "GitHub Copilot token not configured. Set GITHUB_COPILOT_TOKEN to use copilot/* models.",
            "openrouter": "OpenRouter API key not configured. Set OPENROUTER_API_KEY to use openrouter/* models.",
        }
        return messages[requested_provider]

    def _get_requested_provider(self, model: str) -> ProviderType | None:
        if model.startswith("openai/"):
            return "openai"
        if model.startswith("copilot/"):
            return "copilot"
        if model.startswith("openrouter/"):
            return "openrouter"
        return None

    def _model_for_provider(self, model: str, provider_type: ProviderType) -> str:
        model_name = model.split("/", 1)[1] if "/" in model else model
        if provider_type == "openrouter":
            return f"openrouter/{model_name}"
        if provider_type == "copilot":
            return f"copilot/{model_name}"
        return f"openai/{model_name}"

    def _make_provider(
        self, model: str, provider_type: ProviderType
    ) -> CopilotProvider | OpenAIProvider | OpenRouterProvider | None:
        if provider_type == "openrouter" and self.config.openrouter_api_key:
            if model not in self._openrouter_clients:
                self._openrouter_clients[model] = OpenRouterProvider(
                    model, self.config.openrouter_api_key
                )
            return self._openrouter_clients[model]

        if provider_type == "copilot" and self.config.copilot_token:
            if model not in self._copilot_clients:
                self._copilot_clients[model] = CopilotProvider(
                    model, self.config.copilot_token
                )
            return self._copilot_clients[model]

        if provider_type == "openai" and auth_file_exists():
            if model not in self._openai_clients:
                self._openai_clients[model] = OpenAIProvider(model)
            return self._openai_clients[model]

        return None


def create_app(
    openrouter_api_key: str | None = None,
    copilot_token: str | None = None,
) -> FastAPI:
    config = ProxyConfig(
        openrouter_api_key=openrouter_api_key,
        copilot_token=copilot_token,
    )
    return AnthropicBridge(config).app
