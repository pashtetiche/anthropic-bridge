# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

anthropic-bridge is a proxy server that translates Anthropic Messages API requests into OpenRouter API format, enabling use of various LLM providers (Gemini, OpenAI, Grok, DeepSeek, Qwen, MiniMax) through an Anthropic-compatible interface.

## Commands

```bash
# Install dependencies
pip install -e ".[test,dev]"

# Run server (requires OPENROUTER_API_KEY env var)
OPENROUTER_API_KEY=your_key anthropic-bridge --port 8080 --host 127.0.0.1

# Lint
ruff check anthropic_bridge/ tests/

# Type check
mypy anthropic_bridge/

# Run deterministic tests
pytest tests/ -v

# Run live upstream tests
ANTHROPIC_BRIDGE_LIVE_TESTS=1 OPENROUTER_API_KEY=your_key pytest tests/test_integration.py -v
```

## Architecture

**Request Flow**: Anthropic API request → `server.py` → provider client → upstream API → Anthropic JSON or SSE response

**Core Components**:
- `server.py` - FastAPI app exposing `/v1/messages` and `/v1/messages/count_tokens`, with strict provider-prefix routing and non-stream aggregation
- `transform.py` - Converts Anthropic messages/tools/tool_choice into upstream request shapes
- `protocol.py` - Parses Anthropic-style SSE, aggregates non-stream responses, and estimates approximate token counts

**Provider System** (`providers/`):
- `openrouter/client.py` - OpenRouter chat-completions streaming adapter
- `openai/client.py` - OpenAI Responses API adapter via Codex auth
- `copilot/client.py` - GitHub Copilot adapter for Responses API and chat-completions style models
- `openrouter/registry.py` - Selects provider-specific request/response tweaks based on model ID
- Provider implementations (Grok, Gemini, OpenAI, etc.) handle model-specific quirks like XML tool call parsing or reasoning options

**Caching & Reasoning Continuity**:
- **Prompt Caching**: `transform.py` preserves `cache_control` breakpoints across `system`, `messages`, and `tools`. For Claude models on OpenRouter, 5-minute ephemeral caching (`cache_control: {"type": "ephemeral"}`) is enabled by default to trigger upstream prompt caching. Can be extended via `ANTHROPIC_CACHE_TTL=1h`.
- **Reasoning Preservation (Preserved Thinking)**: Chain of Thought traces are preserved across agent tool steps and user turns. Outgoing reasoning is encoded into the Anthropic `thinking` block's `signature` (`or_rd:...`) and reconstructed on subsequent steps. `ReasoningCache` (`cache.py`) acts as a fallback keyed by `tool_call_id`.
- **Debugging & Probe Logs**: Setting `ANTHROPIC_BRIDGE_DEBUG=1` enables granular `[PROBE ...]` logs showing thinking blocks and reasoning details traveling between Client, Bridge, and OpenRouter. All server and provider logs are timestamped at the beginning of each line (`YYYY-MM-DD HH:MM:SS`).
