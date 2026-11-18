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
- **Debugging & Probe Logs**: Setting `ANTHROPIC_BRIDGE_DEBUG=1` enables granular `[PROBE ...]` logs showing thinking blocks, `cache_control` breakpoints, and reasoning details traveling between Client, Bridge, and OpenRouter. Each bridge startup in debug mode also tees all stdout/stderr/uvicorn logs to a fresh per-session file `/tmp/anthropic-bridge-debug-<YYYYmmdd-HHMMSS>-pid<pid>.log` and updates `/tmp/anthropic-bridge-debug-latest.log` (override directory via `ANTHROPIC_BRIDGE_DEBUG_DIR`). All server and provider logs are timestamped at the beginning of each line (`YYYY-MM-DD HH:MM:SS`).

## Autonomous Debugging via Claude Code

Agents can reproduce and verify Claude Code client behavior, caching, subagents, and auto-mode verifiers without manual user clicks:

1. **Isolate sandbox & server**:
   - Use a temporary workspace (e.g., `/tmp/cc_sandbox`) with a dummy `CLAUDE.md` marker and test file; remove all `/tmp` artifacts when done.
   - If running your own mock or test bridge instance, bind an unused port (e.g., `18085`), manage its lifecycle in a `try/finally` block, and verify with `ps` that no stray process remains. Do not kill or spam the user's running bridge unless asked.
   - If live upstream calls are needed and `OPENROUTER_API_KEY` is not in the shell environment (do not read secret files like `~/.envrc.private`), either test protocol/transform behavior with a local mock responder or ask the user to start/restart their debug bridge and read `/tmp/anthropic-bridge-debug-latest.log`.

2. **Drive `claude` headlessly (`-p`)**:
   - Basic tool-loop + multi-turn resume:
     `ANTHROPIC_BASE_URL=http://127.0.0.1:<port> ANTHROPIC_API_KEY=dummy claude -p --model <model> --output-format json --permission-mode bypassPermissions --setting-sources project "<prompt>"`
     Extract `session_id` from JSON output and pass `--resume <session_id>` for cross-turn verification.
   - Subagent & Auto-Mode verifier testing:
     Set `CLAUDE_CODE_SUBAGENT_MODEL`, `CLAUDE_CODE_AUTO_MODE_MODEL`, and `ANTHROPIC_DEFAULT_HAIKU_MODEL` in the subprocess env, use `--permission-mode auto`, and prompt a mutating `Bash` command (read-only commands like `ls` fast-path past the classifier) or an `Agent` call (`Explore` vs `general-purpose`).

3. **Verify in debug logs**:
   - Check `/tmp/anthropic-bridge-debug-latest.log` for `[PROBE Client -> Bridge]`, `[PROBE ReasoningCache -> OpenRouter]`, `[PROBE Bridge -> OpenRouter]`, `[PROBE OpenRouter -> Bridge]`, and the final `in ... | cached ... | out ... | reason ...` summary line.
   - Note that Claude Code's built-in `Explore` subagent intentionally omits `CLAUDE.md` and full system instructions, whereas `general-purpose` includes `CLAUDE.md`. Also note that Auto-Mode classifier calls are single-turn (`[user, user]` without an `assistant` turn), so models whose upstream KV cache requires multi-turn `assistant` history (e.g., `tencent/hy4-flash`) will report `0%` cache hits in `CLAUDE_CODE_AUTO_MODE_MODEL`.
