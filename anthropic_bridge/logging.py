import collections
import copy
import logging
from typing import Any

from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter

_details_by_client: dict[str, str] = {}
_recent_details: collections.deque[str] = collections.deque(maxlen=200)


def extract_client_reasoning_summary(payload: dict[str, Any]) -> str:
    """Summarizes reasoning configuration sent in Anthropic request from Claude Code."""
    if output_config := payload.get("output_config"):
        if isinstance(output_config, dict) and "effort" in output_config:
            return f"effort={output_config['effort']}"
    if thinking := payload.get("thinking"):
        if isinstance(thinking, dict):
            if thinking.get("type") == "disabled":
                return "disabled"
            if budget := thinking.get("budget_tokens"):
                return f"budget={budget}"
            if t_type := thinking.get("type"):
                return f"{t_type}"
            return "enabled"
        return "enabled"
    return "none"


def record_request_log(
    client_addr: str,
    model: str,
    client_reasoning: str,
    bridge_reasoning: str,
) -> None:
    details = (
        f"model: {model} | "
        f"client reasoning: {client_reasoning} | "
        f"bridge reasoning: {bridge_reasoning}"
    )
    if client_addr:
        _details_by_client[client_addr] = details
    _recent_details.append(details)


def pop_request_log(client_addr: str | None = None) -> str:
    if client_addr and client_addr in _details_by_client:
        details = _details_by_client.pop(client_addr)
        try:
            _recent_details.remove(details)
        except ValueError:
            pass
        return details
    if _recent_details:
        details = _recent_details.popleft()
        for k, v in list(_details_by_client.items()):
            if v == details:
                _details_by_client.pop(k, None)
                break
        return details
    return ""


def clear_request_logs() -> None:
    _details_by_client.clear()
    _recent_details.clear()


class BridgeAccessFormatter(AccessFormatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if hasattr(super(), "formatMessage"):
            res = super().formatMessage(record)
        else:
            res = super().format(record)

        client_addr = getattr(record, "client_addr", None)
        if not client_addr and isinstance(record.args, (tuple, list)) and len(record.args) >= 1:
            client_addr = record.args[0]

        details = pop_request_log(str(client_addr) if client_addr else None)
        if details:
            res = f"{res} | {details}"
        return res


def setup_access_logging(
    log_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(log_config or LOGGING_CONFIG)
    if config and "formatters" in config and "access" in config["formatters"]:
        config["formatters"]["access"]["()"] = BridgeAccessFormatter

    access_logger = logging.getLogger("uvicorn.access")
    for handler in access_logger.handlers:
        handler.setFormatter(
            BridgeAccessFormatter(
                '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
            )
        )
    return config
