import collections
import copy
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO, cast

from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import AccessFormatter

_details_by_client: dict[str, str] = {}
_recent_details: collections.deque[str] = collections.deque(maxlen=200)

DEFAULT_ACCESS_FMT = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
DEFAULT_LOG_FMT = "%(asctime)s %(levelprefix)s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def log_timestamp() -> str:
    return time.strftime(DEFAULT_DATEFMT)


def is_debug_enabled() -> bool:
    val = os.environ.get("ANTHROPIC_BRIDGE_DEBUG", "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


class _TeeTextStream:
    """Forwards writes to both the original stream and a per-session debug file."""

    def __init__(self, primary: TextIO, log_file: TextIO, lock: threading.Lock):
        self._primary = primary
        self._log_file = log_file
        self._lock = lock
        self.encoding: str = getattr(primary, "encoding", "utf-8") or "utf-8"

    def isatty(self) -> bool:
        try:
            return bool(self._primary.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._primary.fileno()

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        with self._lock:
            n = self._primary.write(s)
            try:
                self._log_file.write(s)
            except Exception:
                pass
            return n

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            try:
                self._log_file.flush()
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


_active_debug_log_path: Path | None = None
_active_debug_file: TextIO | None = None


def get_debug_log_path() -> Path | None:
    return _active_debug_log_path


def reset_debug_log_file() -> None:
    """Restore stdout/stderr and close active debug file (primarily for tests)."""
    global _active_debug_log_path, _active_debug_file
    if isinstance(sys.stderr, _TeeTextStream):
        sys.stderr = sys.stderr._primary
    if isinstance(sys.stdout, _TeeTextStream):
        sys.stdout = sys.stdout._primary
    if _active_debug_file is not None:
        try:
            _active_debug_file.close()
        except Exception:
            pass
    _active_debug_log_path = None
    _active_debug_file = None


def setup_debug_log_file(log_dir: Path | str | None = None) -> Path | None:
    """Create a fresh per-session debug log in /tmp when ANTHROPIC_BRIDGE_DEBUG is enabled."""
    global _active_debug_log_path, _active_debug_file

    if not is_debug_enabled():
        return None
    if _active_debug_log_path is not None:
        return _active_debug_log_path

    base_dir = Path(
        log_dir
        or os.environ.get("ANTHROPIC_BRIDGE_DEBUG_DIR")
        or "/tmp"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = base_dir / f"anthropic-bridge-debug-{stamp}-pid{os.getpid()}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)

    latest_link = base_dir / "anthropic-bridge-debug-latest.log"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_path)
    except OSError:
        pass

    lock = threading.Lock()
    orig_stderr = sys.stderr
    orig_stdout = sys.stdout
    tee_stderr = cast(TextIO, _TeeTextStream(orig_stderr, log_file, lock))
    tee_stdout = cast(TextIO, _TeeTextStream(orig_stdout, log_file, lock))

    sys.stderr = tee_stderr
    sys.stdout = tee_stdout

    for logger_name in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is orig_stderr:
                handler.setStream(tee_stderr)

    _active_debug_log_path = log_path
    _active_debug_file = log_file
    return log_path


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
    session_id: str | None = None,
) -> None:
    parts = [
        f"model: {model}",
        f"client reasoning: {client_reasoning}",
        f"bridge reasoning: {bridge_reasoning}",
    ]
    if session_id:
        sid_short = f"{session_id[:8]}..{session_id[-4:]}" if len(session_id) > 16 else session_id
        parts.append(f"session: {sid_short}")
    details = " | ".join(parts)
    if client_addr:
        _details_by_client[client_addr] = details
    _recent_details.append(details)


def update_last_request_log(
    bridge_reasoning: str,
    client_addr: str | None = None,
) -> None:
    if client_addr and client_addr in _details_by_client:
        current = _details_by_client[client_addr]
        if " | bridge reasoning: " in current:
            prefix = current.rsplit(" | bridge reasoning: ", 1)[0]
            updated = f"{prefix} | bridge reasoning: {bridge_reasoning}"
        else:
            updated = f"{current} | bridge reasoning: {bridge_reasoning}"
        _details_by_client[client_addr] = updated
        try:
            idx = _recent_details.index(current)
            _recent_details[idx] = updated
        except ValueError:
            pass
        return

    if _recent_details:
        current = _recent_details[-1]
        if " | bridge reasoning: " in current:
            prefix = current.rsplit(" | bridge reasoning: ", 1)[0]
            updated = f"{prefix} | bridge reasoning: {bridge_reasoning}"
        else:
            updated = f"{current} | bridge reasoning: {bridge_reasoning}"
        _recent_details[-1] = updated
        for k, v in list(_details_by_client.items()):
            if v == current:
                _details_by_client[k] = updated


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
    if config and "formatters" in config:
        if "access" in config["formatters"]:
            config["formatters"]["access"]["()"] = BridgeAccessFormatter
            config["formatters"]["access"]["fmt"] = DEFAULT_ACCESS_FMT
            config["formatters"]["access"]["datefmt"] = DEFAULT_DATEFMT
        if "default" in config["formatters"]:
            config["formatters"]["default"]["fmt"] = DEFAULT_LOG_FMT
            config["formatters"]["default"]["datefmt"] = DEFAULT_DATEFMT

    access_logger = logging.getLogger("uvicorn.access")
    for handler in access_logger.handlers:
        handler.setFormatter(
            BridgeAccessFormatter(
                DEFAULT_ACCESS_FMT,
                datefmt=DEFAULT_DATEFMT,
            )
        )
    return config
