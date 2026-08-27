import json
import os
import sys
from typing import Any

ENV_VARS = [
    "OPENROUTER_MANDATORY_REASONING_MODELS",
    "MANDATORY_REASONING_MODELS",
    "OPENROUTER_FORCE_REASONING_MODELS",
]
PRIMARY_ENV_VAR = "OPENROUTER_MANDATORY_REASONING_MODELS"

_in_memory_models: set[str] = set()


def _normalize_model_name(name: str) -> str:
    cleaned = name.strip().strip("'\"").lower()
    cleaned = cleaned.removeprefix("openrouter/")
    return cleaned


def _read_env_models() -> set[str]:
    models: set[str] = set()
    for env_var in ENV_VARS:
        val = os.environ.get(env_var, "").strip()
        if not val:
            continue
        if val.startswith("[") and val.endswith("]"):
            try:
                items = json.loads(val)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str) and item.strip():
                            models.add(_normalize_model_name(item))
                    continue
            except Exception:
                pass
        for item in val.replace(";", ",").split(","):
            norm = _normalize_model_name(item)
            if norm:
                models.add(norm)
    return models


def get_mandatory_reasoning_models() -> set[str]:
    combined = set(_in_memory_models)
    combined.update(_read_env_models())
    return combined


def register_mandatory_reasoning_model(model: str) -> None:
    norm = _normalize_model_name(model)
    if not norm:
        return

    _in_memory_models.add(norm)
    without_preset = norm.removeprefix("@preset/")
    if without_preset:
        _in_memory_models.add(without_preset)

    current_env_models = _read_env_models()
    current_env_models.update(_in_memory_models)

    updated_str = ",".join(sorted(current_env_models))
    os.environ[PRIMARY_ENV_VAR] = updated_str
    print(
        f"[mandatory-reasoning] Added '{model}' to {PRIMARY_ENV_VAR}: {updated_str}",
        file=sys.stderr,
        flush=True,
    )


def is_mandatory_reasoning_model(model: str) -> bool:
    if not model:
        return False
    norm = _normalize_model_name(model)
    without_preset = norm.removeprefix("@preset/")

    mandatory_set = get_mandatory_reasoning_models()
    if norm in mandatory_set or without_preset in mandatory_set:
        return True

    for item in mandatory_set:
        item_without_preset = item.removeprefix("@preset/")
        if norm == item or without_preset == item_without_preset:
            return True
        # Match model slug suffixes (e.g. "glm-5-3-flash" matching "zhipu/glm-5-3-flash")
        if "/" in norm and norm.split("/")[-1] == item_without_preset:
            return True
        if "/" in item and item.split("/")[-1] == without_preset:
            return True

    return False


def is_mandatory_reasoning_error(error_content: str | dict[str, Any]) -> bool:
    if isinstance(error_content, dict):
        msg = str(error_content.get("message") or error_content)
    else:
        msg = str(error_content)
    lower = msg.lower()
    return "reasoning is mandatory" in lower or "mandatory reasoning" in lower


def clear_mandatory_reasoning_models() -> None:
    _in_memory_models.clear()
    for env_var in ENV_VARS:
        os.environ.pop(env_var, None)
