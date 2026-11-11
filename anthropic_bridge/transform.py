import json
from typing import Any

DROP_KEYS = [
    "n",
    "presence_penalty",
    "frequency_penalty",
    "best_of",
    "logit_bias",
    "seed",
    "stream_options",
    "logprobs",
    "top_logprobs",
    "response_format",
    "service_tier",
    "parallel_tool_calls",
    "functions",
    "function_call",
    "developer",
    "strict",
    "reasoning_effort",
]


def remove_uri_format(schema: Any) -> Any:
    if not schema or not isinstance(schema, dict):
        return schema

    if schema.get("type") == "string" and schema.get("format") == "uri":
        schema = {k: v for k, v in schema.items() if k != "format"}

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: remove_uri_format(v) for k, v in value.items()}
        elif key == "items":
            if isinstance(value, list):
                result[key] = remove_uri_format(value[0]) if value else {}
            elif isinstance(value, dict):
                result[key] = remove_uri_format(value)
            else:
                result[key] = {}
        elif key == "additionalProperties" and isinstance(value, dict):
            result[key] = remove_uri_format(value)
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            result[key] = [remove_uri_format(item) for item in value]
        else:
            result[key] = (
                remove_uri_format(value) if isinstance(value, (dict, list)) else value
            )

    # Convert list of types (e.g. ["string", "null"]) to single type + nullable
    type_val = result.get("type")
    if isinstance(type_val, list):
        if "null" in type_val:
            result["nullable"] = True
        non_null = [t for t in type_val if t != "null"]
        result["type"] = non_null[0] if non_null else "string"

    # OpenAPI/Gemini compatibility: array must have items
    if result.get("type") == "array" and "items" not in result:
        result["items"] = {}

    # If items is present but type is not set, set type to array
    if "items" in result and "type" not in result:
        result["type"] = "array"

    # If properties is present but type is not set, set type to object
    if "properties" in result and "type" not in result:
        result["type"] = "object"

    return result


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif block.get("content"):
                    text_parts.append(extract_text_content(block["content"]))
        return "\n".join(text_parts)

    if isinstance(content, dict):
        if content.get("text"):
            return str(content["text"])
        elif content.get("content"):
            return extract_text_content(content["content"])

    return json.dumps(content)


def sanitize_anthropic_request(req: dict[str, Any]) -> list[str]:
    dropped: list[str] = []

    if "stop" in req:
        stop = req.pop("stop")
        req["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    if "user" in req:
        req["metadata"] = {**req.get("metadata", {}), "user_id": req.pop("user")}
        dropped.append("user")

    for key in DROP_KEYS:
        if key in req:
            dropped.append(key)
            del req[key]

    if req.get("max_tokens") is None:
        req["max_tokens"] = 4096

    return dropped


def convert_anthropic_tools_to_openai(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return []

    result: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": remove_uri_format(tool.get("input_schema", {})),
            },
        }
        if "cache_control" in tool:
            entry["cache_control"] = tool["cache_control"]
        result.append(entry)
    return result


def convert_anthropic_tool_choice_to_openai(
    tool_choice: dict[str, Any] | None,
) -> str | dict[str, Any] | None:
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "none":
        return "none"
    elif choice_type == "any":
        return "required"
    elif choice_type == "auto":
        return "auto"
    elif choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def normalize_system_message(
    system: str | list[dict[str, Any]] | None,
) -> str | None:
    if isinstance(system, list):
        return "\n\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in system
        )
    return system


def convert_anthropic_messages_to_openai(
    messages: list[dict[str, Any]], system: str | list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    openai_messages: list[dict[str, Any]] = []

    if isinstance(system, list):
        has_cache_control = any(
            isinstance(item, dict) and "cache_control" in item for item in system
        )
        if has_cache_control:
            system_blocks: list[dict[str, Any]] = []
            for item in system:
                block: dict[str, Any] = {
                    "type": "text",
                    "text": item.get("text", "") if isinstance(item, dict) else str(item),
                }
                if isinstance(item, dict) and "cache_control" in item:
                    block["cache_control"] = item["cache_control"]
                system_blocks.append(block)
            openai_messages.append({"role": "system", "content": system_blocks})
        else:
            system_text = normalize_system_message(system)
            if system_text:
                openai_messages.append({"role": "system", "content": system_text})
    elif isinstance(system, str) and system:
        openai_messages.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, list):
                content_parts: list[dict[str, Any]] = []
                seen_tool_ids: set[str] = set()

                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        text_part: dict[str, Any] = {
                            "type": "text",
                            "text": block.get("text", ""),
                        }
                        if "cache_control" in block:
                            text_part["cache_control"] = block["cache_control"]
                        content_parts.append(text_part)
                    elif block_type == "image":
                        source = block.get("source", {})
                        img_part: dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type')};base64,{source.get('data')}"
                            },
                        }
                        if "cache_control" in block:
                            img_part["cache_control"] = block["cache_control"]
                        content_parts.append(img_part)
                    elif block_type == "tool_result":
                        tool_id = block.get("tool_use_id")
                        if tool_id and tool_id not in seen_tool_ids:
                            if content_parts:
                                openai_messages.append(
                                    {"role": "user", "content": list(content_parts)}
                                )
                                content_parts.clear()
                            seen_tool_ids.add(tool_id)
                            result_content = block.get("content", "")
                            if not isinstance(result_content, str):
                                result_content = json.dumps(result_content)
                            tool_msg: dict[str, Any] = {
                                "role": "tool",
                                "content": result_content,
                                "tool_call_id": tool_id,
                            }
                            if "cache_control" in block:
                                tool_msg["cache_control"] = block["cache_control"]
                            openai_messages.append(tool_msg)

                if content_parts:
                    openai_messages.append({"role": "user", "content": content_parts})
            else:
                openai_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            if isinstance(content, list):
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                reasoning_details_list: list[dict[str, Any]] = []
                tool_calls: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                has_cache_control = any(
                    isinstance(b, dict) and "cache_control" in b for b in content
                )

                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "thinking":
                        t_text = block.get("thinking", "")
                        if t_text:
                            thinking_parts.append(t_text)
                        sig = block.get("signature", "")
                        if sig and sig.startswith("or_rd:"):
                            try:
                                import base64

                                raw_json = base64.b64decode(sig[6:]).decode("utf-8")
                                parsed_rd = json.loads(raw_json)
                                if isinstance(parsed_rd, list):
                                    reasoning_details_list.extend(parsed_rd)
                            except Exception:
                                pass
                        elif sig:
                            reasoning_details_list.append({
                                "type": "reasoning.text",
                                "text": t_text,
                                "signature": sig,
                            })
                    elif block_type == "redacted_thinking":
                        data = block.get("data", "")
                        if data:
                            reasoning_details_list.append({
                                "type": "reasoning.encrypted",
                                "data": data,
                            })
                    elif block_type == "tool_use":
                        block_id = block.get("id")
                        if block_id and block_id not in seen_ids:
                            seen_ids.add(block_id)
                            tc_entry: dict[str, Any] = {
                                "id": block_id,
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                            if "cache_control" in block:
                                tc_entry["cache_control"] = block["cache_control"]
                            tool_calls.append(tc_entry)

                assistant_msg: dict[str, Any] = {"role": "assistant"}

                if thinking_parts:
                    full_thinking = "\n".join(thinking_parts)
                    assistant_msg["reasoning"] = full_thinking
                    assistant_msg["reasoning_content"] = full_thinking

                if reasoning_details_list:
                    assistant_msg["reasoning_details"] = reasoning_details_list
                elif thinking_parts:
                    assistant_msg["reasoning_details"] = [
                        {"type": "reasoning.text", "text": "\n".join(thinking_parts)}
                    ]

                if has_cache_control and text_parts:
                    assistant_blocks: list[dict[str, Any]] = []
                    for block in content:
                        if block.get("type") == "text":
                            t_part: dict[str, Any] = {
                                "type": "text",
                                "text": block.get("text", ""),
                            }
                            if "cache_control" in block:
                                t_part["cache_control"] = block["cache_control"]
                            assistant_blocks.append(t_part)
                    assistant_msg["content"] = assistant_blocks
                elif text_parts:
                    assistant_msg["content"] = " ".join(text_parts)
                elif tool_calls or thinking_parts:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                if (
                    assistant_msg.get("content") is not None
                    or assistant_msg.get("tool_calls")
                    or assistant_msg.get("reasoning")
                ):
                    openai_messages.append(assistant_msg)
            else:
                openai_messages.append({"role": "assistant", "content": content})

        elif role == "system":
            openai_messages.append(
                {"role": "system", "content": extract_text_content(content)}
            )

    return openai_messages
