from anthropic_bridge.transform import (
    convert_anthropic_tools_to_openai,
    remove_uri_format,
)


def test_remove_uri_format():
    schema = {
        "type": "string",
        "format": "uri",
    }
    cleaned = remove_uri_format(schema)
    assert cleaned == {"type": "string"}


def test_array_without_items_gets_items():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
            }
        },
    }
    cleaned = remove_uri_format(schema)
    assert cleaned["properties"]["tags"]["items"] == {}


def test_nested_array_without_items_gets_items():
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "object",
                "properties": {
                    "where": {
                        "type": "array",
                        "items": {
                            "type": "array",
                        },
                    }
                },
            }
        },
    }
    cleaned = remove_uri_format(schema)
    where = cleaned["properties"]["query"]["properties"]["where"]
    assert where["type"] == "array"
    assert "items" in where
    assert where["items"]["type"] == "array"
    assert where["items"]["items"] == {}


def test_tuple_array_items_converted_to_single_schema():
    schema = {
        "type": "array",
        "items": [{"type": "string"}, {"type": "integer"}],
    }
    cleaned = remove_uri_format(schema)
    assert cleaned["items"] == {"type": "string"}


def test_list_type_with_null_converted_to_nullable():
    schema = {
        "type": ["string", "null"],
    }
    cleaned = remove_uri_format(schema)
    assert cleaned["type"] == "string"
    assert cleaned["nullable"] is True


def test_convert_anthropic_tools_to_openai_sanitizes_tools():
    tools = [
        {
            "name": "search",
            "description": "search tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "array",
                    }
                },
            },
        }
    ]
    openai_tools = convert_anthropic_tools_to_openai(tools)
    assert len(openai_tools) == 1
    params = openai_tools[0]["function"]["parameters"]
    assert params["properties"]["filters"]["items"] == {}


def test_convert_anthropic_tools_preserves_cache_control():
    tools = [
        {
            "name": "lookup",
            "description": "lookup tool",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    openai_tools = convert_anthropic_tools_to_openai(tools)
    assert openai_tools[0]["cache_control"] == {"type": "ephemeral"}


def test_convert_anthropic_messages_preserves_cache_control_on_system_and_blocks():
    from anthropic_bridge.transform import convert_anthropic_messages_to_openai

    system = [
        {"type": "text", "text": "Stable instructions", "cache_control": {"type": "ephemeral"}}
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Code context...",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "What is this?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the response",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_123",
                    "content": "output",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    converted = convert_anthropic_messages_to_openai(messages, system=system)

    # System message converted as structured list with cache_control
    assert converted[0]["role"] == "system"
    assert converted[0]["content"] == [
        {"type": "text", "text": "Stable instructions", "cache_control": {"type": "ephemeral"}}
    ]

    # User content blocks preserve cache_control
    assert converted[1]["role"] == "user"
    assert converted[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in converted[1]["content"][1]

    # Assistant content block preserves cache_control
    assert converted[2]["role"] == "assistant"
    assert converted[2]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Tool message preserves cache_control
    assert converted[3]["role"] == "tool"
    assert converted[3]["cache_control"] == {"type": "ephemeral"}
