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
