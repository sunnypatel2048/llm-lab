"""A small JSON Schema validator for tool arguments.

Supports the subset tool definitions actually use: type, properties, required,
additionalProperties, enum, items, minLength, minimum, maximum. Production code would use the
`jsonschema` package; this one is ~60 lines so the behaviour is easy to read and test.

check_schema() rejects any keyword outside that subset when a tool is registered. Silently
ignoring a constraint (say, a "pattern") would let invalid arguments through without anyone
noticing, so an unsupported keyword is a startup error instead.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "enum",
    "items",
    "minLength",
    "minimum",
    "maximum",
    "description",  # documentation only
}

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python, but true/false is not a number in JSON
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}

_JSON_NAMES = {
    dict: "object",
    list: "array",
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    type(None): "null",
}


def check_schema(schema: dict[str, Any], path: str = "$") -> None:
    """Raise ValueError if the schema uses anything this validator doesn't enforce."""
    unknown = set(schema) - SUPPORTED_KEYWORDS
    if unknown:
        raise ValueError(f"{path}: unsupported schema keyword(s): {', '.join(sorted(unknown))}")
    if "type" in schema and schema["type"] not in _TYPE_CHECKS:
        raise ValueError(f"{path}: unsupported type {schema['type']!r}")
    for name, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{path}.{name}")
    if isinstance(schema.get("items"), dict):
        check_schema(schema["items"], f"{path}[]")


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Every way `value` breaks `schema`, as readable messages. Empty list = valid.

    Messages are written for the model to read: they name the field and say what was expected,
    so it can fix its call.
    """
    expected = schema.get("type")
    if expected is not None and not _TYPE_CHECKS[expected](value):
        actual = _JSON_NAMES.get(type(value), type(value).__name__)
        return [f"{path}: expected {expected}, got {actual}"]  # further checks would be noise

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        errors.append(f"{path}: {value!r} is not one of {allowed}")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        errors.append(f"{path}: must be at least {schema['minLength']} characters")
    if "minimum" in schema and isinstance(value, int | float) and value < schema["minimum"]:
        errors.append(f"{path}: must be >= {schema['minimum']}")
    if "maximum" in schema and isinstance(value, int | float) and value > schema["maximum"]:
        errors.append(f"{path}: must be <= {schema['maximum']}")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: missing required field '{name}'")
        for name, item in value.items():
            if name in properties:
                errors += validate(item, properties[name], f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field '{name}'")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors
