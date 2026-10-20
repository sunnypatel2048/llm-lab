import pytest

from llm_lab.schema import check_schema, validate

PERSON = {
    "type": "object",
    "properties": {
        "email": {"type": "string", "minLength": 3},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tier": {"type": "string", "enum": ["free", "pro"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["email"],
    "additionalProperties": False,
}


def test_valid_object_has_no_errors():
    assert validate({"email": "a@b.c", "age": 30, "tier": "pro", "tags": ["x"]}, PERSON) == []


def test_missing_required_field_is_named():
    assert validate({}, PERSON) == ["$: missing required field 'email'"]


def test_unexpected_field_is_named():
    errors = validate({"email": "a@b.c", "nickname": "j"}, PERSON)
    assert errors == ["$: unexpected field 'nickname'"]


def test_wrong_type_reports_json_type_names():
    assert validate({"email": 42}, PERSON) == ["$.email: expected string, got integer"]
    assert validate([], PERSON) == ["$: expected object, got array"]


def test_booleans_are_not_numbers():
    # Python treats True as 1; JSON doesn't, and neither should a tool's arguments.
    assert validate({"email": "a@b.c", "age": True}, PERSON) == [
        "$.age: expected integer, got boolean"
    ]


def test_number_accepts_ints_and_floats_but_integer_only_ints():
    assert validate(3, {"type": "number"}) == validate(3.5, {"type": "number"}) == []
    assert validate(3.5, {"type": "integer"}) == ["$: expected integer, got number"]


def test_enum_bounds_length_and_items():
    errors = validate({"email": "ab", "age": 200, "tier": "gold", "tags": ["ok", 7]}, PERSON)
    assert errors == [
        "$.email: must be at least 3 characters",
        "$.age: must be <= 150",
        "$.tier: 'gold' is not one of 'free', 'pro'",
        "$.tags[1]: expected string, got integer",
    ]


def test_all_problems_are_reported_together():
    # The model gets one chance to repair, so it needs every problem at once, not one per try.
    assert len(validate({"age": -1, "extra": 1}, PERSON)) == 3


def test_check_schema_accepts_the_supported_subset():
    check_schema(PERSON)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "^a"},  # would silently not be enforced
        {"type": "object", "properties": {"x": {"type": "string", "format": "email"}}},
        {"type": "tuple"},
    ],
)
def test_check_schema_rejects_what_it_cannot_enforce(schema):
    with pytest.raises(ValueError):
        check_schema(schema)
