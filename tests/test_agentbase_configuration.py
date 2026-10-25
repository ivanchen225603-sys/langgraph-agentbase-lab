"""Contract evidence for configuration-only assembly; does not claim a working runtime."""

import copy
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from agentbase import schema as schema_module
from agentbase.configuration import AgentPackage, load_package
from agentbase.contracts import (
    AgentError,
    Identity,
    Message,
    ModelResponse,
    ToolCall,
    ToolCallDelta,
)
from agentbase.registry import ComponentRegistry
from agentbase.schema import SchemaContract, canonical_json


def package_data():
    return {
        "models": [{"id": "model-one", "provider": "test", "model": "offline"}],
        "tools": [
            {
                "name": "calculate",
                "handler": "external.calculate",
                "description": "Compute a sum from finite numeric input",
                "input_schema": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                "output_schema": {"type": "number"},
            }
        ],
        "agents": [
            {
                "id": "extract",
                "description": "Return structured fields",
                "model": "model-one",
                "prompt": {"system_template": "Extract declared fields"},
                "input_schema": {"type": "string"},
                "output_schema": {"type": "object"},
            },
            {
                "id": "calculator",
                "description": "Use a declared external calculation tool",
                "model": "model-one",
                "prompt": {"system_template": "Use only registered calculation tools"},
                "input_schema": {"type": "string"},
                "tools": ["calculate"],
            },
        ],
    }


async def handler(arguments, context):
    return arguments["a"] + arguments["b"]


def registry(*, handler_version="1", provider_version="1", workflow_version="1"):
    result = ComponentRegistry()
    result.register_handler("external.calculate", handler, version=handler_version)
    result.register_provider("test", lambda config: None, version=provider_version)
    result.register_workflow("conversation", lambda context: None, version=workflow_version)
    return result


def test_external_declarations_bind_without_business_imports(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps(package_data()), encoding="utf-8")
    package = load_package(path)
    components = registry()
    components.bind(package)
    assert {item["id"] for item in components.agents()} == {"extract", "calculator"}
    assert components.agent("extract").tools == ()
    assert components.agent("calculator").tools[0].description.startswith("Compute")
    assert package.digest == load_package(path).digest
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentbase,sys; assert not any(n=='raglab' or n.startswith('raglab.') for n in sys.modules)",
        ],
        capture_output=True,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode()


@pytest.mark.parametrize(
    "missing,code",
    [
        ("workflow", "unknown_workflow"),
        ("provider", "unknown_provider"),
        ("handler", "unknown_handler"),
    ],
)
def test_missing_registered_executables_cannot_partially_activate(missing, code):
    components = ComponentRegistry()
    if missing != "workflow":
        components.register_workflow("conversation", lambda context: None)
    if missing != "provider":
        components.register_provider("test", lambda config: None)
    if missing != "handler":
        components.register_handler("external.calculate", handler)
    with pytest.raises(AgentError) as error:
        components.bind(AgentPackage.model_validate(package_data()))
    assert error.value.code == code
    assert components.agents() == []


def test_registered_versions_and_schema_are_bound_to_restore_digest():
    package = AgentPackage.model_validate(package_data())
    initial = registry()
    initial.bind(package)
    before = initial.agent("calculator").digest
    for option in ("handler_version", "provider_version", "workflow_version"):
        changed = registry(**{option: "2"})
        changed.bind(package)
        assert changed.agent("calculator").digest != before
    altered = package_data()
    altered["tools"][0]["output_schema"] = {"type": "integer"}
    changed = registry()
    changed.bind(AgentPackage.model_validate(altered))
    assert changed.agent("calculator").digest != before
    retrieved = initial.agent("calculator")
    retrieved.definition.tools.clear()
    retrieved.tools[0].input_schema.clear()
    assert initial.agent("calculator").digest == before
    assert initial.agent("calculator").definition.tools == ["calculate"]
    assert initial.agent("calculator").tools[0].input_schema["type"] == "object"


def test_allowlist_and_server_scope_intersection_and_result_contract():
    components = registry()
    components.bind(AgentPackage.model_validate(package_data()))
    allowed = Identity(owner="user", scopes=frozenset({"read"}))
    with pytest.raises(AgentError, match="not allowed"):
        components.authorized_tool("extract", "calculate", allowed)
    with pytest.raises(AgentError) as error:
        components.authorized_tool("calculator", "calculate", Identity(owner="user"))
    assert error.value.code == "forbidden"
    tool = components.authorized_tool("calculator", "calculate", allowed)
    assert components.validate_arguments(tool, {"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert components.validate_result(tool, 3) == 3
    for arguments in ({"a": 1}, {"a": "1", "b": 2}, {"a": 1, "b": 2, "secret": "x"}):
        with pytest.raises(AgentError) as error:
            components.validate_arguments(tool, arguments)
        assert error.value.code == "tool_arguments"
    with pytest.raises(AgentError) as error:
        components.validate_result(tool, {"secret": "do-not-echo"})
    assert error.value.code == "tool_result"
    assert "do-not-echo" not in str(error.value)


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
@pytest.mark.parametrize(
    "target", ["https://example.test/schema", "file:///private", "other.json#/x"]
)
def test_schema_never_fetches_remote_or_file_references(keyword, target):
    with pytest.raises(AgentError) as error:
        SchemaContract({keyword: target})
    assert error.value.code == "schema_reference"


def test_local_reference_and_available_format_are_actually_validated():
    contract = SchemaContract({"$defs": {"name": {"type": "string"}}, "$ref": "#/$defs/name"})
    assert contract.validate("value") == "value"
    with pytest.raises(AgentError):
        contract.validate(5)
    with pytest.raises(AgentError) as error:
        SchemaContract({"$ref": "#/$defs/missing"}).validate(5)
    assert error.value.code == "schema_reference"
    with pytest.raises(AgentError):
        SchemaContract({"type": "string", "format": "uuid"}).validate("invalid")
    with pytest.raises(AgentError) as error:
        SchemaContract({"type": "string", "format": "not-implemented"})
    assert error.value.code == "schema_format"


def test_schema_property_names_and_literal_values_are_not_schema_keywords():
    data = {"$ref": "https://ordinary-data.test", "format": "not-a-checker"}
    contract = SchemaContract(
        {
            "type": "object",
            "properties": {"$ref": {"type": "string"}, "format": {"type": "string"}},
            "const": data,
        }
    )
    assert contract.validate(data) == data


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "value"}, b"bytes", (1, 2)])
def test_finite_json_contract(value):
    with pytest.raises(AgentError):
        canonical_json(value)


def test_nesting_cycles_schema_draft_and_size_are_bounded():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(AgentError, match="nesting"):
        canonical_json(cyclic)
    with pytest.raises(AgentError, match="byte"):
        canonical_json("large", max_bytes=2)
    for schema in (
        {},
        {"type": "not-a-type"},
        {"$schema": 3},
        {"$schema": "http://json-schema.org/draft-07/schema#"},
    ):
        with pytest.raises(AgentError):
            SchemaContract(schema)


@pytest.fixture
def schema_checks(monkeypatch):
    calls = []
    original = schema_module.Draft202012Validator.check_schema

    def check(schema):
        calls.append(copy.deepcopy(schema))
        original(schema)

    schema_module._check_schema.cache_clear()
    monkeypatch.setattr(schema_module.Draft202012Validator, "check_schema", check)
    try:
        yield calls
    finally:
        schema_module._check_schema.cache_clear()


def test_repeated_schema_checks_keep_declarations_and_values_isolated(schema_checks):
    declaration = {"type": "array", "items": {"type": "integer"}}
    first = SchemaContract(declaration)
    second = SchemaContract({"items": {"type": "integer"}, "type": "array"})
    assert len(schema_checks) == 1
    declaration["items"]["type"] = "string"
    exposed = first.declaration
    exposed["items"]["type"] = "string"
    value = [1]
    result = first.validate(value)
    result.append(2)
    assert value == [1]
    assert second.validate(value) == [1]
    for contract in (first, second):
        with pytest.raises(AgentError) as error:
            contract.validate(["invalid"])
        assert error.value.code == "schema_validation"
    assert SchemaContract(declaration).validate(["valid"]) == ["valid"]
    assert len(schema_checks) == 2
    declaration["items"]["type"] = "invalid-type"
    for _ in range(2):
        with pytest.raises(AgentError) as error:
            SchemaContract(declaration)
        assert error.value.code == "invalid_schema"
    assert len(schema_checks) == 4


def test_cached_meta_schema_checks_do_not_bypass_current_policies(schema_checks, monkeypatch):
    for _ in range(2):
        with pytest.raises(AgentError) as error:
            SchemaContract({"$ref": "https://example.test/schema"})
        assert error.value.code == "schema_reference"
    assert len(schema_checks) == 1
    declaration = {"type": "string", "format": "uuid"}
    contract = SchemaContract(declaration)
    with pytest.raises(AgentError):
        contract.validate("invalid")
    monkeypatch.delitem(schema_module.FormatChecker.checkers, "uuid")
    with pytest.raises(AgentError) as error:
        SchemaContract(declaration)
    assert error.value.code == "schema_format"
    assert len(schema_checks) == 2


def test_schema_check_cache_is_bounded_and_large_declarations_are_not_retained(schema_checks):
    for index in range(129):
        SchemaContract({"type": "string", "title": str(index)})
    assert schema_module._check_schema.cache_info().currsize == 128
    SchemaContract({"type": "string", "title": "0"})
    assert len(schema_checks) == 130
    declaration = {"type": "string", "description": "x" * 65536}
    for _ in range(2):
        assert SchemaContract(declaration, max_bytes=131072).validate("valid") == "valid"
    assert len(schema_checks) == 132
    assert schema_module._check_schema.cache_info().currsize == 128


def test_duplicates_unknown_fields_credentials_and_task_cycles_rejected(tmp_path):
    components = registry()
    with pytest.raises(AgentError, match="already registered"):
        components.register_handler("external.calculate", handler)
    mutations = []
    duplicate = package_data()
    duplicate["agents"].append(copy.deepcopy(duplicate["agents"][0]))
    mutations.append(duplicate)
    extra = package_data()
    extra["agents"][0]["import_module"] = "untrusted.module"
    mutations.append(extra)
    secret = package_data()
    secret["models"][0]["options"] = {"api_key": "do-not-store"}
    mutations.append(secret)
    missing_schema = package_data()
    del missing_schema["tools"][0]["output_schema"]
    mutations.append(missing_schema)
    cycle = package_data()
    cycle["agents"][1]["tasks"] = [{"id": "a", "tool": "calculate", "depends_on": ["a"]}]
    mutations.append(cycle)
    for data in mutations:
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(AgentError):
            load_package(path)
    path.write_text('{"agents": [], "agents": []}', encoding="utf-8")
    with pytest.raises(AgentError, match="duplicate JSON"):
        load_package(path)


def test_stream_fragments_are_not_executable_calls_and_message_roles_are_typed():
    delta = ToolCallDelta(index=0, name="calculate", arguments_delta='{"a":')
    with pytest.raises(ValidationError):
        ToolCall.model_validate(delta.model_dump())
    call = ToolCall(call_id="c1", name="calculate", arguments={"a": 1, "b": 2})
    with pytest.raises(ValidationError):
        Message(role="user", tool_calls=[call])
    with pytest.raises(ValidationError):
        Message(role="tool", content="3")
    with pytest.raises(ValidationError):
        ModelResponse(tool_calls=[call, call])
