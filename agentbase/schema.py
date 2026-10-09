"""有界的 JSON 与离线 JSON Schema 契约；校验过程绝不获取任何外部资源。"""

import copy
import hashlib
import json
import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable

from .contracts import AgentError

DRAFT = "https://json-schema.org/draft/2020-12/schema"


def canonical_json(value: Any, *, max_bytes: int = 262144) -> bytes:
    """拒绝非 JSON、非有限数、嵌套过深、超出体积上限或含循环引用的取值。"""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > 10000:
            raise AgentError("json_limit", "JSON nesting or node limit exceeded")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise AgentError("invalid_json", "JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif item is None or isinstance(item, (str, bool, int)):
            pass
        elif isinstance(item, float) and math.isfinite(item):
            pass
        else:
            raise AgentError("invalid_json", "value must be finite JSON data")
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise AgentError("invalid_json", "value cannot be encoded as JSON") from exc
    if len(encoded) > max_bytes:
        raise AgentError("json_limit", "JSON byte limit exceeded")
    return encoded


def content_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _deny_retrieval(uri: str):
    raise NoSuchResource(uri)


@lru_cache(maxsize=128)
def _check_schema(encoded: bytes) -> None:
    # Cache successful meta-schema checks only, with at most 128 x 64 KiB keys.
    # Mutable declarations, validators, policy decisions and instance values are not cached.
    Draft202012Validator.check_schema(json.loads(encoded))


class SchemaContract:
    def __init__(self, schema: dict, *, max_bytes: int = 65536):
        canonical_json(schema, max_bytes=max_bytes)
        if not isinstance(schema, dict) or not schema:
            raise AgentError("invalid_schema", "an explicit JSON Schema object is required")
        self._schema = copy.deepcopy(schema)
        encoded = canonical_json(self._schema, max_bytes=max_bytes)
        try:
            if len(encoded) <= 65536:
                _check_schema(encoded)
            else:
                Draft202012Validator.check_schema(self._schema)
        except (SchemaError, TypeError, ValueError, RecursionError) as exc:
            raise AgentError("invalid_schema", "JSON Schema declaration is invalid") from exc
        pending = [self._schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for keyword in ("$ref", "$dynamicRef"):
                    if keyword in item and (
                        not isinstance(item[keyword], str) or not item[keyword].startswith("#")
                    ):
                        raise AgentError(
                            "schema_reference", "only document-local references allowed"
                        )
                if "$schema" in item and (
                    not isinstance(item["$schema"], str) or item["$schema"].rstrip("#") != DRAFT
                ):
                    raise AgentError("schema_draft", "only JSON Schema draft 2020-12 is supported")
                if "format" in item and item["format"] not in FormatChecker.checkers:
                    raise AgentError("schema_format", "schema format checker is not available")
                # 只遍历 Schema 的结构位置，不遍历 const/default/examples 中的 JSON 字面量。
                # 名为 "$ref" 或 "format" 的属性只是普通业务数据。
                for key in (
                    "$defs",
                    "definitions",
                    "properties",
                    "patternProperties",
                    "dependentSchemas",
                ):
                    if isinstance(item.get(key), dict):
                        pending.extend(item[key].values())
                for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
                    if isinstance(item.get(key), list):
                        pending.extend(item[key])
                for key in (
                    "items",
                    "additionalProperties",
                    "unevaluatedProperties",
                    "unevaluatedItems",
                    "contains",
                    "propertyNames",
                    "not",
                    "if",
                    "then",
                    "else",
                    "contentSchema",
                ):
                    if isinstance(item.get(key), dict):
                        pending.append(item[key])
        # 显式提供 registry，可避开 jsonschema 旧版本自动远程获取资源的路径。
        # referencing 由 attrs 生成的构造函数对当前安装的 mypy 不可见。
        registry_factory = cast(Callable[..., Registry], Registry)
        self._validator = Draft202012Validator(
            self._schema,
            registry=registry_factory(retrieve=_deny_retrieval),
            format_checker=FormatChecker(),
        )

    @property
    def declaration(self) -> dict:
        return copy.deepcopy(self._schema)

    def validate(self, value: Any, *, code: str = "schema_validation", max_bytes: int = 262144):
        canonical_json(value, max_bytes=max_bytes)
        try:
            self._validator.validate(value)
        except ValidationError as exc:
            # ValidationError.message 可能内嵌机密实例取值，绝不将其返回。
            raise AgentError(code, "value does not satisfy its declared JSON Schema") from exc
        except (Unresolvable, RecursionError) as exc:
            raise AgentError(
                "schema_reference", "schema reference cannot be resolved locally"
            ) from exc
        return copy.deepcopy(value)
