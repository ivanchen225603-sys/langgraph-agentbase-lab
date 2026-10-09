"""可信的历史 tool 发布版本，保存在业务中立的运行时之外。

SQL 记录的是声明（declaration），而不是可执行代码。宿主必须在重启时注册原始的
name/version 实现。绝不导入由 SQL/模型指名的 handler。
"""

import copy
import json
import types
from dataclasses import dataclass, is_dataclass, replace
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Any, ForwardRef, Union, get_args, get_origin

from pydantic import BaseModel, create_model

from agentbase.configuration import ToolDeclaration, ToolExecutionPolicy
from agentbase.schema import SchemaContract, canonical_json, content_digest

from .errors import LabError
from .models import ToolResult

if TYPE_CHECKING:
    from .tools import ToolSpec


@dataclass(frozen=True)
class FrozenTool:
    spec: "ToolSpec"
    legacy: dict
    declaration: ToolDeclaration
    source_schema: type[BaseModel]
    parsing_refs: tuple


@dataclass(frozen=True)
class FrozenManifest:
    digest: str
    payload: dict
    tools: dict[str, FrozenTool]


def frozen_argument_model(root: type[BaseModel]):
    """构造私有的有限模型图（model graph），含嵌套校验器与默认值。

    递归的 Python 模型、未解析的 ref 以及实例/工厂默认值都需要宿主显式提供适配器。
    通用 ToolDeclaration 的 JSON Schema 仍支持嵌套与局部引用；此限制仅针对历史
    Python 桥接层。
    """
    models: dict[type[BaseModel], type[BaseModel]] = {}
    visiting: set[type[BaseModel]] = set()
    refs: list[Any] = []

    def annotation(value):
        if isinstance(value, type) and issubclass(value, BaseModel):
            return model(value)
        if isinstance(value, type) and (
            is_dataclass(value)
            or (hasattr(value, "__required_keys__") and hasattr(value, "__optional_keys__"))
            or (issubclass(value, tuple) and hasattr(value, "_fields"))
        ):
            raise LabError(
                "tool_schema_freeze", "structured Python arguments require an adapter", 409
            )
        if isinstance(value, ForwardRef):
            raise LabError(
                "tool_schema_freeze", "unresolved Python arguments require an adapter", 409
            )
        origin, arguments = get_origin(value), get_args(value)
        if not arguments:
            return value
        revised = tuple(annotation(item) for item in arguments)
        if all(a is b for a, b in zip(arguments, revised, strict=True)):
            return value
        if origin in (Union, types.UnionType):
            return reduce(or_, revised)
        try:
            return origin[revised]
        except (TypeError, AttributeError) as exc:
            raise LabError(
                "tool_schema_freeze", "Python argument annotation requires an adapter", 409
            ) from exc

    def has_model_default(value):
        pending = [value]
        seen = set()
        while pending:
            current = pending.pop()
            if isinstance(current, BaseModel):
                return True
            if isinstance(current, dict):
                if id(current) in seen:
                    continue
                seen.add(id(current))
                pending.extend(current.values())
            elif isinstance(current, (list, tuple, set, frozenset)):
                if id(current) in seen:
                    continue
                seen.add(id(current))
                pending.extend(current)
            if len(seen) > 10000:
                raise LabError(
                    "tool_schema_freeze", "Python argument defaults exceed graph limit", 409
                )
        return False

    def model(source):
        if source in visiting:
            raise LabError(
                "tool_schema_freeze", "recursive Python arguments require an adapter", 409
            )
        if source in models:
            return models[source]
        if len(models) + len(visiting) >= 64:
            raise LabError(
                "tool_schema_freeze", "Python argument model graph exceeds 64 models", 409
            )
        visiting.add(source)
        definitions: dict[str, Any] = {}
        for name, field in source.model_fields.items():
            # 对象类型的默认值/工厂可能包含原始模型实例，或在捕获之后动态构造一个。
            # 一律拒绝，绝不静默共享。
            if has_model_default(field.default) or field.default_factory is not None:
                raise LabError(
                    "tool_schema_freeze", "Python model defaults/factories require an adapter", 409
                )
            definitions[name] = (annotation(field.annotation), copy.deepcopy(field))
            refs.extend(field.metadata)
        private = create_model(
            source.__name__,
            __base__=source,
            __config__=copy.deepcopy(source.model_config),
            __module__=source.__module__,
            **definitions,
        )
        validate, dump = source.model_validate.__func__, source.model_dump
        for attribute, implementation in (
            ("model_validate", classmethod(validate)),
            ("model_dump", dump),
        ):
            setattr(private, attribute, implementation)
        refs.extend(
            (source, validate, dump, source.__pydantic_validator__, source.__pydantic_serializer__)
        )
        visiting.remove(source)
        models[source] = private
        return private

    result = model(root)
    # 自定义类型可能在上述被改写的注解形式之外藏有模型。
    # 因此要检查构造过程，绝不让这类引用静默共享旧类。
    private_models = set(models.values())
    pending, seen = [result.__pydantic_core_schema__], set()
    while pending:
        item = pending.pop()
        if isinstance(item, (dict, list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            if len(seen) > 10000:
                raise LabError("tool_schema_freeze", "Python parser graph exceeds limit", 409)
        if isinstance(item, dict):
            cls = item.get("cls")
            if isinstance(cls, type) and issubclass(cls, BaseModel) and cls not in private_models:
                raise LabError("tool_schema_freeze", "hidden Python model requires an adapter", 409)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return result, tuple(refs)


class LegacyToolCatalog:
    def __init__(self, registry, *, capacity: int = 4096):
        self.registry, self.capacity = registry, capacity
        self._registered: dict[tuple[str, str], ToolSpec] = {}
        self._frozen: dict[tuple[str, str], FrozenTool] = {}

    @staticmethod
    def legacy_declaration(spec):
        # 这个旧 DTO 声明在 requests/tool_contracts 中保持不变。
        return {
            "name": spec.name,
            "version": spec.version,
            "scope": spec.scope,
            "read_only": spec.read_only,
            "retry_safe": spec.retry_safe,
            "timeout": spec.timeout,
            "result_bytes": spec.result_bytes,
            "idempotency": spec.idempotency,
            "compensation": spec.compensation,
            "input_schema": copy.deepcopy(spec.schema.model_json_schema()),
            "output_schema": ToolResult.model_json_schema(),
        }

    @staticmethod
    def snapshot(spec):
        legacy = LegacyToolCatalog.legacy_declaration(spec)
        # StructuredTool 在 ainvoke 中会再次解析参数。仅保留 JSON 快照并不足够，
        # 因为在原类上调用 model_rebuild() 仍能改变实际传入的参数。
        arguments, parsing_refs = frozen_argument_model(spec.schema)
        # 浅拷贝可保留真实的函数/绑定方法，又不会连带复制它们的 service、lock 或
        # runtime。声明中的可变数据则是私有的。
        tool = spec.tool.model_copy(
            update={
                "metadata": copy.deepcopy(spec.tool.metadata),
                "tags": copy.deepcopy(spec.tool.tags),
                "args_schema": arguments,
            },
            deep=False,
        )
        frozen = replace(spec, tool=tool, raw_output_schema=copy.deepcopy(spec.raw_output_schema))
        declaration = ToolDeclaration(
            name=frozen.name,
            handler=frozen.name,
            handler_version=frozen.version,
            version=frozen.version,
            description=tool.description,
            input_schema=copy.deepcopy(legacy["input_schema"]),
            output_schema=copy.deepcopy(frozen.raw_output_schema or {"type": "object"}),
            scope=frozen.scope,
            read_only=frozen.read_only,
            retry_safe=frozen.retry_safe,
            timeout_seconds=frozen.timeout,
            result_bytes=frozen.result_bytes,
            idempotency="read-only" if frozen.read_only else "external-receipt",
            compensation=frozen.compensation,
            execution=ToolExecutionPolicy(
                max_attempts=2,
                backoff_seconds=0.02,
                circuit_failures=3,
                circuit_cooldown_seconds=2.0,
            ),
        )
        return FrozenTool(frozen, legacy, declaration, spec.schema, parsing_refs)

    @staticmethod
    def same(left: FrozenTool, right: FrozenTool):
        return (
            left.legacy == right.legacy
            and left.declaration == right.declaration
            and left.spec.tool.func is right.spec.tool.func
            and left.spec.tool.coroutine is right.spec.tool.coroutine
            and left.source_schema is right.source_schema
            and len(left.parsing_refs) == len(right.parsing_refs)
            and all(a is b for a, b in zip(left.parsing_refs, right.parsing_refs, strict=True))
            and left.spec.tool.response_format == right.spec.tool.response_format
        )

    def register(self, spec):
        key = (spec.name, spec.version)
        if key not in self._registered and len(self._registered) >= self.capacity:
            raise LabError("tool_binding_capacity", "trusted tool release capacity reached", 503)
        prior = self._frozen.get(key)
        if prior is not None and not self.same(prior, self.snapshot(spec)):
            raise LabError(
                "tool_version_conflict", "frozen tool changes require a new version", 409
            )
        self._registered[key] = spec

    def freeze(self, spec):
        candidate = self.snapshot(spec)
        key = (spec.name, spec.version)
        prior = self._frozen.get(key)
        if prior is not None:
            if not self.same(prior, candidate):
                raise LabError(
                    "tool_version_conflict", "frozen tool changes require a new version", 409
                )
            return prior
        if len(self._frozen) >= self.capacity:
            raise LabError("tool_binding_capacity", "frozen tool release capacity reached", 503)
        self.register(spec)
        # 取保守策略：准入失败也保留已捕获的发布版本。任何驱逐或同版本替换
        # 都不能静默改变其他 Run 的 handler。
        self._frozen[key] = candidate
        return candidate

    def capture(self, names):
        names = sorted(set(names))
        if any(name not in self.registry.specs for name in names):
            raise LabError("unknown_tool", "tool is not registered")
        tools = {name: self.freeze(self.registry.specs[name]) for name in names}
        payload = {
            "format": 1,
            "tools": {
                name: {
                    "legacy": copy.deepcopy(tool.legacy),
                    "declaration": tool.declaration.model_dump(mode="json"),
                }
                for name, tool in tools.items()
            },
        }
        return FrozenManifest(content_digest(payload), payload, tools)

    @staticmethod
    def validate_payload(row, manifest):
        request = json.loads(row["payload"])
        names = {task["tool"] for task in request["tasks"]}
        if (
            set(manifest) != {"format", "tools"}
            or manifest["format"] != 1
            or not isinstance(manifest["tools"], dict)
            or set(manifest["tools"]) != names
            or set(request.get("tool_contracts", {})) != names
            or set(request["tool_versions"]) != names
        ):
            raise LabError(
                "tool_manifest_conflict", "tool manifest differs from admitted plan", 409
            )
        for name, item in manifest["tools"].items():
            if set(item) != {"legacy", "declaration"}:
                raise LabError("tool_manifest_conflict", "invalid tool manifest entry", 409)
            declaration = ToolDeclaration.model_validate(item["declaration"])
            legacy = item["legacy"]
            if (
                legacy != request["tool_contracts"][name]
                or declaration.name != name
                or declaration.version != request["tool_versions"][name]
                or declaration.handler != name
                or declaration.handler_version != declaration.version
                or declaration.input_schema != legacy["input_schema"]
                or any(
                    getattr(declaration, field) != legacy[field]
                    for field in (
                        "scope",
                        "read_only",
                        "retry_safe",
                        "result_bytes",
                        "compensation",
                    )
                )
                or declaration.timeout_seconds != legacy["timeout"]
            ):
                raise LabError(
                    "tool_manifest_conflict", "tool declaration differs from admitted plan", 409
                )

    def commit(self, db, row, captured: FrozenManifest):
        if not db.in_transaction:
            raise LabError("tool_manifest_transaction", "manifest needs admission transaction", 503)
        self.validate_payload(row, captured.payload)
        if content_digest(captured.payload) != captured.digest:
            raise LabError("tool_manifest_conflict", "captured manifest changed", 409)
        db.execute(
            "INSERT INTO legacy_tool_manifests VALUES(?,?,?,?)",
            (
                row["run_id"],
                row["binding"],
                captured.digest,
                canonical_json(captured.payload).decode(),
            ),
        )

    def for_run(self, row):
        with self.registry.service.store.connection() as db:
            stored = db.execute(
                "SELECT * FROM legacy_tool_manifests WHERE run_id=?", (row["run_id"],)
            ).fetchone()
        if stored is None:
            raise LabError(
                "legacy_tool_execution_requires_matching_release",
                "old run has no frozen tool execution evidence; query only",
                409,
            )
        payload = json.loads(stored["payload"])
        if stored["binding"] != row["binding"] or content_digest(payload) != stored["digest"]:
            raise LabError("tool_manifest_conflict", "stored tool manifest is inconsistent", 409)
        self.validate_payload(row, payload)
        tools = {}
        for name, item in payload["tools"].items():
            key = (name, item["declaration"]["version"])
            selected = self._frozen.get(key)
            if selected is None:
                # 历史版本的还原绝不回退到当前生效的映射表。
                registered = self._registered.get(key)
                if registered is None:
                    raise LabError(
                        "tool_version_mismatch", "original trusted tool version is unavailable", 409
                    )
                selected = self.snapshot(registered)
                if (
                    selected.legacy != item["legacy"]
                    or selected.declaration.model_dump(mode="json") != item["declaration"]
                ):
                    raise LabError(
                        "tool_version_mismatch", "original trusted tool declaration differs", 409
                    )
                selected = self.freeze(registered)
            if (
                selected.legacy != item["legacy"]
                or selected.declaration.model_dump(mode="json") != item["declaration"]
            ):
                raise LabError("tool_version_mismatch", "original frozen tool differs", 409)
            tools[name] = selected
        return FrozenManifest(stored["digest"], payload, tools)

    def validate_task(self, row, task, principal):
        manifest = self.for_run(row)
        selected = manifest.tools[task.tool]
        if selected.declaration.scope not in principal.scopes:
            raise LabError("forbidden", "original tool scope not authorized", 403)
        SchemaContract(selected.declaration.input_schema).validate(task.arguments)
        return selected
