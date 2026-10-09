"""声明式 Agent 包（declarative package）。所有可执行绑定均来自可信注册。"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    JsonValue,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from .contracts import AgentError, BaseContract, ComponentName, Identifier, Version
from .schema import SchemaContract, canonical_json, content_digest


class ModelDefinition(BaseContract):
    id: Identifier
    provider: ComponentName
    # 未指定的 selector 保持选用器出现之前的序列化 contract/digest。
    provider_version: Version | None = Field(None, exclude_if=lambda value: value is None)
    model: str = Field(min_length=1, max_length=128)
    credential_env: str | None = Field(None, pattern=r"^[A-Z_][A-Z0-9_]{0,127}$")
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def no_embedded_credentials(self):
        pending: list[Any] = [self.options]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if any(
                    key.lower() in {"api_key", "token", "secret", "password", "authorization"}
                    for key in value
                ):
                    raise ValueError("model credentials must use credential_env")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        canonical_json(self.options, max_bytes=16384)
        return self


class PromptDefinition(BaseContract):
    version: Version = "1"
    system_template: str = Field(min_length=1, max_length=16000)
    user_template: str = Field("{input}", min_length=1, max_length=16000)
    variables: dict[str, JsonValue] = Field(default_factory=dict)


class RuntimeContextPolicy(BaseContract):
    """Optional bounded provider views; canonical history and receipts are unchanged."""

    candidate_messages: int = Field(80, ge=1, le=80)
    candidate_chars: int = Field(131072, ge=1, le=262144)
    trigger_ratio: float = Field(0.8, gt=0, le=1)
    retain_recent_units: int = Field(4, ge=1, le=16)
    max_summary_calls: int = Field(3, ge=1, le=8)
    summary_chars: int = Field(2000, ge=32, le=8000)
    offload_bytes: int = Field(4096, ge=512, le=262144)
    preview_chars: int = Field(256, ge=0, le=4096)
    read_chars: int = Field(4096, ge=1, le=4096)
    read_tool: ComponentName = "artifact.read"

    @model_validator(mode="after")
    def bounded_preview(self):
        if self.preview_chars * 4 >= self.offload_bytes:
            raise ValueError("preview must be smaller than the offload threshold in UTF-8")
        return self


class ContextPolicy(BaseContract):
    max_messages: int = Field(20, ge=0, le=80)
    max_chars: int = Field(8192, ge=1, le=65536)
    memory_scope: Literal["none", "agent", "conversation"] = "none"
    max_memories: int = Field(5, ge=0, le=20)
    shared_memory_agents: list[Identifier] = Field(default_factory=list, max_length=16)
    management: RuntimeContextPolicy | None = Field(None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def unique_sharing(self):
        if len(self.shared_memory_agents) != len(set(self.shared_memory_agents)):
            raise ValueError("shared memory Agent IDs must be unique")
        return self


class BudgetPolicy(BaseContract):
    timeout_seconds: float = Field(30.0, gt=0, le=120)
    review_seconds: float = Field(86400.0, gt=0, le=604800)
    max_model_calls: int = Field(8, ge=1, le=64)
    max_steps: int = Field(8, ge=1, le=64)
    max_tool_calls: int = Field(16, ge=0, le=128)
    max_input_tokens: int = Field(8192, ge=1, le=131072)
    max_input_chars: int = Field(65536, ge=1, le=262144)
    max_output_tokens: int = Field(512, ge=1, le=32768)
    max_run_tokens: int = Field(65536, ge=1, le=1048576)
    max_stream_chunks: int = Field(4096, ge=1, le=32768)


class TaskDeclaration(BaseContract):
    id: Identifier
    tool: ComponentName
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    input_bindings: dict[str, str] = Field(default_factory=dict, max_length=64)
    depends_on: list[Identifier] = Field(default_factory=list, max_length=64)
    use_dependencies: bool = False

    @model_validator(mode="after")
    def valid_input_bindings(self):
        import re

        for name, pointer in self.input_bindings.items():
            if (
                not name
                or len(name) > 128
                or len(pointer) > 1024
                or (pointer and not pointer.startswith("/"))
                or re.search(r"~(?![01])", pointer)
            ):
                raise ValueError("input bindings require bounded JSON Pointers")
            if name in self.arguments:
                raise ValueError("input binding cannot overwrite a static argument")
        return self


class ToolExecutionPolicy(BaseContract):
    """受限的 Handler 重试次数，以及进程内的依赖熔断（circuit）。

    只有只读且可安全重试的 Handler 才允许多于一次尝试。Tool 的 timeout_seconds
    约束的是完整执行阶段，其中包含退避（backoff）时间。
    """

    max_attempts: int = Field(2, ge=1, le=8)
    backoff_seconds: float = Field(0.02, ge=0, le=1)
    circuit_failures: int | None = Field(None, ge=1, le=20)
    circuit_cooldown_seconds: float = Field(2.0, gt=0, le=60)


class ToolDeclaration(BaseContract):
    name: ComponentName
    handler: ComponentName
    handler_version: Version | None = Field(None, exclude_if=lambda value: value is None)
    version: Version = "1"
    description: str = Field(min_length=1, max_length=4000)
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    scope: str = Field("read", min_length=1, max_length=64)
    read_only: bool = True
    retry_safe: bool = True
    timeout_seconds: float = Field(15.0, gt=0, le=120)
    execution: ToolExecutionPolicy | None = Field(None, exclude_if=lambda value: value is None)
    result_bytes: int = Field(60000, ge=1, le=262144)
    idempotency: Literal["read-only", "business-key", "external-receipt"] = "read-only"
    compensation: ComponentName | None = None
    compensation_version: Version | None = Field(None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def tool_contracts(self):
        SchemaContract(self.input_schema)
        SchemaContract(self.output_schema)
        if self.input_schema.get("type") != "object":
            raise ValueError("tool arguments must declare an object schema")
        if not self.read_only and self.idempotency == "read-only":
            raise ValueError("write tools require an explicit effect/idempotency contract")
        if self.compensation_version is not None and self.compensation is None:
            raise ValueError("compensation version requires a compensation handler")
        return self


class DelegationTarget(BaseContract):
    agent_id: Identifier
    scopes: frozenset[str] | None = Field(None, exclude_if=lambda value: value is None)

    @field_validator("scopes", mode="before")
    @classmethod
    def json_scopes(cls, value):
        if isinstance(value, list):
            if not all(isinstance(scope, str) for scope in value) or len(set(value)) != len(value):
                raise ValueError("delegation scopes must be unique strings")
            return frozenset(value)
        return value

    @field_serializer("scopes")
    def stable_scopes(self, value):
        return sorted(value) if value is not None else None


class DelegationPolicy(BaseContract):
    """Trusted optional delegation; targets are frozen at admission."""

    targets: list[DelegationTarget] = Field(min_length=1, max_length=32)
    max_depth: int = Field(4, ge=1, le=8)
    max_children: int = Field(8, ge=1, le=64)
    max_descendants: int = Field(64, ge=1, le=256)

    @model_validator(mode="after")
    def unique_targets(self):
        if len({target.agent_id for target in self.targets}) != len(self.targets):
            raise ValueError("delegation target IDs must be unique")
        if any(target.scopes is not None and len(target.scopes) > 64 for target in self.targets):
            raise ValueError("delegation scope ceilings must be bounded")
        return self


class PlanningPolicy(BaseContract):
    """Optional bounded model proposals; execution authority stays in the runtime."""

    max_tasks: int = Field(16, ge=1, le=64)
    max_revisions: int = Field(3, ge=0, le=16)
    max_proposal_bytes: int = Field(32768, ge=512, le=131072)


class AgentDefinition(BaseContract):
    id: Identifier
    version: Version = "1"
    description: str = Field(min_length=1, max_length=4000)
    workflow: ComponentName = "conversation"
    workflow_version: Version | None = Field(None, exclude_if=lambda value: value is None)
    model: Identifier | None = None
    prompt: PromptDefinition
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None = None
    tools: list[ComponentName] = Field(default_factory=list, max_length=64)
    tasks: list[TaskDeclaration] = Field(default_factory=list, max_length=64)
    context: ContextPolicy = Field(default_factory=ContextPolicy)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    stream_policy: Literal["provisional", "final-only"] = "provisional"
    delegation: DelegationPolicy | None = Field(None, exclude_if=lambda value: value is None)
    planning: PlanningPolicy | None = Field(None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def coherent_agent(self):
        SchemaContract(self.input_schema)
        if self.output_schema is not None:
            SchemaContract(self.output_schema)
        if len(self.tools) != len(set(self.tools)):
            raise ValueError("Agent tool names must be unique")
        if self.workflow == "conversation" and self.model is None:
            raise ValueError("conversation workflows require a model binding")
        if self.workflow == "planning" and (
            self.model is None or self.planning is None or self.delegation is None
        ):
            raise ValueError("planning workflows require model, planning and delegation policies")
        if self.planning is not None and self.workflow != "planning":
            raise ValueError("planning policy requires the optional planning workflow")
        if self.id in self.context.shared_memory_agents:
            raise ValueError("Agent memory sharing cannot repeat its own namespace")
        by_id = {task.id: task for task in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        visited, active = set(), set()

        def visit(task_id):
            if task_id in active:
                raise ValueError("task dependencies must not contain cycles")
            if task_id in visited:
                return
            active.add(task_id)
            for dep in by_id[task_id].depends_on:
                if dep not in by_id:
                    raise ValueError("task dependency is not declared")
                visit(dep)
            active.remove(task_id)
            visited.add(task_id)

        for task in self.tasks:
            if task.tool not in self.tools:
                raise ValueError("task tool is not in the Agent allowlist")
            visit(task.id)
        return self


class AgentPackage(BaseContract):
    schema_version: Literal[1] = 1
    models: list[ModelDefinition] = Field(default_factory=list, max_length=64)
    tools: list[ToolDeclaration] = Field(default_factory=list, max_length=128)
    agents: list[AgentDefinition] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def references(self):
        for entries, attr in ((self.models, "id"), (self.tools, "name"), (self.agents, "id")):
            keys = [getattr(entry, attr) for entry in entries]
            if len(keys) != len(set(keys)):
                raise ValueError("package component IDs must be unique")
        model_ids, tool_names = {m.id for m in self.models}, {t.name for t in self.tools}
        agent_ids = {agent.id for agent in self.agents}
        for agent in self.agents:
            if agent.model is not None and agent.model not in model_ids:
                raise ValueError("Agent model is not declared")
            if not set(agent.tools) <= tool_names:
                raise ValueError("Agent tools must be declared in its package")
            if not set(agent.context.shared_memory_agents) <= agent_ids:
                raise ValueError("shared memory Agent must be declared")
            if (
                agent.delegation is not None
                and not {target.agent_id for target in agent.delegation.targets} <= agent_ids
            ):
                raise ValueError("delegation targets must be declared")
        return self

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


def load_package(path: Path, *, max_bytes: int = 262144) -> AgentPackage:
    """加载可信的部署文件；绝不 import 代码，也不解析网络引用。"""
    try:
        with path.open("rb") as file:
            raw = file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise AgentError("configuration_limit", "Agent configuration byte limit exceeded")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise AgentError("invalid_configuration", "duplicate JSON object key")
                result[key] = value
            return result

        data = json.loads(raw, object_pairs_hook=unique_keys)
        canonical_json(data, max_bytes=max_bytes)
        return AgentPackage.model_validate(data)
    except AgentError:
        raise
    except (OSError, ValueError, TypeError, ValidationError, RecursionError) as exc:
        raise AgentError("invalid_configuration", "Agent configuration cannot be loaded") from exc
