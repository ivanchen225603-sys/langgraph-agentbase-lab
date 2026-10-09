"""带版本的可信注册表，历史执行绑定不可变。"""

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from pydantic import Field, TypeAdapter

from .configuration import AgentDefinition, AgentPackage, ModelDefinition, ToolDeclaration
from .contracts import (
    AgentError,
    BaseContract,
    ComponentName,
    Identity,
    ProviderFactory,
    ToolHandler,
    Version,
)
from .effects import AttemptCommit
from .prompts import PromptAssembler
from .schema import SchemaContract, content_digest


@dataclass(frozen=True)
class Registration:
    implementation: Any
    version: str
    attempt_commit: AttemptCommit | None = None


class RegistrationSelection(BaseContract):
    """可信部署侧车，用于未固定组件版本的旧配置。"""

    workflows: dict[ComponentName, Version] = Field(default_factory=dict, max_length=64)
    providers: dict[ComponentName, Version] = Field(default_factory=dict, max_length=64)
    handlers: dict[ComponentName, Version] = Field(default_factory=dict, max_length=128)


@dataclass(frozen=True)
class BoundAgent:
    definition: AgentDefinition
    model: ModelDefinition | None
    tools: tuple[ToolDeclaration, ...]
    digest: str
    workflow_registration: Registration
    provider_registration: Registration | None
    handler_registrations: tuple[tuple[str, Registration], ...]
    compensation_registrations: tuple[tuple[str, Registration], ...]


class ComponentRegistry:
    def __init__(self, *, max_bindings: int = 256):
        if type(max_bindings) is not int or not 1 <= max_bindings <= 4096:
            raise AgentError("binding_capacity", "binding capacity must be 1..4096")
        self.max_bindings = max_bindings
        self._handlers: dict[tuple[str, str], Registration] = {}
        self._providers: dict[tuple[str, str], Registration] = {}
        self._workflows: dict[tuple[str, str], Registration] = {}
        self._agents: dict[str, BoundAgent] = {}
        self._snapshots: dict[tuple[str, str], BoundAgent] = {}
        self._agent_versions: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    @staticmethod
    def _register(entries, name, implementation, version, attempt_commit=None):
        TypeAdapter(ComponentName).validate_python(name)
        TypeAdapter(Version).validate_python(version)
        if (name, version) in entries:
            raise AgentError("duplicate_component", "component version is already registered")
        if not callable(implementation):
            raise AgentError("invalid_component", "registered implementation must be callable")
        if attempt_commit is not None and not isinstance(attempt_commit, AttemptCommit):
            raise AgentError("attempt_commit", "invalid trusted attempt registration", 503)
        entries[(name, version)] = Registration(implementation, version, attempt_commit)

    def register_handler(
        self,
        name: str,
        handler: ToolHandler,
        *,
        version: str = "1",
        attempt_commit: AttemptCommit | None = None,
    ):
        with self._lock:
            self._register(self._handlers, name, handler, version, attempt_commit)

    def register_provider(self, name: str, factory: ProviderFactory, *, version: str = "1"):
        with self._lock:
            self._register(self._providers, name, factory, version)

    def register_workflow(self, name: str, factory: Callable, *, version: str = "1"):
        with self._lock:
            self._register(self._workflows, name, factory, version)

    @staticmethod
    def _select(entries, name, version, code):
        if version is not None:
            entry = entries.get((name, version))
            if entry is None:
                raise AgentError(code, "required component version is not registered")
            return entry
        matches = [entry for (entry_name, _), entry in entries.items() if entry_name == name]
        if not matches:
            raise AgentError(code, "required executable component is not registered")
        if len(matches) != 1:
            raise AgentError(
                "component_version_required",
                "multiple registered versions require an explicit selector",
            )
        return matches[0]

    @staticmethod
    def _copy(bound: BoundAgent) -> BoundAgent:
        # 配置是可变的嵌套 JSON；实现引用不可深拷贝。
        return BoundAgent(
            bound.definition.model_copy(deep=True),
            bound.model.model_copy(deep=True) if bound.model else None,
            tuple(tool.model_copy(deep=True) for tool in bound.tools),
            bound.digest,
            bound.workflow_registration,
            bound.provider_registration,
            bound.handler_registrations,
            bound.compensation_registrations,
        )

    def _stage(self, package: AgentPackage, versions: RegistrationSelection):
        # 重新校验嵌套 JSON，包含受信任调用方在解析之后所做的修改。
        package = AgentPackage.model_validate(package.model_dump(mode="json"))
        models = {model.id: model for model in package.models}
        tools = {tool.name: tool for tool in package.tools}
        used: set[tuple[str, str]] = set()

        def select(group, entries, name, pinned, code):
            sidecar = getattr(versions, group).get(name)
            if sidecar is not None:
                used.add((group, name))
                if pinned is not None and pinned != sidecar:
                    raise AgentError(
                        "component_version_conflict", "config and deployment selectors disagree"
                    )
            return self._select(entries, name, pinned if pinned is not None else sidecar, code)

        additions = {}
        for agent in package.agents:
            workflow = select(
                "workflows",
                self._workflows,
                agent.workflow,
                agent.workflow_version,
                "unknown_workflow",
            )
            PromptAssembler.validate(agent)
            model = models[agent.model] if agent.model is not None else None
            provider = (
                select(
                    "providers",
                    self._providers,
                    model.provider,
                    model.provider_version,
                    "unknown_provider",
                )
                if model
                else None
            )
            selected = [tools[name] for name in agent.tools]
            if agent.context.management is not None:
                from .artifact_tools import artifact_read_registration, artifact_read_tool

                reader = next(
                    (tool for tool in selected if tool.name == agent.context.management.read_tool),
                    None,
                )
                if reader is None or reader != artifact_read_tool(
                    name=reader.name, scope=reader.scope
                ):
                    raise AgentError(
                        "context_reader",
                        "context management requires the declared bounded runtime artifact reader",
                    )
            for task in agent.tasks:
                if not task.input_bindings:
                    SchemaContract(tools[task.tool].input_schema).validate(
                        task.arguments, code="task_arguments"
                    )
            handlers = {
                tool.name: select(
                    "handlers",
                    self._handlers,
                    tool.handler,
                    tool.handler_version,
                    "unknown_handler",
                )
                for tool in selected
            }
            if (
                agent.context.management is not None
                and handlers[agent.context.management.read_tool].implementation
                is not artifact_read_registration
            ):
                raise AgentError(
                    "context_reader", "context reader must use the trusted runtime registration"
                )
            compensations = {
                tool.name: select(
                    "handlers",
                    self._handlers,
                    tool.compensation,
                    tool.compensation_version,
                    "unknown_compensation",
                )
                for tool in selected
                if tool.compensation
            }
            # 未指定的 selector 保持原有描述符结构。
            descriptor = {
                "agent": agent.model_dump(mode="json"),
                "model": model.model_dump(mode="json") if model else None,
                "tools": [tool.model_dump(mode="json") for tool in selected],
                "registrations": {
                    "workflow": workflow.version,
                    "provider": provider.version if provider else None,
                    "handlers": {name: entry.version for name, entry in handlers.items()},
                    "compensations": {
                        tool.compensation: compensations[tool.name].version
                        for tool in selected
                        if tool.compensation
                    },
                },
            }
            sidecars = {
                name: sorted(entry.attempt_commit.tables)
                for name, entry in handlers.items()
                if entry.attempt_commit is not None
            }
            if sidecars:
                descriptor["attempt_sidecars"] = sidecars
            digest = content_digest(descriptor)
            existing = self._agent_versions.get((agent.id, agent.version))
            if existing is not None and existing != digest:
                raise AgentError(
                    "agent_version_conflict", "changed Agent binding requires a new Agent version"
                )
            additions[agent.id] = BoundAgent(
                agent,
                model,
                tuple(selected),
                digest,
                workflow,
                provider,
                tuple(handlers.items()),
                tuple(compensations.items()),
            )
        provided = {
            (group, name)
            for group in ("workflows", "providers", "handlers")
            for name in getattr(versions, group)
        }
        if provided != used:
            raise AgentError(
                "unused_component_selector", "deployment selector is not used by this package"
            )
        fresh = {
            (agent_id, bound.digest) for agent_id, bound in additions.items()
        } - self._snapshots.keys()
        if len(self._snapshots) + len(fresh) > self.max_bindings:
            raise AgentError("binding_capacity", "historical binding capacity exhausted")
        return additions

    def _install(self, package, versions, *, activate, duplicate, expected_digests=None):
        with self._lock:
            if duplicate and any(agent.id in self._agents for agent in package.agents):
                raise AgentError("duplicate_agent", "Agent ID is already bound")
            additions = self._stage(package, versions or RegistrationSelection())
            if expected_digests is not None and expected_digests != {
                agent_id: bound.digest for agent_id, bound in additions.items()
            }:
                raise AgentError(
                    "binding_changed", "restored bindings do not match expected release digests"
                )
            # 所有校验先于发布完成；读取方使用同一把锁。
            self._snapshots = {
                **self._snapshots,
                **{(agent_id, bound.digest): bound for agent_id, bound in additions.items()},
            }
            self._agent_versions = {
                **self._agent_versions,
                **{
                    (agent_id, bound.definition.version): bound.digest
                    for agent_id, bound in additions.items()
                },
            }
            if activate:
                self._agents = {**self._agents, **additions}

    def bind(self, package: AgentPackage, *, versions: RegistrationSelection | None = None):
        """首次激活。已激活的 ID 仍会拒绝意外的重复绑定。"""
        self._install(package, versions, activate=True, duplicate=True)

    def activate(self, package: AgentPackage, *, versions: RegistrationSelection | None = None):
        """原子激活已校验的发布版本，同时保留历史绑定。"""
        self._install(package, versions, activate=True, duplicate=False)

    def restore(
        self,
        package: AgentPackage,
        *,
        versions: RegistrationSelection | None = None,
        expected_digests: dict[str, str] | None = None,
    ):
        """注册可信的历史发布数据，不改变当前激活的发布版本。"""
        self._install(
            package, versions, activate=False, duplicate=False, expected_digests=expected_digests
        )

    def agent(self, agent_id: str) -> BoundAgent:
        with self._lock:
            bound = self._agents.get(agent_id)
            if bound is None:
                raise AgentError("unknown_agent", "Agent has no active release")
            return self._copy(bound)

    def resolve(self, agent_id: str, digest: str) -> BoundAgent:
        with self._lock:
            bound = self._snapshots.get((agent_id, digest))
            if bound is None:
                raise AgentError(
                    "binding_changed", "original Agent configuration binding is unavailable", 409
                )
            return self._copy(bound)

    def agents(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": bound.definition.id,
                    "version": bound.definition.version,
                    "description": bound.definition.description,
                    "digest": bound.digest,
                }
                for bound in self._agents.values()
            ]

    def authorized_tool(
        self,
        agent_id: str,
        name: str,
        identity: Identity,
        *,
        digest: str | None = None,
    ) -> ToolDeclaration:
        bound = self.resolve(agent_id, digest) if digest is not None else self.agent(agent_id)
        tool = next((tool for tool in bound.tools if tool.name == name), None)
        if tool is None:
            raise AgentError("tool_not_allowed", "tool is not allowed for this Agent", 403)
        if tool.scope not in identity.scopes:
            raise AgentError("forbidden", "tool scope is not authorized", 403)
        return tool

    def _tool_entry(self, declaration, bound, *, compensation=False):
        original = self.resolve(bound.definition.id, bound.digest)
        tool = next((tool for tool in original.tools if tool.name == declaration.name), None)
        if tool is None or tool != declaration:
            raise AgentError("binding_changed", "tool does not match the original binding", 409)
        entries = (
            original.compensation_registrations if compensation else original.handler_registrations
        )
        entry = dict(entries).get(declaration.name)
        if entry is None:
            raise AgentError(
                "unknown_compensation" if compensation else "unknown_handler",
                "tool implementation is unavailable",
            )
        return entry

    def handler_registration(
        self, declaration: ToolDeclaration, *, bound: BoundAgent
    ) -> Registration:
        with self._lock:
            return self._tool_entry(declaration, bound)

    def handler(
        self, declaration: ToolDeclaration, *, bound: BoundAgent | None = None
    ) -> ToolHandler:
        with self._lock:
            if bound is not None:
                return self._tool_entry(declaration, bound).implementation
            return self._select(
                self._handlers, declaration.handler, declaration.handler_version, "unknown_handler"
            ).implementation

    def compensation_handler(
        self, declaration: ToolDeclaration, *, bound: BoundAgent
    ) -> ToolHandler:
        with self._lock:
            return self._tool_entry(declaration, bound, compensation=True).implementation

    def provider_factory(
        self, definition: ModelDefinition, *, bound: BoundAgent | None = None
    ) -> ProviderFactory:
        with self._lock:
            if bound is not None:
                original = self.resolve(bound.definition.id, bound.digest)
                if original.model != definition or original.provider_registration is None:
                    raise AgentError(
                        "binding_changed", "model does not match the original binding", 409
                    )
                return original.provider_registration.implementation
            return self._select(
                self._providers,
                definition.provider,
                definition.provider_version,
                "unknown_provider",
            ).implementation

    def workflow_factory(
        self, definition: AgentDefinition, *, bound: BoundAgent | None = None
    ) -> Callable:
        with self._lock:
            if bound is not None:
                original = self.resolve(definition.id, bound.digest)
                if original.definition != definition:
                    raise AgentError(
                        "binding_changed", "workflow does not match the original binding", 409
                    )
                return original.workflow_registration.implementation
            return self._select(
                self._workflows,
                definition.workflow,
                definition.workflow_version,
                "unknown_workflow",
            ).implementation

    def validate_arguments(self, declaration: ToolDeclaration, arguments):
        return SchemaContract(declaration.input_schema).validate(arguments, code="tool_arguments")

    def validate_result(self, declaration: ToolDeclaration, result):
        return SchemaContract(declaration.output_schema).validate(
            result, code="tool_result", max_bytes=declaration.result_bytes
        )
