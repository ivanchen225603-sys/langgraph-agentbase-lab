import asyncio
from dataclasses import dataclass
from typing import Literal, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, ConfigDict, Field

from agentbase.contracts import AgentError, ToolCall

from .errors import LabError, UnknownEffect
from .models import AnswerResult, Question, StrictModel, TaskSpec, ToolResult
from .tool_bridge import LegacyInvocation, LegacyToolBridge, invocation
from .tool_catalog import LegacyToolCatalog


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: Question


class SimulationArgs(StrictModel):
    value: str = Field("synthetic", max_length=200)
    delay: float = Field(0, ge=0, le=10)
    failure: Literal["none", "unknown", "permanent"] = "none"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tool: StructuredTool
    version: str = "1"
    scope: str = "read"
    read_only: bool = True
    retry_safe: bool = True
    timeout: float = 15.0
    result_bytes: int = 60000
    idempotency: str = "read-only"
    compensation: str | None = None
    # handler 的原始返回值；历史 describe() 的输出仍是 DTO Schema。
    raw_output_schema: dict | None = None

    @property
    def schema(self) -> type[BaseModel]:
        return cast(type[BaseModel], self.tool.args_schema)


class ToolRegistry:
    def __init__(self, service):
        self.service = service
        self.specs: dict[str, ToolSpec] = {}
        self._descriptions: dict[str, tuple[ToolSpec, dict]] = {}
        self.catalog = LegacyToolCatalog(self)
        self.bridge = LegacyToolBridge(self)
        self.register(
            ToolSpec(
                "rag.answer",
                StructuredTool.from_function(
                    coroutine=self.rag,
                    name="rag.answer",
                    description="Answer only from authorized knowledge evidence",
                    args_schema=ReadArgs,
                ),
                raw_output_schema=AnswerResult.model_json_schema(),
                version="2",
            )
        )
        self.register(
            ToolSpec(
                "memory.search",
                StructuredTool.from_function(
                    coroutine=self.memory,
                    name="memory.search",
                    description="Read this principal's explicit active memories",
                    args_schema=ReadArgs,
                ),
                raw_output_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "memories": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["query", "memories"],
                    "additionalProperties": False,
                },
                version="2",
            )
        )
        if service.settings.enable_simulation:
            self.register(
                ToolSpec(
                    "demo.write",
                    StructuredTool.from_function(
                        coroutine=self.write,
                        name="demo.write",
                        description="Isolated synthetic write; requires review",
                        args_schema=SimulationArgs,
                    ),
                    scope="memory:write",
                    read_only=False,
                    retry_safe=False,
                    idempotency="stable run_id:task_id + durable intent/receipt",
                    compensation="simulation.compensate",
                    version="2",
                    raw_output_schema={
                        "type": "object",
                        "properties": {
                            "effect": {"const": "applied"},
                            "operation_key": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["effect", "operation_key", "value"],
                        "additionalProperties": False,
                    },
                )
            )

    def register(self, spec, *, activate=True):
        self.catalog.register(spec)
        if activate:
            self.specs[spec.name] = spec

    @property
    def slots(self):
        return self.service.runtime.tools.slots

    def validate(self, tasks: list[TaskSpec], principal, *, manifest=None):
        ids = [t.id for t in tasks]
        if len(set(ids)) != len(ids):
            raise LabError("invalid_plan", "duplicate task IDs")
        visited, active = set(), set()
        by_id = {t.id: t for t in tasks}

        def visit(task_id):
            if task_id in active:
                raise LabError("invalid_plan", "dependency cycle or self dependency")
            if task_id in visited:
                return
            active.add(task_id)
            for dep in by_id[task_id].depends_on:
                if dep not in by_id:
                    raise LabError("invalid_plan", "unknown dependency")
                visit(dep)
            active.remove(task_id)
            visited.add(task_id)

        for task in tasks:
            spec = (
                manifest.tools[task.tool].spec
                if manifest is not None
                else self.specs.get(task.tool)
            )
            if not spec:
                raise LabError("unknown_tool", "tool is not registered")
            if spec.scope not in principal.scopes:
                raise LabError("forbidden", "tool scope not authorized", 403)
            spec.schema.model_validate(task.arguments)
            visit(task.id)

    def describe(self):
        return [self.catalog.legacy_declaration(spec) for spec in self.specs.values()]

    async def rag(self, question: str, config: RunnableConfig, schema_version: int = 1):
        ctx = config["configurable"]["tool_context"]
        history = list(ctx["history"])
        if ctx["use_dependencies"]:
            for r in ctx["dependencies"].values():
                v = self.service.sanitize_value(ctx["principal"], r["value"])
                if "answer" in v and v.get("status") == "answered":
                    history.append(
                        {
                            "question": v["resolved_question"],
                            "answer": v["answer"][:1000],
                            "source_run": ctx["run_id"],
                            "documents": list(
                                dict.fromkeys(
                                    [c["document_id"] for c in v["citations"]]
                                    + v.get("context_documents", [])
                                )
                            ),
                        }
                    )
        result = await self.service.rag.graph.ainvoke(
            {
                "principal": ctx["principal"],
                "question": question,
                "history": history[-4:],
                "index_version": ctx["index_version"],
                "deadline": ctx["deadline"],
                "params": ctx["params"],
                "branches": {},
                "degradation": [],
            },
            config=config,
        )
        return self.service.sanitize_value(ctx["principal"], result["answer"])

    async def memory(self, question: str, config: RunnableConfig, schema_version: int = 1):
        ctx = config["configurable"]["tool_context"]
        return {
            "memories": self.service.memories.search(ctx["principal"], question),
            "query": question,
        }

    async def write(
        self, config: RunnableConfig, value="synthetic", delay=0.0, failure="none", schema_version=1
    ):
        ctx = config["configurable"]["tool_context"]
        key = ctx["operation_key"]
        previous = self.service.store.operation(key, ctx["principal"], ctx["run_id"])
        if previous is not None:
            return previous
        await asyncio.sleep(delay)
        if failure == "permanent":
            result = {"effect": "not_applied"}
            self.service.store.receipt(key, result, "not_applied")
            raise LabError("simulation_failure", "synthetic write failed before effect", 409)
        self.service.store.simulation_effect(key, value)
        if failure == "unknown":
            raise UnknownEffect()
        result = {"effect": "applied", "operation_key": key, "value": value}
        self.service.store.receipt(key, result)
        return result

    async def execute(self, task, state, dependencies, config=None):
        row = self.service.runtime_adapter.validate_task(state, config)
        context = await self.bridge.action_context(row, task, dependencies)
        selected = self.catalog.validate_task(row, task, context.identity)
        operation_key = f"{state['run_id']}:{task.id}"
        call = ToolCall(call_id=task.id, name=task.tool, arguments=task.arguments)
        scope = invocation.set(LegacyInvocation(state, dependencies, config or {}))
        try:
            observed = await self.service.runtime.tools.execute(context, call)
            return ToolResult(
                task_id=task.id,
                tool=task.tool,
                status="succeeded" if observed.succeeded else "failed",
                value=observed.value if observed.succeeded else {},
                error=observed.error,
                operation_key=None if selected.declaration.read_only else operation_key,
            )
        except (asyncio.CancelledError, GraphInterrupt):
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, (LabError, AgentError)) else "tool_failure"
            return ToolResult(
                task_id=task.id,
                tool=task.tool,
                status="needs_reconciliation" if code == "needs_reconciliation" else "failed",
                error=code,
                operation_key=None if selected.declaration.read_only else operation_key,
            )
        finally:
            invocation.reset(scope)
            self.bridge.settle_unstarted(row)
