"""声明的任务依赖以 LangGraph Send 波次执行，而不是引入第二套调度器。"""

import copy
import json
import re
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import JsonValue

from .contracts import AgentError, Message, ToolCall
from .registry import BoundAgent
from .runtime_contracts import FinalProjection, GraphBinding
from .schema import canonical_json
from .tools import ToolObservation


def input_pointer(value, pointer: str):
    """仅有界的 RFC 6901 查询。不做求值、不访问属性、不抓取 URI。"""
    current = value
    if pointer == "":
        return copy.deepcopy(current)
    if not pointer.startswith("/") or len(pointer) > 1024 or re.search(r"~(?![01])", pointer):
        raise AgentError("input_binding", "invalid configured input pointer")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and re.fullmatch(r"0|[1-9][0-9]*", token)
            and len(token) < 10
            and int(token) < len(current)
        ):
            current = current[int(token)]
        else:
            raise AgentError("input_binding", "configured input pointer does not resolve", 422)
    return copy.deepcopy(current)


def merge_results(left: dict, right: dict) -> dict:
    merged = dict(left)
    for key, value in right.items():
        if key in merged and canonical_json(merged[key]) != canonical_json(value):
            raise AgentError("task_result_conflict", "completed task observation changed", 409)
        merged[key] = value
    return merged


class FixedState(TypedDict, total=False):
    run_id: str
    agent_id: str
    binding: str
    status: str
    error: str
    results: Annotated[dict[str, dict], merge_results]
    task_id: str
    dependencies: dict[str, dict]
    input_review_done: bool


def fixed_factory(engine, saver, bound: BoundAgent) -> GraphBinding:
    definition = bound.definition
    by_id = {task.id: task for task in definition.tasks}

    def ready(state):
        completed = state.get("results", {})
        eligible = [
            task.id
            for task in definition.tasks
            if task.id not in completed and all(dep in completed for dep in task.depends_on)
        ]
        declarations = {tool.name: tool for tool in bound.tools}
        reads = [task_id for task_id in eligible if declarations[by_id[task_id].tool].read_only]
        # 只读波次并行执行；写操作保持单一且明确的 effect 归属方。
        return reads or eligible[:1]

    def call_for(row, task_id):
        return ToolCall(
            call_id=task_id,
            name=by_id[task_id].tool,
            arguments=json.loads(row["payload"])["task_arguments"][task_id],
        )

    def sources(results):
        unique = {}
        for result in results.values():
            for source in result.get("sources", []):
                unique[canonical_json(source)] = source
        if len(unique) > 100:
            raise AgentError("source_limit", "task source references exceed the limit", 409)
        return list(unique.values())

    async def task_context(run_id, task, dependencies):
        context = await engine.context(run_id)
        if task.use_dependencies:
            context = context.model_copy(
                update={
                    "dependency_results": {
                        key: value["value"] for key, value in dependencies.items()
                    }
                },
                deep=True,
            )
        return context

    async def prepare(identity, value: JsonValue):
        arguments = {}
        for task in definition.tasks:
            declaration = engine.registry.authorized_tool(
                definition.id, task.tool, identity, digest=bound.digest
            )
            args = copy.deepcopy(task.arguments)
            args.update(
                {
                    name: input_pointer(value, pointer)
                    for name, pointer in task.input_bindings.items()
                }
            )
            engine.registry.validate_arguments(declaration, args)
            arguments[task.id] = args
        return {"input": value, "task_arguments": arguments}

    async def review(state: FixedState, config: RunnableConfig):
        run_id = config["configurable"]["runtime_run_id"]
        row = engine.store.internal_request(run_id)
        results = state.get("results", {})
        if (
            not state.get("input_review_done")
            and json.loads(engine.store.control(run_id)["options"])["pause_for_review"]
        ):
            from .schema import content_digest

            payload = json.loads(row["payload"])
            digest = content_digest(
                {
                    "kind": "input_review",
                    "run": run_id,
                    "binding": bound.digest,
                    "input": payload["input"],
                    "arguments": payload["task_arguments"],
                }
            )
            await engine.require_review(
                run_id,
                digest,
                {
                    "kind": "input_review",
                    "input": payload["input"],
                    "arguments": payload["task_arguments"],
                },
            )
        for task_id in ready(state):
            task = by_id[task_id]
            dependencies = {key: results[key] for key in task.depends_on}
            if any(not value["succeeded"] for value in dependencies.values()):
                continue
            context = await task_context(run_id, task, dependencies)
            call = call_for(row, task_id)
            declaration = engine.tools.declaration(context, call)
            if not declaration.read_only:
                digest = engine.tools.challenge(context, call)
                await engine.require_review(
                    run_id,
                    digest,
                    {
                        "kind": "tool_action",
                        "tool": call.name,
                        "call_id": call.call_id,
                        "arguments": call.arguments,
                        "dependencies": context.dependency_results,
                    },
                )
        return {"status": "running", "input_review_done": True}

    def dispatch(state):
        results = state.get("results", {})
        return [
            Send(
                "execute_task",
                {
                    "task_id": task_id,
                    "dependencies": {key: results[key] for key in by_id[task_id].depends_on},
                },
            )
            for task_id in ready(state)
        ]

    async def execute_task(state: FixedState, config: RunnableConfig):
        run_id = config["configurable"]["runtime_run_id"]
        row = engine.store.internal_request(run_id)
        task_id, dependencies = state["task_id"], state["dependencies"]
        task = by_id[task_id]
        step_context = await engine.context(run_id)
        engine.store.reserve_step(
            row,
            f"task:{task_id}",
            engine.budget_policy(step_context).max_steps,
            context=step_context,
        )
        if any(not value["succeeded"] for value in dependencies.values()):
            result = ToolObservation(call_id=task_id, name=task.tool, error="dependency_failed")
        else:
            context = await task_context(run_id, task, dependencies)
            call = call_for(row, task_id)
            refs = sources(dependencies)
            if not await engine.authorized_sources(row, context.identity, refs):
                raise AgentError("source_revoked", "dependency source is no longer authorized", 403)
            await engine.emit(
                run_id, "tool.started", {"call_id": task_id, "name": task.tool}, sources=refs
            )
            result = await engine.tools.execute(context, call)
            inherited = list(
                {canonical_json(ref): ref for ref in [*refs, *result.sources]}.values()
            )
            result = result.model_copy(update={"sources": inherited})
            await engine.emit(
                run_id,
                "tool.completed",
                {
                    "call_id": task_id,
                    "name": task.tool,
                    "succeeded": result.succeeded,
                    "error": result.error,
                },
                sources=inherited,
            )
        return {"results": {task_id: result.model_dump(mode="json")}}

    async def finish(state: FixedState, config: RunnableConfig):
        run_id = config["configurable"]["runtime_run_id"]
        results = state.get("results", {})
        status = (
            "succeeded"
            if all(value["succeeded"] for value in results.values())
            else "partial"
            if any(value["succeeded"] for value in results.values())
            else "failed"
        )
        row = engine.store.internal_request(run_id)
        value = json.loads(row["payload"])["input"]
        refs = sources(results)
        messages = [
            Message(
                role="user",
                content=value if isinstance(value, str) else canonical_json(value).decode(),
            ),
            Message(role="assistant", content=canonical_json(results).decode(), sources=refs),
        ]
        await engine.commit_final(
            run_id,
            FinalProjection(
                value=cast(JsonValue, results),
                status=cast(Literal["succeeded", "partial", "failed"], status),
                messages=messages,
                sources=refs,
            ),
        )
        return {"status": status}

    graph = StateGraph(FixedState)
    graph.add_node("review", review)
    graph.add_node("execute_task", execute_task)
    graph.add_node("collect", lambda state: {})
    graph.add_node("finish", finish)
    graph.add_edge(START, "review")
    graph.add_conditional_edges(
        "review",
        lambda state: dispatch(state) if ready(state) else "finish",
        ["execute_task", "finish"],
    )
    graph.add_edge("execute_task", "collect")
    graph.add_conditional_edges(
        "collect", lambda state: "review" if ready(state) else "finish", ["review", "finish"]
    )
    graph.add_edge("finish", END)

    def initial(row):
        return {
            "run_id": row["run_id"],
            "agent_id": row["agent_id"],
            "binding": row["binding"],
            "status": "queued",
            "results": {},
        }

    async def project(row, values, control):
        return {"error": values.get("error"), "completed_tasks": len(values.get("results", {}))}

    return GraphBinding(
        graph=graph.compile(checkpointer=saver),
        initial=initial,
        project=project,
        prepare=prepare,
        visible_sources=lambda row, values: sources(values.get("results", {})),
    )
