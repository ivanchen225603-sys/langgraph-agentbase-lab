"""可复用的 LangGraph 会话循环；领域数据保留在声明与 Handler 中。"""

import json
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import JsonValue

from .context import ContextBundle
from .context_management import RuntimeContextManager
from .contracts import AgentError, Message, ModelChunk, ModelResponse
from .prompts import PromptAssembler
from .registry import BoundAgent, ComponentRegistry
from .runtime_contracts import FinalProjection, GraphBinding
from .schema import canonical_json, content_digest


class ConversationState(TypedDict, total=False):
    run_id: str
    agent_id: str
    binding: str
    input: Any
    status: str
    error: str
    messages: list[dict]
    sources: list[dict]
    bundle: dict
    turn_start: int
    cycle: int
    response: dict
    output: Any


def _run_id(config: RunnableConfig) -> str:
    return config["configurable"]["runtime_run_id"]


def _sources(messages: list[Message]) -> list[dict]:
    unique = {}
    for message in messages:
        for source in message.sources:
            unique[canonical_json(source)] = source
    if len(unique) > 100:
        raise AgentError("source_limit", "run source references exceed the limit", 409)
    return list(unique.values())


def conversation_factory(engine, saver, bound: BoundAgent) -> GraphBinding:
    definition = bound.definition

    async def prepare(state: ConversationState, config: RunnableConfig):
        context = await engine.context(_run_id(config))
        bundle = await engine.contexts.load(definition, context)
        tools = [
            {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}
            for tool in bound.tools
            if tool.scope in context.effective_scopes
        ]
        request = PromptAssembler.assemble(definition, state["input"], bundle, tools)
        return {
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "bundle": bundle.model_dump(mode="json"),
            "sources": _sources(request.messages),
            "turn_start": len(request.messages) - 1,
            "cycle": 0,
            "status": "running",
        }

    async def plan_review(state: ConversationState, config: RunnableConfig):
        run_id = _run_id(config)
        control = engine.store.control(run_id)
        if json.loads(control["options"])["pause_for_review"]:
            digest = content_digest(
                {
                    "run": run_id,
                    "binding": bound.digest,
                    "input": state["input"],
                    "kind": "input_review",
                }
            )
            await engine.require_review(
                run_id, digest, {"kind": "input_review", "input": state["input"]}
            )
        return {}

    async def model(state: ConversationState, config: RunnableConfig):
        run_id = _run_id(config)
        context = await engine.context(run_id)
        bundle = ContextBundle.model_validate(state["bundle"])
        if not await engine.contexts.revalidate(bundle, definition, context):
            raise AgentError("source_revoked", "assembled context changed before model use", 403)
        messages = [Message.model_validate(value) for value in state["messages"]]
        sources = _sources(messages)
        row = engine.store.internal_request(run_id)
        if not await engine.authorized_sources(row, context.identity, sources):
            raise AgentError(
                "source_revoked", "tool or context sources changed before model use", 403
            )
        context = await engine.context(run_id)
        cycle = state["cycle"] + 1
        budget = engine.budget_policy(context)
        engine.store.reserve_step(row, f"model:{cycle}", budget.max_steps, context=context)
        request = await RuntimeContextManager(engine).prepare(
            bound, context, messages, current_index=state["turn_start"]
        )
        sources = _sources(request.messages)
        context = await engine.context(run_id)

        async def sink(chunk: ModelChunk):
            await engine.emit(
                run_id,
                "model.delta",
                {"text": chunk.text_delta, "cycle": cycle},
                provisional=True,
                sources=sources,
            )

        response = await engine.model_caller.call(
            engine.provider(bound),
            context,
            budget,
            request,
            sink=sink,
            provisional=definition.stream_policy == "provisional",
            request_guard=lambda: engine.model_request_current(context, sources),
        )
        # 模型调用 ID 只需在单次响应内唯一，因此把它们绑定到本轮次上。
        calls = [
            call.model_copy(update={"call_id": f"c{cycle}_" + content_digest(call.call_id)[7:39]})
            for call in response.tool_calls
        ]
        for call in calls:
            engine.tools.declaration(await engine.context(run_id), call)
        response = response.model_copy(update={"tool_calls": calls})
        messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=calls,
                sources=cast(list[dict[str, JsonValue]], sources),
            )
        )
        return {
            "cycle": cycle,
            "response": response.model_dump(mode="json"),
            "messages": [message.model_dump(mode="json") for message in messages],
            "sources": _sources(messages),
        }

    async def review(state: ConversationState, config: RunnableConfig):
        run_id = _run_id(config)
        response = ModelResponse.model_validate(state["response"])
        for call in response.tool_calls:
            context = await engine.context(run_id)
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
                        "idempotency": declaration.idempotency,
                    },
                )
        return {}

    async def tool(state: ConversationState, config: RunnableConfig):
        run_id = _run_id(config)
        messages = [Message.model_validate(value) for value in state["messages"]]
        for call in ModelResponse.model_validate(state["response"]).tool_calls:
            context = await engine.context(run_id)
            await engine.emit(
                run_id,
                "tool.started",
                {"name": call.name, "call_id": call.call_id},
                sources=state["sources"],
            )
            observation = await engine.tools.execute(context, call)
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=call.call_id,
                    content=canonical_json(
                        {"value": observation.value, "error": observation.error}
                    ).decode(),
                    sources=observation.sources,
                )
            )
            await engine.emit(
                run_id,
                "tool.completed",
                {
                    "name": call.name,
                    "call_id": call.call_id,
                    "succeeded": observation.succeeded,
                    "error": observation.error,
                },
                sources=[*state["sources"], *observation.sources],
            )
        return {
            "messages": [message.model_dump(mode="json") for message in messages],
            "sources": _sources(messages),
        }

    async def finish(state: ConversationState, config: RunnableConfig):
        response = ModelResponse.model_validate(state["response"])
        value = engine.model_caller.final_value(response, definition.output_schema)
        messages = [
            Message.model_validate(message) for message in state["messages"][state["turn_start"] :]
        ]
        await engine.commit_final(
            _run_id(config),
            FinalProjection(value=value, messages=messages, sources=state["sources"]),
        )
        return {"status": "succeeded", "output": value}

    graph = StateGraph(ConversationState)
    for name, node in (
        ("prepare", prepare),
        ("plan_review", plan_review),
        ("model", model),
        ("review", review),
        ("tool", tool),
        ("finish", finish),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "plan_review")
    graph.add_edge("plan_review", "model")
    graph.add_conditional_edges(
        "model",
        lambda state: "review" if state["response"]["tool_calls"] else "finish",
        ["review", "finish"],
    )
    graph.add_edge("review", "tool")
    graph.add_edge("tool", "model")
    graph.add_edge("finish", END)

    def initial(row):
        return {
            "run_id": row["run_id"],
            "agent_id": row["agent_id"],
            "binding": row["binding"],
            "input": json.loads(row["payload"])["input"],
            "status": "queued",
        }

    async def project(row, values, control):
        return {"cycles": values.get("cycle", 0), "error": values.get("error")}

    return GraphBinding(
        graph=graph.compile(checkpointer=saver),
        initial=initial,
        project=project,
        visible_sources=lambda row, values: values.get("sources", []),
    )


def register_workflows(registry: ComponentRegistry):
    from .artifact_tools import artifact_read_registration
    from .fixed_workflow import fixed_factory
    from .planning import planning_factory

    registry.register_workflow("conversation", conversation_factory)
    registry.register_workflow("fixed", fixed_factory)
    registry.register_workflow("planning", planning_factory)
    registry.register_handler("agentbase.artifact_read", artifact_read_registration)
