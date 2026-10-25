"""An external trusted catalog uses the public executor and original SQL plan.

Run: python -m examples.agentbase_catalog. Only disposable databases are used.
This is a host workflow extension; it does not add automatic model tool discovery
or demonstrate legacy RAG migration, real model quality or process crash recovery.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import JsonValue

from agentbase import ComponentRegistry, ResolvedTool, ToolDeclaration, load_package
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, RunContext, ToolCall
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings, FinalProjection, GraphBinding
from agentbase.schema import content_digest
from agentbase.storage import SqliteContextStore, SqliteDatabase


class State(TypedDict, total=False):
    run_id: str
    agent_id: str
    binding: str
    status: str
    result: JsonValue


class CatalogHost:
    def __init__(self, store: SqliteRunStore):
        self.store, self.calls = store, 0
        self.tool = ToolDeclaration(
            name="catalog.upper",
            handler="host.upper",
            handler_version="1",
            description="Convert the admitted text to uppercase",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 200}},
                "required": ["text"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )
        # Retain one actual reference: bound methods created on each lookup differ
        # in object identity. A real host retains all original manifest versions.
        self.handler = self.upper
        self.manifest = content_digest({"tools": [self.tool.model_dump(mode="json")]})

    async def upper(self, arguments: dict[str, JsonValue], context: RunContext) -> JsonValue:
        text = arguments["text"]
        if not isinstance(text, str):
            raise AgentError("tool_arguments", "text must be a string")
        self.calls += 1
        return {"text": text.upper()}

    async def prepare(self, actor: Identity, value: JsonValue) -> dict:
        return {
            "input": value,
            "manifest": self.manifest,
            "tool": self.tool.model_dump(mode="json"),
        }

    def resolve(self, context: RunContext, call: ToolCall) -> ResolvedTool:
        row = self.store.request(context.identity.owner, context.run_id)
        payload = json.loads(row["payload"])
        if (
            row["agent_id"] != context.agent_id
            or row["binding"] != context.config_digest
            or payload.get("manifest") != self.manifest
            or payload.get("tool") != self.tool.model_dump(mode="json")
            or call.call_id != "uppercase"
            or call.name != self.tool.name
            or call.arguments != payload["input"]
        ):
            raise AgentError("tool_not_allowed", "call does not match the original catalog", 403)
        return ResolvedTool(self.tool, self.handler, self.manifest)

    def factory(self, engine: RunEngine, saver, bound) -> GraphBinding:
        async def execute(state: State, config):
            run_id = config["configurable"]["runtime_run_id"]
            row = self.store.internal_request(run_id)
            payload = json.loads(row["payload"])
            observation = await engine.tools.execute(
                await engine.context(run_id),
                ToolCall(call_id="uppercase", name=self.tool.name, arguments=payload["input"]),
            )
            if not observation.succeeded:
                raise AgentError(observation.error or "tool_failure", "catalog action failed", 409)
            return {"result": observation.value}

        async def finish(state: State, config):
            await engine.commit_final(
                config["configurable"]["runtime_run_id"], FinalProjection(value=state["result"])
            )
            return {"status": "succeeded"}

        async def project(row, values, control):
            return {}

        graph = StateGraph(State)
        graph.add_node("execute", execute)
        graph.add_node("finish", finish)
        graph.add_edge(START, "execute")
        graph.add_edge("execute", "finish")
        graph.add_edge("finish", END)
        return GraphBinding(
            graph=graph.compile(checkpointer=saver),
            initial=lambda row: {
                "run_id": row["run_id"],
                "agent_id": row["agent_id"],
                "binding": row["binding"],
                "status": "queued",
            },
            project=project,
            prepare=self.prepare,
        )


async def run_demo():
    actor = Identity(owner="demo", scopes=frozenset({"read"}))

    async def identity(owner):
        return actor if owner == actor.owner else Identity(owner=owner)

    with tempfile.TemporaryDirectory(prefix="agentbase-catalog-") as folder:
        database = SqliteDatabase(Path(folder) / "business.sqlite")
        store = SqliteRunStore(database.connection, EngineSettings())
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            store.install(db)
        contexts = SqliteContextStore(database.connection)
        host = CatalogHost(store)
        registry = ComponentRegistry()
        registry.register_workflow("host.catalog", host.factory)
        registry.bind(load_package(Path(__file__).with_suffix(".json")))
        async with AsyncSqliteSaver.from_conn_string(
            str(Path(folder) / "checkpoints.sqlite")
        ) as saver:
            engine = RunEngine(
                registry,
                store,
                saver,
                ContextProvider(contexts, contexts),
                identity,
                tool_resolver=host.resolve,
            )
            await engine.start()
            try:
                run = await engine.submit(actor, "catalog", "sample", {"text": "Hello Agent"})
                async with asyncio.timeout(10):
                    await engine.queue.join()
                view = await engine.view(actor.owner, run["run_id"])
                assert view["status"] == "succeeded" and view["result"] == {
                    "text": "HELLO AGENT"
                }, view
                assert (await engine.submit(actor, "catalog", "sample", {"text": "Hello Agent"}))[
                    "run_id"
                ] == run["run_id"]
                operations = engine.actions.operations(actor.owner, run["run_id"])
                assert len(operations) == 1 and operations[0]["attempts"] == 1
                assert registry.agent("catalog").definition.tools == [] and host.calls == 1
                return {
                    "result": view["result"],
                    "handler_calls": host.calls,
                    "logical_actions": len(operations),
                    "attempts": operations[0]["attempts"],
                    "uses_public_executor": True,
                    "model_calls": 0,
                }
            finally:
                await engine.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), ensure_ascii=False))
