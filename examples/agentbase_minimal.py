"""Smallest offline extension: register tools, bind configuration, run the shared engine.

Run: python -m examples.agentbase_minimal
No model credentials or persistent user data are needed.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import JsonValue

from agentbase import AgentPackage, ComponentRegistry, MemoryLimits
from agentbase.context import ContextProvider
from agentbase.contracts import Identity
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import TERMINAL, EngineSettings
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.workflows import register_workflows


async def run_demo():
    registry = ComponentRegistry()
    register_workflows(registry)
    calls = []

    async def echo(arguments, context):
        assert context.effective_scopes == frozenset({"read"})
        calls.append("echo")
        return arguments["text"]

    async def result(arguments, context):
        calls.append("result")
        return {"text": context.dependency_results["echo"]}

    registry.register_handler("example.echo", echo)
    registry.register_handler("example.result", result)
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "maxLength": 100}},
        "required": ["text"],
        "additionalProperties": False,
    }
    registry.bind(
        AgentPackage.model_validate(
            {
                "tools": [
                    {
                        "name": "echo",
                        "handler": "example.echo",
                        "description": "Return input text",
                        "input_schema": schema,
                        "output_schema": {"type": "string"},
                    },
                    {
                        "name": "result",
                        "handler": "example.result",
                        "description": "Collect the preceding tool result",
                        "input_schema": {"type": "object", "additionalProperties": False},
                        "output_schema": schema,
                    },
                ],
                "agents": [
                    {
                        "id": "minimal",
                        "description": "Generic fixed workflow example",
                        "workflow": "fixed",
                        "prompt": {"system_template": "Execute the declared tasks."},
                        "input_schema": schema,
                        "tools": ["echo", "result"],
                        "tasks": [
                            {"id": "echo", "tool": "echo", "input_bindings": {"text": "/text"}},
                            {
                                "id": "result",
                                "tool": "result",
                                "depends_on": ["echo"],
                                "use_dependencies": True,
                            },
                        ],
                    }
                ],
            }
        )
    )
    actor = Identity(owner="demo", scopes=frozenset({"read"}))

    async def identity(owner):
        return actor if owner == actor.owner else None

    with tempfile.TemporaryDirectory(prefix="agentbase-minimal-") as directory:
        database = SqliteDatabase(Path(directory) / "runs.sqlite")
        store = SqliteRunStore(database.connection, EngineSettings())
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            store.install(db)
        contexts = SqliteContextStore(database.connection, memory_limits=MemoryLimits())
        async with AsyncSqliteSaver.from_conn_string(
            str(Path(directory) / "checkpoints.sqlite")
        ) as saver:
            engine = RunEngine(
                registry, store, saver, ContextProvider(contexts, contexts), identity
            )
            await engine.start()
            try:
                payload: JsonValue = {"text": "hello"}
                run = await engine.submit(actor, "minimal", "request-1", payload)
                async with asyncio.timeout(10):
                    while True:
                        view = await engine.view(actor.owner, run["run_id"])
                        if view["status"] == "succeeded":
                            break
                        if view["status"] in TERMINAL or view["interrupts"]:
                            raise RuntimeError(view)
                        await asyncio.sleep(0.02)
                assert view["result"]["result"]["value"] == payload
                replay = await engine.submit(actor, "minimal", "request-1", payload)
                assert replay["run_id"] == run["run_id"] and calls == ["echo", "result"]
                return {
                    "status": view["status"],
                    "result": payload,
                    "tool_calls": len(calls),
                    "idempotency_verified": True,
                }
            finally:
                await engine.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), ensure_ascii=False, indent=2))
