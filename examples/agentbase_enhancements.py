"""Run four generic enhancements with two Agents and disposable SQLite stores.

The deterministic provider verifies orchestration and provenance, not model quality.
Run with Python from a source distribution, or copy this file beside an installed wheel.
"""

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentbase import AgentPackage, ComponentRegistry, artifact_read_tool
from agentbase.artifacts import ArtifactScope
from agentbase.budgets import SqliteBudgetLedger
from agentbase.context import ContextProvider
from agentbase.contracts import Identity, Message, ModelChunk, ModelRequest, ModelResponse
from agentbase.planning import PlanningStore
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings
from agentbase.schema import canonical_json
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.workflows import register_workflows


class FixtureProvider:
    native_stream = False

    def __init__(self):
        self.requests: list[ModelRequest] = []
        self.proposals = 0

    async def complete(self, request):
        self.requests.append(request)
        properties = (request.output_schema or {}).get("properties", {})
        if isinstance(properties, dict) and "summary" in properties:
            return ModelResponse(text='{"summary":"Prior generic context."}', finish_reason="stop")
        tasks = [
            {"id": "a", "agent_id": "worker", "input": {"value": 7}, "depends_on": []},
            {"id": "b", "agent_id": "worker", "input": {"value": 9}, "depends_on": ["a"]},
        ]
        self.proposals += 1
        if self.proposals > 3:
            raise AssertionError("unexpected extra planning request")
        proposal = {
            "action": "finish" if self.proposals == 3 else "execute",
            "tasks": tasks,
            "result": {"ready": True} if self.proposals == 3 else None,
        }
        return ModelResponse(text=canonical_json(proposal).decode(), finish_reason="stop")

    def count_input_tokens(self, request):
        return len(canonical_json(request.model_dump(mode="json"))) // 4 + 1

    def count_output_tokens(self, text):
        return len(text.encode()) // 4 + 1

    async def stream(self, request):
        raise AssertionError("fixture has no native stream")
        yield ModelChunk()


def configuration():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    reader = artifact_read_tool().model_dump(mode="json")
    return AgentPackage.model_validate(
        {
            "models": [{"id": "local", "provider": "example.plan", "model": "fixture"}],
            "tools": [
                reader,
                {
                    "name": "value",
                    "handler": "example.value",
                    "description": "Return the supplied value",
                    "input_schema": schema,
                    "output_schema": schema,
                    "scope": "read",
                    "read_only": True,
                    "retry_safe": True,
                    "idempotency": "read-only",
                },
            ],
            "agents": [
                {
                    "id": "coordinator",
                    "description": "Coordinate bounded generic tasks",
                    "workflow": "planning",
                    "model": "local",
                    "prompt": {
                        "system_template": "Complete the generic task through the permitted worker."
                    },
                    "input_schema": {"type": "object", "additionalProperties": False},
                    "output_schema": {
                        "type": "object",
                        "properties": {"ready": {"type": "boolean"}},
                        "required": ["ready"],
                        "additionalProperties": False,
                    },
                    "tools": [reader["name"]],
                    "context": {
                        "management": {
                            "retain_recent_units": 1,
                            "summary_chars": 128,
                            "offload_bytes": 1024,
                            "preview_chars": 64,
                        }
                    },
                    "planning": {"max_tasks": 4, "max_revisions": 3},
                    "delegation": {"targets": [{"agent_id": "worker", "scopes": ["read"]}]},
                    "budget": {
                        "max_steps": 32,
                        "max_model_calls": 16,
                        "max_run_tokens": 100000,
                        "max_input_chars": 9000,
                        "max_input_tokens": 3000,
                    },
                },
                {
                    "id": "worker",
                    "description": "Return a typed value",
                    "workflow": "fixed",
                    "prompt": {"system_template": "Use the declared fixed task."},
                    "input_schema": schema,
                    "tools": ["value"],
                    "tasks": [
                        {"id": "value", "tool": "value", "input_bindings": {"value": "/value"}}
                    ],
                },
            ],
        }
    )


async def run_demo():
    provider = FixtureProvider()
    calls = []

    async def value(arguments, context):
        assert context.effective_scopes == frozenset({"read"})
        calls.append(arguments["value"])
        return {"value": arguments["value"]}

    registry = ComponentRegistry()
    register_workflows(registry)
    registry.register_provider("example.plan", lambda options: provider)
    registry.register_handler("example.value", value)
    registry.bind(configuration())
    identity = Identity(owner="example", scopes=frozenset({"read"}))

    async def resolve(owner):
        assert owner == identity.owner
        return identity

    async def allowed(actor, agent_id):
        return actor.owner == identity.owner and agent_id in {"coordinator", "worker"}

    completed = asyncio.Event()

    def observe(kind, row, data):
        if row["agent_id"] == "coordinator" and kind in {"run.final", "run.terminal"}:
            completed.set()

    with tempfile.TemporaryDirectory(prefix="agentbase-enhancements-") as directory:
        folder = Path(directory)
        database = SqliteDatabase(folder / "runs.sqlite")
        settings = EngineSettings(workers=1, queue_size=1)
        store = SqliteRunStore(database.connection, settings)
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            store.install(db)
        history = SqliteContextStore(database.connection)
        for index in range(2):
            history.append_turn(
                identity.owner,
                "coordinator",
                "default",
                0,
                f"history{index}",
                [
                    Message(role="user", content="Earlier input " + "x" * 2000),
                    Message(role="assistant", content="Earlier output " + "y" * 2000),
                ],
                guard=lambda db: True,
            )
        async with AsyncSqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver:
            engine = RunEngine(
                registry,
                store,
                saver,
                ContextProvider(history, history),
                resolve,
                agent_authorizer=allowed,
                observer=observe,
            )
            await engine.start()
            try:
                run = await engine.submit(identity, "coordinator", "example", {})
                row = engine.store.request(identity.owner, run["run_id"])
                async with asyncio.timeout(max(0.0, row["deadline"] - time.time())):
                    await completed.wait()
                    while True:
                        view = await engine.view(identity.owner, run["run_id"])
                        if view["status"] == "succeeded":
                            break
                        if view["status"] in {
                            "failed",
                            "timed_out",
                            "cancelled",
                            "needs_reconciliation",
                        }:
                            raise AssertionError(view)
                        await asyncio.sleep(0.05)
                children = engine.task_coordinator.children(run["run_id"])
                archives = engine.artifacts.list(
                    ArtifactScope(owner=identity.owner, agent_id="coordinator", namespace="default")
                )
                revision = PlanningStore(engine).revision(run["run_id"], 0)
                assert revision is not None and any(
                    source["kind"] == "artifact" for source in revision.sources
                )
                assert calls == [7, 9] and len(children) == 2 and archives
                for child in children:
                    metadata = engine.task_coordinator.metadata(child["run_id"])
                    assert metadata is not None and metadata.input_sources
                assert all(
                    len(canonical_json(request.model_dump(mode="json")).decode()) <= 9000
                    for request in provider.requests
                )
                assert isinstance(engine.model_caller.ledger, SqliteBudgetLedger)
                assert engine.model_caller.ledger.snapshot(run["run_id"])["calls"] == len(
                    provider.requests
                )
                return {
                    "status": view["status"],
                    "result": view["result"],
                    "agents": ["coordinator", "worker"],
                    "handler_values": calls,
                    "children": len(children),
                    "artifacts": len(archives),
                    "model_calls": len(provider.requests),
                    "planning_calls": provider.proposals,
                    "provenance_verified": True,
                    "single_worker": True,
                }
            finally:
                await engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_demo())
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
