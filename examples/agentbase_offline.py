"""Two configured businesses share one engine. The provider is an offline fixture.

Run from the repository: python -m examples.agentbase_offline
Temporary databases only. This proves registration and stream protocol, not model
quality, HTTP streaming, process crash recovery or legacy RAG migration.
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentbase import ComponentRegistry, load_package
from agentbase.context import ContextProvider
from agentbase.contracts import Identity, ModelChunk
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings
from agentbase.schema import canonical_json
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.workflows import register_workflows


class OfflineFixture:
    native_stream = True

    def __init__(self):
        self.calls = 0
        self.release = asyncio.Event()
        self.finished = False

    async def complete(self, request):
        raise AssertionError("fixture uses native async streaming")

    async def stream(self, request):
        self.calls += 1
        # Scripted fragments are produced before the fixture is allowed to finish.
        yield ModelChunk(text_delta='{"category":')
        await self.release.wait()
        text = request.messages[-1].content.rstrip()
        yield ModelChunk(text_delta='"question"}' if text.endswith(("?", "？")) else '"statement"}')
        self.finished = True
        yield ModelChunk(finish_reason="stop")

    def count_input_tokens(self, request):
        return len(canonical_json(request.model_dump(mode="json")))

    def count_output_tokens(self, text):
        return len(text.encode("utf-8"))


async def number(arguments, context):
    if context.dependency_results:
        return {"value": sum(value["value"] for value in context.dependency_results.values())}
    return {"value": arguments["x"] * 2}


async def run_demo(package_path: Path | None = None):
    registry = ComponentRegistry()
    fixture = OfflineFixture()
    register_workflows(registry)
    registry.register_provider("demo.fixture", lambda config: fixture)
    registry.register_handler("demo.number", number)
    registry.bind(load_package(package_path or Path(__file__).with_name("agentbase.json")))
    identity = Identity(owner="demo", scopes=frozenset({"read"}))

    async def resolve(owner):
        return identity if owner == identity.owner else Identity(owner=owner)

    # A new temporary directory supplies exclusive ownership. Reusable data directories
    # require an application process lock before schema migration and engine startup.
    with tempfile.TemporaryDirectory(prefix="agentbase-demo-") as directory:
        database = SqliteDatabase(Path(directory) / "business.sqlite")
        store = SqliteRunStore(database.connection, EngineSettings())
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            store.install(db)
        contexts = SqliteContextStore(database.connection)
        async with AsyncSqliteSaver.from_conn_string(
            str(Path(directory) / "checkpoints.sqlite")
        ) as saver:
            engine = RunEngine(registry, store, saver, ContextProvider(contexts, contexts), resolve)
            await engine.start()
            try:

                async def final(run_id):
                    async with asyncio.timeout(10):
                        while True:
                            view = await engine.view("demo", run_id)
                            if view["status"] in {"succeeded", "failed", "partial", "timed_out"}:
                                return view
                            await asyncio.sleep(0.005)

                calculation = await engine.submit(
                    identity, "calculation", "same-key", {"left": 2, "right": 3}
                )
                result = await final(calculation["run_id"])
                assert result["status"] == "succeeded" and result["result"]["total"]["value"] == {
                    "value": 10
                }
                assert fixture.calls == 0
                classification = await engine.submit(
                    identity, "classification", "same-key", "Is this a question?"
                )
                async with asyncio.timeout(10):
                    for _ in range(1000):
                        if any(
                            event.kind == "model.delta"
                            for event in await engine.events("demo", classification["run_id"])
                        ):
                            break
                        await asyncio.sleep(0.005)
                    else:
                        raise AssertionError("fixture delta did not reach the durable event log")
                assert not fixture.finished
                fixture.release.set()
                classified = await final(classification["run_id"])
                assert classified["status"] == "succeeded" and classified["result"] == {
                    "category": "question"
                }
                return {
                    "calculation": result["result"]["total"]["value"],
                    "classification": classified["result"],
                    "fixture_model_calls": fixture.calls,
                    "delta_before_fixture_finish": True,
                    "model_quality_verified": False,
                }
            finally:
                await engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="A package using the demo's registered components"
    )
    print(json.dumps(asyncio.run(run_demo(parser.parse_args().config)), ensure_ascii=False))
