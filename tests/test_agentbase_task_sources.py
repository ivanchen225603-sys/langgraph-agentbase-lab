import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from langgraph.graph import END, START, StateGraph

from agentbase.artifacts import ArtifactOperation, ArtifactScope
from agentbase.configuration import AgentPackage
from agentbase.contracts import Message, ModelRequest, ModelResponse
from agentbase.runtime_contracts import FinalProjection, GraphBinding
from agentbase.task_tree import ChildTask
from agentbase.tools import ToolResult
from tests.test_agentbase_task_tree import ParentState, package, system, wait


def factory(ready, release, nested=False):
    def build(engine, saver, bound):
        async def finish(state, config):
            run_id = state["run_id"]
            target = "child" if state["agent_id"] == "parent" else "grandchild"
            children = await engine.dispatch_children(
                run_id, [ChildTask(id="one", agent_id=target, input={"x": 1})]
            )
            result = await engine.wait_children(run_id, "all", children)
            if state["agent_id"] == "parent":
                ready.set()
                await release.wait()
            await engine.commit_final(
                run_id,
                FinalProjection(
                    value=result, sources=[ref for item in result for ref in item["sources"]]
                ),
            )
            return {"status": "succeeded"}

        async def project(row, values, info):
            return info

        graph = StateGraph(ParentState)
        graph.add_node("finish", finish)
        graph.add_edge(START, "finish")
        graph.add_edge("finish", END)
        return GraphBinding(
            graph=graph.compile(checkpointer=saver),
            initial=lambda row: {
                "run_id": row["run_id"],
                "agent_id": row["agent_id"],
                "binding": row["binding"],
                "status": "running",
            },
            project=project,
        )

    return build


async def echo(arguments, context):
    return {"value": arguments["x"]}


async def breadth_probe(counts, expected):
    ready, release = asyncio.Event(), asyncio.Event()
    seen_external, seen_local = [], []
    config = package().model_dump(mode="json")
    config["agents"][1]["context"] = {"memory_scope": "conversation", "max_memories": 20}

    async def policy(context, declaration, value):
        records = [
            engine.contexts.memory_store.create_memory(
                context.identity.owner,
                context.agent_id,
                context.conversation_id,
                f"memory {i}",
                sources=[{"kind": "external", "id": f"{i}_{j}"} for j in range(n)],
            )
            for i, n in enumerate(counts)
        ]
        return ToolResult(
            value=value, sources=[engine.contexts.memory_reference(record) for record in records]
        )

    with tempfile.TemporaryDirectory(prefix="source_breadth_") as folder:
        async with system(
            Path(folder),
            echo,
            configuration=AgentPackage.model_validate(config),
            workflow_factory=factory(ready, release),
            result_policy=policy,
        ) as (engine, identity):

            async def check(identity, agent_id, source):
                seen_external.append((agent_id, source["id"]))
                return True

            engine.contexts.source_checker = check
            engine.local_source_guard = lambda db, row, source: (
                seen_local.append((row["agent_id"], source["id"])) is None
            )
            parent = await engine.submit(identity, "parent", "breadth", {})
            if not expected:
                done = await wait(engine, parent["run_id"])
                assert done["status"] == "failed", done
                assert not engine.task_coordinator.consumed(parent["run_id"])
                print(
                    json.dumps(
                        {
                            "probe": "breadth_limit",
                            "visits": 1 + len(counts) + sum(counts),
                            "status": done["status"],
                            "consumed": 0,
                        }
                    )
                )
                return
            await asyncio.wait_for(ready.wait(), 10)
            row = engine.store.internal_request(parent["run_id"])
            refs = engine.task_coordinator.consumed(parent["run_id"])
            assert len(refs) == 1
            for supplied in ([], refs, refs + refs):
                seen_external.clear()
                seen_local.clear()
                assert await engine.authorized_sources(row, identity, supplied)
                assert len(seen_external) == sum(counts), len(seen_external)
                assert len(seen_local) == sum(counts), len(seen_local)
            current = await engine.context(parent["run_id"])
            seen_local.clear()
            assert engine.model_request_current(current, refs + refs)
            assert len(seen_local) == sum(counts)

            class Provider:
                native_stream = False
                calls = 0

                def count_input_tokens(self, request):
                    return 1

                def count_output_tokens(self, value):
                    return 1

                async def complete(self, request):
                    self.calls += 1
                    return ModelResponse(text="okay", finish_reason="stop")

            provider = Provider()
            budget = engine.budget_policy(current)
            await engine.model_caller.call(
                provider,
                current,
                budget,
                ModelRequest(
                    messages=[Message(role="user", content="probe", sources=refs)],
                    max_output_tokens=budget.max_output_tokens,
                ),
                streaming=False,
                request_guard=lambda: engine.model_request_current(current, refs),
            )
            assert provider.calls == 1
            event = await engine.emit(
                parent["run_id"], "result.projection", {"value": "derived"}, sources=refs + refs
            )
            assert event.data["sources"] == refs
            assert (await engine.events("user", parent["run_id"], event.seq - 1))[0] == event
            engine.check_event_frame(identity, event)
            release.set()
            done = await wait(engine, parent["run_id"])
            assert done["status"] == "succeeded", done
            final = engine.store.final_receipt("user", parent["run_id"])
            assert final["sources"] == refs, final
            print(
                json.dumps(
                    {
                        "probe": "breadth_dedup",
                        "visits": 1 + len(counts) + sum(counts),
                        "external_checks_per_authorization": sum(counts),
                        "supplied_ref_counts": [0, 1, 2],
                        "provider_calls": provider.calls,
                        "event_sources": len(event.data["sources"]),
                        "status": done["status"],
                    }
                )
            )


async def depth_probe():
    ready, release = asyncio.Event(), asyncio.Event()
    with tempfile.TemporaryDirectory(prefix="source_depth_") as folder:
        async with system(Path(folder), echo, workflow_factory=factory(ready, release)) as (
            engine,
            identity,
        ):
            parent = await engine.submit(identity, "parent", "depth", {})
            await asyncio.wait_for(ready.wait(), 10)
            row = engine.store.internal_request(parent["run_id"])
            context = await engine.context(parent["run_id"])
            refs = engine.task_coordinator.consumed(parent["run_id"])
            sources = refs
            results = []
            for n in range(1, 18):
                artifact = engine.artifacts.create(
                    ArtifactScope(
                        owner="user", agent_id="parent", namespace=context.conversation_id
                    ),
                    ArtifactOperation(
                        run_id=context.run_id, key=f"depth_{n}", config_digest=context.config_digest
                    ),
                    {"n": n},
                    sources=sources,
                    guard=lambda db: True,
                )
                sources = [artifact.reference()]
                if n >= 16:
                    async_ok = await engine.authorized_sources(row, identity, sources)
                    sql_ok = engine.model_request_current(context, sources)
                    assert async_ok == (n == 16), (n, async_ok)
                    assert sql_ok == (n == 16), (n, sql_ok)
                    results.append({"artifacts": n, "async": async_ok, "sql": sql_ok})
            print(json.dumps({"probe": "empty_receipt_depth", "results": results}))
            release.set()
            assert (await wait(engine, parent["run_id"]))["status"] == "succeeded"


async def scope_probe():
    ready, release = asyncio.Event(), asyncio.Event()
    config = package().model_dump(mode="json")
    grandchild = dict(config["agents"][1])
    grandchild["id"] = "grandchild"
    config["agents"].append(grandchild)
    config["agents"][1]["workflow"] = "fixture.parent"
    config["agents"][1]["tasks"] = []
    config["agents"][1]["delegation"] = {"targets": [{"agent_id": "grandchild"}]}
    config["agents"][0]["context"] = {"memory_scope": "conversation"}

    async def policy(context, declaration, value):
        return ToolResult(value=value, sources=[{"kind": "external", "required": "write"}])

    with tempfile.TemporaryDirectory(prefix="source_scope_") as folder:
        async with system(
            Path(folder),
            echo,
            configuration=AgentPackage.model_validate(config),
            workflow_factory=factory(ready, release, True),
            result_policy=policy,
        ) as (engine, identity):
            external_scopes = []

            async def checker(identity, agent_id, source):
                external_scopes.append((agent_id, sorted(identity.scopes)))
                return source["required"] in identity.scopes

            engine.contexts.source_checker = checker
            parent = await engine.submit(identity, "parent", "scope", {})
            await asyncio.wait_for(ready.wait(), 10)
            row = engine.store.internal_request(parent["run_id"])
            context = await engine.context(parent["run_id"])
            refs = engine.task_coordinator.consumed(parent["run_id"])
            narrowed = identity.model_copy(update={"scopes": frozenset({"read"})})
            external_scopes.clear()
            assert not await engine.authorized_sources(row, narrowed, refs)
            assert external_scopes == [("grandchild", ["read"])], external_scopes
            memory = engine.contexts.memory_store.create_memory(
                "user", "parent", context.conversation_id, "root wrapper", sources=refs
            )
            memory_ref = engine.contexts.memory_reference(memory)
            artifact = engine.artifacts.create(
                ArtifactScope(owner="user", agent_id="parent", namespace=context.conversation_id),
                ArtifactOperation(
                    run_id=context.run_id, key="scope_wrapper", config_digest=context.config_digest
                ),
                {"value": "wrapper"},
                sources=[memory_ref],
                guard=lambda db: True,
            )
            wrapped = [artifact.reference()]
            external_scopes.clear()
            assert not await engine.authorized_sources(row, narrowed, wrapped)
            assert external_scopes == [("grandchild", ["read"])], external_scopes
            original_expand = engine._expand_child_source_in
            expanded_scopes = []

            def capture(db, frame, source):
                expansion = original_expand(db, frame, source)
                expanded_scopes.append(
                    (
                        frame.agent_id,
                        sorted(frame.identity.scopes),
                        None if expansion is None else expansion.frame.agent_id,
                        None if expansion is None else sorted(expansion.frame.identity.scopes),
                    )
                )
                return expansion

            engine._expand_child_source_in = capture
            with engine.store.connection() as db:
                db.execute("BEGIN")
                frame = engine._source_frame_in(db, row, narrowed)
                assert engine._local_sources(db, row, wrapped, frame=frame)
            assert expanded_scopes == [
                ("parent", ["read"], "child", ["read"]),
                ("child", ["read"], "grandchild", ["read"]),
            ], expanded_scopes
            print(
                json.dumps(
                    {
                        "probe": "scope_narrowing",
                        "async_full": True,
                        "async_narrowed_direct": False,
                        "async_narrowed_artifact_memory": False,
                        "sql_expanded_frames": expanded_scopes,
                    }
                )
            )
            engine._expand_child_source_in = original_expand
            release.set()
            assert (await wait(engine, parent["run_id"]))["status"] == "succeeded"


@pytest.mark.parametrize(
    ("counts", "expected"), [([60] * 10, True), ([63] * 15 + [62], True), ([63] * 16, False)]
)
async def test_source_visit_budget_and_receipt_dedup(counts, expected):
    await breadth_probe(counts, expected)


async def test_empty_receipt_at_exact_depth_limit():
    await depth_probe()


async def test_nested_source_frames_do_not_regain_scopes():
    await scope_probe()
