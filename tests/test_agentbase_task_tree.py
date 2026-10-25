"""Actual parent/child execution through one shared engine and durable SQL state."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agentbase.artifacts import ArtifactOperation, ArtifactScope
from agentbase.budgets import SqliteBudgetLedger
from agentbase.configuration import AgentPackage
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, Message, ModelRequest, ModelResponse, ToolCall
from agentbase.registry import ComponentRegistry
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings, FinalProjection, GraphBinding
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.task_tree import ChildTask
from agentbase.tools import ToolResult
from agentbase.workflows import register_workflows


class ParentState(TypedDict, total=False):
    run_id: str
    agent_id: str
    binding: str
    status: str


def parent_factory(engine, saver, bound, *, after_consumption=None):
    async def finish(state, config):
        run_id = state["run_id"]
        children = await engine.dispatch_children(
            run_id,
            [
                ChildTask(id=f"task{index}", agent_id="child", input={"x": index})
                for index in range(3)
            ],
        )
        outcomes = await engine.wait_children(run_id, "all", children)
        if after_consumption is not None:
            await after_consumption()
        await engine.commit_final(
            run_id,
            FinalProjection(value=outcomes),
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


def package(*, write=False, timeout=30):
    return AgentPackage.model_validate(
        {
            "tools": [
                {
                    "name": "echo",
                    "handler": "fixture.echo",
                    "description": "Returns its argument",
                    "input_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                    "scope": "write" if write else "read",
                    "read_only": not write,
                    "retry_safe": not write,
                    "idempotency": "business-key" if write else "read-only",
                }
            ],
            "agents": [
                {
                    "id": "parent",
                    "description": "Coordinates generic child tasks",
                    "workflow": "fixture.parent",
                    "prompt": {"system_template": "Run bounded tasks"},
                    "input_schema": {"type": "object"},
                    "budget": {"timeout_seconds": timeout},
                    "delegation": {"targets": [{"agent_id": "child"}]},
                },
                {
                    "id": "child",
                    "description": "Runs a deterministic task",
                    "workflow": "fixed",
                    "prompt": {"system_template": "Run bounded tasks"},
                    "input_schema": {"type": "object"},
                    "tools": ["echo"],
                    "tasks": [{"id": "echo", "tool": "echo", "input_bindings": {"x": "/x"}}],
                },
            ],
        }
    )


@asynccontextmanager
async def system(
    tmp_path,
    handler,
    *,
    configuration=None,
    workflow_factory=parent_factory,
    result_policy=None,
    provider=None,
    agent_authorizer=None,
    frame_authorizer=None,
):
    registry = ComponentRegistry()
    register_workflows(registry)
    registry.register_workflow("fixture.parent", workflow_factory)
    registry.register_handler("fixture.echo", handler)
    if provider is not None:
        registry.register_provider("fixture.plan", lambda configuration: provider)
    registry.bind(configuration or package())
    settings = EngineSettings(workers=1, queue_size=1)
    database = SqliteDatabase(tmp_path / "business.sqlite")
    store = SqliteRunStore(database.connection, settings)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        store.install(db)
    contexts = SqliteContextStore(database.connection)
    identity = Identity(owner="user", scopes=frozenset({"read", "write"}))

    async def resolver(owner):
        assert owner == identity.owner
        return identity

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        engine = RunEngine(
            registry,
            store,
            saver,
            ContextProvider(contexts, contexts, frame_authorizer=frame_authorizer),
            resolver,
            result_policy=result_policy,
            agent_authorizer=agent_authorizer,
        )
        await engine.start()
        try:
            yield engine, identity
        finally:
            await engine.close()


async def wait(
    engine, run_id, statuses=("succeeded", "failed", "needs_reconciliation", "awaiting_review")
):
    for _ in range(300):
        view = await engine.view("user", run_id)
        if view["status"] in statuses:
            return view
        await asyncio.sleep(0.01)
    raise AssertionError(view)


async def wait_event(engine, identity, run_id, event):
    """Wait for a correctness precondition within the original runtime deadline."""
    row = engine.store.request(identity.owner, run_id)
    try:
        async with asyncio.timeout(max(0.0, row["deadline"] - time.time())):
            await event.wait()
    except TimeoutError as exc:
        raise AssertionError(
            {
                "run_id": run_id,
                "deadline": row["deadline"],
                "children": engine.task_coordinator.children(run_id),
                "dispatch": engine.task_coordinator.state(run_id),
                "outcome": engine.task_coordinator.outcome(run_id),
            }
        ) from exc


async def test_one_worker_one_slot_runs_three_children_and_replays_dispatch_once(tmp_path):
    first_started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        if len(calls) == 1:
            first_started.set()
            await release.wait()
        return {"value": arguments["x"]}

    async with system(tmp_path, echo) as (engine, identity):
        parent = await engine.submit(identity, "parent", "first", {})
        await asyncio.wait_for(first_started.wait(), 2)
        assert parent["run_id"] not in engine.active
        view = await engine.view("user", parent["run_id"])
        assert view["status"] == "waiting_children"
        assert len(view["children"]) == 3
        with engine.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4
        release.set()
        result = await wait(engine, parent["run_id"])
        assert result["status"] == "succeeded", result
        assert sorted(calls) == [0, 1, 2]
        assert sorted(item["value"]["echo"]["value"]["value"] for item in result["result"]) == [
            0,
            1,
            2,
        ]
        with engine.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4
            assert db.execute("SELECT COUNT(*) FROM agent_tool_operations").fetchone()[0] == 3
            assert db.execute("SELECT COUNT(*) FROM agent_final_receipts").fetchone()[0] == 4
            assert not db.execute(
                "SELECT 1 FROM agent_task_dispatch WHERE state!='terminal'"
            ).fetchone()
            records = db.execute("SELECT payload FROM agent_final_receipts").fetchall()
            assert all(json.loads(row[0])["status"] == "succeeded" for row in records)


@pytest.mark.parametrize("kind", ["model", "tool", "step"])
async def test_siblings_compete_for_last_ancestor_quota_atomically(tmp_path, kind):
    started, release = asyncio.Event(), asyncio.Event()

    async def echo(arguments, context):
        started.set()
        await release.wait()
        return {"value": arguments["x"]}

    config = package().model_dump(mode="json")
    config["agents"][0]["budget"].update(max_model_calls=1, max_tool_calls=2, max_steps=3)
    async with system(tmp_path, echo, configuration=AgentPackage.model_validate(config)) as (
        engine,
        identity,
    ):
        parent = await engine.submit(identity, "parent", "quota", {})
        await asyncio.wait_for(started.wait(), 2)
        children = (await engine.view("user", parent["run_id"]))["children"]
        contexts = [await engine.context(child["run_id"]) for child in children[1:]]

        def reserve(context):
            try:
                budget = engine.budget_policy(context)
                if kind == "model":
                    SqliteBudgetLedger(engine.store.connection).reserve(context, budget, 10)
                elif kind == "tool":
                    call = ToolCall(call_id="quota", name="echo", arguments={"x": 4})
                    tool = engine.registry.resolve(context.agent_id, context.config_digest).tools[0]
                    engine.actions.begin(context, tool, call, budget.max_tool_calls)
                else:
                    row = engine.store.internal_request(context.run_id)
                    engine.store.reserve_step(row, "quota", budget.max_steps, context=context)
                return "reserved"
            except AgentError as error:
                return error.code

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(reserve, contexts))
        expected = {"model": "model_run_budget", "tool": "tool_call_budget", "step": "step_budget"}[
            kind
        ]
        assert sorted(results) == sorted(["reserved", expected])
        if kind == "model":
            assert engine.model_caller.ledger.snapshot(parent["run_id"])["calls"] == 1


async def test_child_reviews_survive_restart_and_resume_without_duplicate_effects(tmp_path):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    config = package(write=True)
    async with system(tmp_path, echo, configuration=config) as (engine, identity):
        parent = await engine.submit(identity, "parent", "reviews", {})
        parent_id = parent["run_id"]
        for _ in range(200):
            children = engine.task_coordinator.children(parent_id)
            if children and any(child["state"] == "awaiting_review" for child in children):
                break
            await asyncio.sleep(0.01)
        assert not calls
        child_id = children[0]["run_id"]
        previous = await wait(engine, child_id)
        assert previous["status"] == "awaiting_review"
        review = previous["interrupts"][0]
        deadline = previous["deadline"]

    async with system(tmp_path, echo, configuration=config) as (engine, identity):
        current = await wait(engine, child_id)
        assert current["interrupts"][0] == review
        assert current["deadline"] == deadline
        assert (await engine.view("user", parent_id))["status"] == "waiting_children"
        for child in engine.task_coordinator.children(parent_id):
            current = await wait(engine, child["run_id"])
            review = current["interrupts"][0]
            await engine.resume(
                identity, child["run_id"], review_id=review["review_id"], digest=review["digest"]
            )
            done = await wait(
                engine, child["run_id"], ("succeeded", "failed", "needs_reconciliation")
            )
            assert done["status"] == "succeeded", done
        done = await wait(engine, parent_id)
        assert done["status"] == "succeeded", done
        assert sorted(calls) == [0, 1, 2]


async def test_parent_cancel_preserves_unknown_child_write_across_restart(tmp_path):
    effect_started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        effect_started.set()
        await release.wait()
        return {"value": arguments["x"]}

    config = package(write=True)
    async with system(tmp_path, echo, configuration=config) as (engine, identity):
        parent = await engine.submit(identity, "parent", "cancel", {})
        parent_id = parent["run_id"]
        for _ in range(200):
            children = engine.task_coordinator.children(parent_id)
            if children:
                break
            await asyncio.sleep(0.01)
        child_id = children[0]["run_id"]
        pending = await wait(engine, child_id)
        review = pending["interrupts"][0]
        await engine.resume(
            identity, child_id, review_id=review["review_id"], digest=review["digest"]
        )
        await asyncio.wait_for(effect_started.wait(), 2)
        cancelled = await engine.cancel("user", parent_id)
        assert cancelled["status"] == "needs_reconciliation", cancelled
        assert (await engine.view("user", child_id))["status"] == "needs_reconciliation"
        with engine.store.connection() as db:
            assert (
                db.execute(
                    "SELECT phase FROM agent_tool_operations WHERE run_id=?", (child_id,)
                ).fetchone()[0]
                == "unknown"
            )
            assert not db.execute(
                "SELECT 1 FROM agent_final_receipts WHERE run_id=?", (parent_id,)
            ).fetchone()

    async with system(tmp_path, echo, configuration=config) as (engine, identity):
        assert (await engine.view("user", parent_id))["status"] == "needs_reconciliation"
        assert (await engine.view("user", child_id))["status"] == "needs_reconciliation"
        await asyncio.sleep(0.1)
        assert calls == [0]


async def test_review_deadline_expires_while_parent_releases_worker(tmp_path):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    async with system(tmp_path, echo, configuration=package(write=True, timeout=0.8)) as (
        engine,
        identity,
    ):
        parent = await engine.submit(identity, "parent", "deadline", {})
        done = await wait(engine, parent["run_id"], ("timed_out", "failed", "needs_reconciliation"))
        assert done["status"] == "timed_out", done
        assert not calls
        assert (
            engine.task_coordinator.dispatcher is not None
            and not engine.task_coordinator.dispatcher.done()
        )


async def test_accepted_child_approval_command_survives_post_commit_ack_loss(tmp_path):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    config = package(write=True)
    async with system(tmp_path, echo, configuration=config) as (engine, identity):
        parent = await engine.submit(identity, "parent", "ack", {})
        for _ in range(200):
            children = engine.task_coordinator.children(parent["run_id"])
            if children:
                break
            await asyncio.sleep(0.01)
        child_id = children[0]["run_id"]
        pending = await wait(engine, child_id)
        review = pending["interrupts"][0]
        original = engine.store.accept_review

        def lose_ack(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("synthetic post-commit acknowledgement loss")

        engine.store.accept_review = lose_ack
        with pytest.raises(RuntimeError, match="post-commit"):
            await engine.resume(
                identity, child_id, review_id=review["review_id"], digest=review["digest"]
            )

    async with system(tmp_path, echo, configuration=config) as (engine, identity):
        done = await wait(engine, child_id, ("succeeded", "failed", "needs_reconciliation"))
        assert done["status"] == "succeeded", done
        assert calls == [0]


@pytest.mark.parametrize("revoke_before_consumption", [True, False])
async def test_child_source_revoke_blocks_consumption_provider_final_and_sse(
    tmp_path, revoke_before_consumption
):
    consumption_started, release = asyncio.Event(), asyncio.Event()
    artifacts = []

    async def echo(arguments, context):
        return {"value": arguments["x"]}

    async def policy(context, declaration, value):
        record = engine.artifacts.create(
            ArtifactScope(
                owner=context.identity.owner,
                agent_id=context.agent_id,
                namespace=context.conversation_id,
            ),
            ArtifactOperation(
                run_id=context.run_id, key="source", config_digest=context.config_digest
            ),
            {"value": value},
            sources=[],
            guard=lambda db: engine.model_request_current_in(db, context, []),
        )
        artifacts.append((context, record))
        return ToolResult(value=value, sources=[record.reference()])

    async def hold():
        consumption_started.set()
        await release.wait()

    factory = (
        parent_factory
        if revoke_before_consumption
        else lambda engine, saver, bound: parent_factory(
            engine, saver, bound, after_consumption=hold
        )
    )
    async with system(tmp_path, echo, workflow_factory=factory, result_policy=policy) as (
        engine,
        identity,
    ):
        if revoke_before_consumption:
            original = engine.task_coordinator._consume

            async def intercepted(*args):
                await hold()
                return await original(*args)

            engine.task_coordinator._consume = intercepted
        parent = await engine.submit(identity, "parent", "sources", {})
        await wait_event(engine, identity, parent["run_id"], consumption_started)
        assert len(artifacts) == 3
        context, record = artifacts[0]
        if not revoke_before_consumption:
            event = await engine.emit(parent["run_id"], "result.projection", {"text": "derived"})
            assert event.data["sources"]
        engine.artifacts.revoke(
            ArtifactScope(
                owner=context.identity.owner,
                agent_id=context.agent_id,
                namespace=context.conversation_id,
            ),
            ArtifactOperation(
                run_id=context.run_id, key="revoke", config_digest=context.config_digest
            ),
            record.id,
            record.version,
            guard=lambda db: True,
        )
        if not revoke_before_consumption:
            with pytest.raises(AgentError, match="sources"):
                await engine.events("user", parent["run_id"], event.seq - 1)
            with pytest.raises(AgentError):
                engine.check_event_frame(identity, event)

            class Provider:
                native_stream = False
                calls = 0

                def count_input_tokens(self, request):
                    return 1

                def count_output_tokens(self, value):
                    return 1

                async def complete(self, request):
                    self.calls += 1
                    return ModelResponse(text="unused")

            provider = Provider()
            current = await engine.context(parent["run_id"])
            budget = engine.budget_policy(current)
            with pytest.raises(AgentError):
                await engine.model_caller.call(
                    provider,
                    current,
                    budget,
                    ModelRequest(
                        messages=[Message(role="user", content="use consumed results")],
                        max_output_tokens=budget.max_output_tokens,
                    ),
                    streaming=False,
                )
            assert provider.calls == 0
        release.set()
        done = await wait(engine, parent["run_id"])
        assert done["status"] == "failed", done
        assert engine.store.final_receipt("user", parent["run_id"]) is None


async def test_child_unknown_without_user_cancel_cannot_commit_parent_success(tmp_path):
    async def echo(arguments, context):
        raise RuntimeError("external acknowledgement unavailable")

    async with system(tmp_path, echo, configuration=package(write=True)) as (engine, identity):
        parent = await engine.submit(identity, "parent", "unknown", {})
        for _ in range(200):
            children = engine.task_coordinator.children(parent["run_id"])
            if children:
                break
            await asyncio.sleep(0.01)
        first = await wait(engine, children[0]["run_id"])
        review = first["interrupts"][0]
        await engine.resume(
            identity, children[0]["run_id"], review_id=review["review_id"], digest=review["digest"]
        )
        done = await wait(engine, children[0]["run_id"], ("needs_reconciliation",))
        assert done["status"] == "needs_reconciliation"
        with pytest.raises(AgentError):
            await engine.commit_final(parent["run_id"], FinalProjection(value="must not publish"))
        assert engine.store.final_receipt("user", parent["run_id"]) is None
