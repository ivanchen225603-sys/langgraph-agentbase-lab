"""Event-controlled cancellation races; no real providers or external effects."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agentbase.configuration import AgentPackage
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, ToolCall
from agentbase.registry import ComponentRegistry
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import (
    TERMINAL,
    EngineSettings,
    FinalProjection,
    GraphBinding,
    RunOptions,
)
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.task_tree import ChildTask
from agentbase.workflows import register_workflows


class State(TypedDict, total=False):
    run_id: str
    agent_id: str
    binding: str
    status: str


def parent_factory(engine, saver, bound):
    async def finish(state):
        run_id = state["run_id"]
        row = engine.store.internal_request(run_id)
        value = json.loads(row["payload"])["input"]
        children = await engine.dispatch_children(
            run_id, [ChildTask(id="child", agent_id="worker", input=value)]
        )
        result = await engine.wait_children(run_id, "all", children)
        await engine.commit_final(run_id, FinalProjection(value=result))
        return {"status": "succeeded"}

    async def project(row, values, info):
        return {}

    graph = StateGraph(State)
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


@asynccontextmanager
async def system(tmp_path, handler, *, write=False, managed=False, wait_seconds=0.0, workers=2):
    registry = ComponentRegistry()
    register_workflows(registry)
    registry.register_handler("fixture.handler", handler)
    worker = {
        "id": "worker",
        "description": "Synthetic worker",
        "workflow": "fixed",
        "prompt": {"system_template": "Test"},
        "input_schema": {"type": "object"},
        "tools": ["fixture"],
        "tasks": [{"id": "effect", "tool": "fixture", "input_bindings": {"text": "/text"}}],
    }
    agents = [worker]
    if managed:
        registry.register_workflow("fixture.parent", parent_factory)
        agents.append(
            {
                "id": "parent",
                "description": "Synthetic coordinator",
                "workflow": "fixture.parent",
                "prompt": {"system_template": "Test"},
                "input_schema": {"type": "object"},
                "delegation": {"targets": [{"agent_id": "worker"}]},
            }
        )
    registry.bind(
        AgentPackage.model_validate(
            {
                "tools": [
                    {
                        "name": "fixture",
                        "handler": "fixture.handler",
                        "description": "Synthetic tool",
                        "scope": "write" if write else "read",
                        "read_only": not write,
                        "retry_safe": not write,
                        "idempotency": "business-key" if write else "read-only",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                        "output_schema": {"type": "string"},
                    }
                ],
                "agents": agents,
            }
        )
    )
    database = SqliteDatabase(tmp_path / "runs.sqlite")
    settings = EngineSettings(
        workers=workers, queue_size=8, tools_concurrency=1, cancel_wait_seconds=wait_seconds
    )
    store = SqliteRunStore(database.connection, settings)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        store.install(db)
    contexts = SqliteContextStore(database.connection)
    identity = Identity(owner="user", scopes=frozenset({"read", "write"}))

    async def resolve(owner):
        assert owner == "user"
        return identity

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        engine = RunEngine(
            registry, store, saver, ContextProvider(contexts, contexts), resolve, settings=settings
        )
        await engine.start()
        try:
            yield engine, identity
        finally:
            await engine.close()


async def until(engine, run_id, *statuses):
    async with asyncio.timeout(5):
        while True:
            view = await engine.view("user", run_id)
            if view["status"] in statuses:
                return view
            if view["status"] in TERMINAL:
                raise AssertionError(view)
            await asyncio.sleep(0)


async def approve(engine, identity, run_id):
    pending = await until(engine, run_id, "awaiting_review")
    review = pending["interrupts"][0]
    await engine.resume(identity, run_id, review_id=review["review_id"], digest=review["digest"])


@pytest.mark.parametrize("effect", ["read", "applied", "unknown"])
async def test_cancel_admits_new_run_but_keeps_tool_slot_and_receipts(tmp_path, effect):
    entered, cleaning, release, new_ready, new_started = [asyncio.Event() for _ in range(5)]
    calls, denied = [], []

    async def handler(arguments, context):
        calls.append(arguments["text"])
        if arguments["text"] != "old":
            new_started.set()
            return "new result"
        if effect == "applied":
            engine.actions.complete(
                context.operation_key, "committed receipt", context.attempt_token
            )
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            # Even a handler suppressing cancellation cannot start another runtime tool.
            try:
                await engine.tools.execute(
                    context, ToolCall(call_id="late", name="fixture", arguments={"text": "late"})
                )
            except AgentError as error:
                denied.append(error.code)
            return "OLD_RESULT_MUST_NOT_PUBLISH"

    async with system(tmp_path, handler, write=effect != "read") as (engine, identity):

        async def authorize(context, tool, call):
            if call.arguments["text"] == "new":
                new_ready.set()
            return True

        engine.tool_authorizer = authorize
        old = await engine.submit(identity, "worker", "old", {"text": "old"})
        if effect != "read":
            await approve(engine, identity, old["run_id"])
        await asyncio.wait_for(entered.wait(), 3)
        try:
            response = await asyncio.wait_for(engine.cancel("user", old["run_id"]), 2)
            await asyncio.wait_for(cleaning.wait(), 2)
            assert response["status"] == "cancelling" and response["cleanup_pending"]
            assert response["cancel_requested"] and not response["recoverable"]
            assert not engine.admission_lock.locked()
            assert engine.store.internal_request(old["run_id"])["cancel"] == 1
            finalizer = engine._cancellations[old["run_id"]]
            for _ in range(3):
                await engine.cancel("user", old["run_id"])
                assert engine._cancellations[old["run_id"]] is finalizer
            assert engine.active[old["run_id"]].cancelling() == 1
            new = await asyncio.wait_for(
                engine.submit(identity, "worker", "new", {"text": "new"}), 2
            )
            if effect != "read":
                await approve(engine, identity, new["run_id"])
            await asyncio.wait_for(new_ready.wait(), 3)
            assert not new_started.is_set() and calls == ["old"]
        finally:
            release.set()
        expected = {
            "read": "cancelled",
            "applied": "effect_applied_result_unavailable",
            "unknown": "needs_reconciliation",
        }[effect]
        done = await until(engine, old["run_id"], expected)
        assert not done["cleanup_pending"] and done["result"] is None
        assert denied == ["cancelled_or_old_topic"]
        assert len(engine.actions.operations("user", old["run_id"])) == 1
        operation = engine.actions.operations("user", old["run_id"])[0]
        if effect != "read":
            assert operation["phase"] == effect and operation["attempts"] == 1
        assert (await until(engine, new["run_id"], "succeeded"))["result"]
        assert calls == ["old", "new"]
        history = engine.contexts.history.turns("user", "worker", "default", 0, 10)
        assert len(history) == 1 and history[0][0].content == '{"text":"new"}'


async def test_repeated_replacement_fences_old_results_and_queued_tools(tmp_path):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]
    calls = []

    async def handler(arguments, context):
        calls.append(arguments["text"])
        if arguments["text"] == "old":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                return "old late output"
        return arguments["text"]

    async with system(tmp_path, handler) as (engine, identity):
        old = await engine.submit(identity, "worker", "old", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        try:
            middle = await engine.submit(
                identity, "worker", "middle", {"text": "middle"}, RunOptions(switch_topic=True)
            )
            await asyncio.wait_for(cleaning.wait(), 2)
            newest = await engine.submit(
                identity, "worker", "newest", {"text": "newest"}, RunOptions(switch_topic=True)
            )
            assert newest["epoch"] == 2
            assert (await engine.view("user", old["run_id"]))["status"] == "cancelling"
            assert calls == ["old"]
        finally:
            release.set()
        await until(engine, old["run_id"], "cancelled")
        await until(engine, middle["run_id"], "cancelled")
        await until(engine, newest["run_id"], "succeeded")
        assert calls == ["old", "newest"]
        for epoch in (0, 1):
            assert engine.contexts.history.turns("user", "worker", "default", epoch, 10) == []


@pytest.mark.parametrize("winner", ["cancel", "completion"])
async def test_cancel_completion_race_uses_durable_receipt(tmp_path, winner):
    reached, cleaning, release = [asyncio.Event() for _ in range(3)]

    async def handler(arguments, context):
        return "result"

    async with system(tmp_path, handler) as (engine, identity):
        original_commit = engine.commit_final

        async def commit(run_id, projection):
            if winner == "completion":
                await original_commit(run_id, projection)
            reached.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
            if winner == "cancel":
                await original_commit(run_id, projection)

        engine.commit_final = commit
        run = await engine.submit(identity, "worker", "race", {"text": "result"})
        await asyncio.wait_for(reached.wait(), 3)
        try:
            view = await engine.cancel("user", run["run_id"])
            if winner == "cancel":
                await asyncio.wait_for(cleaning.wait(), 2)
                assert view["status"] == "cancelling" and view["result"] is None
            else:
                assert not view["cancel_requested"] and view["result"] is not None
                assert engine.active[run["run_id"]].cancelling() == 0
        finally:
            release.set()
        done = await until(
            engine, run["run_id"], "cancelled" if winner == "cancel" else "succeeded"
        )
        assert (engine.store.final_receipt("user", run["run_id"]) is not None) == (
            winner == "completion"
        )
        assert (done["result"] is not None) == (winner == "completion")


async def test_cancel_caller_disconnect_does_not_interrupt_cleanup_or_close(tmp_path):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]

    async def handler(arguments, context):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async with system(tmp_path, handler, wait_seconds=5.0) as (engine, identity):
        run = await engine.submit(identity, "worker", "disconnect", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        caller = asyncio.create_task(engine.cancel("user", run["run_id"]))
        try:
            await asyncio.wait_for(cleaning.wait(), 2)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert not engine._cancellations[run["run_id"]].cancelled()
            newer = await engine.submit(identity, "worker", "admitted", {"text": "new"})
            assert newer["run_id"] != run["run_id"]
            close = asyncio.create_task(engine.close())
            # An event-loop barrier, not an assumed cleanup duration.
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await barrier.wait()
            assert not close.done()
            assert engine.active[run["run_id"]].cancelling() == 1
            owner = engine._cancellations[run["run_id"]]
            engine.settings = engine.settings.model_copy(update={"cancel_wait_seconds": 0.0})
            repeated = await engine.cancel("user", run["run_id"])
            assert repeated["cleanup_pending"] and engine._cancellations[run["run_id"]] is owner
        finally:
            release.set()
        await asyncio.wait_for(close, 3)
        assert not engine.active and not engine._cancellations


@pytest.mark.parametrize("failure", ["handler_cleanup", "checkpoint_cleanup"])
async def test_cleanup_failure_is_observed_and_keeps_cancel_fence(tmp_path, failure):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]

    async def handler(arguments, context):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            if failure == "handler_cleanup":
                raise RuntimeError("synthetic cleanup failure") from None
            raise

    async with system(tmp_path, handler) as (engine, identity):
        run = await engine.submit(identity, "worker", "failure", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        if failure == "checkpoint_cleanup":

            async def broken(*args):
                raise AgentError("checkpoint_unavailable", "synthetic", 503)

            engine.terminal_checkpoint = broken
        try:
            await engine.cancel("user", run["run_id"])
            await asyncio.wait_for(cleaning.wait(), 2)
            finalizer = engine._cancellations[run["run_id"]]
        finally:
            release.set()
        if failure == "checkpoint_cleanup":
            with pytest.raises(AgentError, match="repair"):
                await finalizer
            assert not engine.available
            with pytest.raises(AgentError):
                await engine.submit(identity, "worker", "blocked", {"text": "new"})
        else:
            await finalizer
            await until(engine, run["run_id"], "cancelled")
        assert engine.store.internal_request(run["run_id"])["cancel"] == 1
        assert engine.store.final_receipt("user", run["run_id"]) is None


async def test_parent_cancel_fences_child_tree_without_blocking_new_parent(tmp_path):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]
    calls = []

    async def handler(arguments, context):
        calls.append(arguments["text"])
        if arguments["text"] == "old":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
        return arguments["text"]

    async with system(tmp_path, handler, managed=True) as (engine, identity):
        old = await engine.submit(identity, "parent", "old", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        try:
            response = await engine.cancel("user", old["run_id"])
            await asyncio.wait_for(cleaning.wait(), 2)
            assert response["status"] == "cancelling" and response["cleanup_pending"]
            child = engine.task_coordinator.children(old["run_id"])[0]["run_id"]
            assert engine.store.internal_request(child)["cancel"] == 1
            new = await engine.submit(identity, "parent", "new", {"text": "new"})
            assert new["run_id"] != old["run_id"] and calls == ["old"]
        finally:
            release.set()
        await until(engine, old["run_id"], "cancelled")
        await until(engine, child, "cancelled")
        await until(engine, new["run_id"], "succeeded")
        assert calls == ["old", "new"]


async def test_cancel_registration_does_not_wait_for_unrelated_admission_prepare(tmp_path):
    entered, cleaning, release, preparing, admit_release = [asyncio.Event() for _ in range(5)]

    async def handler(arguments, context):
        if arguments["text"] == "old":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
        return arguments["text"]

    async with system(tmp_path, handler) as (engine, identity):
        old = await engine.submit(identity, "worker", "old", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        bound = engine.registry.agent("worker")
        binding = engine.binding({"agent_id": "worker", "binding": bound.digest})
        original_prepare = binding.prepare

        async def prepare(identity, value):
            preparing.set()
            await admit_release.wait()
            return await original_prepare(identity, value)

        binding.prepare = prepare
        submitting = asyncio.create_task(engine.submit(identity, "worker", "new", {"text": "new"}))
        try:
            await asyncio.wait_for(preparing.wait(), 2)
            assert engine.admission_lock.locked()
            stopped = await asyncio.wait_for(engine.cancel("user", old["run_id"]), 2)
            await asyncio.wait_for(cleaning.wait(), 2)
            assert stopped["cancel_requested"] and stopped["cleanup_pending"]
            assert not submitting.done() and engine.admission_lock.locked()
        finally:
            release.set()
            admit_release.set()
        new = await submitting
        await until(engine, old["run_id"], "cancelled")
        await until(engine, new["run_id"], "succeeded")


@pytest.mark.parametrize("value", [-0.1, 5.1, float("nan"), float("inf")])
def test_cancel_response_wait_budget_is_bounded(value):
    with pytest.raises(ValueError):
        EngineSettings(cancel_wait_seconds=value)


async def test_shutdown_does_not_accept_new_cancellation_owner(tmp_path):
    worker_stopping, release_worker = asyncio.Event(), asyncio.Event()
    finalizing, release_finalizer = asyncio.Event(), asyncio.Event()

    async def handler(arguments, context):
        raise AssertionError("unapproved tool must not execute")

    async with system(tmp_path, handler, write=True) as (engine, identity):
        run = await engine.submit(identity, "worker", "paused", {"text": "old"})
        await until(engine, run["run_id"], "awaiting_review")
        original_terminal = engine.terminal_checkpoint

        async def terminal(*args):
            finalizing.set()
            await release_finalizer.wait()
            return await original_terminal(*args)

        engine.terminal_checkpoint = terminal

        async def worker_tail():
            try:
                await asyncio.Event().wait()
            finally:
                worker_stopping.set()
                await release_worker.wait()

        tail = asyncio.create_task(worker_tail())
        engine.workers.append(tail)
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        closing = asyncio.create_task(engine.close())
        try:
            await asyncio.wait_for(worker_stopping.wait(), 2)
            assert engine.closing and not closing.done()
            try:
                await engine.cancel("user", run["run_id"])
            except AgentError as error:
                assert error.code == "not_accepting"
            else:
                await asyncio.wait_for(finalizing.wait(), 2)
                release_worker.set()
                await asyncio.wait_for(closing, 3)
                assert not engine._cancellations, "close returned with a live late finalizer"
            assert run["run_id"] not in engine._cancellations
            assert not finalizing.is_set()
        finally:
            release_finalizer.set()
            release_worker.set()
            await asyncio.wait_for(closing, 3)
            await asyncio.gather(*list(engine._cancellations.values()), return_exceptions=True)


@pytest.mark.parametrize("managed", [False, True])
async def test_shutdown_rechecks_admission_after_slow_prepare(tmp_path, managed):
    preparing, release = asyncio.Event(), asyncio.Event()

    async def handler(arguments, context):
        raise AssertionError("shutdown request must not execute")

    async with system(tmp_path, handler, managed=managed) as (engine, identity):
        agent = "parent" if managed else "worker"
        binding = engine.binding(
            {"agent_id": agent, "binding": engine.registry.agent(agent).digest}
        )
        original = binding.prepare

        async def prepare(identity, value):
            preparing.set()
            await release.wait()
            return await original(identity, value) if original else {"input": value}

        binding.prepare = prepare
        submitting = asyncio.create_task(
            engine.submit(identity, agent, "late", {"text": "late"}, RunOptions(switch_topic=True))
        )
        try:
            await asyncio.wait_for(preparing.wait(), 2)
            await asyncio.wait_for(engine.close(), 3)
        finally:
            release.set()
        with pytest.raises(AgentError) as error:
            await submitting
        assert error.value.code == "not_accepting"
        assert engine.store.lookup("user", agent, "late") is None
        with engine.store.connection() as db:
            assert engine.store.epoch(db, "user", agent, "default") == 0
            assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
            assert db.execute("SELECT COUNT(*) FROM agent_run_events").fetchone()[0] == 0
        assert not engine._cancellations and not engine.pending


async def test_shutdown_caller_cancellation_does_not_interrupt_handler_cleanup(tmp_path):
    entered, cleaning, release, cleaned = [asyncio.Event() for _ in range(4)]

    async def handler(arguments, context):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    async with system(tmp_path, handler) as (engine, identity):
        run = await engine.submit(identity, "worker", "close-caller", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        executing = engine.active[run["run_id"]]
        caller = asyncio.create_task(engine.close())
        try:
            await asyncio.wait_for(cleaning.wait(), 2)
            with pytest.raises(AgentError) as error:
                await engine.start()
            assert error.value.code == "not_accepting"
            close_owner = engine._close_task
            second_caller = asyncio.create_task(engine.close())
            caller.cancel()
            assert executing.cancelling() == 1, "shutdown caller re-cancelled handler cleanup"
            assert engine._close_task is close_owner
            assert not cleaned.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
        await second_caller
        assert cleaned.is_set() and not engine.active and not engine.workers


@pytest.mark.parametrize("queued", [False, True])
async def test_concurrent_parent_child_cancel_preserves_terminal_fences(tmp_path, queued):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]
    calls = []

    async def handler(arguments, context):
        calls.append(arguments["text"])
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async with system(tmp_path, handler, managed=True, workers=1 if queued else 2) as (
        engine,
        identity,
    ):
        if queued:
            original = engine.dispatch_children

            async def dispatch(*args):
                children = await original(*args)
                entered.set()
                await release.wait()
                return children

            engine.dispatch_children = dispatch
        parent = await engine.submit(identity, "parent", "parent", {"text": "old"})
        await asyncio.wait_for(entered.wait(), 3)
        child = engine.task_coordinator.children(parent["run_id"])[0]["run_id"]
        try:
            await asyncio.gather(
                engine.cancel("user", parent["run_id"]), engine.cancel("user", child)
            )
            if not queued:
                await asyncio.wait_for(cleaning.wait(), 2)
            assert engine.store.internal_request(parent["run_id"])["cancel"] == 1
            assert engine.store.internal_request(child)["cancel"] == 1
        finally:
            release.set()
        await until(engine, parent["run_id"], "cancelled")
        await until(engine, child, "cancelled")
        await asyncio.gather(*list(engine._cancellations.values()))
        assert not engine._cancellations
        assert calls == ([] if queued else ["old"])
        assert engine.store.final_receipt("user", parent["run_id"]) is None
        assert engine.store.final_receipt("user", child) is None
