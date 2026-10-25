"""Real SQLite/RunEngine integration; synthetic tools only, no model calls."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentbase.configuration import AgentPackage
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, Message
from agentbase.intent import IntentContext, IntentSession, read_intent_target
from agentbase.registry import ComponentRegistry
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings, RunOptions
from agentbase.schema import content_digest
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.workflows import register_workflows


@asynccontextmanager
async def runtime(tmp_path):
    registry = ComponentRegistry()
    register_workflows(registry)
    effects, started, release = [], asyncio.Event(), asyncio.Event()

    async def write(arguments, context):
        if arguments["x"] == 1:
            started.set()
            await release.wait()
        effects.append(arguments["x"])
        return arguments["x"]

    registry.register_handler("fixture.write", write)
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    registry.bind(
        AgentPackage.model_validate(
            {
                "tools": [
                    {
                        "name": "write",
                        "handler": "fixture.write",
                        "description": "Synthetic write",
                        "scope": "write",
                        "read_only": False,
                        "retry_safe": False,
                        "idempotency": "business-key",
                        "input_schema": schema,
                        "output_schema": {"type": "integer"},
                    }
                ],
                "agents": [
                    {
                        "id": "worker",
                        "description": "Synthetic fixed Agent",
                        "workflow": "fixed",
                        "prompt": {"system_template": "Execute declared tasks"},
                        "input_schema": schema,
                        "tools": ["write"],
                        "tasks": [{"id": "write", "tool": "write", "input_bindings": {"x": "/x"}}],
                    }
                ],
            }
        )
    )
    database = SqliteDatabase(tmp_path / "runs.sqlite")
    store = SqliteRunStore(database.connection, EngineSettings())
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        store.install(db)
    contexts = SqliteContextStore(database.connection)
    actor = Identity(owner="user", scopes=frozenset({"read", "write"}))

    async def identity(owner):
        return actor if owner == actor.owner else None

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        engine = RunEngine(registry, store, saver, ContextProvider(contexts, contexts), identity)
        await engine.start()
        try:
            yield engine, actor, effects, started, release
        finally:
            release.set()
            await engine.close()


def current(engine, actor, request="request1", conversation="same"):
    with engine.store.connection() as db:
        epoch = engine.store.epoch(db, actor.owner, "worker", conversation)
    return IntentContext(
        request_id=request,
        identity=actor,
        agent_id="worker",
        conversation_id=conversation,
        epoch=epoch,
    )


def session(context, text, target=None):
    return IntentSession(
        context, Message(role="user", content=text), origin="authenticated-user", target=target
    )


def proposal(s, action="propose_task", interpretation="explicit", **changes):
    data = s.model_input()
    result = {
        "request_id": data["request_id"],
        "request_digest": data["request_digest"],
        "action": action,
        "interpretation": interpretation,
        "message": "请确认本次提案。",
    }
    if action in {"propose_task", "propose_replace"}:
        result["task"] = {"input": {"x": 2}}
    if action in {"propose_replace", "propose_cancel"}:
        result.update(
            target_run_id=data["target"]["run_id"],
            target_input_digest=data["target"]["input_digest"],
        )
    return {**result, **changes}


def state(engine):
    with engine.store.connection() as db:
        db.execute("BEGIN")
        return {
            "runs": [tuple(row) for row in db.execute("SELECT * FROM requests ORDER BY run_id")],
            "operations": [
                tuple(row) for row in db.execute("SELECT * FROM agent_tool_operations ORDER BY key")
            ],
            "reviews": [
                tuple(row) for row in db.execute("SELECT * FROM agent_reviews ORDER BY id")
            ],
            "memories": [
                tuple(row) for row in db.execute("SELECT * FROM agent_memories ORDER BY id")
            ],
        }


async def wait(engine, run_id, status):
    async with asyncio.timeout(10):
        while True:
            view = await engine.view("user", run_id)
            if view["status"] == status:
                return view
            if view["status"] in {"failed", "timed_out", "needs_reconciliation"}:
                raise AssertionError(view)
            await asyncio.sleep(0.01)


async def approve(engine, actor, view):
    review = view["interrupts"][0]
    return await engine.resume(
        actor, view["run_id"], review_id=review["review_id"], digest=review["digest"]
    )


@pytest.mark.parametrize(
    "text,interpretation,invalid",
    [
        ("不要创建任务", "explicit", False),
        ("Do not write", "explicit", False),
        ("可能创建一个任务", "ambiguous", False),
        ("取消，但也继续执行", "conflicting", False),
        ("创建任务", "negated", False),
        ("创建任务", "explicit", True),
        ("创建任务", "explicit", False),
    ],
)
async def test_review_never_creates_runs_attempts_approvals_or_memories(
    tmp_path, text, interpretation, invalid
):
    async with runtime(tmp_path) as (engine, actor, effects, *_):
        ctx = current(engine, actor)
        s = session(ctx, text)
        before = state(engine)
        raw = proposal(s, interpretation=interpretation)
        if invalid:
            raw["approval"] = True
        reviewed = s.review(raw, current=ctx)
        assert reviewed.action == (
            "proposal"
            if text == "创建任务" and interpretation == "explicit" and not invalid
            else "clarify"
        )
        assert (
            state(engine) == before == {"runs": [], "operations": [], "reviews": [], "memories": []}
        )
        assert effects == []


async def test_target_snapshot_is_owner_scoped_and_read_only(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, *_):
        run = await engine.submit(
            actor, "worker", "old", {"x": 1}, RunOptions(conversation_id="same")
        )
        await wait(engine, run["run_id"], "awaiting_review")
        before = state(engine)
        target = read_intent_target(engine.store, actor, run["run_id"])
        assert target.input_digest == content_digest({"x": 1}) and target.active
        with pytest.raises(AgentError) as error:
            read_intent_target(engine.store, Identity(owner="other"), run["run_id"])
        assert error.value.code == "not_found"
        assert state(engine) == before and effects == []


async def test_cancel_is_only_proposed_until_host_uses_existing_control(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, *_):
        run = await engine.submit(
            actor, "worker", "old", {"x": 1}, RunOptions(conversation_id="same")
        )
        await wait(engine, run["run_id"], "awaiting_review")
        bound = read_intent_target(engine.store, actor, run["run_id"])
        ctx = current(engine, actor)
        s = session(ctx, "取消当前任务", bound)
        before = state(engine)
        reviewed = s.review(
            proposal(s, "propose_cancel"),
            current=ctx,
            target=read_intent_target(engine.store, actor, run["run_id"]),
        )
        assert reviewed.action == "proposal" and state(engine) == before
        # A separate, explicit host control action; review never calls this method.
        await engine.cancel(actor.owner, run["run_id"])
        await wait(engine, run["run_id"], "cancelled")
        assert not read_intent_target(engine.store, actor, run["run_id"]).active
        assert effects == []


async def test_midflight_replacement_keeps_old_approval_out_of_new_run(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, started, _):
        old = await engine.submit(
            actor, "worker", "old", {"x": 1}, RunOptions(conversation_id="same")
        )
        pending = await wait(engine, old["run_id"], "awaiting_review")
        old_review = pending["interrupts"][0]
        await approve(engine, actor, pending)
        async with asyncio.timeout(10):
            await started.wait()
        ctx = current(engine, actor)
        bound = read_intent_target(engine.store, actor, old["run_id"])
        s = session(ctx, "将当前任务替换为处理数字2", bound)
        before = state(engine)
        reviewed = s.review(proposal(s, "propose_replace"), current=ctx, target=bound)
        assert reviewed.action == "proposal" and state(engine) == before and effects == []

        # The host separately confirms a new requirement and submits through RunEngine.
        # switch_topic affects the conversation epoch, not just the named target.
        replacement = await engine.submit(
            actor, "worker", "new", {"x": 2}, RunOptions(conversation_id="same", switch_topic=True)
        )
        fresh = await wait(engine, replacement["run_id"], "awaiting_review")
        assert engine.store.request(actor.owner, old["run_id"])["cancel"] == 1
        assert current(engine, actor).epoch == ctx.epoch + 1
        assert replacement["run_id"] != old["run_id"]
        assert fresh["interrupts"][0]["review_id"] != old_review["review_id"]
        assert effects == []
        with pytest.raises(AgentError):
            await engine.resume(
                actor,
                replacement["run_id"],
                review_id=old_review["review_id"],
                digest=old_review["digest"],
            )
        assert effects == []
        await approve(engine, actor, fresh)
        await wait(engine, replacement["run_id"], "succeeded")
        assert effects == [2]
        operations = engine.actions.operations(actor.owner, replacement["run_id"])
        assert len(operations) == 1 and operations[0]["attempts"] == 1
        assert not read_intent_target(engine.store, actor, replacement["run_id"]).active


async def test_late_model_response_after_topic_switch_cannot_create_third_run(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, *_):
        old = await engine.submit(
            actor, "worker", "old", {"x": 1}, RunOptions(conversation_id="same")
        )
        await wait(engine, old["run_id"], "awaiting_review")
        ctx = current(engine, actor)
        bound = read_intent_target(engine.store, actor, old["run_id"])
        s = session(ctx, "替换当前任务", bound)
        raw = proposal(s, "propose_replace")
        new = await engine.submit(
            actor, "worker", "new", {"x": 2}, RunOptions(conversation_id="same", switch_topic=True)
        )
        await wait(engine, new["run_id"], "awaiting_review")
        before = state(engine)
        with pytest.raises(AgentError) as error:
            s.review(
                raw,
                current=current(engine, actor, request="request2"),
                target=read_intent_target(engine.store, actor, old["run_id"]),
            )
        assert error.value.code == "intent_stale"
        assert state(engine) == before and len(before["runs"]) == 2 and effects == []


async def test_cross_conversation_target_cannot_be_proposed(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, *_):
        run = await engine.submit(
            actor, "worker", "old", {"x": 1}, RunOptions(conversation_id="other")
        )
        await wait(engine, run["run_id"], "awaiting_review")
        before = state(engine)
        with pytest.raises(AgentError) as error:
            session(
                current(engine, actor),
                "取消任务",
                read_intent_target(engine.store, actor, run["run_id"]),
            )
        assert error.value.code == "intent_target"
        assert state(engine) == before and effects == []


async def test_visible_paused_review_remains_a_target_after_execution_clock_expires(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, *_):
        run = await engine.submit(
            actor, "worker", "old", {"x": 1}, RunOptions(conversation_id="same")
        )
        await wait(engine, run["run_id"], "awaiting_review")
        # A sealed pause-review can outlive the original execution deadline.
        with engine.store.connection() as db:
            db.execute("UPDATE requests SET deadline=0 WHERE run_id=?", (run["run_id"],))
        bound = read_intent_target(engine.store, actor, run["run_id"])
        assert bound.active
        ctx = current(engine, actor)
        s = session(ctx, "取消当前任务", bound)
        before = state(engine)
        result = s.review(proposal(s, "propose_cancel"), current=ctx, target=bound)
        assert result.action == "proposal" and state(engine) == before and effects == []
