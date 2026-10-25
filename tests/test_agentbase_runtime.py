"""Real temporary SQLite checkpoints exercise the shared run lifecycle."""

import asyncio
import copy
import json
import time
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentbase.configuration import AgentPackage, ToolDeclaration
from agentbase.context import ContextProvider
from agentbase.contracts import (
    AgentError,
    Identity,
    Message,
    ModelChunk,
    ToolCall,
    ToolCallDelta,
)
from agentbase.fixed_workflow import input_pointer
from agentbase.registry import ComponentRegistry
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings, FinalProjection, RunOptions
from agentbase.schema import content_digest
from agentbase.storage import SqliteContextStore, SqliteDatabase
from agentbase.tools import ResolvedTool, ToolResult
from agentbase.workflows import register_workflows


class ScriptProvider:
    native_stream = True

    def __init__(self, scripts, gate=None):
        self.scripts, self.requests, self.gate = scripts, [], gate

    async def complete(self, request):
        raise AssertionError("native stream required")

    async def stream(self, request):
        self.requests.append(request)
        script = self.scripts[len(self.requests) - 1]
        if isinstance(script, str):
            yield ModelChunk(text_delta=script)
            if self.gate:
                await self.gate.wait()
            yield ModelChunk(finish_reason="stop")
        else:
            for index, (name, args) in enumerate(script):
                yield ModelChunk(
                    tool_call_deltas=[
                        ToolCallDelta(
                            index=index,
                            call_id=f"call{index}",
                            name=name,
                            arguments_delta=json.dumps(args),
                        )
                    ]
                )
            yield ModelChunk(finish_reason="tool_calls")

    def count_input_tokens(self, request):
        return 1

    def count_output_tokens(self, text):
        return len(text) // 4


def package(*, tools=None, agents=1, budget=None, stream_policy="provisional"):
    return AgentPackage.model_validate(
        {
            "models": [{"id": "local", "provider": "offline", "model": "fixture"}],
            "tools": tools or [],
            "agents": [
                {
                    "id": f"agent{index}",
                    "description": "Synthetic non-RAG Agent",
                    "model": "local",
                    "prompt": {"system_template": "Use declared tools; return a final response."},
                    "input_schema": {"type": "string"},
                    "tools": [tool["name"] for tool in tools or []],
                    "budget": budget or {},
                    "stream_policy": stream_policy,
                }
                for index in range(agents)
            ],
        }
    )


def declaration(write=False):
    return {
        "name": "custom",
        "handler": "custom.handler",
        "description": "Synthetic action",
        "scope": "write" if write else "read",
        "read_only": not write,
        "idempotency": "business-key" if write else "read-only",
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
            "additionalProperties": False,
        },
    }


@asynccontextmanager
async def system(
    tmp_path,
    scripts,
    *,
    config=None,
    handler=None,
    settings=None,
    gate=None,
    identities=None,
    tool_resolver=None,
):
    registry = ComponentRegistry()
    provider = ScriptProvider(scripts, gate)
    registry.register_provider("offline", lambda config: provider)
    register_workflows(registry)
    if handler:
        registry.register_handler("custom.handler", handler)
    registry.bind(config or package())
    database = SqliteDatabase(tmp_path / "business.sqlite")
    settings = settings or EngineSettings()
    store = SqliteRunStore(database.connection, settings)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        store.install(db)
    contexts = SqliteContextStore(database.connection)
    identities = (
        identities
        if identities is not None
        else {"user": Identity(owner="user", scopes=frozenset({"read", "write"}))}
    )

    async def resolve(owner):
        return identities.get(owner, Identity(owner=owner))

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        engine = RunEngine(
            registry,
            store,
            saver,
            ContextProvider(contexts, contexts),
            resolve,
            settings=settings,
            tool_resolver=tool_resolver,
        )
        await engine.start()
        try:
            yield engine, provider, contexts, identities
        finally:
            await engine.close()


async def wait(
    engine,
    run_id,
    statuses=(
        "succeeded",
        "failed",
        "awaiting_review",
        "cancelled",
        "timed_out",
        "needs_reconciliation",
        "effect_applied_result_unavailable",
    ),
):
    for _ in range(300):
        result = await engine.view("user", run_id)
        if result["status"] in statuses:
            return result
        await asyncio.sleep(0.01)
    raise AssertionError(result)


@pytest.mark.parametrize("workflow", ["conversation", "fixed"])
async def test_builtin_workflow_custom_manifest_uses_real_review_digest_and_writes_once(
    tmp_path, workflow
):
    effects = []
    tool = declaration(write=True)
    manifest = content_digest({"tool_catalog": "trusted-release-1"})

    async def handler(args, context):
        effects.append(args["x"])
        return {"value": args["x"]}

    selected = ResolvedTool(ToolDeclaration.model_validate(tool), handler, manifest)

    def resolver(context, call):
        assert context.agent_id == "agent0" and call.name == "custom"
        return selected

    configuration = package(tools=[tool]).model_dump(mode="json")
    configuration["agents"][0]["workflow"] = workflow
    if workflow == "fixed":
        configuration["agents"][0]["tasks"] = [
            {"id": "one", "tool": "custom", "arguments": {"x": 3}}
        ]
    async with system(
        tmp_path,
        [[("custom", {"x": 3})], "done"],
        config=AgentPackage.model_validate(configuration),
        handler=handler,
        tool_resolver=resolver,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "manifest", "run")
        pending = await wait(engine, created["run_id"])
        assert pending["status"] == "awaiting_review" and not effects
        review = pending["interrupts"][0]
        action = ToolCall(call_id=review["preview"]["call_id"], name="custom", arguments={"x": 3})
        context = await engine.context(created["run_id"])
        assert review["digest"] == engine.tools.challenge(context, action)
        assert review["digest"] != engine.tools.action_digest(context, selected.declaration, action)
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=review["review_id"],
            digest=review["digest"],
        )
        finished = await wait(engine, created["run_id"])
        assert finished["status"] == "succeeded" and effects == [3]
        rows = engine.actions.operations("user", created["run_id"])
        assert len(rows) == 1 and rows[0]["phase"] == "applied" and rows[0]["attempts"] == 1
        assert (await engine.submit(identities["user"], "agent0", "manifest", "run"))[
            "run_id"
        ] == created["run_id"]
        assert effects == [3] and len(provider.requests) == (0 if workflow == "fixed" else 2)


@pytest.mark.parametrize("write", [False, True])
async def test_custom_resolver_runtime_rechecks_scope_after_real_slot_wait(tmp_path, write):
    effects = []
    tool = declaration(write=write)

    async def handler(args, context):
        effects.append(args)
        return {"value": args["x"]}

    selected = ResolvedTool(
        ToolDeclaration.model_validate(tool), handler, content_digest({"trusted_release": 1})
    )

    def resolver(context, call):
        return selected

    async with system(
        tmp_path,
        [[("custom", {"x": 3})], "done"],
        config=package(tools=[tool]),
        handler=handler,
        tool_resolver=resolver,
    ) as (engine, provider, contexts, identities):
        await engine.tools.slots.acquire()
        # Hold all public slots so the actual tool must wait after admission/review.
        for _ in range(engine.settings.tools_concurrency - 1):
            await engine.tools.slots.acquire()
        created = await engine.submit(identities["user"], "agent0", "scope", "run")
        if write:
            pending = await wait(engine, created["run_id"])
            review = pending["interrupts"][0]
            await engine.resume(
                identities["user"],
                created["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
        async with asyncio.timeout(2):
            while not engine.tools.slots._waiters:  # noqa: ASYNC110 -- bounded wait for actual queue
                await asyncio.sleep(0)
        identities["user"] = Identity(
            owner="user", scopes=frozenset({"read" if write else "write"})
        )
        for _ in range(engine.settings.tools_concurrency):
            engine.tools.slots.release()
        await engine.queue.join()
        finished = await engine.view("user", created["run_id"])
        assert finished["resume_reason"] == "permission_changed" and effects == []
        rows = engine.actions.operations("user", created["run_id"])
        # Slot waiting grants no attempt. Revocation is detected before the
        # first claim, so no intent, token or execution window may be created.
        assert len(rows) == 1 and rows[0]["phase"] == "pending"
        assert rows[0]["attempts"] == 0 and rows[0]["attempt_token"] is None
        assert rows[0]["call_deadline"] is None


async def test_conversation_history_agent_isolation_and_client_idempotency(tmp_path):
    async with system(tmp_path, ["first", "second", "other"], config=package(agents=2)) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        identity = identities["user"]
        first = await engine.submit(identity, "agent0", "key", "Hello")
        final = await wait(engine, first["run_id"])
        assert final["result"] == "first" and final["status"] == "succeeded"
        assert len(contexts.turns("user", "agent0", "default", 0, 10)) == 1
        engine.available = False
        assert (await engine.submit(identity, "agent0", "key", "Hello"))["run_id"] == first[
            "run_id"
        ]
        with pytest.raises(AgentError, match="different"):
            await engine.submit(identity, "agent0", "key", "Changed")
        engine.available = True
        second = await engine.submit(identity, "agent0", "next", "Again")
        await wait(engine, second["run_id"])
        assert [m.content for m in provider.requests[1].messages if m.role != "system"] == [
            "Hello",
            "first",
            "Again",
        ]
        third = await engine.submit(identity, "agent1", "key", "Hello")
        await wait(engine, third["run_id"])
        assert [m.content for m in provider.requests[2].messages if m.role != "system"] == ["Hello"]
        with pytest.raises(AgentError) as denied:
            await engine.view("other", first["run_id"])
        assert denied.value.status == 404


async def test_read_tool_observation_and_history_flow_through_model(tmp_path):
    invoked = []

    async def handler(args, context):
        invoked.append(context.run_id)
        return {"value": args["x"] * 2}

    async with system(
        tmp_path,
        [[("custom", {"x": 3})], "answer"],
        config=package(tools=[declaration()]),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Calculate")
        final = await wait(engine, created["run_id"])
        assert final["status"] == "succeeded" and final["result"] == "answer" and len(invoked) == 1
        tool = next(message for message in provider.requests[1].messages if message.role == "tool")
        assert json.loads(tool.content)["value"] == {"value": 6}
        assert tool.tool_call_id == provider.requests[1].messages[-2].tool_calls[0].call_id
        assert len(contexts.turns("user", "agent0", "default", 0, 10)[0]) == 4


async def test_write_waits_for_exact_visible_review_and_preserves_budget(tmp_path):
    invoked = []

    async def handler(args, context):
        invoked.append((context.operation_key, args))
        return {"value": args["x"]}

    async with system(
        tmp_path,
        [[("custom", {"x": 4})], "done"],
        config=package(tools=[declaration(True)]),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Write")
        paused = await wait(engine, created["run_id"])
        assert paused["status"] == "awaiting_review" and invoked == []
        review = paused["interrupts"][0]
        with pytest.raises(AgentError, match="visible"):
            await engine.resume(
                identities["user"], created["run_id"], review_id=review["review_id"], digest="wrong"
            )
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=review["review_id"],
            digest=review["digest"],
        )
        final = await wait(engine, created["run_id"])
        assert (
            final["status"] == "succeeded" and len(invoked) == 1 and len(provider.requests) == 2
        ), [message.content for message in provider.requests[-1].messages]
        with engine.store.connection() as db:
            assert (
                db.execute(
                    "SELECT calls FROM agent_model_budgets WHERE run_id=?", (created["run_id"],)
                ).fetchone()[0]
                == 2
            )
        assert engine.actions.operations("user", created["run_id"])[0]["phase"] == "applied"


async def test_provisional_event_precedes_provider_finish_and_cancel_has_no_history(tmp_path):
    gate = asyncio.Event()
    async with system(tmp_path, ["candidate"], gate=gate) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        for _ in range(100):
            events = await engine.events("user", created["run_id"])
            if any(event.kind == "model.delta" for event in events):
                break
            await asyncio.sleep(0.01)
        delta = next(event for event in events if event.kind == "model.delta")
        assert delta.provisional and not gate.is_set() and created["run_id"] in engine.active
        assert contexts.turns("user", "agent0", "default", 0, 10) == []
        final = await engine.cancel("user", created["run_id"])
        assert (
            final["status"] == "cancelled"
            and contexts.turns("user", "agent0", "default", 0, 10) == []
        )
        assert engine.store.final_receipt("user", created["run_id"]) is None


async def test_final_only_and_scope_revocation_block_review_replay(tmp_path):
    identities = {"user": Identity(owner="user", scopes=frozenset({"write"}))}

    async def handler(args, context):
        raise AssertionError("revoked action must not run")

    async with system(
        tmp_path,
        [[("custom", {"x": 1})]],
        config=package(tools=[declaration(True)], stream_policy="final-only"),
        handler=handler,
        identities=identities,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        await wait(engine, created["run_id"])
        identities["user"] = Identity(owner="user", permission_version=2)
        view = await engine.view("user", created["run_id"])
        assert view["result_visibility"] == "revoked" and view["interrupts"] == []
        with pytest.raises(AgentError):
            await engine.resume(identities["user"], created["run_id"])
        with pytest.raises(AgentError):
            await engine.events("user", created["run_id"])


@pytest.fixture
def store(tmp_path):
    database = SqliteDatabase(tmp_path / "ledger.sqlite")
    result = SqliteRunStore(database.connection, EngineSettings(event_count=2))
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        result.install(db)
    return result


def admit(store, key="key", options=None):
    return store.admit(
        Identity(owner="user"),
        "agent0",
        key,
        key,
        {"input": "data"},
        "sha256:" + "1" * 64,
        options or RunOptions(),
        time.time() + 60,
        "pause-review",
    )[0]


def test_final_transaction_receipt_identity_history_conflict_and_large_value(store):
    row = admit(store)
    projection = FinalProjection(
        value="v" * 70000,
        messages=[
            Message(role="user", content="input"),
            Message(role="assistant", content="answer"),
        ],
    )
    assert store.commit_final(row, projection)
    assert store.commit_final(row, copy.deepcopy(projection))
    for field, other in (
        ("owner", "other"),
        ("agent_id", "other"),
        ("binding", "sha256:" + "2" * 64),
    ):
        with pytest.raises(AgentError, match="binding"):
            store.commit_final({**row, field: other}, projection)
    assert store.events("user", row["run_id"], 0)[0].kind == "run.final"
    assert not store.cancel("user", row["run_id"])
    conflict = admit(store, "conflict")
    with store.connection() as db:
        db.execute(
            "INSERT INTO agent_history VALUES(?,?,?,?,?,?,?)",
            ("user", "agent0", "default", 0, conflict["run_id"], "[]", time.time()),
        )
    with pytest.raises(AgentError, match="different history"):
        store.commit_final(conflict, projection)
    assert store.final_receipt("user", conflict["run_id"]) is None
    assert store.events("user", conflict["run_id"], 0) == []


def test_final_guard_and_extra_commit_failure_rollback_all_records(store):
    row = admit(store)
    assert not store.commit_final(row, FinalProjection(value="value", local_guard=lambda db: False))

    def fail(db):
        db.execute(
            "INSERT INTO agent_run_steps VALUES(?,?,?)", (row["run_id"], "side", row["binding"])
        )
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError):
        store.commit_final(row, FinalProjection(value="value", extra_commit=fail))
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_run_steps").fetchone()[0] == 0
        db.execute(
            "UPDATE requests SET deadline=? WHERE run_id=?", (time.time() - 1, row["run_id"])
        )
    assert not store.commit_final(row, FinalProjection(value="value"))
    assert store.final_receipt("user", row["run_id"]) is None


def test_current_review_pointer_expiry_and_duplicate_approval_do_not_refresh(store):
    row = admit(store)
    first = store.prepare_review(row, content_digest("first"), {"x": 1}, 60)
    second = store.prepare_review(row, content_digest("second"), {"x": 2}, 60)
    store.seal_review(row, second["id"], second["digest"])
    with pytest.raises(AgentError, match="current run"):
        store.accept_review("user", row["run_id"], first["id"], first["digest"])
    store.accept_review("user", row["run_id"], second["id"], second["digest"])
    deadline = store.request("user", row["run_id"])["deadline"]
    store.accept_review("user", row["run_id"], second["id"], second["digest"])
    assert store.request("user", row["run_id"])["deadline"] == deadline
    with store.connection() as db:
        db.execute(
            "UPDATE agent_reviews SET expires_at=? WHERE id=?", (time.time() - 1, second["id"])
        )
    with pytest.raises(AgentError, match="expired"):
        store.accept_review("user", row["run_id"], second["id"], second["digest"])


def test_event_cursor_retention_and_restart_sequence(store):
    row = admit(store)
    for index in range(4):
        assert store.append_event(row, "tool.completed", {"index": index}).seq == index + 1
    with pytest.raises(AgentError) as gap:
        store.events("user", row["run_id"], 1)
    assert gap.value.code == "event_cursor_expired"
    assert [event.seq for event in store.events("user", row["run_id"], 2)] == [3, 4]
    restarted = SqliteRunStore(store.connection, store.settings)
    assert restarted.append_event(row, "run.terminal", {}).seq == 5
    with store.connection() as db:
        db.execute("UPDATE agent_run_events SET created_at=0")
    with pytest.raises(AgentError):
        store.events("user", row["run_id"], 3)


def test_install_requires_transaction_and_unknown_version_leaves_no_new_tables(tmp_path):
    database = SqliteDatabase(tmp_path / "unknown.sqlite")
    with database.connection() as db:
        with pytest.raises(AgentError, match="transaction"):
            SqliteRunStore.install(db)
        db.execute("CREATE TABLE agent_runtime_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("INSERT INTO agent_runtime_meta VALUES('schema','999')")
    with pytest.raises(AgentError, match="migration"), database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
    with database.connection() as db:
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()[0][0]
            == "agent_runtime_meta"
        )
        assert len(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()) == 1


def fixed_package(tasks, tools=None):
    data = package(tools=tools or [declaration()]).model_dump(mode="json")
    data["agents"][0].update(
        {"workflow": "fixed", "model": None, "input_schema": {"type": "object"}, "tasks": tasks}
    )
    return AgentPackage.model_validate(data)


async def test_fixed_input_bindings_parallel_reads_and_dependency_observations(tmp_path):
    entered = []
    both = asyncio.Event()

    async def handler(args, context):
        if not context.dependency_results:
            entered.append(args["x"])
            if len(entered) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 1)
            return {"value": args["x"] * 2}
        assert set(context.dependency_results) == {"left", "right"}
        return {"value": sum(value["value"] for value in context.dependency_results.values())}

    config = fixed_package(
        [
            {"id": "left", "tool": "custom", "input_bindings": {"x": "/a"}},
            {"id": "right", "tool": "custom", "input_bindings": {"x": "/b"}},
            {
                "id": "combine",
                "tool": "custom",
                "arguments": {"x": 0},
                "depends_on": ["left", "right"],
                "use_dependencies": True,
            },
        ]
    )
    async with system(tmp_path, [], config=config, handler=handler) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "key", {"a": 2, "b": 3})
        final = await wait(engine, created["run_id"])
        assert final["status"] == "succeeded" and both.is_set()
        assert final["result"]["combine"]["value"] == {"value": 10}
        assert provider.requests == []
        with pytest.raises(AgentError, match="pointer"):
            await engine.submit(identities["user"], "agent0", "bad", {"a": 2})
        assert engine.store.lookup("user", "agent0", "bad") is None


async def test_fixed_public_input_review_precedes_read_tool(tmp_path):
    invoked = []

    async def handler(args, context):
        invoked.append(args)
        return {"value": args["x"]}

    config = fixed_package([{"id": "task", "tool": "custom", "input_bindings": {"x": "/x"}}])
    async with system(tmp_path, [], config=config, handler=handler) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(
            identities["user"], "agent0", "key", {"x": 2}, RunOptions(pause_for_review=True)
        )
        paused = await wait(engine, created["run_id"])
        assert paused["status"] == "awaiting_review" and invoked == []
        review = paused["interrupts"][0]
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=review["review_id"],
            digest=review["digest"],
        )
        assert (await wait(engine, created["run_id"]))["status"] == "succeeded" and invoked == [
            {"x": 2}
        ]


def test_json_pointer_lookup_and_binding_reject_evaluation_or_invalid_indices():
    source = {"a/b": {"m~n": [1, {"x": 2}]}}
    selected = input_pointer(source, "/a~1b/m~0n/1")
    selected["x"] = 5
    assert source["a/b"]["m~n"][1]["x"] == 2
    for pointer in ("/a~1b/m~0n/-", "/a~1b/m~0n/01", "/a~1b/m~0n/-1", "__class__", "/bad~2"):
        with pytest.raises(AgentError):
            input_pointer(source, pointer)
    with pytest.raises(ValueError):
        fixed_package(
            [{"id": "task", "tool": "custom", "arguments": {"x": 1}, "input_bindings": {"x": "/x"}}]
        )


async def test_sql_final_ack_loss_restart_past_deadline_repairs_checkpoint_only(
    tmp_path, monkeypatch
):
    async with system(tmp_path, ["done"]) as (engine, provider, contexts, identities):
        original = engine.store.commit_final

        def lost_ack(row, projection):
            original(row, projection)
            raise RuntimeError("synthetic final receipt acknowledgement loss")

        monkeypatch.setattr(engine.store, "commit_final", lost_ack)
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        gap = await wait(engine, created["run_id"], ("finalization_incomplete",))
        assert gap["result"] == "done"
        await engine.close()
        monkeypatch.setattr(engine.store, "commit_final", original)
        with engine.store.connection() as db:
            db.execute(
                "UPDATE requests SET deadline=? WHERE run_id=?",
                (time.time() - 1, created["run_id"]),
            )
        restarted = RunEngine(
            engine.registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        await restarted.start()
        try:
            await restarted.resume(identities["user"], created["run_id"])
            final = await wait(restarted, created["run_id"], ("succeeded",))
            assert final["result"] == "done" and len(provider.requests) == 1
            assert len(contexts.turns("user", "agent0", "default", 0, 10)) == 1
        finally:
            await restarted.close()


async def test_approved_review_restart_before_command_and_expired_original_clock(tmp_path):
    invoked = []

    async def handler(args, context):
        invoked.append(args)
        return {"value": args["x"]}

    async with system(
        tmp_path,
        [[("custom", {"x": 1})], "done"],
        config=package(tools=[declaration(True)]),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        paused = await wait(engine, created["run_id"])
        review = paused["interrupts"][0]
        with engine.store.connection() as db:
            db.execute(
                "UPDATE requests SET deadline=? WHERE run_id=?",
                (time.time() - 1, created["run_id"]),
            )
        # Simulate approval transaction committed, but no Command has entered the graph.
        engine.store.accept_review("user", created["run_id"], review["review_id"], review["digest"])
        approved_deadline = engine.store.request("user", created["run_id"])["deadline"]
        await engine.close()
        restarted = RunEngine(
            engine.registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        await restarted.start()
        try:
            await restarted.resume(
                identities["user"],
                created["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
            assert engine.store.request("user", created["run_id"])["deadline"] == approved_deadline
            assert (await wait(restarted, created["run_id"]))["status"] == "succeeded"
            assert invoked == [{"x": 1}] and len(provider.requests) == 2
        finally:
            await restarted.close()


async def test_prepared_review_without_visible_interrupt_cannot_grant_approval(tmp_path):
    gate = asyncio.Event()
    async with system(tmp_path, ["candidate"], gate=gate) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        row = engine.store.request("user", created["run_id"])
        review = engine.store.prepare_review(row, content_digest("prepared"), {"x": 1}, 60)
        with pytest.raises(AgentError):
            await engine.resume(
                identities["user"],
                created["run_id"],
                review_id=review["id"],
                digest=review["digest"],
            )
        assert engine.store.review("user", created["run_id"], review["id"])["approved_at"] is None
        await engine.cancel("user", created["run_id"])


async def test_topic_switch_cancels_active_and_queued_runs(tmp_path):
    gate = asyncio.Event()
    async with system(tmp_path, ["old", "new"], gate=gate, settings=EngineSettings(workers=1)) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        first = await engine.submit(identities["user"], "agent0", "first", "Old active")
        for _ in range(100):
            if provider.requests:
                break
            await asyncio.sleep(0.01)
        second = await engine.submit(identities["user"], "agent0", "second", "Old queued")
        newer = await engine.submit(
            identities["user"], "agent0", "new", "New input", RunOptions(switch_topic=True)
        )
        gate.set()
        assert (await wait(engine, first["run_id"]))["status"] == "cancelled"
        assert (await wait(engine, second["run_id"]))["status"] == "cancelled"
        assert (await wait(engine, newer["run_id"]))["status"] == "succeeded"
        assert [
            message.content
            for message in provider.requests[-1].messages
            if message.role != "system"
        ] == ["New input"]
        assert contexts.turns("user", "agent0", "default", 0, 10) == []


async def test_write_applied_then_final_validation_failure_reports_effect(tmp_path):
    invoked = []

    async def handler(args, context):
        invoked.append(args)
        return {"value": args["x"]}

    config = package(tools=[declaration(True)]).model_dump(mode="json")
    config["agents"][0]["output_schema"] = {"type": "object"}
    async with system(
        tmp_path,
        [[("custom", {"x": 1})], "not JSON"],
        config=AgentPackage.model_validate(config),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        paused = await wait(engine, created["run_id"])
        review = paused["interrupts"][0]
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=review["review_id"],
            digest=review["digest"],
        )
        final = await wait(engine, created["run_id"])
        assert (
            final["status"] == "effect_applied_result_unavailable"
            and final["effects"][0]["phase"] == "applied"
        )
        assert (
            invoked == [{"x": 1}] and engine.store.final_receipt("user", created["run_id"]) is None
        )


async def test_tool_sources_propagate_to_model_history_view_and_event_replay(tmp_path):
    invoked, revoked = [], False
    tool = declaration()
    tool["output_schema"]["properties"].update(
        {"source_id": {"type": "string"}, "source_version": {"type": "integer"}}
    )

    async def handler(args, context):
        invoked.append(args)
        return {"value": args["x"], "source_id": "document1", "source_version": 1}

    async with system(
        tmp_path, [[("custom", {"x": 2})], "done"], config=package(tools=[tool]), handler=handler
    ) as (engine, provider, contexts, identities):

        async def policy(context, declaration, value):
            return ToolResult(
                value=value,
                sources=[
                    {
                        "kind": "fixture",
                        "id": value["source_id"],
                        "version": value["source_version"],
                    }
                ],
            )

        async def authorized(*args):
            return not revoked

        engine.tools.result_policy = policy
        engine.contexts.source_checker = authorized
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        assert (await wait(engine, created["run_id"]))["status"] == "succeeded"
        tool_message = next(
            message for message in provider.requests[1].messages if message.role == "tool"
        )
        assert tool_message.sources == [{"kind": "fixture", "id": "document1", "version": 1}]
        assert (
            contexts.turns("user", "agent0", "default", 0, 10)[0][-1].sources
            == tool_message.sources
        )
        revoked = True
        assert (await engine.view("user", created["run_id"]))["result_visibility"] == "revoked"
        with pytest.raises(AgentError, match="sources"):
            await engine.events("user", created["run_id"])
        assert invoked == [{"x": 2}]


async def test_fixed_writes_are_serial_and_each_new_action_requires_its_review(tmp_path):
    invoked = []

    async def handler(args, context):
        invoked.append((args["x"], context.dependency_results))
        return {"value": args["x"]}

    read, write = declaration(), declaration(True)
    write["name"] = "write"
    config = fixed_package(
        [
            {"id": "read", "tool": "custom", "arguments": {"x": 1}},
            {
                "id": "write1",
                "tool": "write",
                "arguments": {"x": 2},
                "depends_on": ["read"],
                "use_dependencies": True,
            },
            {"id": "write2", "tool": "write", "arguments": {"x": 3}},
        ],
        [read, write],
    )
    async with system(tmp_path, [], config=config, handler=handler) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "key", {})
        first = (await wait(engine, created["run_id"]))["interrupts"][0]
        assert invoked == [(1, {})] and first["preview"]["dependencies"] == {"read": {"value": 1}}
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=first["review_id"],
            digest=first["digest"],
        )
        second = (await wait(engine, created["run_id"]))["interrupts"][0]
        assert second["review_id"] != first["review_id"] and len(invoked) == 2
        with pytest.raises(AgentError):
            await engine.resume(
                identities["user"],
                created["run_id"],
                review_id=first["review_id"],
                digest=first["digest"],
            )
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=second["review_id"],
            digest=second["digest"],
        )
        assert (await wait(engine, created["run_id"]))["status"] == "succeeded"
        assert [args for args, deps in invoked] == [1, 2, 3]


async def test_shutdown_recovery_preserves_model_call_reservation(tmp_path):
    gate = asyncio.Event()
    async with system(
        tmp_path, ["candidate"], gate=gate, config=package(budget={"max_model_calls": 1})
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        for _ in range(100):
            if provider.requests:
                break
            await asyncio.sleep(0.01)
        await engine.close()
        restarted = RunEngine(
            engine.registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        await restarted.start()
        try:
            await restarted.resume(identities["user"], created["run_id"])
            final = await wait(restarted, created["run_id"])
            assert final["status"] == "failed" and final["error"] == "model_run_budget"
            assert (
                len(provider.requests) == 1
                and contexts.turns("user", "agent0", "default", 0, 10) == []
            )
        finally:
            await restarted.close()


async def test_topic_switch_invalidates_waiting_review_without_applying_tool(tmp_path):
    async def handler(args, context):
        raise AssertionError("old topic must not apply")

    async with system(
        tmp_path,
        [[("custom", {"x": 1})], "new"],
        config=package(tools=[declaration(True)]),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        first = await engine.submit(identities["user"], "agent0", "old", "Old")
        review = (await wait(engine, first["run_id"]))["interrupts"][0]
        newer = await engine.submit(
            identities["user"], "agent0", "new", "New", RunOptions(switch_topic=True)
        )
        assert (await wait(engine, first["run_id"]))["status"] == "cancelled"
        with pytest.raises(AgentError):
            await engine.resume(
                identities["user"],
                first["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
        assert (await wait(engine, newer["run_id"]))["status"] == "succeeded"


async def test_scope_change_without_version_increment_still_blocks_replay(tmp_path):
    async with system(tmp_path, ["done"]) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "key", "Input")
        await wait(engine, created["run_id"])
        identities["user"] = Identity(owner="user", scopes=frozenset())
        assert (await engine.view("user", created["run_id"]))["result_visibility"] == "revoked"
        with pytest.raises(AgentError):
            await engine.events("user", created["run_id"])


def test_owned_intent_requires_exact_token_and_never_exempts_unknown(store):
    from agentbase.contracts import RunContext, ToolCall
    from agentbase.effects import SqliteActionLedger

    row = admit(store)
    ledger = SqliteActionLedger(store.connection)
    context = RunContext(
        identity=Identity(owner="user"),
        agent_id="agent0",
        run_id=row["run_id"],
        conversation_id="default",
        deadline=row["deadline"],
        config_digest=row["binding"],
    )
    tool = package(tools=[declaration(True)]).tools[0]
    call = ToolCall(call_id="call", name="custom", arguments={"x": 1})
    action = ledger.begin(context, tool, call, 10)
    token = ledger.claim_attempt(action["key"], 1)
    assert ledger.uncertain(row["run_id"])
    assert not ledger.uncertain(row["run_id"], owned_intent=(action["key"], token))
    assert ledger.uncertain(row["run_id"], owned_intent=(action["key"], "wrong"))
    ledger.failed(action["key"], "lost", handler_started=True, attempt_token=token)
    assert ledger.uncertain(row["run_id"], owned_intent=(action["key"], token))


async def test_memory_mutation_after_async_checks_fails_final_local_transaction_guard(tmp_path):
    config = package().model_dump(mode="json")
    config["agents"][0]["context"]["memory_scope"] = "agent"
    async with system(tmp_path, [], config=AgentPackage.model_validate(config)) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        identity = identities["user"]
        memory = contexts.create_memory("user", "agent0", "", "Original")
        row = engine.store.admit(
            identity,
            "agent0",
            "key",
            "key",
            {"input": "Input", "_identity_digest": engine.identity_digest(identity)},
            engine.registry.agent("agent0").digest,
            RunOptions(),
            time.time() + 30,
            "pause-review",
        )[0]
        calls = 0

        async def resolve(owner):
            nonlocal calls
            calls += 1
            if calls == 2:
                with contexts.connection() as db:
                    db.execute(
                        "UPDATE agent_memories SET version=version+1 WHERE id=?", (memory.id,)
                    )
            await asyncio.sleep(0)
            return identity

        engine.identity_resolver = resolve
        with pytest.raises(AgentError, match="race"):
            await engine.commit_final(
                row["run_id"],
                FinalProjection(value="result", sources=[ContextProvider.memory_reference(memory)]),
            )
        assert engine.store.final_receipt("user", row["run_id"]) is None
        assert contexts.turns("user", "agent0", "default", 0, 10) == []


def test_review_clock_pauses_only_after_visibility_and_seals_once(store):
    row = admit(store)
    prepared = store.prepare_review(row, content_digest("review"), {"x": 1}, 60)
    with pytest.raises(AgentError):
        store.accept_review("user", row["run_id"], prepared["id"], prepared["digest"])
    with store.connection() as db:
        db.execute(
            "UPDATE requests SET deadline=? WHERE run_id=?", (time.time() + 3, row["run_id"])
        )
    sealed = store.seal_review(row, prepared["id"], prepared["digest"])
    assert 0 < sealed["remaining"] <= 3 < prepared["remaining"]
    assert store.seal_review(row, prepared["id"], prepared["digest"]) == sealed
    store.accept_review("user", row["run_id"], prepared["id"], prepared["digest"])
    deadline = store.request("user", row["run_id"])["deadline"]
    store.accept_review("user", row["run_id"], prepared["id"], prepared["digest"])
    assert store.request("user", row["run_id"])["deadline"] == deadline
    unsealed = admit(store, "unsealed")
    review = store.prepare_review(unsealed, content_digest("unsealed"), {}, 60)
    with store.connection() as db:
        db.execute(
            "UPDATE requests SET deadline=? WHERE run_id=?", (time.time() - 1, unsealed["run_id"])
        )
    with pytest.raises(AgentError, match="unsealed"):
        store.seal_review(unsealed, review["id"], review["digest"])


async def test_topic_switch_keeps_already_committed_run_final(tmp_path):
    async with system(tmp_path, ["first", "next"]) as (engine, provider, contexts, identities):
        first = await engine.submit(identities["user"], "agent0", "first", "First")
        await wait(engine, first["run_id"])
        second = await engine.submit(
            identities["user"], "agent0", "next", "Next", RunOptions(switch_topic=True)
        )
        await wait(engine, second["run_id"])
        old = await engine.view("user", first["run_id"])
        assert old["status"] == "succeeded" and old["result"] == "first"
        assert engine.store.request("user", first["run_id"])["cancel"] == 0


async def test_two_business_configuration_example_runs_without_core_changes():
    from examples.agentbase_offline import run_demo

    result = await run_demo()
    assert result == {
        "calculation": {"value": 10},
        "classification": {"category": "question"},
        "fixture_model_calls": 1,
        "delta_before_fixture_finish": True,
        "model_quality_verified": False,
    }
