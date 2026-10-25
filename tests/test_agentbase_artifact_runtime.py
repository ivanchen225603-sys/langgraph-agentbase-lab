"""Artifacts remain authorized through actual runs, final transactions and event delivery."""

import asyncio
import time

import pytest
from langgraph.graph import START

from agentbase.artifacts import ArtifactOperation, ArtifactScope, SqliteArtifactStore
from agentbase.configuration import AgentPackage
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Message
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import EngineSettings, FinalProjection, RunOptions
from agentbase.storage import SqliteDatabase
from agentbase.tools import ToolResult
from tests.test_agentbase_runtime import declaration, package, system, wait

SCOPE = ArtifactScope(owner="user", agent_id="agent0", namespace="default")


def operation(engine, key):
    return ArtifactOperation(
        run_id="host", key=key, config_digest=engine.registry.agent("agent0").digest
    )


def artifact(engine, *, sources=None):
    return engine.artifacts.create(
        SCOPE,
        operation(engine, "create"),
        {"items": [1]},
        sources=sources or [],
        guard=lambda db: True,
    )


def admit(engine, identity):
    return engine.store.admit(
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


@pytest.mark.parametrize("mutation", ["update", "revoke", "restore"])
async def test_actual_model_run_final_history_view_and_event_frames_use_current_references(
    tmp_path, mutation
):
    async with system(tmp_path, ["result"]) as (engine, provider, contexts, identities):
        record = artifact(engine)
        contexts.append_turn(
            "user",
            "agent0",
            "default",
            0,
            "earlier",
            [
                Message(role="user", content="Earlier input"),
                Message(role="assistant", content="Earlier result", sources=[record.reference()]),
            ],
            guard=lambda db: True,
        )
        created = await engine.submit(identities["user"], "agent0", "actual", "Current input")
        final = await wait(engine, created["run_id"])
        assert final["status"] == "succeeded" and final["result"] == "result"
        assert any(
            record.reference() in message.sources for message in provider.requests[0].messages
        )
        events = await engine.events("user", created["run_id"])
        delta = next(event for event in events if event.kind == "model.delta")
        engine.check_event_frame(identities["user"], delta)
        if mutation == "update":
            engine.artifacts.update(
                SCOPE,
                operation(engine, "change"),
                record.id,
                1,
                "changed",
                sources=[],
                guard=lambda db: True,
            )
        elif mutation == "revoke":
            engine.artifacts.revoke(
                SCOPE, operation(engine, "change"), record.id, 1, guard=lambda db: True
            )
        else:
            engine.artifacts.restore(
                SCOPE, operation(engine, "change"), record.id, 1, 1, guard=lambda db: True
            )
        hidden = await engine.view("user", created["run_id"])
        assert hidden["result"] is None and hidden["result_visibility"] == "revoked"
        with pytest.raises(AgentError) as error:
            await engine.events("user", created["run_id"])
        assert error.value.code == "source_revoked"
        with pytest.raises(AgentError) as error:
            engine.check_event_frame(identities["user"], delta)
        assert error.value.code == "source_revoked"
        assert (
            engine.store.final_receipt("user", created["run_id"]) is not None
        )  # Real completion is retained, visibility changes.


@pytest.mark.parametrize("placement", ["projection", "messages", "memory"])
async def test_mutation_after_async_checks_cannot_commit_final_history_or_final_event(
    tmp_path, placement
):
    config = package().model_dump(mode="json")
    config["agents"][0]["context"]["memory_scope"] = "agent"
    async with system(tmp_path, [], config=AgentPackage.model_validate(config)) as (
        engine,
        _,
        contexts,
        identities,
    ):
        identity = identities["user"]
        row = admit(engine, identity)
        record = artifact(engine)
        source = record.reference()
        if placement == "memory":
            memory = contexts.create_memory("user", "agent0", "", "value", sources=[source])
            source = ContextProvider.memory_reference(memory)
        calls = 0

        async def resolve(owner):
            nonlocal calls
            calls += 1
            if calls == 2:
                engine.artifacts.revoke(
                    SCOPE, operation(engine, "race"), record.id, 1, guard=lambda db: True
                )
            await asyncio.sleep(0)
            return identity

        engine.identity_resolver = resolve
        messages = [
            Message(role="user", content="Input"),
            Message(
                role="assistant",
                content="Result",
                sources=[source] if placement == "messages" else [],
            ),
        ]
        with pytest.raises(AgentError) as error:
            await engine.commit_final(
                row["run_id"],
                FinalProjection(
                    value="result",
                    messages=messages,
                    sources=[] if placement == "messages" else [source],
                ),
            )
        assert error.value.code == "final_control"
        assert engine.store.final_receipt("user", row["run_id"]) is None
        assert contexts.turns("user", "agent0", "default", 0, 10) == []
        with engine.store.connection() as db:
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM agent_run_events WHERE run_id=?", (row["run_id"],)
                ).fetchone()[0]
                == 0
            )


@pytest.mark.parametrize("phase", ["emit", "replay"])
async def test_event_publication_and_replay_recheck_artifacts_in_their_sql_transaction(
    tmp_path, phase
):
    async with system(tmp_path, []) as (engine, _, _, identities):
        identity = identities["user"]
        row = admit(engine, identity)
        record = artifact(engine)
        if phase == "replay":
            await engine.emit(
                row["run_id"], "model.delta", {"text": "protected"}, sources=[record.reference()]
            )
        calls = 0

        async def resolve(owner):
            nonlocal calls
            calls += 1
            if calls == 2:
                engine.artifacts.revoke(
                    SCOPE, operation(engine, "race"), record.id, 1, guard=lambda db: True
                )
            await asyncio.sleep(0)
            return identity

        engine.identity_resolver = resolve
        with pytest.raises(AgentError) as error:
            if phase == "emit":
                await engine.emit(
                    row["run_id"],
                    "model.delta",
                    {"text": "protected"},
                    sources=[record.reference()],
                )
            else:
                await engine.events("user", row["run_id"])
        assert error.value.code == ("event_control" if phase == "emit" else "source_revoked")
        with engine.store.connection() as db:
            assert db.execute(
                "SELECT COUNT(*) FROM agent_run_events WHERE run_id=?", (row["run_id"],)
            ).fetchone()[0] == (phase == "replay")
            assert (
                db.execute(
                    "SELECT next_seq FROM agent_event_heads WHERE run_id=?", (row["run_id"],)
                ).fetchone()
                is None
                if phase == "emit"
                else True
            )


def test_runtime_artifacts_cannot_use_a_different_database(tmp_path):
    from agentbase.registry import ComponentRegistry
    from agentbase.storage import SqliteContextStore

    first = SqliteDatabase(tmp_path / "first.sqlite")
    second = SqliteDatabase(tmp_path / "second.sqlite")
    contexts = SqliteContextStore(first.connection)
    provider = ContextProvider(contexts, contexts, artifacts=SqliteArtifactStore(second.connection))
    with pytest.raises(AgentError) as error:
        RunEngine(
            ComponentRegistry(),
            SqliteRunStore(first.connection, EngineSettings()),
            None,
            provider,
            None,
        )
    assert error.value.code == "artifact_binding"
    assert not first.path.exists() and not second.path.exists()


def test_runtime_schema4_upgrade_is_explicit_and_transactional(tmp_path):
    database = SqliteDatabase(tmp_path / "state.sqlite")
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
        for table in [
            "agent_artifact_mutations",
            "agent_artifact_revisions",
            "agent_artifact_heads",
            "agent_artifact_meta",
        ]:
            db.execute(f"DROP TABLE {table}")
        db.execute("UPDATE agent_runtime_meta SET value='4' WHERE key='schema'")
        db.execute("INSERT INTO agent_conversations VALUES('user','agent0','default',9)")
    with pytest.raises(RuntimeError, match="interrupt"):
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            SqliteRunStore.install(db)
            raise RuntimeError("interrupt migration before commit")
    with database.connection() as db:
        assert (
            db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
            == "4"
        )
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name='agent_artifact_heads'"
        ).fetchone()
        assert db.execute("SELECT epoch FROM agent_conversations").fetchone()[0] == 9
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
    with database.connection() as db:
        assert (
            db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
            == "8"
        )
        assert db.execute("SELECT epoch FROM agent_conversations").fetchone()[0] == 9
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_heads").fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["update", "revoke", "restore"])
async def test_source_change_in_last_identity_await_never_reaches_model_provider(
    tmp_path, mutation
):
    async with system(tmp_path, ["must not execute"]) as (engine, provider, contexts, identities):
        identity = identities["user"]
        record = artifact(engine)
        contexts.append_turn(
            "user",
            "agent0",
            "default",
            0,
            "earlier",
            [
                Message(role="user", content="Earlier"),
                Message(
                    role="assistant", content="Sensitive history", sources=[record.reference()]
                ),
            ],
            guard=lambda db: True,
        )
        row = admit(engine, identity)
        await engine.update_checkpoint(row, engine.binding(row).initial(row), as_node=START)
        source_authorized, changed = False, False
        original_authorize = engine.authorized_sources

        async def authorize(request, current_identity, sources):
            nonlocal source_authorized
            valid = await original_authorize(request, current_identity, sources)
            if valid and record.reference() in sources:
                source_authorized = True
            return valid

        async def resolve(owner):
            nonlocal changed, source_authorized
            if source_authorized and not changed:
                source_authorized, changed = False, True
                if mutation == "update":
                    engine.artifacts.update(
                        SCOPE,
                        operation(engine, "before-provider"),
                        record.id,
                        1,
                        "changed",
                        sources=[],
                        guard=lambda db: True,
                    )
                elif mutation == "revoke":
                    engine.artifacts.revoke(
                        SCOPE,
                        operation(engine, "before-provider"),
                        record.id,
                        1,
                        guard=lambda db: True,
                    )
                else:
                    engine.artifacts.restore(
                        SCOPE,
                        operation(engine, "before-provider"),
                        record.id,
                        1,
                        1,
                        guard=lambda db: True,
                    )
            await asyncio.sleep(0)
            return identity

        engine.authorized_sources, engine.identity_resolver = authorize, resolve
        await engine.execute(row)
        assert changed and provider.requests == []
        assert engine.store.final_receipt("user", row["run_id"]) is None
        snapshot = await engine.checkpoint_state(row)
        assert snapshot.values["error"] == "source_revoked"


async def test_sql_source_guard_handles_cross_kind_equal_ids_cycles_and_depth(tmp_path):
    config = package().model_dump(mode="json")
    config["agents"][0]["context"]["memory_scope"] = "conversation"
    async with system(tmp_path, [], config=AgentPackage.model_validate(config)) as (
        engine,
        _,
        contexts,
        identities,
    ):
        row = admit(engine, identities["user"])
        record = artifact(engine)
        memory = contexts.create_memory(
            "user", "agent0", "default", "value", sources=[record.reference()]
        )
        same_id = memory.model_copy(update={"id": record.id})
        with contexts.connection() as db:
            db.execute(
                "UPDATE agent_memories SET id=?,payload=? WHERE id=?",
                (same_id.id, same_id.model_dump_json(), memory.id),
            )
        source = ContextProvider.memory_reference(same_id)
        with engine.store.connection() as db:
            db.execute("BEGIN")
            assert engine._local_sources(db, row, [source])
        cyclic = engine.artifacts.update(
            SCOPE,
            operation(engine, "cycle"),
            record.id,
            1,
            "value",
            sources=[source],
            guard=lambda db: True,
        )
        with contexts.connection() as db:
            db.execute(
                "UPDATE agent_memories SET payload=? WHERE id=?",
                (
                    same_id.model_copy(update={"sources": [cyclic.reference()]}).model_dump_json(),
                    same_id.id,
                ),
            )
        with engine.store.connection() as db:
            db.execute("BEGIN")
            assert not engine._local_sources(db, row, [cyclic.reference()])
        leaf = engine.artifacts.create(
            SCOPE, operation(engine, "leaf"), 1, sources=[], guard=lambda db: True
        )
        current = leaf
        for index in range(17):
            current = engine.artifacts.create(
                SCOPE,
                operation(engine, f"depth{index}"),
                index,
                sources=[current.reference()],
                guard=lambda db: True,
            )
            with engine.store.connection() as db:
                db.execute("BEGIN")
                assert engine._local_sources(db, row, [current.reference()]) == (index < 16)


@pytest.mark.parametrize("invalid", ["digest", "expiry", "shared-agent"])
async def test_local_artifact_guard_rejects_invalid_digest_expiry_and_implicit_sharing(
    tmp_path, invalid
):
    config = package(agents=2).model_dump(mode="json")
    config["agents"][0]["context"] = {"memory_scope": "agent", "shared_memory_agents": ["agent1"]}
    async with system(tmp_path, [], config=AgentPackage.model_validate(config)) as (
        engine,
        _,
        _,
        identities,
    ):
        row = admit(engine, identities["user"])
        record = engine.artifacts.create(
            SCOPE.model_copy(update={"agent_id": "agent1"}) if invalid == "shared-agent" else SCOPE,
            operation(engine, "invalid"),
            "value",
            sources=[],
            valid_until=time.time() - 1 if invalid == "expiry" else None,
            guard=lambda db: True,
        )
        source = record.reference()
        if invalid == "digest":
            source["digest"] = "sha256:" + "f" * 64
        assert not await engine.authorized_sources(row, identities["user"], [source])
        with engine.store.connection() as db:
            db.execute("BEGIN")
            assert not engine._local_sources(db, row, [source])


async def test_actual_tool_observation_retains_artifact_provenance_in_model_history_and_receipt(
    tmp_path,
):
    calls = []

    async def handler(arguments, context):
        calls.append(context.run_id)
        scope = ArtifactScope(
            owner=context.identity.owner,
            agent_id=context.agent_id,
            namespace=context.conversation_id,
        )
        current = engine.artifacts.read(scope, record.id)
        return {"value": current.value["items"][0] + arguments["x"]}

    async with system(
        tmp_path,
        [[("custom", {"x": 2})], "3"],
        config=package(tools=[declaration()]),
        handler=handler,
    ) as (engine, provider, _, identities):
        record = artifact(engine)

        async def policy(context, tool, value):
            return ToolResult(value=value, sources=[record.reference()])

        engine.tools.result_policy = policy
        created = await engine.submit(identities["user"], "agent0", "tool-artifact", "Input")
        final = await wait(engine, created["run_id"])
        assert final["status"] == "succeeded" and final["result"] == "3"
        assert calls == [created["run_id"]] and len(provider.requests) == 2
        observation = next(
            message for message in provider.requests[1].messages if message.role == "tool"
        )
        assert observation.sources == [record.reference()] and '"value":3' in observation.content
        assert (
            record.reference() in engine.store.final_receipt("user", created["run_id"])["sources"]
        )
