"""Shared-runtime migration and compatibility authorization fault windows."""

import asyncio
import copy
import sqlite3
import time
from types import SimpleNamespace

import pytest

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError
from agentbase.run_store import SqliteRunStore
from agentbase.workflows import register_workflows
from raglab.config import Settings
from raglab.errors import LabError
from raglab.models import RunInput, TaskSpec
from raglab.persistence import Store, digest, encode
from raglab.service import LabService
from tests.conftest import LOCAL, ROOT, wait_run
from tests.test_runtime import register


async def test_repeated_immediate_cancel_keeps_durable_terminal_checkpoint_after_reopen(tmp_path):
    settings = Settings(data_dir=tmp_path, knowledge_root=ROOT / "data/knowledge")
    lab = await LabService(settings).start()
    snapshots = []
    try:
        for number in range(12):
            run = await lab.submit(
                LOCAL,
                f"immediate-cancel-{number}",
                RunInput(
                    tasks=[
                        TaskSpec(id="m", tool="memory.search", arguments={"question": "sample"})
                    ],
                    pause_for_review=True,
                    timeout_seconds=60,
                ),
            )
            # Do not wait for a node/review; exercise the admitted checkpoint window.
            await lab.cancel(LOCAL.owner, run["run_id"])
            async with asyncio.timeout(5):
                await lab.runtime.queue.join()
            view = await lab.view(LOCAL.owner, run["run_id"])
            row = lab.store.request(LOCAL.owner, run["run_id"])
            assert view["status"] == "cancelled" and row["cancel"] == 1
            assert view["checkpoint_id"] and row["deadline"] == run["deadline"]
            assert not lab.active and not lab.pending
            snapshots.append((run["run_id"], view["checkpoint_id"], row["deadline"]))
    finally:
        await lab.close()
    reopened = await LabService(settings).start()
    try:
        for run_id, checkpoint, deadline in snapshots:
            view = await reopened.view(LOCAL.owner, run_id)
            assert view["status"] == "cancelled" and view["checkpoint_id"] == checkpoint
            assert view["deadline"] == deadline and not view["recoverable"]
            with pytest.raises(LabError) as denied:
                await reopened.resume(LOCAL, run_id)
            assert denied.value.code == "not_recoverable"
    finally:
        await reopened.close()


def old_database(path, owner="local"):
    request = RunInput(
        tasks=[TaskSpec(id="m", tool="memory.search", arguments={"question": "sample"})],
        pause_for_review=True,
    )
    payload = request.model_dump()
    payload.update(
        permission_version=1,
        tool_versions={"memory.search": "1"},
        index_version="",
        state_version=1,
        graph_version="agent-v1",
        client_digest=digest(request.model_dump()),
    )
    original = (
        "legacy-run",
        owner,
        "legacy-key",
        digest(payload),
        encode(payload),
        "original-thread",
        time.time() + 60,
        3,
        0,
    )
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.executemany(
            "INSERT INTO meta VALUES(?,?)", [("schema", "1"), ("acl", "1"), ("memory", "1")]
        )
        db.execute(
            "CREATE TABLE conversations(owner TEXT,id TEXT,epoch INTEGER,PRIMARY KEY(owner,id))"
        )
        db.execute("INSERT INTO conversations VALUES(?,'default',3)", (owner,))
        db.execute(
            "CREATE TABLE requests(run_id TEXT PRIMARY KEY,owner TEXT,key TEXT,digest TEXT,payload TEXT,thread TEXT,deadline REAL,epoch INTEGER,cancel INTEGER DEFAULT 0,UNIQUE(owner,key))"
        )
        db.execute("INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?)", original)
    return original


def test_unknown_business_version_does_not_install_tables(tmp_path):
    path = tmp_path / "business.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")
        db.execute("INSERT INTO meta VALUES('schema','999')")
    with pytest.raises(LabError, match="unsupported"):
        Store(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("meta",)
        ]
        assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == "999"


def test_failed_migration_rolls_back_ddl_and_original_records(tmp_path, monkeypatch):
    path = tmp_path / "business.sqlite"
    original = old_database(path)
    install = SqliteRunStore.install

    def fail(db, **kwargs):
        install(db, **kwargs)
        raise AgentError("injected", "migration failure")

    monkeypatch.setattr(SqliteRunStore, "install", staticmethod(fail))
    with pytest.raises(AgentError, match="migration failure"):
        Store(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM requests").fetchone() == original
        assert len(db.execute("PRAGMA table_info(requests)").fetchall()) == 9
        assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == "1"
        assert {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        } == {"meta", "requests", "conversations"}


async def test_schema_one_records_are_preserved_and_not_automatically_executed(tmp_path):
    original = old_database(tmp_path / "business.sqlite")
    lab = await LabService(
        Settings(data_dir=tmp_path, knowledge_root=ROOT / "data/knowledge")
    ).start()
    try:
        with lab.store.connection() as db:
            row = tuple(db.execute("SELECT * FROM requests").fetchone())
            assert row[:9] == original
            assert row[9] == "rag" and row[10]
            assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == "4"
        assert lab.store.epoch("local", "default") == 3
        view = await lab.view("local", "legacy-run")
        assert not view["recoverable"]
        assert view["resume_reason"] == "legacy_checkpoint_requires_matching_release"
        assert view["deadline"] == original[6]
        with pytest.raises(LabError, match="matching_release"):
            await lab.resume(LOCAL, "legacy-run", approve=True)
        assert lab.queue.empty()
    finally:
        await lab.close()


async def test_removed_old_principal_does_not_block_startup(tmp_path):
    original = old_database(tmp_path / "business.sqlite", "removed")
    lab = await LabService(
        Settings(data_dir=tmp_path, auth_mode="bearer", bearer_principals={"b" * 24: "bob"})
    ).start()
    try:
        with lab.store.connection() as db:
            assert tuple(db.execute("SELECT * FROM requests").fetchone())[:9] == original
            assert not db.execute(
                "SELECT 1 FROM agent_run_controls WHERE run_id='legacy-run'"
            ).fetchone()
        assert lab.available
    finally:
        await lab.close()


async def test_new_agent_and_rag_share_runtime_and_legacy_namespace_is_filtered(service):
    runtime = service.runtime
    for name in ("queue", "workers", "active", "pending", "admission_lock"):
        assert getattr(service, name) is getattr(runtime, name)
    register_workflows(runtime.registry)

    async def constant(arguments, context):
        return {"value": 7}

    runtime.registry.register_handler("test.constant", constant)
    runtime.registry.bind(
        AgentPackage.model_validate(
            {
                "tools": [
                    {
                        "name": "constant",
                        "description": "A separate business",
                        "handler": "test.constant",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    }
                ],
                "agents": [
                    {
                        "id": "separate",
                        "description": "Package outside RAG",
                        "workflow": "fixed",
                        "prompt": {"system_template": "Configured business."},
                        "input_schema": {"type": "object"},
                        "tools": ["constant"],
                        "tasks": [{"id": "one", "tool": "constant"}],
                    }
                ],
            }
        )
    )
    other = await runtime.submit(await runtime.identity("local"), "separate", "same-key", {})
    rag = await service.submit(LOCAL, "same-key", RunInput(message="RAG 分块"))
    assert other["run_id"] != rag["run_id"]
    await wait_run(service, rag["run_id"])
    for _ in range(100):
        view = await runtime.view("local", other["run_id"])
        if view["status"] == "succeeded":
            break
        await asyncio.sleep(0.01)
    assert view["result"]["one"]["value"] == {"value": 7}
    for method in (service.view, service.cancel):
        with pytest.raises(LabError) as error:
            await method("local", other["run_id"])
        assert error.value.status == 404
    with pytest.raises(LabError):
        await service.resume(LOCAL, other["run_id"])
    assert runtime.store.request("local", other["run_id"])["cancel"] == 0
    assert all(row["agent_id"] == "rag" for row in service.store.requests("local"))
    for row in service.store.requests("local"):
        await service.view("local", row["run_id"])


async def paused_write(service, key="review"):
    run = await service.submit(
        LOCAL,
        key,
        RunInput(tasks=[TaskSpec(id="write", tool="demo.write", arguments={"value": "approved"})]),
    )
    return await wait_run(service, run["run_id"])


async def test_forged_checkpoint_approved_does_not_authorize_effect(service):
    run = await paused_write(service)
    row = service.store.request("local", run["run_id"])
    state = dict((await service.checkpoint_state(row)).values)
    state["approved"] = True
    with pytest.raises(LabError, match="durable"):
        await service.agent.approval(state)
    with service.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM simulation").fetchone()[0] == 0


async def test_pending_task_cannot_change_approved_arguments(service):
    run = await paused_write(service)
    row = service.store.request("local", run["run_id"])
    review = run["interrupts"][0]
    service.runtime.store.accept_review(
        "local", run["run_id"], review["review_id"], review["digest"]
    )
    state = copy.deepcopy((await service.checkpoint_state(row)).values)
    state.update(
        task={**state["tasks"][0], "arguments": {"value": "changed"}}, dependency_results={}
    )
    with pytest.raises(AgentError, match="approved plan"):
        await service.agent.execute_task(state, service.runtime.config(row))
    with service.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM simulation").fetchone()[0] == 0


async def test_review_anchor_changes_digest_even_with_identical_plan(service):
    run = await paused_write(service)
    row = service.store.request("local", run["run_id"])
    snapshot = await service.checkpoint_state(row)
    descriptor = service.runtime_adapter.review_descriptor(row, snapshot)[0]
    alternate = SimpleNamespace(
        tasks=snapshot.tasks, config={"configurable": {"checkpoint_id": "another-checkpoint"}}
    )
    changed = service.runtime_adapter.review_descriptor(row, alternate)[0]
    assert descriptor["digest"] != changed["digest"]


async def test_approval_recorded_before_command_survives_restart_without_new_deadline(service):
    run = await paused_write(service)
    row = service.store.request("local", run["run_id"])
    review = run["interrupts"][0]
    service.runtime.store.accept_review(
        "local", run["run_id"], review["review_id"], review["digest"]
    )
    await service.close()
    reopened = await LabService(service.settings).start()
    try:
        await reopened.resume(LOCAL, run["run_id"], approve=True)
        final = await wait_run(reopened, run["run_id"])
        assert final["status"] == "succeeded"
        assert final["deadline"] == row["deadline"]
        with reopened.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM simulation").fetchone()[0] == 1
    finally:
        await reopened.close()


async def test_scope_revocation_while_waiting_tool_slot_prevents_write(service):
    for _ in range(service.settings.tools_concurrency):
        await service.registry.slots.acquire()
    run = await paused_write(service)
    await service.resume(LOCAL, run["run_id"], approve=True)
    await asyncio.sleep(0.1)
    service.runtime_adapter.identities["local"] = LOCAL.model_copy(update={"scopes": set()})
    for _ in range(service.settings.tools_concurrency):
        service.registry.slots.release()
    for _ in range(100):
        if not service.active and not service.pending:
            break
        await asyncio.sleep(0.01)
    with service.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM simulation").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0


async def test_document_revocation_during_async_projection_blocks_publication(service, monkeypatch):
    run = await service.submit(LOCAL, "projection", RunInput(message="RAG 为什么分块？"))
    final = await wait_run(service, run["run_id"])
    document = final["results"]["q1"]["value"]["citations"][0]["document_id"]
    row = service.store.request("local", run["run_id"])
    binding = service.runtime.binding(row)
    project = binding.project

    async def revoke_after_project(*args):
        result = await project(*args)
        await asyncio.sleep(0)
        service.store.revoke("local", document)
        return result

    monkeypatch.setattr(binding, "project", revoke_after_project)
    public = await service.runtime.view("local", run["run_id"])
    assert public["result"] is None
    assert public["result_visibility"] == "revoked"
    assert "results" not in public
    receipt = service.runtime.store.final_receipt("local", run["run_id"])
    assert {source["kind"] for source in receipt["sources"]} == {"rag.document"}


async def test_identity_revocation_during_last_source_check_blocks_publication(
    service, monkeypatch
):
    run = await service.submit(LOCAL, "last-source", RunInput(message="RAG 为什么分块？"))
    await wait_run(service, run["run_id"])
    checker = service.runtime.authorized_sources
    calls = 0

    async def revoke_identity(*args):
        nonlocal calls
        allowed = await checker(*args)
        calls += 1
        if calls == 2:
            service.runtime_adapter.identities["local"] = LOCAL.model_copy(update={"scopes": set()})
            await asyncio.sleep(0)
        return allowed

    monkeypatch.setattr(service.runtime, "authorized_sources", revoke_identity)
    public = await service.runtime.view("local", run["run_id"])
    assert calls == 2
    assert public["result"] is None and public["result_visibility"] == "revoked"
    assert "results" not in public


async def test_unsealed_expired_legacy_review_cannot_refresh_clock(service, monkeypatch):
    run = await paused_write(service)
    row = service.store.request("local", run["run_id"])
    review = run["interrupts"][0]
    recorded = service.runtime.store.review("local", run["run_id"], review["review_id"])
    with service.store.connection() as db:
        db.execute("DELETE FROM agent_review_visibility WHERE review_id=?", (review["review_id"],))
    with monkeypatch.context() as clocks:
        clocks.setattr("time.time", lambda: row["deadline"] + 0.01)
        view = await service.view("local", run["run_id"])
        assert view["resume_reason"] == "original_deadline_expired" and not view["recoverable"]
        with pytest.raises(LabError):
            await service.resume(LOCAL, run["run_id"], approve=True)
    assert service.runtime.store.visibility(review["review_id"]) is None
    assert (
        service.runtime.store.review("local", run["run_id"], review["review_id"])["expires_at"]
        == recorded["expires_at"]
    )
    assert service.store.request("local", run["run_id"])["deadline"] == row["deadline"]


async def test_domain_history_and_final_receipt_roll_back_together(service, monkeypatch):
    binding = service.runtime.binding(
        {"agent_id": "rag", "binding": service.runtime.registry.agent("rag").digest}
    )
    complete = binding.on_complete

    async def fail_after_history(*args):
        projection = await complete(*args)
        commit = projection.extra_commit

        def fail(db):
            commit(db)
            raise AgentError("injected", "final transaction interrupted")

        projection.extra_commit = fail
        return projection

    monkeypatch.setattr(binding, "on_complete", fail_after_history)
    run = await service.submit(LOCAL, "atomic-final", RunInput(message="RAG 为什么分块？"))
    final = await wait_run(service, run["run_id"], {"failed"})
    assert final["status"] == "failed"
    with service.store.connection() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM history WHERE run_id LIKE ?", (run["run_id"] + ":%",)
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT COUNT(*) FROM agent_final_receipts WHERE run_id=?", (run["run_id"],)
            ).fetchone()[0]
            == 0
        )
    assert not any(
        event.kind == "run.final" for event in await service.runtime.events("local", run["run_id"])
    )


async def test_runtime_metrics_count_admission_once_and_terminal_outcomes(service):
    request = RunInput(message="RAG 为什么分块？")
    run = await service.submit(LOCAL, "metrics-success", request)
    await wait_run(service, run["run_id"])
    await service.submit(LOCAL, "metrics-success", request)
    assert service.telemetry.counts["runs_submitted"] == 1
    assert service.telemetry.counts["runs_completed"] == 1
    paused = await service.submit(
        LOCAL, "metrics-cancel", RunInput(message="RAG", pause_for_review=True)
    )
    await wait_run(service, paused["run_id"])
    await service.cancel("local", paused["run_id"])
    assert service.telemetry.counts["runs_cancelled"] == 1

    async def slow(question):
        await asyncio.sleep(2)
        return {"value": 1}

    register(service, "test.metrics", slow)
    timed = await service.submit(
        LOCAL,
        "metrics-timeout",
        RunInput(
            tasks=[TaskSpec(id="slow", tool="test.metrics", arguments={"question": "test"})],
            timeout_seconds=0.5,
        ),
    )
    await wait_run(service, timed["run_id"], {"timed_out"})
    assert service.telemetry.counts["timeouts"] == 1
    assert service.telemetry.counts["runs_submitted"] == 3


async def test_shutdown_cancels_execution_once_and_waits_for_cleanup(service, monkeypatch):
    entered, cleaned = asyncio.Event(), asyncio.Event()
    cancellations = 0

    async def execution(row, command=None):
        nonlocal cancellations
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            await asyncio.sleep(0.03)
            cleaned.set()
            raise

    monkeypatch.setattr(service.runtime, "execute", execution)
    await service.submit(LOCAL, "shutdown", RunInput(message="RAG"))
    await entered.wait()
    await service.runtime.close()
    assert cleaned.is_set() and cancellations == 1
    assert not service.active and not service.workers and not service.pending


async def test_shutdown_waits_for_cleanup_of_previously_cancelled_execution(service, monkeypatch):
    entered, cleanup_started, allow_cleanup, cleaned = [asyncio.Event() for _ in range(4)]

    async def execution(row, command=None):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleaned.set()

    monkeypatch.setattr(service.runtime, "execute", execution)
    run = await service.submit(LOCAL, "pre-cancelled-close", RunInput(message="RAG"))
    await entered.wait()
    child = service.active[run["run_id"]]
    service.runtime.cancel_once(child)
    await cleanup_started.wait()
    close = asyncio.create_task(service.runtime.close())
    await asyncio.sleep(0.03)
    assert not close.done() and child.cancelling() == 1
    allow_cleanup.set()
    await close
    assert cleaned.is_set() and child.cancelling() == 1
    assert not service.active and not service.workers


async def test_observer_mutation_and_failure_do_not_change_authority(service, monkeypatch):
    def mutate_and_fail(kind, row, data):
        row["binding"], row["owner"] = "changed", "intruder"
        data["status"] = "failed"
        raise RuntimeError("broken observer")

    monkeypatch.setattr(service.runtime, "observer", mutate_and_fail)
    run = await service.submit(LOCAL, "broken-observer", RunInput(message="RAG 为什么分块？"))
    final = await wait_run(service, run["run_id"])
    assert final["status"] == "succeeded"
    row = service.store.request("local", run["run_id"])
    assert row["binding"] == service.runtime.registry.agent("rag").digest
    assert service.runtime.observer_errors == 2
