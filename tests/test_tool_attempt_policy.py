"""Atomic sidecars and real handler executions under durable bounded policies."""

import asyncio
import copy
import sqlite3
import threading
import time

import pytest
from pydantic import ValidationError

from agentbase import AttemptCommit, ToolExecutionPolicy
from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Identity, RunContext, ToolCall
from agentbase.effects import AttemptEvidence, NeedsReconciliation, SqliteActionLedger
from agentbase.registry import ComponentRegistry
from agentbase.run_store import SqliteRunStore
from agentbase.storage import SqliteDatabase
from agentbase.tool_control import ToolCircuit
from agentbase.tools import ResolvedTool, ToolExecutor, TransientToolError


@pytest.fixture
def ledger(tmp_path):
    database = SqliteDatabase(tmp_path / "actions.sqlite")
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteActionLedger.install(db)
        db.execute("CREATE TABLE receipt_links(key TEXT,token TEXT,attempt INTEGER,payload TEXT)")
    return SqliteActionLedger(database.connection)


def configured(ledger, handler, *, hook=None, execution=None, timeout=1.0, write=False, **kwargs):
    tool = {
        "name": "custom",
        "handler": "custom.handler",
        "description": "Synthetic tool",
        "scope": "write" if write else "read",
        "read_only": not write,
        "idempotency": "business-key" if write else "read-only",
        "timeout_seconds": timeout,
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        "output_schema": {"type": "integer"},
    }
    if execution is not None:
        tool["execution"] = execution
    package = AgentPackage.model_validate(
        {
            "tools": [tool],
            "agents": [
                {
                    "id": "alpha",
                    "description": "Synthetic Agent",
                    "workflow": "fixed",
                    "prompt": {"system_template": "Policy"},
                    "input_schema": {"type": "string"},
                    "tools": ["custom"],
                }
            ],
        }
    )
    registry = ComponentRegistry()
    registry.register_workflow("fixed", lambda: None)
    registry.register_handler("custom.handler", handler, attempt_commit=hook)
    registry.bind(package)
    context = RunContext(
        identity=Identity(owner="user", scopes=frozenset({"read", "write"})),
        agent_id="alpha",
        run_id="run",
        conversation_id="topic",
        deadline=time.time() + 3,
        config_digest=registry.agent("alpha").digest,
        dependency_results={"prior": {"value": 9}},
    )
    if write:

        async def approve(*args):
            return True

        kwargs.setdefault("approval_checker", approve)
    return ToolExecutor(registry, ledger, **kwargs), context, package


def call(name="one"):
    return ToolCall(call_id=name, name="custom", arguments={"x": 1})


def operation(ledger, name="one"):
    return next(row for row in ledger.operations("user", "run") if row["call_id"] == name)


async def test_same_transaction_sidecar_uses_claim_authority_and_private_snapshots(ledger):
    observations = []

    def hook(db, row, evidence):
        assert (
            dict(
                db.execute(
                    "SELECT * FROM agent_tool_operations WHERE key=?", (row["key"],)
                ).fetchone()
            )
            == row
        )
        assert row["attempts"] == 1 and row["phase"] == "executing"
        assert evidence.context.operation_key == row["key"]
        assert evidence.context.attempt_token == row["attempt_token"]
        assert evidence.context.dependency_results == {"prior": {"value": 9}}
        db.execute(
            "INSERT INTO receipt_links VALUES(?,?,?,?)",
            (row["key"], row["attempt_token"], row["attempts"], "original"),
        )
        observations.append(copy.deepcopy(row))
        row["attempt_token"] = "changed"
        evidence.call.arguments["x"] = 99
        evidence.tool.input_schema["type"] = "string"
        evidence.context.dependency_results["prior"]["value"] = 100

    async def handler(args, context):
        assert args == {"x": 1} and context.dependency_results == {"prior": {"value": 9}}
        with ledger.connection() as db:
            link = db.execute("SELECT * FROM receipt_links").fetchone()
            assert tuple(link)[:3] == (context.operation_key, context.attempt_token, 1)
        return 1

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    result = await executor.execute(context, call())
    assert result.succeeded and result.value == 1
    assert operation(ledger)["attempt_token"] == observations[0]["attempt_token"]
    assert context.dependency_results["prior"]["value"] == 9


@pytest.mark.parametrize(
    "statement",
    [
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT forbidden",
        "PRAGMA user_version=9",
        "CREATE TABLE forbidden(x)",
        "DROP TABLE receipt_links",
        "ALTER TABLE receipt_links ADD COLUMN forbidden TEXT",
        "ATTACH ':memory:' AS other",
        "UPDATE agent_tool_operations SET attempts=8",
        "DELETE FROM agent_tool_operations",
        "UPDATE agent_runtime_meta SET value='9'",
        "script",
    ],
    ids=[
        "commit",
        "rollback",
        "savepoint",
        "pragma",
        "create",
        "drop",
        "alter",
        "attach",
        "core_update",
        "core_delete",
        "controls",
        "script",
    ],
)
async def test_hook_forbidden_sql_rolls_back_claim_and_sidecar(ledger, statement):
    with ledger.connection() as db:
        db.execute("CREATE TABLE agent_runtime_meta(key TEXT,value TEXT)")
        db.execute("INSERT INTO agent_runtime_meta VALUES('schema','4')")

    def hook(db, row, evidence):
        db.execute(
            "INSERT INTO receipt_links VALUES(?,?,?,?)",
            (row["key"], row["attempt_token"], 1, "rollback"),
        )
        if statement == "script":
            db.executescript("INSERT INTO receipt_links VALUES('bad','bad',99,'bad');")
        else:
            db.execute(statement)

    invoked = []

    async def handler(*args):
        invoked.append(1)
        return 1

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert not (await executor.execute(context, call())).succeeded
    row = operation(ledger)
    assert (
        row["phase"] == "pending"
        and row["attempts"] == 0
        and row["attempt_token"] is None
        and row["call_deadline"] is None
    )
    assert invoked == []
    with ledger.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM receipt_links").fetchone()[0] == 0
        assert db.execute("SELECT value FROM agent_runtime_meta").fetchone()[0] == "4"


@pytest.mark.parametrize("mode", ["error", "awaitable", "value"])
async def test_hook_exception_or_invalid_return_rolls_back_without_coroutine_warning(ledger, mode):
    async def asynchronous():
        return None

    def hook(db, row, evidence):
        db.execute(
            "INSERT INTO receipt_links VALUES(?,?,1,'rollback')", (row["key"], row["attempt_token"])
        )
        if mode == "error":
            raise TransientToolError()
        return asynchronous() if mode == "awaitable" else True

    async def handler(*args):
        raise AssertionError("failed claim cannot run handler")

    executor, context, _ = configured(
        ledger,
        handler,
        hook=AttemptCommit(frozenset({"receipt_links"}), hook),
        execution={"circuit_failures": 1},
    )
    assert not (await executor.execute(context, call())).succeeded
    assert operation(ledger)["attempts"] == 0
    with ledger.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM receipt_links").fetchone()[0] == 0
    assert next(iter(executor._circuits.values())).failures == 0


async def test_second_hook_failure_preserves_first_failed_attempt_exactly(ledger):
    first = []

    def hook(db, row, evidence):
        db.execute(
            "INSERT INTO receipt_links VALUES(?,?,?,?)",
            (row["key"], row["attempt_token"], row["attempts"], "receipt"),
        )
        if row["attempts"] == 2:
            first.append(
                copy.deepcopy(dict(db.execute("SELECT * FROM agent_tool_operations").fetchone()))
            )
            raise AgentError("sidecar_unavailable", "synthetic", 503)

    invoked = []

    async def handler(*args):
        invoked.append(1)
        raise TransientToolError()

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    expected = []
    failed = ledger.failed

    def record(*args, **kwargs):
        failed(*args, **kwargs)
        expected.append(operation(ledger))

    ledger.failed = record
    result = await executor.execute(context, call())
    assert result.error == "sidecar_unavailable" and invoked == [1]
    assert operation(ledger) == expected[0] and len(expected) == 1
    assert first[0]["attempts"] == 2
    with ledger.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM receipt_links").fetchone()[0] == 1


@pytest.mark.parametrize(
    "tables",
    [
        frozenset(),
        frozenset({"requests"}),
        frozenset({"AGENT_run_controls"}),
        frozenset({"sqlite_master"}),
        frozenset({"bad;sql"}),
        {"receipt_links"},
    ],
)
def test_invalid_sidecar_registration_rejected(tables):
    with pytest.raises(AgentError):
        AttemptCommit(tables, lambda *args: None)


async def test_attempt_evidence_must_match_original_fingerprint(ledger):
    async def handler(*args):
        return 1

    executor, context, _ = configured(ledger, handler)
    selected = executor.resolve(context, call())
    row = ledger.begin(context, selected.declaration, call(), 16)
    hook = AttemptCommit(frozenset({"receipt_links"}), lambda *args: None)
    bad = AttemptEvidence(
        context=context,
        tool=selected.declaration,
        call=call().model_copy(update={"arguments": {"x": 2}}),
    )
    with pytest.raises(AgentError, match="evidence"):
        ledger.claim_attempt(
            row["key"], 2, call_deadline=time.time() + 1, evidence=bad, commit=hook
        )
    assert operation(ledger) == row


async def test_original_registered_hook_survives_new_active_release(ledger):
    seen = []

    def first(*args):
        seen.append("v1")

    def second(*args):
        seen.append("v2")

    async def handler(*args):
        return 1

    executor, context, package = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), first)
    )
    registry = executor.registry
    registry.register_handler(
        "custom.handler",
        handler,
        version="2",
        attempt_commit=AttemptCommit(frozenset({"receipt_links"}), second),
    )
    data = package.model_dump(mode="json")
    data["agents"][0]["version"] = "2"
    data["tools"][0]["handler_version"] = "2"
    registry.activate(AgentPackage.model_validate(data))
    assert (await executor.execute(context, call())).succeeded
    assert seen == ["v1"]
    newer = context.model_copy(
        update={"run_id": "newrun", "config_digest": registry.agent("alpha").digest}
    )
    assert (await executor.execute(newer, call())).succeeded and seen == ["v1", "v2"]


async def test_resolver_explicit_none_delegates_but_error_and_hook_change_never_do(ledger):
    invoked = []

    async def handler(*args):
        invoked.append(1)
        return 1

    executor, context, _ = configured(ledger, handler, resolver=lambda *args: None)
    assert (await executor.execute(context, call())).succeeded

    def error(*args):
        raise AgentError("history_missing", "synthetic", 409)

    executor.resolver = error
    with pytest.raises(AgentError, match="synthetic"):
        await executor.execute(context, call("other"))
    selected = executor.registry.agent("alpha").tools[0]
    manifest = "sha256:" + "a" * 64
    first = AttemptCommit(frozenset({"receipt_links"}), lambda *args: None)
    executor.resolver = lambda *args: ResolvedTool(selected, handler, manifest, first)
    executor.resolve(context, call())
    executor.resolver = lambda *args: ResolvedTool(
        selected, handler, manifest, AttemptCommit(frozenset({"receipt_links"}), lambda *args: None)
    )
    with pytest.raises(AgentError, match="implementation changed"):
        executor.resolve(context, call())
    assert invoked == [1]


async def test_all_attempts_backoff_and_projection_share_one_persisted_deadline(ledger):
    invoked = []

    async def handler(*args):
        invoked.append(1)
        await asyncio.sleep(0.15 if len(invoked) == 1 else 0.4)
        if len(invoked) == 1:
            raise TransientToolError()
        return 1

    executor, context, _ = configured(ledger, handler, timeout=0.5)
    result = await executor.execute(context, call())
    row = operation(ledger)
    assert result.error == "tool_timeout" and invoked == [1, 1] and row["attempts"] == 2
    deadline = row["call_deadline"]
    reopened = ToolExecutor(executor.registry, SqliteActionLedger(ledger.connection))
    assert (await reopened.execute(context, call())).error == "tool_attempt_budget"
    assert operation(ledger)["call_deadline"] == deadline and len(invoked) == 2


async def test_backoff_expiry_prevents_next_claim_and_reopen_cannot_refresh_window(ledger):
    invoked = []

    async def handler(*args):
        invoked.append(1)
        raise TransientToolError()

    executor, context, _ = configured(
        ledger, handler, timeout=0.15, execution={"backoff_seconds": 0.5}
    )
    assert (await executor.execute(context, call())).error == "tool_timeout"
    before = operation(ledger)
    assert before["attempts"] == 1
    reopened = ToolExecutor(executor.registry, SqliteActionLedger(ledger.connection))
    assert (await reopened.execute(context, call())).error == "tool_timeout"
    assert operation(ledger) == before and invoked == [1]


async def test_slot_wait_run_expiry_never_claims_or_invokes_hook(ledger):
    seen = []

    async def handler(*args):
        seen.append("handler")
        return 1

    executor, context, _ = configured(
        ledger,
        handler,
        hook=AttemptCommit(frozenset({"receipt_links"}), lambda *args: seen.append("hook")),
        concurrency=1,
    )
    await executor.slots.acquire()
    try:
        result = await executor.execute(
            context.model_copy(update={"deadline": time.time() + 0.25}), call()
        )
    finally:
        executor.slots.release()
    assert result.error == "tool_timeout" and seen == []
    assert operation(ledger)["attempts"] == 0 and operation(ledger)["call_deadline"] is None


@pytest.mark.parametrize("mode", ["cancel", "revoke"])
async def test_cancel_or_revoke_during_backoff_prevents_next_attempt(ledger, mode):
    ready = asyncio.Event()
    permission = True
    invoked = []

    async def authorize(*args):
        return permission

    async def handler(*args):
        invoked.append(1)
        raise TransientToolError()

    executor, context, _ = configured(
        ledger,
        handler,
        execution={"backoff_seconds": 0.2, "circuit_failures": 1},
        authorizer=authorize,
    )
    failed = ledger.failed

    def mark(*args, **kwargs):
        failed(*args, **kwargs)
        ready.set()

    ledger.failed = mark
    task = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(ready.wait(), 1)
    if mode == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        permission = False
        assert (await task).error == "forbidden"
    assert operation(ledger)["attempts"] == 1 and invoked == [1]
    assert next(iter(executor._circuits.values())).failures == 0


@pytest.mark.parametrize("where", ["authorization", "policy", "schema", "permanent"])
async def test_non_handler_transient_and_permanent_errors_never_retry_or_trip(ledger, where):
    invoked = []
    checks = []

    async def authorize(*args):
        checks.append(1)
        if where == "authorization" and len(checks) == 3:
            raise TransientToolError()
        return True

    async def policy(context, tool, value):
        if where == "policy":
            raise TransientToolError()
        return value

    async def handler(*args):
        invoked.append(1)
        if where == "permanent":
            raise AgentError("permanent", "synthetic")
        return "invalid" if where == "schema" else 1

    executor, context, _ = configured(
        ledger,
        handler,
        execution={"circuit_failures": 1},
        authorizer=authorize,
        result_policy=policy,
    )
    assert not (await executor.execute(context, call())).succeeded
    assert operation(ledger)["attempts"] == 1 and len(invoked) <= 1
    circuit = next(iter(executor._circuits.values()))
    assert circuit.failures == 0 and circuit.open_until is None


async def test_write_even_retry_safe_has_one_attempt_and_unknown_blocks_replay(ledger):
    invoked = []

    async def handler(*args):
        invoked.append(1)
        raise TransientToolError()

    executor, context, _ = configured(ledger, handler, write=True, execution={"max_attempts": 8})
    for _ in range(2):
        with pytest.raises(NeedsReconciliation):
            await executor.execute(context, call())
    assert invoked == [1] and operation(ledger)["attempts"] == 1


async def test_circuit_counts_one_logical_failure_and_open_skips_hook_and_attempt(ledger):
    invoked = []
    hooks = []

    async def handler(*args):
        invoked.append(1)
        raise TransientToolError()

    executor, context, _ = configured(
        ledger,
        handler,
        execution={"circuit_failures": 2, "backoff_seconds": 0.0},
        hook=AttemptCommit(frozenset({"receipt_links"}), lambda *args: hooks.append(1)),
    )
    assert (await executor.execute(context, call("first"))).error == "tool_transient"
    circuit = next(iter(executor._circuits.values()))
    assert circuit.failures == 1 and len(invoked) == 2
    assert (await executor.execute(context, call("second"))).error == "tool_transient"
    assert circuit.failures == 2 and len(invoked) == 4
    assert (await executor.execute(context, call("third"))).error == "tool_circuit_open"
    assert operation(ledger, "third")["attempts"] == 0 and len(invoked) == 4 and len(hooks) == 4


async def test_real_half_open_probe_is_exclusive_and_cancel_releases_it(ledger):
    entered = asyncio.Event()
    release = asyncio.Event()
    mode = "fail"
    invoked = []

    async def handler(*args):
        invoked.append(1)
        if mode == "fail":
            raise TransientToolError()
        entered.set()
        await release.wait()
        return 1

    executor, context, _ = configured(
        ledger, handler, execution={"max_attempts": 1, "circuit_failures": 1}
    )
    assert (await executor.execute(context, call("fail"))).error == "tool_transient"
    circuit = next(iter(executor._circuits.values()))
    circuit.open_until = 0.0
    mode = "probe"
    probe = asyncio.create_task(executor.execute(context, call("probe")))
    await asyncio.wait_for(entered.wait(), 1)
    assert (await executor.execute(context, call("blocked"))).error == "tool_circuit_open"
    assert operation(ledger, "blocked")["attempts"] == 0
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    assert circuit.probe is None and circuit.failures == 1
    release.set()
    assert (await executor.execute(context, call("reprobe"))).succeeded
    assert circuit.open_until is None and circuit.failures == 0 and len(invoked) == 3


@pytest.mark.parametrize("stale_outcome", ["success", "failure", "neutral"])
def test_stale_generation_cannot_release_or_close_new_probe(stale_outcome):
    circuit = ToolCircuit(1, 1.0)
    stale = circuit.acquire()
    circuit.finish(circuit.acquire(), "failure")
    circuit.open_until = 0.0
    first = circuit.acquire()
    circuit.finish(first, "neutral")
    second = circuit.acquire()
    expected = (circuit.generation, circuit.probe, circuit.failures, circuit.open_until)
    circuit.finish(stale, stale_outcome)
    circuit.finish(first, stale_outcome)
    assert (circuit.generation, circuit.probe, circuit.failures, circuit.open_until) == expected
    circuit.finish(second, "failure")
    assert circuit.probe is None and circuit.open_until > time.monotonic()


async def test_open_circuit_cached_done_still_checks_current_permission(ledger):
    allowed = True
    failed = False
    invoked = []

    async def authorize(*args):
        return allowed

    async def handler(*args):
        invoked.append(1)
        if failed:
            raise TransientToolError()
        return 1

    executor, context, _ = configured(
        ledger, handler, execution={"max_attempts": 1, "circuit_failures": 1}, authorizer=authorize
    )
    assert (await executor.execute(context, call("done"))).succeeded
    failed = True
    assert (await executor.execute(context, call("fail"))).error == "tool_transient"
    assert (await executor.execute(context, call("done"))).succeeded and len(invoked) == 2
    allowed = False
    with pytest.raises(AgentError, match="authorized"):
        await executor.execute(context, call("done"))
    assert len(invoked) == 2


async def test_runtime_three_migration_preserves_nonempty_rows_and_no_old_window(ledger):
    async def handler(*args):
        return 1

    executor, context, _ = configured(ledger, handler)
    assert (await executor.execute(context, call("done"))).succeeded
    tool = executor.resolve(context, call("old")).declaration
    row = ledger.begin(context, tool, call("old"), 16)
    token = ledger.claim_attempt(row["key"], 2)
    ledger.failed(row["key"], "tool_transient", handler_started=True, attempt_token=token)
    with ledger.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN call_deadline")
        db.execute("UPDATE agent_runtime_meta SET value='3' WHERE key='schema'")
        before = [tuple(r) for r in db.execute("SELECT * FROM agent_tool_operations ORDER BY key")]
    with ledger.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
        after = [tuple(r) for r in db.execute("SELECT * FROM agent_tool_operations ORDER BY key")]
        assert [r[:-1] for r in after] == before and all(r[-1] is None for r in after)
        assert (
            db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
            == "8"
        )
        SqliteRunStore.install(db)
    assert (await executor.execute(context, call("done"))).succeeded
    old = operation(ledger, "old")
    assert (await executor.execute(context, call("old"))).error == "tool_deadline_missing"
    assert operation(ledger, "old") == old and old["attempt_token"] == token


@pytest.mark.parametrize(
    "invalid",
    [
        {"max_attempts": 0},
        {"max_attempts": 9},
        {"backoff_seconds": 1.1},
        {"circuit_failures": 0},
        {"circuit_cooldown_seconds": float("inf")},
    ],
)
def test_execution_policy_is_bounded(invalid):
    with pytest.raises(ValidationError):
        ToolExecutionPolicy.model_validate(invalid)


def test_absent_execution_preserves_old_descriptor_and_explicit_policy_changes_binding(ledger):
    async def handler(*args):
        return 1

    executor, _, package = configured(ledger, handler)
    assert "execution" not in package.model_dump(mode="json")["tools"][0]
    data = package.model_dump(mode="json")
    data["tools"][0]["execution"] = {"max_attempts": 1}
    with pytest.raises(AgentError, match="new Agent version"):
        executor.registry.activate(AgentPackage.model_validate(data))


def blocking_extension():
    # A synchronous host callback occupies the loop; the deadline callback
    # cannot run until the extension returns to an actual asynchronous yield.
    threading.Event().wait(0.15)


async def test_synchronous_slow_hook_rolls_back_before_handler(ledger):
    invoked = []

    def hook(db, row, evidence):
        db.execute(
            "INSERT INTO receipt_links VALUES(?,?,1,'late')", (row["key"], row["attempt_token"])
        )
        blocking_extension()

    async def handler(*args):
        invoked.append(1)
        return 1

    executor, context, _ = configured(
        ledger, handler, timeout=0.1, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert (await executor.execute(context, call())).error == "tool_timeout"
    assert (
        invoked == []
        and operation(ledger)["attempts"] == 0
        and operation(ledger)["phase"] == "pending"
    )
    with ledger.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM receipt_links").fetchone()[0] == 0


@pytest.mark.parametrize("write", [False, True])
async def test_synchronous_slow_handler_cannot_publish_after_deadline(ledger, write):
    invoked = []

    async def handler(*args):
        invoked.append(1)
        blocking_extension()
        return 1

    executor, context, _ = configured(ledger, handler, timeout=0.1, write=write)
    if write:
        with pytest.raises(NeedsReconciliation):
            await executor.execute(context, call())
        assert operation(ledger)["phase"] == "unknown"
    else:
        assert (await executor.execute(context, call())).error == "tool_timeout"
        assert operation(ledger)["phase"] == "failed"
    assert invoked == [1] and operation(ledger)["attempts"] == 1


async def test_synchronous_slow_projection_keeps_applied_fact_without_publishing(ledger):
    from agentbase.effects import ActionReceipt, AppliedResultUnavailable

    async def handler(*args):
        return 1

    def receipt(db, row):
        return ActionReceipt(phase="applied", value=1) if row["attempts"] else None

    async def policy(*args):
        blocking_extension()
        return 1

    executor, context, _ = configured(
        ledger, handler, timeout=0.1, write=True, receipt_inspector=receipt, result_policy=policy
    )
    with pytest.raises(AppliedResultUnavailable):
        await executor.execute(context, call())
    assert operation(ledger)["phase"] == "applied" and operation(ledger)["attempts"] == 1


@pytest.mark.parametrize("route", ["connection", "cursor", "cached", "parameters"])
async def test_implicit_rollback_never_leaks_later_autocommit_sidecar(ledger, route):
    with ledger.connection() as db:
        db.execute("CREATE UNIQUE INDEX receipt_key ON receipt_links(key)")
        db.execute("INSERT INTO receipt_links VALUES('taken','old',1,'old')")
    sql = "WITH entry AS (SELECT 'leaked','partial',99,'partial') INSERT INTO receipt_links SELECT * FROM entry"
    retained = []

    def hook(db, row, evidence):
        retained.append(db)
        target = db.cursor() if route == "cursor" else db
        if route == "cached":
            target.execute(sql)

        def conflict():
            try:
                target.execute(
                    "INSERT OR ROLLBACK INTO receipt_links VALUES('taken','collision',99,'collision')"
                )
            except sqlite3.IntegrityError:
                pass

        if route == "parameters":

            def items():
                conflict()
                yield ("leaked", "partial", 99, "partial")

            target.executemany("INSERT INTO receipt_links VALUES(?,?,?,?)", items())
        else:
            conflict()
            target.execute(sql)

    async def handler(*args):
        raise AssertionError("implicit rollback cannot start handler")

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert (await executor.execute(context, call())).error == "attempt_commit"
    assert operation(ledger)["phase"] == "pending" and operation(ledger)["attempts"] == 0
    with ledger.connection() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM receipt_links")] == [
            ("taken", "old", 1, "old")
        ]
    with pytest.raises(AgentError):
        retained[0].execute(sql)


async def test_hook_facade_and_cursor_expire_after_success(ledger):
    retained = []

    def hook(db, row, evidence):
        cursor = db.cursor()
        assert cursor.connection is db
        retained.extend([db, cursor])
        cursor.executemany(
            "INSERT INTO receipt_links VALUES(?,?,?,?)",
            [(row["key"], row["attempt_token"], 1, "ok")],
        )
        assert cursor.rowcount == 1
        rows = db.execute("SELECT * FROM receipt_links")
        assert rows.description and len(list(rows)) == 1

    async def handler(*args):
        for target in retained:
            with pytest.raises(AgentError):
                target.execute("INSERT INTO receipt_links VALUES('late','late',2,'late')")
        return 1

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert (await executor.execute(context, call())).succeeded


@pytest.mark.parametrize("where", ["policy", "authorization"])
@pytest.mark.parametrize("latest", ["applied", "unknown", "compensated"])
async def test_ack_recovery_async_wait_uses_original_deadline_and_latest_fact(
    ledger, where, latest
):
    from agentbase.effects import ActionReceipt, AppliedResultUnavailable

    entered = asyncio.Event()
    never = asyncio.Event()
    fact = "applied"
    invoked = []
    checks = []

    def receipt(db, row):
        return (
            ActionReceipt(phase=fact, value=1 if fact == "applied" else None)
            if row["attempts"]
            else None
        )

    async def handler(*args):
        invoked.append(1)
        raise AgentError("ack_lost", "synthetic")

    async def policy(context, tool, value):
        if where == "policy":
            entered.set()
            await never.wait()
        return value

    async def authorize(*args):
        checks.append(1)
        if where == "authorization" and invoked:
            entered.set()
            await never.wait()
        return True

    executor, context, _ = configured(
        ledger,
        handler,
        timeout=0.25,
        write=True,
        receipt_inspector=receipt,
        result_policy=policy,
        authorizer=authorize,
    )
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        # The recovery's entry was applied; a later fact must supersede it.
        fact = latest
        error = (
            AppliedResultUnavailable
            if latest == "applied"
            else NeedsReconciliation
            if latest == "unknown"
            else AgentError
        )
        with pytest.raises(error) as caught:
            await asyncio.wait_for(asyncio.shield(task), 1)
        if latest == "compensated":
            assert caught.value.code == "effect_compensated"
        assert (
            invoked == [1]
            and operation(ledger)["attempts"] == 1
            and operation(ledger)["phase"] == latest
        )
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("many", [False, True])
async def test_sql_parameter_adapter_never_runs_inside_claim(many, ledger):
    called = []

    def hook(db, row, evidence):
        class Dangerous:
            def __conform__(self, protocol):
                called.append(1)
                db.execute(
                    "INSERT OR ROLLBACK INTO receipt_links VALUES('taken','collision',99,'collision')"
                )
                return "leaked"

        sql = "WITH entry AS (SELECT ?, 'partial', 99, 'partial') INSERT INTO receipt_links SELECT * FROM entry"
        if many:
            db.executemany(sql, [(Dangerous(),)])
        else:
            db.execute(sql, (Dangerous(),))

    async def handler(*args):
        raise AssertionError("invalid parameters cannot start handler")

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert (await executor.execute(context, call())).error == "attempt_commit"
    assert called == [] and operation(ledger)["attempts"] == 0
    with ledger.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM receipt_links").fetchone()[0] == 0


@pytest.mark.parametrize("many", [False, True])
async def test_custom_sql_parameter_sequence_cannot_run_python(many, ledger):
    called = []

    def hook(db, row, evidence):
        class Parameters:
            def __len__(self):
                called.append(1)
                return 1

            def __getitem__(self, index):
                called.append(1)
                return "leaked"

        if many:
            db.executemany("INSERT INTO receipt_links(key) VALUES(?)", [Parameters()])
        else:
            db.execute("INSERT INTO receipt_links(key) VALUES(?)", Parameters())

    async def handler(*args):
        raise AssertionError("invalid parameters cannot start handler")

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert (await executor.execute(context, call())).error == "attempt_commit"
    assert called == [] and operation(ledger)["attempts"] == 0


@pytest.mark.parametrize("kind", [str, int])
async def test_registered_builtin_scalar_adapter_is_rejected_before_driver(ledger, kind):
    called = []
    key = (kind, sqlite3.PrepareProtocol)
    prior = sqlite3.adapters.get(key)

    def adapter(value):
        called.append(1)
        return value

    def hook(db, row, evidence):
        sqlite3.register_adapter(kind, adapter)
        try:
            db.execute("INSERT INTO receipt_links(key) VALUES(?)", (kind(1),))
        finally:
            # Scope this driver-adapter probe to the hook, avoiding changes to
            # unrelated public-ledger SQL after the callback has returned.
            if prior is None:
                sqlite3.adapters.pop(key, None)
            else:
                sqlite3.adapters[key] = prior

    async def handler(*args):
        raise AssertionError("adapted parameters cannot start handler")

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    try:
        assert (await executor.execute(context, call())).error == "attempt_commit"
    finally:
        if prior is None:
            sqlite3.adapters.pop(key, None)
        else:
            sqlite3.adapters[key] = prior
    assert called == [] and operation(ledger)["attempts"] == 0


async def test_host_custom_sql_function_cannot_run_in_hook(ledger):
    from contextlib import contextmanager

    invoked = []
    base = ledger.connection

    @contextmanager
    def connection():
        with base() as db:
            db.create_function("host_extension", 0, lambda: invoked.append(1) or 1)
            yield db

    ledger.connection = connection

    def hook(db, row, evidence):
        db.execute("INSERT INTO receipt_links(key) VALUES(host_extension())")

    async def handler(*args):
        raise AssertionError("custom SQL function cannot start handler")

    executor, context, _ = configured(
        ledger, handler, hook=AttemptCommit(frozenset({"receipt_links"}), hook)
    )
    assert not (await executor.execute(context, call())).succeeded
    assert invoked == [] and operation(ledger)["attempts"] == 0
