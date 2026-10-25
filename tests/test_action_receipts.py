"""Host-verified SQL receipts classify facts without replaying business handlers."""

import asyncio
import copy
import json

import pytest

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Identity
from agentbase.effects import (
    ActionReceipt,
    AppliedResultUnavailable,
    NeedsReconciliation,
    SqliteActionLedger,
)
from agentbase.run_store import SqliteRunStore
from agentbase.schema import canonical_json
from agentbase.storage import SqliteDatabase
from tests.test_agentbase_runtime import declaration, package, system
from tests.test_agentbase_tools import call, setup


@pytest.fixture
def ledger(tmp_path):
    database = SqliteDatabase(tmp_path / "business.sqlite")
    result = SqliteActionLedger(database.connection)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        result.install(db)
        db.execute(
            "CREATE TABLE host_receipts(key TEXT PRIMARY KEY,owner TEXT,agent_id TEXT,run_id TEXT,call_id TEXT,binding TEXT,fingerprint TEXT,attempt_token TEXT,phase TEXT,value TEXT)"
        )
    return result


LINK = ("key", "owner", "agent_id", "run_id", "call_id", "binding", "fingerprint", "attempt_token")


def persist(ledger, key, phase, value=None, **overrides):
    operation = ledger.operation(key)
    operation.update(overrides)
    with ledger.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT OR REPLACE INTO host_receipts VALUES(?,?,?,?,?,?,?,?,?,?)",
            (*(operation[name] for name in LINK), phase, canonical_json(value).decode()),
        )


def inspect_receipt(db, operation):
    row = db.execute("SELECT * FROM host_receipts WHERE key=?", (operation["key"],)).fetchone()
    if row is None or row["phase"] == "intent":
        return None
    if any(row[name] != operation[name] for name in LINK):
        raise AgentError(
            "host_receipt_conflict", "receipt does not match the original attempt", 409
        )
    return ActionReceipt(
        phase=row["phase"],
        value=json.loads(row["value"]) if row["phase"] == "applied" else None,
    )


async def approved(*args):
    return True


def original(ledger, executor, context, action=None):
    action = action or call()
    tool = executor.declaration(context, action)
    row = ledger.begin(context, tool, action, 16)
    token = ledger.claim_attempt(row["key"], 1 if not tool.read_only else 2)
    return row["key"], token


def reconcile(ledger, context, inspector=inspect_receipt):
    ledger.reconcile(context.identity.owner, context.agent_id, context.run_id, inspector)


@pytest.mark.parametrize("phase", ["applied", "not_applied", "unknown", "compensated"])
async def test_trusted_receipt_adopted_without_another_handler_or_attempt(ledger, phase):
    calls = []

    async def handler(args, context):
        calls.append(args)
        return {"value": 999}

    executor, context = setup(ledger, handler, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    key, token = original(ledger, executor, context)
    persist(ledger, key, phase, {"value": 7} if phase == "applied" else None)
    if phase == "unknown":
        with pytest.raises(NeedsReconciliation):
            await executor.execute(context, call())
    else:
        result = await executor.execute(context, call())
        assert result.succeeded is (phase == "applied")
        assert result.value == ({"value": 7} if phase == "applied" else None)
    row = ledger.operation(key)
    assert row["phase"] == phase and row["attempts"] == 1 and row["attempt_token"] == token
    assert row["effect_applied"] == int(phase in {"applied", "compensated"})
    assert row["effect_compensated"] == int(phase == "compensated")
    assert calls == []


@pytest.mark.parametrize("phase", [None, "intent"])
async def test_absent_or_active_domain_intent_does_not_clear_or_reclassify_public_intent(
    ledger, phase
):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    key, _ = original(ledger, executor, context)
    if phase:
        persist(ledger, key, phase)
    before = ledger.operation(key)
    reconcile(ledger, context)
    assert ledger.operation(key) == before
    assert not ledger.uncertain(context.run_id, owned_intent=(key, before["attempt_token"]))
    assert ledger.uncertain(context.run_id)


@pytest.mark.parametrize("failure", ["handler", "schema", "policy", "cancel"])
async def test_done_fact_wins_over_handler_output_policy_or_cancel_failure(ledger, failure):
    entered = []

    async def handler(args, context):
        entered.append(args)
        value = {"value": "invalid"} if failure == "schema" else {"value": 7}
        persist(ledger, context.operation_key, "applied", value)
        if failure == "handler":
            raise AgentError("business_ack_lost", "receipt persisted before acknowledgement", 503)
        if failure == "cancel":
            raise asyncio.CancelledError
        return value

    async def policy(*args):
        raise AgentError("policy_denied", "result publication denied", 403)

    executor, context = setup(
        ledger,
        handler,
        write=True,
        approval=approved,
        policy=policy if failure == "policy" else None,
    )
    executor.receipt_inspector = inspect_receipt
    if failure in {"schema", "policy"}:
        with pytest.raises(AppliedResultUnavailable):
            await executor.execute(context, call())
    elif failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await executor.execute(context, call())
    else:
        assert (await executor.execute(context, call())).value == {"value": 7}
    row = ledger.operations("user", "run1")[0]
    assert row["phase"] == "applied" and row["effect_applied"] == 1 and row["attempts"] == 1
    assert len(entered) == 1
    if failure == "cancel":
        assert (await executor.execute(context, call())).value == {"value": 7}
        assert len(entered) == 1


async def test_known_not_applied_preserves_real_error_and_never_replays(ledger):
    calls = []

    async def handler(args, context):
        calls.append(args)
        persist(ledger, context.operation_key, "not_applied")
        raise AgentError("business_permanent", "operation failed before effect", 409)

    executor, context = setup(ledger, handler, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    first = await executor.execute(context, call())
    row = ledger.operations("user", "run1")[0]
    assert not first.succeeded and first.error == "business_permanent"
    assert row["phase"] == "not_applied" and row["error"] == "business_permanent"
    second = await executor.execute(context, call())
    assert second.error == "business_permanent" and len(calls) == 1
    assert ledger.operations("user", "run1")[0] == row


@pytest.mark.parametrize("phase", ["unknown", "compensated"])
async def test_new_domain_uncertainty_or_compensation_overrides_success_cache(ledger, phase):
    calls = []

    async def handler(args, context):
        calls.append(args)
        persist(ledger, context.operation_key, "applied", {"value": 7})
        return {"value": 7}

    executor, context = setup(ledger, handler, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    first = await executor.execute(context, call())
    assert first.succeeded
    key = first.operation_key
    persist(ledger, key, phase)
    if phase == "unknown":
        with pytest.raises(NeedsReconciliation):
            await executor.execute(context, call())
    else:
        assert not (await executor.execute(context, call())).succeeded
    row = ledger.operation(key)
    assert row["phase"] == phase and row["effect_applied"] == 1
    assert row["effect_compensated"] == int(phase == "compensated")
    assert len(calls) == 1


@pytest.mark.parametrize("first,last", [("applied", "not_applied"), ("compensated", "applied")])
async def test_sticky_history_blocks_contradictions_even_through_unknown_and_reopen(
    ledger, first, last
):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    key, _ = original(ledger, executor, context)
    persist(ledger, key, first, {"value": 7} if first == "applied" else None)
    reconcile(ledger, context)
    persist(ledger, key, "unknown")
    reconcile(ledger, context)
    before = ledger.operation(key)
    persist(ledger, key, last, {"value": 7} if last == "applied" else None)
    reopened = SqliteActionLedger(ledger.connection)
    with pytest.raises(AgentError, match="durable effect history"):
        reconcile(reopened, context)
    assert ledger.operation(key) == before


async def test_not_applied_is_terminal_and_applied_raw_receipt_cannot_change_via_unknown(ledger):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    key, _ = original(ledger, executor, context)
    persist(ledger, key, "not_applied")
    reconcile(ledger, context)
    persist(ledger, key, "unknown")
    with pytest.raises(AgentError, match="durable effect history"):
        reconcile(ledger, context)
    key2, _ = original(ledger, executor, context, call("call2"))
    persist(ledger, key2, "applied", {"value": 7})
    # Isolate the second independent action after the first rejected fact.
    persist(ledger, key, "not_applied")
    reconcile(ledger, context)
    persist(ledger, key2, "unknown")
    reconcile(ledger, context)
    persist(ledger, key2, "applied", {"value": 8})
    with pytest.raises(AgentError, match="value changed"):
        reconcile(ledger, context)
    assert ledger.operation(key2)["phase"] == "unknown"


@pytest.mark.parametrize(
    "field", ["owner", "agent_id", "run_id", "call_id", "binding", "fingerprint", "attempt_token"]
)
async def test_mismatched_host_link_rejected_and_entire_group_rolls_back(ledger, field):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    first, _ = original(ledger, executor, context)
    second, _ = original(ledger, executor, context, call("call2"))
    persist(ledger, first, "applied", {"value": 7})
    persist(ledger, second, "applied", {"value": 8}, **{field: "wrong"})
    before = ledger.operations("user", "run1")
    with pytest.raises(AgentError, match="original attempt"):
        reconcile(ledger, context)
    assert ledger.operations("user", "run1") == before


@pytest.mark.parametrize(
    "mode",
    [
        "async",
        "invalid",
        "write",
        "ddl",
        "script",
        "commit",
        "rollback",
        "savepoint",
        "pragma",
        "unclaimed",
    ],
)
async def test_receipt_hook_contract_failure_does_not_publish_any_partial_fact(ledger, mode):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    if mode == "unclaimed":
        tool = executor.declaration(context, call())
        key = ledger.begin(context, tool, call(), 16)["key"]
    else:
        key, _ = original(ledger, executor, context)
    before = ledger.operation(key)

    async def async_hook(*args):
        return ActionReceipt(phase="applied", value={"value": 7})

    def hook(db, operation):
        if mode == "async":
            return async_hook(db, operation)
        if mode == "invalid":
            return {"phase": "applied"}
        if mode == "write":
            db.execute("UPDATE agent_tool_operations SET phase='unknown' WHERE key=?", (key,))
            return None
        if mode == "ddl":
            db.execute("CREATE TABLE forbidden_receipt_sideeffect(value TEXT)")
        if mode == "script":
            db.executescript("UPDATE agent_tool_operations SET phase='compensated'")
        if mode == "commit":
            db.commit()
        if mode == "rollback":
            db.rollback()
        if mode == "savepoint":
            db.execute("SAVEPOINT forbidden")
        if mode == "pragma":
            db.execute("PRAGMA user_version=55")
        return ActionReceipt(phase="applied", value={"value": 7})

    with pytest.raises(AgentError):
        reconcile(ledger, context, hook)
    assert ledger.operation(key) == before
    with ledger.connection() as db:
        assert (
            db.execute(
                "SELECT name FROM sqlite_master WHERE name='forbidden_receipt_sideeffect'"
            ).fetchone()
            is None
        )
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0


async def test_mutated_hook_snapshot_and_foreign_namespace_do_not_change_original_action(ledger):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    key, _ = original(ledger, executor, context)
    before = ledger.operation(key)

    def hook(db, operation):
        operation.clear()
        return None

    reconcile(ledger, context, hook)
    ledger.reconcile("another", "alpha", "run1", lambda *args: pytest.fail("foreign action"))
    ledger.reconcile("user", "another", "run1", lambda *args: pytest.fail("foreign action"))
    assert ledger.operation(key) == before


@pytest.mark.parametrize("failure", ["scope", "deadline", "authorizer"])
async def test_known_exact_applied_fact_survives_current_control_denial(ledger, failure):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    key, _ = original(ledger, executor, context)
    executor.receipt_inspector = inspect_receipt
    persist(ledger, key, "applied", {"value": 7})
    if failure == "scope":
        context = context.model_copy(update={"identity": Identity(owner="user")})
    elif failure == "deadline":
        context = context.model_copy(update={"deadline": 0.0})
    else:

        async def deny(*args):
            return False

        executor.authorizer = deny
    with pytest.raises(AppliedResultUnavailable):
        await executor.execute(context, call())
    assert ledger.operation(key)["phase"] == "applied"
    with pytest.raises(AgentError) as conflict:
        await executor.execute(context, call(x=2))
    assert conflict.value.code == "operation_conflict"


async def test_runtime_reconciles_before_permission_and_deadline_guard_and_before_uncertain(
    tmp_path,
):
    async def handler(args, context):
        raise AssertionError("classification must not execute a handler")

    async with system(tmp_path, [], config=package(tools=[declaration(True)]), handler=handler) as (
        engine,
        _,
        _,
        identities,
    ):
        # Admission is durable but a bounded initializer failure keeps it out of
        # the queue while this test creates the exact original action evidence.
        async def no_checkpoint(*args, **kwargs):
            raise AgentError("fixture_checkpoint_unavailable", "fixture", 503)

        engine.update_checkpoint = no_checkpoint
        with pytest.raises(AgentError):
            await engine.submit(identities["user"], "agent0", "fact", "input")
        row = engine.store.lookup("user", "agent0", "fact")
        context = await engine.context(row["run_id"])
        binding = engine.binding(row)
        key, token = original(engine.actions, engine.tools, context)
        expected = copy.deepcopy(engine.actions.operation(key))

        def receipt(db, operation):
            assert operation == expected
            assert (
                db.execute(
                    "SELECT owner FROM requests WHERE run_id=?", (row["run_id"],)
                ).fetchone()[0]
                == "user"
            )
            return ActionReceipt(phase="applied", value={"value": 7})

        binding.receipt_inspector = receipt
        identities["user"] = Identity(owner="user")
        assert engine._reason(row, identities["user"]) == "permission_changed"
        stored = engine.actions.operation(key)
        assert (
            stored["phase"] == "applied"
            and stored["attempts"] == 1
            and stored["attempt_token"] == token
        )

        # Updated snapshots for a later known unknown verdict retain the original link.
        def unknown(db, operation):
            assert all(operation[field] == expected[field] for field in LINK)
            return ActionReceipt(phase="unknown")

        binding.receipt_inspector = unknown
        assert engine._uncertain(row)
        assert engine._applied_effects(row) == []
        assert engine.actions.operation(key)["effect_applied"] == 1


def test_runtime_schema_two_migrates_original_action_history_and_rolls_back_on_failure(tmp_path):
    database = SqliteDatabase(tmp_path / "old.sqlite")
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
        db.execute("UPDATE agent_runtime_meta SET value='2' WHERE key='schema'")
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN effect_applied")
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN effect_compensated")
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN call_deadline")
        db.execute(
            "INSERT INTO agent_tool_operations VALUES('key','user','alpha','run','call','binding','fingerprint',0,'applied',1,'null',NULL,'original-token')"
        )
        before = tuple(db.execute("SELECT * FROM agent_tool_operations").fetchone())
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
        after = tuple(db.execute("SELECT * FROM agent_tool_operations").fetchone())
        assert after[:-3] == before and after[-3:] == (1, 0, None)
        assert (
            db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
            == "8"
        )
    # A fresh old layout is migrated inside a transaction which then fails.
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE agent_runtime_meta SET value='2' WHERE key='schema'")
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN effect_applied")
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN effect_compensated")
        db.execute("ALTER TABLE agent_tool_operations DROP COLUMN call_deadline")
    with pytest.raises(RuntimeError), database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteRunStore.install(db)
        raise RuntimeError("migration failure")
    with database.connection() as db:
        assert (
            db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
            == "2"
        )
        assert tuple(db.execute("SELECT * FROM agent_tool_operations").fetchone()) == before
        assert "effect_applied" not in {
            row[1] for row in db.execute("PRAGMA table_info(agent_tool_operations)")
        }
    with pytest.raises(AgentError), database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteActionLedger.install(db)


def test_action_receipt_rejects_unbounded_or_non_applied_payloads():
    with pytest.raises(ValueError):
        ActionReceipt(phase="unknown", value={"value": 7})
    with pytest.raises(AgentError):
        ActionReceipt(phase="applied", value="x" * 300000)


@pytest.mark.parametrize("path", ["cached", "ack_recovery"])
@pytest.mark.parametrize("phase", ["unknown", "compensated"])
async def test_domain_phase_change_during_cached_or_ack_policy_blocks_success(ledger, path, phase):
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(args, context):
        calls.append(args)
        persist(ledger, context.operation_key, "applied", {"value": 7})
        if path == "ack_recovery":
            raise AgentError("ack_lost", "receipt committed", 503)
        return {"value": 7}

    async def policy(context, tool, value):
        entered.set()
        await release.wait()
        return value

    executor, context = setup(ledger, handler, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    if path == "cached":
        first = await executor.execute(context, call())
        key = first.operation_key
    else:
        key = ledger.key(context, call())
    executor.result_policy = policy
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        persist(ledger, key, phase)
        release.set()
        with pytest.raises(AgentError) as error:
            await task
        assert error.value.code == (
            "needs_reconciliation" if phase == "unknown" else "effect_compensated"
        )
        row = ledger.operation(key)
        assert row["phase"] == phase and row["effect_applied"] == 1
        assert row["attempts"] == 1 and len(calls) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_first_authoritative_raw_value_arriving_during_policy_is_reprojected_once(ledger):
    calls, projected = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(args, context):
        calls.append(args)
        return {"value": 7}

    async def policy(context, tool, value):
        projected.append(value["value"])
        if len(projected) == 1:
            entered.set()
            await release.wait()
        return {"value": value["value"] * 10}

    executor, context = setup(ledger, handler, write=True, approval=approved, policy=policy)
    executor.receipt_inspector = inspect_receipt
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        key = ledger.key(context, call())
        persist(ledger, key, "applied", {"value": 8})
        release.set()
        result = await task
        assert result.succeeded and result.value == {"value": 80}
        assert ledger.previous_value(ledger.operation(key)) == {"value": 8}
        assert projected == [7, 8] and len(calls) == 1
        assert ledger.operation(key)["attempts"] == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("raw", [True, 1.0], ids=["boolean", "float"])
async def test_late_receipt_raw_comparison_preserves_json_types(ledger, raw):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handler(args, context):
        calls.append(args)
        return {"value": 1}

    async def policy(context, tool, value):
        entered.set()
        await release.wait()
        return value

    executor, context = setup(ledger, handler, write=True, approval=approved, policy=policy)
    executor.receipt_inspector = inspect_receipt
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        key = ledger.key(context, call())
        persist(ledger, key, "applied", {"value": raw})
        release.set()
        if type(raw) is bool:
            with pytest.raises(AppliedResultUnavailable):
                await task
        else:
            result = await task
            assert type(result.value["value"]) is float
        assert ledger.operation(key)["phase"] == "applied"
        assert type(ledger.previous_value(ledger.operation(key))["value"]) is type(raw)
        assert len(calls) == 1 and ledger.operation(key)["attempts"] == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_authorizer_failure_classifies_current_unknown_instead_of_old_applied(ledger):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    key, _ = original(ledger, executor, context)
    persist(ledger, key, "applied", {"value": 7})
    entered, release = asyncio.Event(), asyncio.Event()

    async def authorizer(*args):
        entered.set()
        await release.wait()
        return False

    executor.authorizer = authorizer
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        persist(ledger, key, "unknown")
        release.set()
        with pytest.raises(NeedsReconciliation):
            await task
        assert ledger.operation(key)["phase"] == "unknown"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["unknown", "compensated"])
async def test_cancel_during_cached_policy_preserves_new_domain_phase(ledger, phase):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    key, _ = original(ledger, executor, context)
    persist(ledger, key, "applied", {"value": 7})
    entered = asyncio.Event()

    async def policy(*args):
        entered.set()
        await asyncio.Event().wait()

    executor.result_policy = policy
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        persist(ledger, key, phase)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ledger.operation(key)["phase"] == phase
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["unknown", "compensated"])
async def test_cancel_during_initial_authorizer_keeps_current_domain_fact(ledger, phase):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    executor.receipt_inspector = inspect_receipt
    key, _ = original(ledger, executor, context)
    persist(ledger, key, "applied", {"value": 7})
    entered = asyncio.Event()

    async def authorizer(*args):
        entered.set()
        await asyncio.Event().wait()

    executor.authorizer = authorizer
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        persist(ledger, key, phase)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ledger.operation(key)["phase"] == phase
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("available", [True, False])
async def test_applied_unavailable_is_distinct_from_successful_json_null(ledger, available):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    bound = executor.registry.agent("alpha")
    tool = bound.tools[0].model_dump(mode="json")
    tool["output_schema"] = {"type": ["object", "null"]}
    definition = bound.definition.model_dump(mode="json")
    definition["version"] = "2"
    executor.registry.activate(
        AgentPackage.model_validate({"tools": [tool], "agents": [definition]})
    )
    context = context.model_copy(update={"config_digest": executor.registry.agent("alpha").digest})
    key, _ = original(ledger, executor, context)
    executor.receipt_inspector = lambda db, row: ActionReceipt(
        phase="applied", result_available=available
    )
    if available:
        result = await executor.execute(context, call())
        assert result.succeeded and result.value is None
    else:
        with pytest.raises(AppliedResultUnavailable):
            await executor.execute(context, call())
    row = ledger.operation(key)
    assert row["phase"] == "applied" and row["effect_applied"] == 1 and row["attempts"] == 1
    assert row["result"] == ("null" if available else None)
    assert row["error"] == (None if available else "effect_applied_result_unavailable")


@pytest.mark.parametrize(
    "value", [object(), float("nan"), "x" * 300000], ids=["non_json", "nonfinite", "oversized"]
)
async def test_known_applied_non_json_or_unbounded_raw_never_replays_or_publishes_null(
    ledger, value
):
    calls = []

    async def handler(args, context):
        calls.append(args)
        return value

    executor, context = setup(ledger, handler, write=True, approval=approved)

    def fact(db, row):
        return ActionReceipt(phase="applied", result_available=False) if row["attempts"] else None

    executor.receipt_inspector = fact
    for _ in range(2):
        with pytest.raises(AppliedResultUnavailable):
            await executor.execute(context, call())
    row = ledger.operations("user", "run1")[0]
    assert row["phase"] == "applied" and row["effect_applied"] == 1 and row["result"] is None
    assert row["attempts"] == 1 and len(calls) == 1


async def test_temporarily_unavailable_fact_retains_old_raw_and_rejects_later_changed_value(ledger):
    executor, context = setup(ledger, lambda: None, write=True, approval=approved)
    key, _ = original(ledger, executor, context)
    persist(ledger, key, "applied", {"value": 7})
    reconcile(ledger, context)
    reconcile(
        ledger, context, lambda db, row: ActionReceipt(phase="applied", result_available=False)
    )
    row = ledger.operation(key)
    assert row["result"] == canonical_json({"value": 7}).decode()
    with pytest.raises(AppliedResultUnavailable):
        ledger.previous_value(row)
    persist(ledger, key, "applied", {"value": 8})
    with pytest.raises(AgentError, match="value changed"):
        reconcile(ledger, context)
    persist(ledger, key, "applied", {"value": 7})
    reconcile(ledger, context)
    assert ledger.previous_value(ledger.operation(key)) == {"value": 7}
