"""Temporary action ledgers prove tool contracts and uncertain-write recovery."""

import asyncio
import time

import pytest

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Identity, RunContext, ToolCall
from agentbase.effects import AppliedResultUnavailable, NeedsReconciliation, SqliteActionLedger
from agentbase.registry import ComponentRegistry
from agentbase.storage import SqliteDatabase
from agentbase.tools import ToolExecutor, TransientToolError


@pytest.fixture
def ledger(tmp_path):
    database = SqliteDatabase(tmp_path / "business.sqlite")
    result = SqliteActionLedger(database.connection)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        result.install(db)
    return result


def setup(
    ledger,
    handler,
    *,
    write=False,
    retry=True,
    max_calls=16,
    result_bytes=60000,
    approval=None,
    policy=None,
):
    package = AgentPackage.model_validate(
        {
            "tools": [
                {
                    "name": "custom",
                    "handler": "custom.handler",
                    "description": "Synthetic test tool",
                    "scope": "write" if write else "read",
                    "read_only": not write,
                    "retry_safe": retry,
                    "idempotency": "business-key" if write else "read-only",
                    "result_bytes": result_bytes,
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
            ],
            "agents": [
                {
                    "id": "alpha",
                    "description": "Synthetic Agent",
                    "workflow": "fixed",
                    "prompt": {"system_template": "Policy"},
                    "input_schema": {"type": "string"},
                    "tools": ["custom"],
                    "budget": {"max_tool_calls": max_calls},
                }
            ],
        }
    )
    registry = ComponentRegistry()
    registry.register_handler("custom.handler", handler)
    registry.register_workflow("fixed", lambda: None)
    registry.bind(package)
    context = RunContext(
        identity=Identity(owner="user", scopes=frozenset({"read", "write"})),
        agent_id="alpha",
        run_id="run1",
        conversation_id="same",
        deadline=time.time() + 10,
        config_digest=registry.agent("alpha").digest,
    )
    executor = ToolExecutor(registry, ledger, approval_checker=approval, result_policy=policy)
    return executor, context


def call(call_id="call1", x=1):
    return ToolCall(call_id=call_id, name="custom", arguments={"x": x})


async def test_tool_allowlist_scope_input_output_and_stable_cached_observation(ledger):
    invoked = []

    async def handler(args, context):
        invoked.append(args)
        assert context.operation_key
        return {"value": args["x"]}

    executor, context = setup(ledger, handler)
    for denied in (
        call().model_copy(update={"name": "other"}),
        call().model_copy(update={"arguments": {"x": "wrong"}}),
    ):
        with pytest.raises(AgentError):
            await executor.execute(context, denied)
    with pytest.raises(AgentError):
        await executor.execute(
            context.model_copy(update={"identity": Identity(owner="user")}), call()
        )
    assert invoked == [] and ledger.operations("user", "run1") == []
    first = await executor.execute(context, call())
    assert first.succeeded and first.value == {"value": 1}
    assert await executor.execute(context, call()) == first
    assert len(invoked) == 1
    with pytest.raises(AgentError, match="different"):
        await executor.execute(context, call(x=2))
    with pytest.raises(AgentError, match="binding"):
        await executor.execute(
            context.model_copy(update={"config_digest": "sha256:" + "0" * 64}), call()
        )


async def test_read_retry_attempts_are_finite_persistent_and_output_errors_are_safe(ledger):
    invoked = []

    async def handler(args, context):
        invoked.append(args)
        if len(invoked) == 1:
            raise TransientToolError()
        return {"value": args["x"]}

    executor, context = setup(ledger, handler)
    assert (await executor.execute(context, call())).succeeded
    assert len(invoked) == 2 and ledger.operations("user", "run1")[0]["attempts"] == 2
    assert (await executor.execute(context, call())).succeeded and len(invoked) == 2


@pytest.mark.parametrize("failure", ["schema", "size", "receipt", "handler"])
async def test_write_after_effect_postprocessing_failure_blocks_automatic_replay(
    ledger, failure, monkeypatch
):
    effects = []

    async def approved(context, tool, action, digest):
        return digest == ToolExecutor.action_digest(context, tool, action)

    async def handler(args, context):
        effects.append(args["x"])
        if failure == "schema":
            return {"unexpected": args["x"]}
        if failure == "handler":
            raise RuntimeError("private payload after effect")
        return {"value": args["x"]}

    executor, context = setup(
        ledger,
        handler,
        write=True,
        result_bytes=1 if failure == "size" else 60000,
        approval=approved,
    )
    if failure == "receipt":

        def crash_receipt(key, value, attempt_token):
            raise AgentError("storage_unavailable", "synthetic receipt failure", 503)

        monkeypatch.setattr(ledger, "complete", crash_receipt)
    with pytest.raises(NeedsReconciliation):
        await executor.execute(context, call())
    assert effects == [1] and ledger.uncertain("run1")
    restored = SqliteActionLedger(ledger.connection)
    executor.ledger = restored
    with pytest.raises(NeedsReconciliation):
        await executor.execute(context, call())
    assert effects == [1] and restored.operations("user", "run1")[0]["phase"] == "unknown"


async def test_write_requires_trusted_exact_action_confirmation_and_keeps_receipt(ledger):
    effects = []

    async def handler(args, context):
        effects.append(args["x"])
        return {"value": args["x"]}

    executor, context = setup(ledger, handler, write=True)
    with pytest.raises(AgentError) as error:
        await executor.execute(context, call())
    assert error.value.code == "approval_required"
    assert not effects and not ledger.operations("user", "run1")
    expected = ToolExecutor.action_digest(context, executor.declaration(context, call()), call())

    async def approved(context, tool, action, digest):
        return digest == expected

    executor.approval_checker = approved
    with pytest.raises(AgentError, match="review"):
        await executor.execute(context, call(x=2))
    first = await executor.execute(context, call())
    assert first.succeeded and first.operation_key and not ledger.uncertain("run1")
    assert await executor.execute(context, call()) == first and effects == [1]


async def test_write_cancellation_keeps_unknown_intent_and_is_never_implicit_undo(ledger):
    entered = asyncio.Event()
    effects = []

    async def approved(*args):
        return True

    async def handler(args, context):
        effects.append(args["x"])
        entered.set()
        await asyncio.Event().wait()

    executor, context = setup(ledger, handler, write=True, approval=approved)
    pending = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert effects == [1] and ledger.uncertain("run1")
    with pytest.raises(NeedsReconciliation):
        await executor.execute(context, call())
    assert effects == [1]


async def test_logical_action_quota_scope_and_current_result_policy_cannot_be_bypassed(ledger):
    async def handler(args, context):
        return {"value": args["x"]}

    revoked = False

    async def policy(context, tool, value):
        if revoked:
            raise AgentError("source_revoked", "result source is no longer authorized", 403)
        return value

    executor, context = setup(ledger, handler, max_calls=1, policy=policy)
    assert (await executor.execute(context, call())).succeeded
    with pytest.raises(AgentError, match="budget"):
        await executor.execute(context, call("second"))
    revoked = True
    with pytest.raises(AgentError, match="authorized"):
        await executor.execute(context, call())  # Cached output still runs current policy.
    assert not ledger.operations("other", "run1")


async def test_read_invalid_output_and_handler_errors_never_become_success_or_leak_raw(ledger):
    async def handler(args, context):
        if args["x"] == 1:
            return {"unexpected": "private data"}
        raise RuntimeError("private-secret")

    executor, context = setup(ledger, handler)
    bad_schema = await executor.execute(context, call())
    assert (
        not bad_schema.succeeded and bad_schema.value is None and bad_schema.error == "tool_result"
    )
    failed = await executor.execute(context, call("second", 2))
    assert not failed.succeeded and failed.error == "tool_failure"
    assert "private" not in failed.model_dump_json()


@pytest.mark.parametrize("write", [False, True])
async def test_same_action_competitor_never_overwrites_active_attempt(ledger, write, monkeypatch):
    entered, release, both_pending = asyncio.Event(), asyncio.Event(), asyncio.Event()
    effects = []

    async def approved(*args):
        return True

    async def handler(args, context):
        effects.append(args["x"])
        entered.set()
        await release.wait()
        return {"value": args["x"]}

    executor, context = setup(ledger, handler, write=write, approval=approved)
    executor.slots = asyncio.Semaphore(0)
    original_begin = ledger.begin
    admissions = 0

    def observed_begin(*args):
        nonlocal admissions
        row = original_begin(*args)
        admissions += 1
        if admissions == 2:
            both_pending.set()
        return row

    monkeypatch.setattr(ledger, "begin", observed_begin)
    first = asyncio.create_task(executor.execute(context, call()))
    second = asyncio.create_task(executor.execute(context, call()))
    try:
        await asyncio.wait_for(both_pending.wait(), 1)
        executor.slots.release()
        await asyncio.wait_for(entered.wait(), 1)
        executor.slots.release()
        if write:
            with pytest.raises(NeedsReconciliation):
                await second
        else:
            assert (await second).error == "operation_in_progress"
        current = ledger.operations("user", "run1")[0]
        assert current["phase"] == ("intent" if write else "executing")
        assert current["attempts"] == 1 and ledger.uncertain("run1") is write
        release.set()
        assert (await first).succeeded and effects == [1]
        assert not ledger.uncertain("run1")
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_unclaimed_competitor_cancel_does_not_change_write_intent(ledger, monkeypatch):
    entered, release, both_pending = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def approved(*args):
        return True

    async def handler(args, context):
        entered.set()
        await release.wait()
        return {"value": args["x"]}

    executor, context = setup(ledger, handler, write=True, approval=approved)
    executor.slots = asyncio.Semaphore(0)
    original_begin = ledger.begin
    admissions = 0

    def observed_begin(*args):
        nonlocal admissions
        result = original_begin(*args)
        admissions += 1
        if admissions == 2:
            both_pending.set()
        return result

    monkeypatch.setattr(ledger, "begin", observed_begin)
    first = asyncio.create_task(executor.execute(context, call()))
    competitor = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(both_pending.wait(), 1)
    executor.slots.release()
    await asyncio.wait_for(entered.wait(), 1)
    competitor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await competitor
    assert ledger.uncertain("run1") and ledger.operations("user", "run1")[0]["phase"] == "intent"
    release.set()
    assert (await first).succeeded


async def test_committed_receipt_ack_failure_uses_trusted_receipt(ledger, monkeypatch):
    effects = []

    async def approved(*args):
        return True

    async def handler(args, context):
        effects.append(args["x"])
        return {"value": args["x"]}

    executor, context = setup(ledger, handler, write=True, approval=approved)
    original_complete = ledger.complete

    def lost_ack(key, value, attempt_token):
        original_complete(key, value, attempt_token)
        raise RuntimeError("private receipt acknowledgement loss")

    monkeypatch.setattr(ledger, "complete", lost_ack)
    observed = await executor.execute(context, call())
    assert observed.succeeded and observed.value == {"value": 1}
    assert (
        not ledger.uncertain("run1") and ledger.operations("user", "run1")[0]["phase"] == "applied"
    )
    assert (await executor.execute(context, call())).succeeded and effects == [1]


async def test_applied_result_revocation_preserves_known_effect_and_blocks_replay(ledger):
    effects, revoked = [], False

    async def approved(*args):
        return True

    async def handler(args, context):
        effects.append(args["x"])
        return {"value": args["x"]}

    async def current_policy(context, tool, value):
        if revoked:
            raise AgentError("source_revoked", "result is no longer publishable", 403)
        return value

    executor, context = setup(ledger, handler, write=True, approval=approved, policy=current_policy)
    assert (await executor.execute(context, call())).succeeded
    revoked = True
    with pytest.raises(AppliedResultUnavailable):
        await executor.execute(context, call())
    assert effects == [1] and not ledger.uncertain("run1")
    assert ledger.operations("user", "run1")[0]["phase"] == "applied"


async def test_cached_action_cannot_accept_changed_dependency_inputs(ledger):
    invoked = []

    async def handler(args, context):
        invoked.append(context.dependency_results)
        return {"value": 1}

    executor, context = setup(ledger, handler)
    first = context.model_copy(update={"dependency_results": {"previous": 1}})
    assert (await executor.execute(first, call())).succeeded
    with pytest.raises(AgentError, match="different parameters"):
        await executor.execute(
            context.model_copy(update={"dependency_results": {"previous": 2}}), call()
        )
    assert invoked == [{"previous": 1}]


def test_configured_action_and_fingerprint_keep_pre_resolver_snapshot_digests(ledger):
    async def handler(args, context):
        return {"value": args["x"]}

    executor, context = setup(ledger, handler)
    # Captured with the actual pre-resolver tools/effects modules in the source snapshot.
    assert executor.challenge(context, call()) == (
        "sha256:a6373733fc83caffd02669349f2f1551e74d07728fce07613593056998ac8bf6"
    )
    tool = executor.declaration(context, call())
    assert ledger.begin(context, tool, call(), 16)["fingerprint"] == (
        "sha256:c5e2399c0705502cc3b046de3805beee5415faedb60295631f48aeb585573872"
    )
