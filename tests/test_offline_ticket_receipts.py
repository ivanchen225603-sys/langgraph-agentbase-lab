"""Synthetic external receipt recovery; no real service, model or production configuration."""

import asyncio
import copy
import json
import os
import subprocess
import sys
from contextlib import contextmanager

import pytest

from agentbase.contracts import AgentError
from agentbase.effects import AppliedResultUnavailable, AttemptEvidence
from examples.offline_ticket_receipts import HostPrincipal, OfflineTicketDemo, OriginalAttempt


@pytest.fixture
async def demo(tmp_path):
    result = OfflineTicketDemo(tmp_path / "fixture")
    row = await result.submit()
    assert row["phase"] == "unknown" and row["attempts"] == 1
    assert result.remote.submissions == 1
    assert result.remote.committed_tickets == 1
    return result


def receipt(demo):
    with demo.database.connection() as db:
        return db.execute(
            "SELECT receipt FROM ticket_attempt_evidence WHERE key=?", (demo.key,)
        ).fetchone()[0]


async def test_response_lost_then_lookup_reconcile_never_resubmits(demo):
    for _ in range(2):
        await demo.submit()
        assert demo.ledger.operation(demo.key)["phase"] == "unknown"
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    for _ in range(2):
        value = await demo.executor.execute(demo.context, demo.call)
        assert value.succeeded and value.value["ticket_id"].startswith("ticket_")
    row = demo.ledger.operation(demo.key)
    assert row["phase"] == "applied" and row["effect_applied"] == 1
    assert row["attempts"] == demo.remote.submissions == 1


async def test_remote_request_counter_detects_a_duplicate_even_when_ticket_insert_rolls_back(demo):
    with demo.database.connection() as db:
        original = OriginalAttempt.model_validate_json(
            db.execute(
                "SELECT original FROM ticket_attempt_evidence WHERE key=?", (demo.key,)
            ).fetchone()[0]
        )
    with pytest.raises(AgentError):
        await demo.remote.create(original)  # Deliberately bypass the adapter's replay fence.
    assert demo.remote.submissions == 2 and demo.remote.committed_tickets == 1
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    assert demo.remote.submissions == 2


async def test_missing_remote_evidence_is_unknown_not_not_applied(tmp_path):
    demo = OfflineTicketDemo(tmp_path / "absent")
    tool = demo.executor.declaration(demo.context, demo.call)
    demo.ledger.begin(demo.context, tool, demo.call, 16)
    token = demo.ledger.claim_attempt(
        demo.key,
        1,
        evidence=AttemptEvidence(context=demo.context, tool=tool, call=demo.call),
        commit=demo.adapter.attempt_commit,
    )
    demo.ledger.failed(demo.key, "response_missing", handler_started=True, attempt_token=token)
    assert await demo.adapter.query(demo.principal, demo.key) == "unknown"
    assert receipt(demo) is None and (await demo.submit())["phase"] == "unknown"
    assert demo.remote.submissions == 0


@pytest.mark.parametrize("mode", ["temporary", "timeout", "unknown"])
async def test_query_failure_never_retries_submission(demo, mode):
    entered = asyncio.Event()

    async def unavailable(original):
        entered.set()
        if mode == "temporary":
            raise OSError("synthetic service unavailable")
        if mode == "unknown":
            return None
        await asyncio.Event().wait()

    demo.remote.lookup = unavailable
    before = demo.ledger.operation(demo.key)
    task = asyncio.create_task(demo.adapter.query(demo.principal, demo.key, query_seconds=0.02))
    await asyncio.wait_for(entered.wait(), 2)
    assert await asyncio.wait_for(task, 2) == "unknown"
    assert receipt(demo) is None and demo.ledger.operation(demo.key) == before
    await demo.submit()
    assert demo.remote.submissions == 1


async def test_cancel_query_keeps_unknown_and_remote_fact_for_later_query(demo):
    entered, release = asyncio.Event(), asyncio.Event()
    lookup = demo.remote.lookup

    async def gated(original):
        entered.set()
        await release.wait()
        return await lookup(original)

    demo.remote.lookup = gated
    task = asyncio.create_task(demo.adapter.query(demo.principal, demo.key))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert receipt(demo) is None and demo.ledger.operation(demo.key)["phase"] == "unknown"
    assert demo.remote.submissions == 1
    release.set()
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    assert demo.remote.submissions == 1


async def test_no_local_connection_is_open_at_asynchronous_transport_boundary(demo):
    connections = []
    original_connection = demo.ledger.connection
    lookup = demo.remote.lookup

    @contextmanager
    def tracked():
        with original_connection() as db:
            connections.append(db)
            try:
                yield db
            finally:
                connections.remove(db)

    async def checked(original):
        assert connections == []
        await asyncio.sleep(0)
        assert connections == []
        return await lookup(original)

    demo.ledger.connection = tracked
    demo.remote.lookup = checked
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    assert connections == [] and demo.remote.submissions == 1


@pytest.mark.parametrize(
    "field",
    [
        "tenant",
        "subject",
        "owner",
        "key",
        "agent_id",
        "run_id",
        "call_id",
        "binding",
        "fingerprint",
        "attempt_token",
        "attempts",
        "arguments_digest",
    ],
)
async def test_old_attempt_cross_tenant_or_parameter_receipt_rejected(demo, field):
    lookup = demo.remote.lookup

    async def tampered(original):
        payload = copy.deepcopy(await lookup(original))
        payload["original"][field] = 2 if field == "attempts" else "foreign"
        return payload

    demo.remote.lookup = tampered
    before = demo.ledger.operation(demo.key)
    with pytest.raises(AgentError, match="differs from original"):
        await demo.adapter.query(demo.principal, demo.key)
    assert receipt(demo) is None and demo.ledger.operation(demo.key) == before
    assert demo.remote.submissions == 1


@pytest.mark.parametrize("field", ["tenant", "subject"])
async def test_foreign_authenticated_principal_rejected_before_transport(demo, field):
    principal = HostPrincipal(**(demo.principal.model_dump() | {field: "other"}))
    with pytest.raises(AgentError):
        await demo.adapter.query(principal, demo.key)
    assert demo.remote.lookups == 0 and receipt(demo) is None
    assert demo.remote.submissions == 1


@pytest.mark.parametrize(
    "change",
    [
        "attempt_token",
        "attempts",
        "fingerprint",
        "subject",
        "compensated",
        "sticky_compensated",
        "historical_result",
    ],
)
async def test_query_rechecks_live_binding_after_async_wait(demo, change):
    entered, release = asyncio.Event(), asyncio.Event()
    lookup = demo.remote.lookup

    async def gated(original):
        entered.set()
        await release.wait()
        return await lookup(original)

    demo.remote.lookup = gated
    task = asyncio.create_task(demo.adapter.query(demo.principal, demo.key))
    await asyncio.wait_for(entered.wait(), 2)
    # Acquiring another writer while transport is suspended proves the adapter
    # holds neither the local connection nor its transaction across the await.
    with demo.database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if change == "subject":
            original = json.loads(
                db.execute("SELECT original FROM ticket_attempt_evidence").fetchone()[0]
            )
            original["subject"] = "other"
            db.execute("UPDATE ticket_attempt_evidence SET original=?", (json.dumps(original),))
        elif change == "compensated":
            db.execute("UPDATE agent_tool_operations SET phase='compensated',effect_compensated=1")
        elif change == "sticky_compensated":
            db.execute(
                "UPDATE agent_tool_operations SET phase='unknown',effect_applied=1,effect_compensated=1"
            )
        elif change == "historical_result":
            db.execute(
                "UPDATE agent_tool_operations SET phase='unknown',effect_applied=1,result=?",
                ('{"ticket_id":"other"}',),
            )
        else:
            value = 2 if change == "attempts" else "changed"
            db.execute(f"UPDATE agent_tool_operations SET {change}=?", (value,))
    release.set()
    with pytest.raises(AgentError):
        await task
    assert receipt(demo) is None and demo.remote.submissions == 1


async def test_concurrent_duplicate_queries_are_idempotent_and_conflicting_fact_rejected(demo):
    entered, release = asyncio.Event(), asyncio.Event()
    lookup = demo.remote.lookup
    count = 0

    async def gated(original):
        nonlocal count
        count += 1
        if count == 4:
            entered.set()
        await release.wait()
        return await lookup(original)

    demo.remote.lookup = gated
    tasks = [asyncio.create_task(demo.adapter.query(demo.principal, demo.key)) for _ in range(4)]
    await asyncio.wait_for(entered.wait(), 2)
    with demo.database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        assert db.execute("SELECT COUNT(*) FROM ticket_attempt_evidence").fetchone()[0] == 1
    release.set()
    assert await asyncio.gather(*tasks) == ["applied"] * 4
    before = receipt(demo)

    async def contradictory(original):
        payload = await lookup(original)
        payload["ticket_id"] = "a_different_ticket"
        return payload

    demo.remote.lookup = contradictory
    with pytest.raises(AgentError):
        await demo.adapter.query(demo.principal, demo.key)
    assert receipt(demo) == before
    assert demo.ledger.operation(demo.key)["attempts"] == demo.remote.submissions == 1


@pytest.mark.parametrize("restriction", ["approval_revoked", "current_run_expired"])
async def test_current_write_restriction_does_not_erase_submitted_fact(demo, restriction):
    if restriction == "approval_revoked":
        demo.approved = False
    else:
        demo.context = demo.context.model_copy(update={"deadline": 1.0})
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    with pytest.raises(AppliedResultUnavailable):
        await demo.executor.execute(demo.context, demo.call)
    assert demo.ledger.operation(demo.key)["effect_applied"] == 1
    assert receipt(demo) is not None and demo.remote.submissions == 1


async def test_historical_attempt_deadline_does_not_expire_trusted_fact(demo):
    with demo.database.connection() as db:
        db.execute("UPDATE agent_tool_operations SET call_deadline=1")
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    observation = await demo.executor.execute(demo.context, demo.call)
    assert observation.succeeded
    assert demo.ledger.operation(demo.key)["effect_applied"] == 1
    assert demo.remote.submissions == 1


async def test_cancel_after_receipt_commit_does_not_delete_fact(demo):
    original_reconcile = demo.adapter.reconcile

    def cancelled_after_commit(original):
        raise asyncio.CancelledError

    demo.adapter.reconcile = cancelled_after_commit
    with pytest.raises(asyncio.CancelledError):
        await demo.adapter.query(demo.principal, demo.key)
    assert receipt(demo) is not None and demo.ledger.operation(demo.key)["phase"] == "unknown"
    demo.adapter.reconcile = original_reconcile
    assert await demo.adapter.query(demo.principal, demo.key) == "applied"
    assert demo.remote.submissions == 1


async def test_restart_reconciles_committed_evidence_without_remote_query(demo):
    def crashed(original):
        raise RuntimeError("synthetic crash between receipt commit and reconcile")

    demo.adapter.reconcile = crashed
    with pytest.raises(RuntimeError):
        await demo.adapter.query(demo.principal, demo.key)
    restored = OfflineTicketDemo(demo.database.path.parent)
    restored.ledger.reconcile(
        restored.principal.owner,
        restored.context.agent_id,
        restored.context.run_id,
        restored.adapter.inspect,
    )
    assert restored.ledger.operation(restored.key)["phase"] == "applied"
    assert restored.remote.lookups == 0 and restored.remote.submissions == 1


async def test_inspector_revalidates_durable_receipt_binding(demo):
    def crashed(original):
        raise RuntimeError("synthetic crash")

    demo.adapter.reconcile = crashed
    with pytest.raises(RuntimeError):
        await demo.adapter.query(demo.principal, demo.key)
    with demo.database.connection() as db:
        payload = json.loads(receipt(demo))
        payload["original"]["attempt_token"] = "old_attempt"
        db.execute("UPDATE ticket_attempt_evidence SET receipt=?", (json.dumps(payload),))
    with pytest.raises(AgentError):
        demo.ledger.reconcile(
            demo.principal.owner, demo.context.agent_id, demo.context.run_id, demo.adapter.inspect
        )
    assert demo.ledger.operation(demo.key)["phase"] == "unknown"
    assert demo.remote.submissions == 1


async def test_run_call_key_has_no_cross_run_business_deduplication(demo):
    next_run = demo.context.model_copy(update={"run_id": "another_run"})
    next_call = demo.call.model_copy(update={"call_id": "another_call"})
    assert (
        len(
            {
                demo.key,
                demo.ledger.key(next_run, demo.call),
                demo.ledger.key(demo.context, next_call),
            }
        )
        == 3
    )
    assert demo.remote.submissions == 1


@pytest.mark.parametrize(
    "payload", ["tool claims success", {"phase": "applied"}, {"phase": "unknown"}]
)
async def test_unbound_or_incomplete_response_is_not_a_receipt(demo, payload):
    async def untrusted(original):
        return payload

    demo.remote.lookup = untrusted
    with pytest.raises(AgentError):
        await demo.adapter.query(demo.principal, demo.key)
    assert receipt(demo) is None and demo.ledger.operation(demo.key)["phase"] == "unknown"
    assert demo.remote.submissions == 1


@pytest.mark.parametrize("change", ["phase", "extra"])
async def test_outer_receipt_schema_rejected_without_persisting(demo, change):
    lookup = demo.remote.lookup

    async def invalid(original):
        payload = await lookup(original)
        payload["phase" if change == "phase" else "tool_claim"] = "unknown"
        return payload

    demo.remote.lookup = invalid
    with pytest.raises(AgentError):
        await demo.adapter.query(demo.principal, demo.key)
    assert receipt(demo) is None and demo.ledger.operation(demo.key)["phase"] == "unknown"
    assert demo.remote.submissions == demo.remote.committed_tickets == 1


async def test_tool_self_report_is_not_a_trusted_receipt(tmp_path):
    demo = OfflineTicketDemo(tmp_path / "self-report")

    async def self_claimed(original):
        return {"ticket_id": "tool_claims_success"}

    demo.remote.create = self_claimed
    assert (await demo.submit())["phase"] == "unknown"
    assert await demo.adapter.query(demo.principal, demo.key) == "unknown"
    assert receipt(demo) is None and demo.remote.submissions == 0


def test_real_process_restart_preserves_one_submission(tmp_path):
    directory = tmp_path / "two-processes"
    env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1"}
    results = []
    for phase in ["submit", "recover", "recover"]:
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.offline_ticket_receipts",
                phase,
                "--directory",
                str(directory),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        assert child.returncode == 0, child.stderr
        results.append(json.loads(child.stdout))
    assert [value["phase"] for value in results] == ["unknown", "applied", "applied"]
    assert all(value["submissions"] == value["attempts"] == 1 for value in results)
    assert all(value["committed_tickets"] == 1 for value in results)
