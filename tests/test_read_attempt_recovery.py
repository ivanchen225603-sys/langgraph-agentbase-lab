"""Startup repair proves original read claims and never refreshes their policy."""

import json
import time

import pytest

from agentbase import AttemptCommit
from agentbase.contracts import AgentError
from agentbase.effects import AttemptEvidence
from tests.test_tool_attempt_policy import call, configured
from tests.test_tool_attempt_policy import ledger as ledger


def persist(db, row, evidence):
    db.execute(
        "INSERT INTO receipt_links VALUES(?,?,?,?)",
        (row["key"], row["attempt_token"], row["attempts"], evidence.model_dump_json()),
    )


def seed(ledger, *, name="one", write=False):
    calls = []

    async def handler(arguments, context):
        calls.append(context.model_copy(deep=True))
        return arguments["x"]

    executor, context, _ = configured(
        ledger,
        handler,
        hook=AttemptCommit(frozenset({"receipt_links"}), persist),
        timeout=60,
        write=write,
    )
    context = context.model_copy(update={"deadline": time.time() + 60})
    action = call(name)
    selected = executor.resolve(context, action)
    operation = ledger.begin(context, selected.declaration, action, 10)
    ledger.claim_attempt(
        operation["key"],
        2 if not write else 1,
        call_deadline=context.deadline - 1,
        commit=selected.attempt_commit,
        evidence=AttemptEvidence(context=context, tool=selected.declaration, call=action),
    )
    return executor, context, action, calls, ledger.operation(operation["key"])


def proof(ledger):
    def inspect(db, operation):
        link = db.execute(
            "SELECT * FROM receipt_links WHERE key=? AND token=?",
            (operation["key"], operation["attempt_token"]),
        ).fetchone()
        if link is None:
            return False
        evidence = AttemptEvidence.model_validate_json(link["payload"])
        ledger._check_evidence(operation, evidence)
        if (
            evidence.context.operation_key != operation["key"]
            or evidence.context.attempt_token != operation["attempt_token"]
            or link["attempt"] != operation["attempts"]
        ):
            raise AgentError("read_recovery_conflict", "original claim differs", 409)
        return evidence.tool.read_only

    return inspect


async def test_recovered_attempt_preserves_window_fences_old_token_and_uses_remaining_budget(
    ledger,
):
    executor, context, action, calls, original = seed(ledger)
    assert ledger.recover_read_attempts(proof(ledger)) == 1
    recovered = ledger.operation(original["key"])
    expected = {**original, "phase": "failed", "error": "process_lost", "attempt_token": None}
    assert recovered == expected
    with pytest.raises(AgentError):
        ledger.complete(original["key"], 99, original["attempt_token"])
    ledger.failed(
        original["key"], "late", handler_started=True, attempt_token=original["attempt_token"]
    )
    assert ledger.operation(original["key"]) == expected
    assert (await executor.execute(context, action)).value == 1
    completed = ledger.operation(original["key"])
    assert completed["attempts"] == 2 and completed["attempt_token"] != original["attempt_token"]
    assert completed["call_deadline"] == original["call_deadline"] and len(calls) == 1
    assert (await executor.execute(context, action)).value == 1 and len(calls) == 1
    with ledger.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM receipt_links").fetchone()[0] == 2


async def test_recovery_keeps_expired_original_tool_window(ledger, monkeypatch):
    executor, context, action, calls, original = seed(ledger)
    with monkeypatch.context() as clocks:
        clocks.setattr("time.time", lambda: original["call_deadline"] + 0.1)
        assert ledger.recover_read_attempts(proof(ledger)) == 1
        observed = await executor.execute(context, action)
    assert observed.error == "tool_timeout" and not observed.succeeded and calls == []
    recovered = ledger.operation(original["key"])
    assert recovered["call_deadline"] == original["call_deadline"] and recovered["attempts"] == 1


@pytest.mark.parametrize("phase", ["intent", "unknown", "applied"])
def test_recovery_never_touches_writes_or_terminal_reads(ledger, phase):
    _, _, _, _, read = seed(ledger)
    _, _, _, _, write = seed(ledger, name="write", write=True)
    _, _, _, _, terminal = seed(ledger, name="terminal")
    ledger.complete(terminal["key"], 1, terminal["attempt_token"])
    with ledger.connection() as db:
        db.execute("UPDATE agent_tool_operations SET phase=? WHERE key=?", (phase, write["key"]))
    untouched = [ledger.operation(write["key"]), ledger.operation(terminal["key"])]
    observed = []
    original_proof = proof(ledger)

    def inspect(db, operation):
        observed.append(operation["key"])
        return original_proof(db, operation)

    assert ledger.recover_read_attempts(inspect) == 1 and observed == [read["key"]]
    assert [ledger.operation(write["key"]), ledger.operation(terminal["key"])] == untouched


@pytest.mark.parametrize("field", ["attempt_token", "call_deadline"])
def test_missing_original_control_skips_proof_and_keeps_query_only(ledger, field):
    _, _, _, _, original = seed(ledger)
    with ledger.connection() as db:
        db.execute(f"UPDATE agent_tool_operations SET {field}=NULL")
    before = ledger.operation(original["key"])

    def forbidden(*args):
        raise AssertionError("missing original control cannot be authorized")

    assert ledger.recover_read_attempts(forbidden) == 0
    assert ledger.operation(original["key"]) == before


def test_second_conflicting_proof_rolls_back_whole_batch_and_missing_link_skips(ledger):
    _, _, _, _, first = seed(ledger)
    _, _, _, _, second = seed(ledger, name="two")
    with ledger.connection() as db:
        payload = json.loads(
            db.execute(
                "SELECT payload FROM receipt_links WHERE key=?", (second["key"],)
            ).fetchone()[0]
        )
        payload["call"]["arguments"]["x"] = 2
        db.execute(
            "UPDATE receipt_links SET payload=? WHERE key=?", (json.dumps(payload), second["key"])
        )
    with pytest.raises(AgentError):
        ledger.recover_read_attempts(proof(ledger))
    assert ledger.operation(first["key"]) == first and ledger.operation(second["key"]) == second
    with ledger.connection() as db:
        db.execute("DELETE FROM receipt_links WHERE key=?", (second["key"],))
    assert ledger.recover_read_attempts(proof(ledger)) == 1
    assert ledger.operation(second["key"]) == second


@pytest.mark.parametrize("kind", ["integer", "none", "async"])
def test_proof_requires_strict_synchronous_boolean_and_rolls_back(ledger, kind):
    _, _, _, _, first = seed(ledger)
    _, _, _, _, second = seed(ledger, name="two")
    inspect = proof(ledger)

    async def pending():
        return True

    def invalid(db, operation):
        if operation["key"] == first["key"]:
            return inspect(db, operation)
        return 1 if kind == "integer" else None if kind == "none" else pending()

    with pytest.raises(AgentError) as error:
        ledger.recover_read_attempts(invalid)
    assert error.value.code == "read_recovery_contract"
    assert ledger.operation(first["key"]) == first and ledger.operation(second["key"]) == second


@pytest.mark.parametrize(
    "sql",
    [
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT bad",
        "CREATE TABLE bad(x)",
        "UPDATE agent_tool_operations SET attempts=0",
    ],
)
def test_proof_cannot_write_or_change_transaction(ledger, sql):
    _, _, _, _, original = seed(ledger)

    def invalid(db, operation):
        db.execute(sql)
        return True

    with pytest.raises(AgentError) as error:
        ledger.recover_read_attempts(invalid)
    assert (
        error.value.code == "read_recovery_contract"
        and ledger.operation(original["key"]) == original
    )


def test_proof_private_snapshot_and_whole_batch_validation_order(ledger):
    _, _, _, _, first = seed(ledger)
    _, _, _, _, second = seed(ledger, name="two")
    inspect = proof(ledger)

    def private(db, operation):
        approved = inspect(db, operation)
        assert (
            dict(
                db.execute(
                    "SELECT * FROM agent_tool_operations WHERE key=?", (first["key"],)
                ).fetchone()
            )
            == first
        )
        operation.update(attempts=0, attempt_token="forged", call_deadline=time.time() + 1000)
        return approved

    assert ledger.recover_read_attempts(private) == 2
    for original in (first, second):
        assert ledger.operation(original["key"]) == {
            **original,
            "phase": "failed",
            "error": "process_lost",
            "attempt_token": None,
        }


async def test_started_and_closed_real_host_cannot_run_startup_recovery(service):
    for operation in (service.runtime.recover_startup_actions,):
        with pytest.raises(AgentError) as error:
            operation()
        assert error.value.code == "read_recovery_state"
    await service.runtime.close()
    with pytest.raises(AgentError) as error:
        service.runtime.recover_startup_actions()
    assert error.value.code == "read_recovery_state"
