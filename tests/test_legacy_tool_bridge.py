"""Original StructuredTools use public attempts, proof and exact approvals."""

import asyncio
import copy
import json

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from agentbase.contracts import AgentError
from agentbase.runtime_contracts import FinalProjection
from raglab.errors import LabError, TransientError
from raglab.models import RunInput, TaskSpec
from raglab.tools import ReadArgs, ToolSpec
from tests.conftest import LOCAL, wait_run


def register(service, name, handler, *, write=False):
    service.registry.register(
        ToolSpec(
            name,
            StructuredTool.from_function(
                coroutine=handler, name=name, description="Bridge regression", args_schema=ReadArgs
            ),
            read_only=not write,
            retry_safe=not write,
            scope="memory:write" if write else "read",
            idempotency="business receipt" if write else "read-only",
        )
    )


async def native_review(service, run_id, original):
    for _ in range(200):
        view = await service.view("local", run_id)
        if (
            view["status"] == "awaiting_review"
            and view["interrupts"]
            and view["interrupts"][0]["review_id"] != original
        ):
            assert view["interrupts"][0]["preview"]["kind"] == "tool_action"
            return view
        await asyncio.sleep(0.01)
    raise AssertionError("dependency action did not expose its own exact review")


async def setup_plan(service, *, consume=True):
    observed = []

    async def read(question, config: RunnableConfig):
        observed.append(("read", copy.deepcopy(config["configurable"]["tool_context"])))
        return {"answer": question}

    async def write(question, config: RunnableConfig):
        ctx = config["configurable"]["tool_context"]
        observed.append(("write", copy.deepcopy(ctx)))
        with service.store.connection() as db:
            public = dict(
                db.execute(
                    "SELECT * FROM agent_tool_operations WHERE key=?",
                    (ctx["public_operation_key"],),
                ).fetchone()
            )
            domain = dict(
                db.execute(
                    "SELECT * FROM operations WHERE key=?", (ctx["operation_key"],)
                ).fetchone()
            )
            assert public["phase"] == domain["phase"] == "intent"
            assert public["attempt_token"] == ctx["attempt_token"]
            assert (
                db.execute(
                    "SELECT public_key FROM legacy_write_actions WHERE domain_key=?",
                    (ctx["operation_key"],),
                ).fetchone()[0]
                == public["key"]
            )
        assert (
            service.store.operation(ctx["operation_key"], ctx["principal"], ctx["run_id"]) is None
        )
        result = {"effect": "applied", "answer": question}
        service.store.receipt(ctx["operation_key"], result)
        return result

    register(service, "test.read", read)
    register(service, "test.write", write, write=True)
    tasks = [
        TaskSpec(id="a", tool="test.read", arguments={"question": "trusted"}),
        TaskSpec(
            id="b",
            tool="test.write",
            arguments={"question": "apply"},
            depends_on=["a"],
            use_dependencies=consume,
        ),
    ]
    run = await service.submit(LOCAL, "bridge-plan", RunInput(tasks=tasks))
    plan = (await wait_run(service, run["run_id"], {"awaiting_review"}))["interrupts"][0]
    assert plan["preview"]["kind"] == "legacy_plan" and observed == []
    assert service.runtime.actions.operations("local", run["run_id"]) == []
    await service.resume(LOCAL, run["run_id"], True)
    return run["run_id"], tasks, plan, observed


async def test_real_structured_tool_retry_has_one_public_policy_and_committed_links(service):
    observed = []

    async def read(question, config: RunnableConfig):
        configured, ctx = config["configurable"], config["configurable"]["tool_context"]
        with service.store.connection() as db:
            public = dict(
                db.execute(
                    "SELECT * FROM agent_tool_operations WHERE key=?",
                    (ctx["public_operation_key"],),
                ).fetchone()
            )
            link = dict(
                db.execute(
                    "SELECT * FROM legacy_action_links WHERE public_key=? AND attempt_token=?",
                    (public["key"], ctx["attempt_token"]),
                ).fetchone()
            )
        assert public["phase"] == "executing"
        assert link["attempt_token"] == public["attempt_token"] == ctx["attempt_token"]
        assert link["attempt"] == public["attempts"]
        assert link["domain_key"] == ctx["operation_key"] == f"{ctx['run_id']}:a"
        assert public["key"] != link["domain_key"]
        assert configured["runtime_run_id"] == ctx["run_id"]
        assert "execute_task" in configured["checkpoint_ns"]
        observed.append((public, link, configured["thread_id"]))
        if len(observed) == 1:
            raise TransientError()
        return {"answer": question}

    register(service, "test.retry", read)
    run = await service.submit(
        LOCAL,
        "retry",
        RunInput(tasks=[TaskSpec(id="a", tool="test.retry", arguments={"question": "q"})]),
    )
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["a"]["value"] == {"answer": "q"}
    assert len(observed) == 2
    first, second = observed[0][0], observed[1][0]
    assert [first["attempts"], second["attempts"]] == [1, 2]
    for field in ("key", "fingerprint", "call_deadline"):
        assert first[field] == second[field]
    assert first["attempt_token"] != second["attempt_token"]
    row = service.store.request("local", run["run_id"])
    assert all(value[2] == row["thread"] for value in observed)
    operations = service.runtime.actions.operations("local", run["run_id"])
    assert (
        len(operations) == 1
        and operations[0]["phase"] == "read_success"
        and operations[0]["attempts"] == 2
    )
    with service.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM legacy_action_links").fetchone()[0] == 2


@pytest.mark.parametrize("consume", [False, True])
async def test_plan_and_dependency_review_are_distinct_and_no_intent_precedes_confirmation(
    service, consume
):
    run_id, tasks, plan, observed = await setup_plan(service, consume=consume)
    if consume:
        view = await native_review(service, run_id, plan["review_id"])
        review = view["interrupts"][0]
        assert review["preview"]["dependencies"] == {"a": {"answer": "trusted"}}
        assert review["preview"]["call"] == {
            "call_id": "b",
            "name": "test.write",
            "arguments": tasks[1].arguments,
        }
        assert [name for name, _ in observed] == ["read"]
        with service.store.connection() as db:
            grant = dict(
                db.execute("SELECT * FROM legacy_plan_grants WHERE run_id=?", (run_id,)).fetchone()
            )
            assert grant["review_id"] == plan["review_id"] and grant["digest"] == plan["digest"]
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM agent_tool_operations WHERE read_only=0"
                ).fetchone()[0]
                == 0
            )
            assert db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0
        actor = await service.runtime_adapter.resolve_identity("local")
        with pytest.raises(AgentError):
            await service.runtime.resume(
                actor, run_id, review_id=plan["review_id"], digest=plan["digest"]
            )
        await service.resume(LOCAL, run_id, True)
    final = await wait_run(service, run_id, {"succeeded"})
    assert [name for name, _ in observed] == ["read", "write"]
    write_context = observed[-1][1]
    assert write_context["dependencies"] == ({"a": final["results"]["a"]} if consume else {})
    with service.store.connection() as db:
        evidence = json.loads(
            db.execute("SELECT evidence FROM legacy_action_links WHERE call_id='b'").fetchone()[0]
        )
        assert evidence["context"]["dependency_results"] == (
            {"a": {"answer": "trusted"}} if consume else {}
        )
        assert db.execute("SELECT COUNT(*) FROM agent_reviews").fetchone()[0] == (
            2 if consume else 1
        )
        assert db.execute("SELECT phase FROM operations").fetchone()[0] == "done"
        assert (
            db.execute("SELECT phase FROM agent_tool_operations WHERE read_only=0").fetchone()[0]
            == "applied"
        )


@pytest.mark.parametrize("tamper", ["missing", "owner", "domain_key", "attempt", "deadline"])
async def test_cache_requires_full_original_link_without_reexecuting(service, tamper):
    calls = []

    async def read(question, config: RunnableConfig):
        calls.append(True)
        return {"answer": question}

    register(service, "test.cache", read)
    task = TaskSpec(id="a", tool="test.cache", arguments={"question": "cached"})
    run = await service.submit(LOCAL, "cached", RunInput(tasks=[task]))
    await wait_run(service, run["run_id"], {"succeeded"})
    row = service.store.request("local", run["run_id"])
    state = service.initial(row)
    state["task"] = task.model_dump()
    config = service.runtime.config(row)
    assert (await service.registry.execute(task, state, {}, config)).status == "succeeded"
    before = service.runtime.actions.operations("local", row["run_id"])
    with service.store.connection() as db:
        if tamper == "missing":
            db.execute("DELETE FROM legacy_action_links")
        elif tamper == "deadline":
            evidence = json.loads(
                db.execute("SELECT evidence FROM legacy_action_links").fetchone()[0]
            )
            evidence["context"]["deadline"] += 1
            db.execute("UPDATE legacy_action_links SET evidence=?", (json.dumps(evidence),))
        else:
            db.execute(
                f"UPDATE legacy_action_links SET {tamper}=?",
                (2 if tamper == "attempt" else "wrong",),
            )
    cached = await service.registry.execute(task, state, {}, config)
    assert cached.status == "failed" and cached.error == "receipt_conflict"
    assert calls == [True] and service.runtime.actions.operations("local", row["run_id"]) == before


@pytest.mark.parametrize("field", ["task_id", "tool", "status", "value"])
async def test_dependency_checkpoint_dto_cannot_replace_sql_success(service, field):
    run_id, tasks, plan, observed = await setup_plan(service)
    view = await native_review(service, run_id, plan["review_id"])
    row = service.store.request("local", run_id)
    dto = copy.deepcopy(view["results"]["a"])
    assert (
        await service.registry.bridge.action_context(row, tasks[1], {"a": dto})
    ).dependency_results == {"a": {"answer": "trusted"}}
    dto[field] = {} if field == "value" else "wrong"
    with pytest.raises(AgentError, match="stale or forged"):
        await service.registry.bridge.action_context(row, tasks[1], {"a": dto})
    assert [name for name, _ in observed] == ["read"]
    assert all(
        operation["read_only"] for operation in service.runtime.actions.operations("local", run_id)
    )


@pytest.mark.parametrize("field", ["kind", "manifest", "owner", "run_id", "binding", "epoch"])
async def test_plan_grant_rechecks_complete_original_preview(service, field):
    run_id, _, plan, observed = await setup_plan(service)
    await native_review(service, run_id, plan["review_id"])
    with service.store.connection() as db:
        review = db.execute(
            "SELECT * FROM agent_reviews WHERE id=?", (plan["review_id"],)
        ).fetchone()
        preview = json.loads(review["preview"])
        preview[field] = 999 if field == "epoch" else "wrong"
        db.execute(
            "UPDATE agent_reviews SET preview=? WHERE id=?", (json.dumps(preview), review["id"])
        )
    assert not service.runtime_adapter.approved(service.store.request("local", run_id))
    assert [name for name, _ in observed] == ["read"]


async def test_claim_hook_failure_rolls_back_domain_intent_and_all_links(service, monkeypatch):
    bridge = service.registry.bridge
    original = bridge.claim

    def failed(db, operation, evidence):
        original(db, operation, evidence)
        if not evidence.tool.read_only:
            raise AgentError("injected_claim", "synthetic transaction failure", 503)

    monkeypatch.setattr(bridge, "claim", failed)
    run_id, _, _, observed = await setup_plan(service, consume=False)
    await wait_run(service, run_id, {"partial"})
    assert [name for name, _ in observed] == ["read"]
    with service.store.connection() as db:
        public = dict(
            db.execute("SELECT * FROM agent_tool_operations WHERE read_only=0").fetchone()
        )
        assert public["phase"] == "pending" and public["attempts"] == 0
        assert public["attempt_token"] is None and public["call_deadline"] is None
        assert (
            db.execute("SELECT COUNT(*) FROM legacy_action_links WHERE call_id='b'").fetchone()[0]
            == 0
        )
        for table in ("operations", "legacy_write_actions"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert (
            db.execute("SELECT COUNT(*) FROM audit WHERE event='operation.intent'").fetchone()[0]
            == 0
        )


async def test_native_query_manifest_conflict_returns_metadata_and_cannot_resume(service):
    run_id, _, plan, observed = await setup_plan(service)
    await native_review(service, run_id, plan["review_id"])
    with service.store.connection() as db:
        db.execute("UPDATE legacy_tool_manifests SET digest='wrong'")
    view = await service.view("local", run_id)
    assert not view["recoverable"] and view["resume_reason"] == "tool_manifest_conflict"
    assert view["interrupts"] == [] and view["results"] == {}
    assert (await service.runtime.view("local", run_id))["result"] is None
    with pytest.raises(LabError):
        await service.resume(LOCAL, run_id, True)
    assert [name for name, _ in observed] == ["read"]


@pytest.mark.parametrize(
    "payload",
    [None, json.dumps({"value": "x" * 262144}), '{"value": NaN}'],
    ids=["missing", "oversized", "nonfinite"],
)
async def test_unpublishable_applied_receipt_blocks_dependency_and_hides_old_result(
    service, payload
):
    tasks = [
        TaskSpec(id="a", tool="demo.write", arguments={"value": "original"}),
        TaskSpec(
            id="b",
            tool="demo.write",
            arguments={"value": "downstream"},
            depends_on=["a"],
            use_dependencies=True,
        ),
    ]
    run = await service.submit(LOCAL, "lost-result", RunInput(tasks=tasks))
    plan = (await wait_run(service, run["run_id"], {"awaiting_review"}))["interrupts"][0]
    await service.resume(LOCAL, run["run_id"], True)
    await native_review(service, run["run_id"], plan["review_id"])
    with service.store.connection() as db:
        db.execute("UPDATE operations SET payload=? WHERE key=?", (payload, f"{run['run_id']}:a"))
    await service.runtime.context(run["run_id"])
    operation = service.runtime.actions.operations("local", run["run_id"])[0]
    assert operation["phase"] == "applied" and operation["effect_applied"] == 1
    assert (
        operation["error"] == "effect_applied_result_unavailable"
        and operation["result"] is not None
    )
    row = service.store.request("local", run["run_id"])
    with service.store.connection() as db, pytest.raises(AgentError):
        service.registry.bridge.dependencies(db, row, tasks[1])
    view = await service.view("local", run["run_id"])
    assert not view["recoverable"] and view["results"] == {} and view["interrupts"] == []
    with pytest.raises(LabError):
        await service.resume(LOCAL, run["run_id"], True)
    with service.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM simulation").fetchone()[0] == 1


@pytest.mark.parametrize("phase", ["compensated", "unknown", "unavailable"])
async def test_query_checks_current_receipt_after_last_visibility_await(
    service, monkeypatch, phase
):
    run = await service.submit(
        LOCAL,
        "query-race",
        RunInput(tasks=[TaskSpec(id="a", tool="demo.write", arguments={"value": "old"})]),
    )
    await wait_run(service, run["run_id"], {"awaiting_review"})
    await service.resume(LOCAL, run["run_id"], True)
    await wait_run(service, run["run_id"], {"succeeded"})
    entered, released = asyncio.Event(), asyncio.Event()
    original = service.runtime.authorized_sources
    calls = 0

    async def gate(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            await released.wait()
        return await original(*args)

    monkeypatch.setattr(service.runtime, "authorized_sources", gate)
    pending = asyncio.create_task(service.runtime.view("local", run["run_id"]))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with service.store.connection() as db:
            if phase == "unavailable":
                db.execute("UPDATE operations SET payload=NULL")
            else:
                db.execute("UPDATE operations SET phase=?", (phase,))
        service.runtime._reconcile_actions(service.store.request("local", run["run_id"]))
    finally:
        released.set()
    view = await asyncio.wait_for(pending, 3)
    assert view["result"] is None and "results" not in view and not view["recoverable"]
    assert view["resume_reason"] == "result_unavailable"


async def test_new_release_cannot_lose_protocol_and_fall_back_to_legacy_final_guard(service):
    run = await service.submit(
        LOCAL, "missing-proof", RunInput(message="RAG", pause_for_review=True)
    )
    await wait_run(service, run["run_id"], {"awaiting_review"})
    row = service.store.request("local", run["run_id"])
    assert service.runtime_adapter.requires_bridge(row)
    old = {**row, "binding": service.runtime_adapter.legacy_query_binding}
    assert not service.runtime_adapter.requires_bridge(old)
    committed = []
    projection = FinalProjection(
        value={"synthetic": True},
        local_guard=lambda db: service.runtime_adapter.results_guard(db, row, {}),
        extra_commit=lambda db: committed.append(True),
    )
    with service.store.connection() as db:
        db.execute("DELETE FROM legacy_execution_protocol WHERE run_id=?", (row["run_id"],))
    assert not service.runtime.store.commit_final(row, projection)
    assert service.runtime.store.final_receipt("local", row["run_id"]) is None and committed == []
