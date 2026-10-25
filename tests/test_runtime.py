import asyncio
import dataclasses
import time

import pytest
from langchain_core.tools import StructuredTool

from raglab.errors import LabError, UnknownEffect
from raglab.models import Principal, QueryInput, RunInput, TaskSpec
from raglab.service import LabService
from raglab.tools import ReadArgs, ToolSpec
from tests.conftest import LOCAL, wait_run


def register(service, name, function):
    service.registry.register(
        ToolSpec(
            name,
            StructuredTool.from_function(
                coroutine=function,
                name=name,
                description="synthetic test tool",
                args_schema=ReadArgs,
            ),
        )
    )


async def test_paused_history_revoked_before_resume(service):
    first = await service.answer(
        LOCAL, QueryInput(question="Orion 如何备份？", conversation_id="revoked-history")
    )
    run = await service.submit(
        LOCAL,
        "history-review",
        RunInput(
            message="它的超时是多少？", conversation_id="revoked-history", pause_for_review=True
        ),
    )
    await wait_run(service, run["run_id"])
    for cite in first["citations"]:
        service.store.revoke("local", cite["document_id"])
    await service.resume(LOCAL, run["run_id"], approve=True)
    final = await wait_run(service, run["run_id"])
    value = final["results"]["q1"]["value"]
    assert value["status"] == "clarify" and value["resolved_question"] == "它的超时是多少？"
    assert "Orion" not in str(value)


async def test_concurrent_idempotency_and_upgrade_retransmission(service):
    request = RunInput(message="RAG 为什么分块？", switch_topic=True, conversation_id="switch")
    runs = await asyncio.gather(*[service.submit(LOCAL, "same", request) for _ in range(10)])
    assert len({r["run_id"] for r in runs}) == 1
    assert service.store.epoch("local", "switch") == 1
    run = await wait_run(service, runs[0]["run_id"])
    assert run["status"] == "succeeded"
    service.registry.specs["rag.answer"] = dataclasses.replace(
        service.registry.specs["rag.answer"], version="3"
    )
    assert (await service.submit(LOCAL, "same", request))["run_id"] == run["run_id"]
    with pytest.raises(LabError, match="different"):
        await service.submit(LOCAL, "same", RunInput(message="different"))


async def test_principal_and_thread_authorization(service):
    one = await service.submit(
        Principal(owner="alice"), "shared", RunInput(message="RAG 为什么分块？")
    )
    two = await service.submit(
        Principal(owner="bob"), "shared", RunInput(message="RAG 为什么分块？")
    )
    assert one["run_id"] != two["run_id"]
    for operation in (service.view, service.cancel):
        with pytest.raises(LabError, match="not found"):
            await operation("bob", one["run_id"])
    with pytest.raises(LabError, match="not found"):
        await service.resume(Principal(owner="bob"), one["run_id"], True)


async def test_pause_resume_and_terminal_cancel(service):
    run = await service.submit(
        LOCAL, "review", RunInput(message="RAG 为什么分块？", pause_for_review=True)
    )
    paused = await wait_run(service, run["run_id"], {"awaiting_review"})
    assert paused["interrupts"] and not paused["results"]
    with pytest.raises(LabError, match="approve"):
        await service.resume(LOCAL, run["run_id"])
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["q1"]["status"] == "succeeded"
    assert (await service.cancel("local", run["run_id"]))["status"] == "succeeded"
    with pytest.raises(LabError):
        await service.resume(LOCAL, run["run_id"], True)


async def test_dependency_structured_result_and_nested_checkpoint(service):
    tasks = [
        TaskSpec(id="a", tool="rag.answer", arguments={"question": "Orion 服务如何备份？"}),
        TaskSpec(
            id="b",
            tool="rag.answer",
            arguments={"question": "它的超时时间是多少？"},
            depends_on=["a"],
            use_dependencies=True,
        ),
    ]
    run = await service.submit(LOCAL, "dag", RunInput(tasks=tasks))
    final = await wait_run(service, run["run_id"])
    assert final["status"] == "succeeded", final
    assert "8 秒" in final["results"]["b"]["value"]["answer"]
    # Check real nested namespaces; an imported-only subgraph would not create these.
    import sqlite3

    with sqlite3.connect(service.settings.data_dir / "checkpoints.sqlite") as db:
        rows = db.execute(
            "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id=?",
            (final["thread_id"],),
        ).fetchall()
    assert any("execute_task" in row[0] for row in rows), rows


async def test_parallel_barrier_and_partial_results(service):
    entered, release = asyncio.Event(), asyncio.Event()
    calls, in_flight, maximum = [], 0, 0

    async def tool(question):
        nonlocal in_flight, maximum
        calls.append(question)
        in_flight += 1
        maximum = max(maximum, in_flight)
        try:
            if len(calls) >= 2:
                entered.set()
            await release.wait()
            if question == "fail":
                raise LabError("permanent", "synthetic", 409)
            return {"question": question}
        finally:
            in_flight -= 1

    register(service, "test.parallel", tool)
    tasks = [
        TaskSpec(id="a", tool="test.parallel", arguments={"question": "ok"}),
        TaskSpec(id="b", tool="test.parallel", arguments={"question": "fail"}),
        TaskSpec(
            id="c", tool="test.parallel", arguments={"question": "dependent"}, depends_on=["a", "b"]
        ),
    ]
    run = await service.submit(LOCAL, "parallel", RunInput(tasks=tasks))
    await asyncio.wait_for(entered.wait(), 2)
    assert "dependent" not in calls
    release.set()
    final = await wait_run(service, run["run_id"])
    assert maximum == 2 <= service.settings.tools_concurrency
    assert final["status"] == "partial"
    assert final["results"]["c"]["status"] == "skipped"
    assert "dependent" not in calls


async def test_cancel_preserves_completed_parallel_pending_write(service):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def fast(question):
        return {"completed": question}

    async def slow(question):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    register(service, "test.fast", fast)
    register(service, "test.slow", slow)
    run = await service.submit(
        LOCAL,
        "cancel",
        RunInput(
            tasks=[
                TaskSpec(id="a", tool="test.fast", arguments={"question": "ok"}),
                TaskSpec(id="b", tool="test.slow", arguments={"question": "wait"}),
            ]
        ),
    )
    await asyncio.wait_for(entered.wait(), 2)
    # Wait for the real checkpointer's pending write of A, not a timing guess.
    row = service.store.request("local", run["run_id"])
    saver = service.agent.graph.checkpointer
    for _ in range(100):
        checkpoint = await saver.aget_tuple(service.config(row))
        if any(
            channel == "results" and "a" in value for _, channel, value in checkpoint.pending_writes
        ):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("no completed pending write")
    await service.cancel("local", run["run_id"])
    final = await wait_run(service, run["run_id"], {"cancelled"})
    assert cancelled.is_set()
    assert final["results"]["a"]["value"] == {"completed": "ok"}, final
    assert "b" not in final["results"]


async def test_restart_checkpoint_authority_and_deadline(service, monkeypatch):
    # Model one second of downtime without relying on SQLite/scheduler wall time.
    # asyncio still uses its real monotonic clock and enforces the remaining timeout.
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    entered = asyncio.Event()

    async def waiting(question):
        entered.set()
        await asyncio.Event().wait()

    register(service, "test.wait", waiting)
    run = await service.submit(
        LOCAL,
        "restart",
        RunInput(
            tasks=[TaskSpec(id="a", tool="test.wait", arguments={"question": "wait"})],
            timeout_seconds=5,
        ),
    )
    assert run["deadline"] == clock[0] + 5
    await asyncio.wait_for(entered.wait(), 2)
    await service.close()
    clock[0] += 1
    restarted = await LabService(service.settings).start()
    try:

        async def immediate(question):
            return {"answer": "resumed"}

        register(restarted, "test.wait", immediate)
        interrupted = await restarted.view("local", run["run_id"])
        assert interrupted["status"] == "interrupted"
        assert interrupted["deadline"] == run["deadline"]
        resumed = await restarted.resume(LOCAL, run["run_id"])
        assert resumed["deadline"] == run["deadline"]
        final = await wait_run(restarted, run["run_id"])
        assert final["status"] == "succeeded"
        assert final["deadline"] == run["deadline"]
        # Requests table has no execution status and cannot fake success.
        with restarted.store.connection() as db:
            assert "status" not in [r[1] for r in db.execute("PRAGMA table_info(requests)")]
            assert (
                db.execute(
                    "SELECT deadline FROM requests WHERE run_id=?", (run["run_id"],)
                ).fetchone()[0]
                == run["deadline"]
            )
            assert (
                db.execute(
                    "SELECT original_deadline FROM agent_run_controls WHERE run_id=?",
                    (run["run_id"],),
                ).fetchone()[0]
                == run["deadline"]
            )
    finally:
        await restarted.close()


async def test_expired_resume_and_version_mismatch(service, monkeypatch):
    run = await service.submit(
        LOCAL, "expire", RunInput(message="RAG", pause_for_review=True, timeout_seconds=30)
    )
    await wait_run(service, run["run_id"], {"awaiting_review"})
    # Verify expiration independently of scheduler/SQLite latency while reaching the interrupt.
    with monkeypatch.context() as clocks:
        clocks.setattr("time.time", lambda: run["deadline"] + 0.01)
        view = await service.view("local", run["run_id"])
        assert not view["recoverable"] and view["resume_reason"] == "original_deadline_expired"
        assert view["deadline"] == run["deadline"]
        with pytest.raises(LabError):
            await service.resume(LOCAL, run["run_id"], True)
    other = await service.submit(LOCAL, "version", RunInput(message="RAG", pause_for_review=True))
    await wait_run(service, other["run_id"], {"awaiting_review"})
    service.settings.graph_version = "agent-v3"
    assert (await service.view("local", other["run_id"]))[
        "resume_reason"
    ] == "checkpoint_version_mismatch"


async def test_admission_lock_wait_consumes_budget(service):
    await service.admission_lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            await service.submit(LOCAL, "deadline", RunInput(message="RAG", timeout_seconds=0.1))
        assert not service.store.requests()
    finally:
        service.admission_lock.release()


async def test_topic_switch_late_result_and_history(service):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def old(question):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    register(service, "test.old", old)
    one = await service.submit(
        LOCAL,
        "old",
        RunInput(
            tasks=[TaskSpec(id="a", tool="test.old", arguments={"question": "Orion"})],
            conversation_id="topic",
        ),
    )
    await entered.wait()
    new = await service.submit(
        LOCAL,
        "new",
        RunInput(message="Atlas 服务超时时间是多少？", conversation_id="topic", switch_topic=True),
    )
    assert new["topic_epoch"] == 1
    final = await wait_run(service, new["run_id"])
    assert final["status"] == "succeeded" and cancelled.is_set()
    assert (await wait_run(service, one["run_id"], {"cancelled"}))["status"] == "cancelled"
    assert all(
        "Atlas" in h["question"] for h in service.store.history("local", "topic", 1, 4, 3000)
    )


async def test_unknown_write_forbids_resume_and_compensation(service):
    request = RunInput(
        tasks=[TaskSpec(id="write", tool="demo.write", arguments={"failure": "unknown"})]
    )
    run = await service.submit(LOCAL, "unknown", request)
    await wait_run(service, run["run_id"], {"awaiting_review"})
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"needs_reconciliation"})
    assert not final["recoverable"]
    assert service.store.operations("local", run["run_id"])[0]["phase"] == "intent"
    with pytest.raises(LabError):
        await service.resume(LOCAL, run["run_id"])
    with pytest.raises(LabError):
        service.compensate("local", run["run_id"])


async def test_receipt_failure_semantics_and_compensation_stop(service):
    key = "synthetic:key"
    service.store.operation(key, "local", "synthetic")
    service.store.receipt(key, {"effect": "not_applied"}, "not_applied")
    with pytest.raises(LabError, match="did not apply"):
        service.store.operation(key, "local", "synthetic")
    run = await service.submit(
        LOCAL,
        "compensate",
        RunInput(
            tasks=[
                TaskSpec(id="a", tool="demo.write", arguments={"value": "one"}),
                TaskSpec(id="b", tool="demo.write", arguments={"value": "two"}, depends_on=["a"]),
            ]
        ),
    )
    await wait_run(service, run["run_id"], {"awaiting_review"})
    await service.resume(LOCAL, run["run_id"], True)
    await wait_run(service, run["run_id"], {"succeeded"})
    with pytest.raises(UnknownEffect):
        service.compensate("local", run["run_id"], inject_unknown=True)
    operations = service.store.operations("local", run["run_id"])
    assert {x["key"].split(":")[-1]: x["phase"] for x in operations} == {
        "a": "done",
        "b": "compensating",
    }
    assert (await service.view("local", run["run_id"]))["status"] == "needs_reconciliation"
    with pytest.raises(LabError):
        service.compensate("local", run["run_id"])


async def test_cancel_storage_failure_stops_admission(service, monkeypatch):
    entered = asyncio.Event()

    async def slow(question):
        entered.set()
        await asyncio.Event().wait()

    register(service, "test.slow", slow)
    run = await service.submit(
        LOCAL,
        "cancel-store",
        RunInput(tasks=[TaskSpec(id="a", tool="test.slow", arguments={"question": "wait"})]),
    )
    await entered.wait()

    async def fail(*args):
        raise LabError("checkpoint_unavailable", "synthetic", 503)

    monkeypatch.setattr(service, "terminal_checkpoint", fail)
    with pytest.raises(LabError, match="cleanup"):
        await service.cancel("local", run["run_id"])
    assert not service.available
    with pytest.raises(LabError, match="unavailable"):
        await service.submit(LOCAL, "later", RunInput(message="RAG"))


async def test_queued_cancel_and_backpressure(service):
    for task in service.workers:
        task.cancel()
    await asyncio.gather(*service.workers, return_exceptions=True)
    runs = []
    for i in range(service.settings.queue_size):
        runs.append(await service.submit(LOCAL, f"queue{i}", RunInput(message="RAG")))
    with pytest.raises(LabError, match="queue is full"):
        await service.submit(LOCAL, "overflow", RunInput(message="RAG"))
    # Duplicate is accepted even while full.
    assert (await service.submit(LOCAL, "queue0", RunInput(message="RAG")))["run_id"] == runs[0][
        "run_id"
    ]
    await service.cancel("local", runs[0]["run_id"])
    service.workers = [asyncio.create_task(service.worker())]
    cancelled = await wait_run(service, runs[0]["run_id"], {"cancelled"})
    assert not cancelled["results"]
