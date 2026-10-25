"""Additional admission fault/race cases using only the tracked synthetic host."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agentbase.contracts import AgentError
from agentbase.intent_execution import AGENT
from agentbase.intent_proposals import IntentConfirmation
from agentbase.runtime_contracts import AdmissionFence, PreparedAdmission, RunOptions
from tests.test_intent_synthetic_execution import BASE, completed, counts, host, proposal


async def test_expiry_crossing_at_atomic_admission_leaves_no_run(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, record = await proposal(session)
        valid = record["expires_at"] - 0.01
        times = iter([valid, valid, valid, valid, record["expires_at"]])
        app.state.intents.clock = lambda: next(times)
        response = await session.post(uri + "/execute", json=body)
        app.state.intents.clock = time.time
        assert response.status_code == 410
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0


async def test_epoch_change_and_cross_conversation_ids_are_rejected(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, record = await proposal(session)
        other = await session.post(BASE, json={"conversation_id": "other"})
        swapped = BASE + "/" + other.json()["session_id"] + "/proposals/" + record["proposal_id"]
        assert (await session.post(swapped + "/execute", json=body)).status_code == 404
        assert (await session.get(swapped + "/execution")).status_code == 404
        app.state.engine.contexts.history.switch("alice", AGENT, "case")
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert counts(app)["requests"] == 0


async def test_even_a_declared_read_only_foreign_handler_is_not_executable(tmp_path, monkeypatch):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        invoked = []

        async def foreign(arguments, context):
            invoked.append(True)
            return {"doubled": 0}

        monkeypatch.setattr(app.state.engine.registry, "handler", lambda *args, **kwargs: foreign)
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert not invoked and counts(app)["requests"] == 0


@pytest.mark.parametrize("action", ["propose_cancel", "propose_replace"])
async def test_control_proposals_cannot_change_an_existing_run(tmp_path, action):
    async with host(tmp_path) as (session, app, _):
        engine = app.state.engine
        actor = await engine.identity("alice")
        bound = engine.registry.agent(AGENT)
        options = RunOptions(conversation_id="case")
        target, _, _ = engine.store.admit(
            actor,
            AGENT,
            "target",
            "synthetic-target",
            {"input": {"number": 1}},
            bound.digest,
            options,
            time.time() + 30,
            "pause-review",
        )
        draft = {
            "text": "为指定目标生成控制提案",
            "action": action,
            "target_run_id": target["run_id"],
        }
        if action == "propose_replace":
            draft["task"] = {"input": {"number": 9}}
        _, uri, body, _ = await proposal(session, draft=draft)
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert engine.store.request("alice", target["run_id"])["cancel"] == 0
        assert counts(app)["requests"] == 1 and counts(app)["agent_intent_executions"] == 0


@pytest.mark.parametrize("failure", ["async", "return_value"])
async def test_prepared_admission_contract_failures_do_not_consume_intent(tmp_path, failure):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        engine = app.state.engine
        bound = engine.registry.agent(AGENT)
        workflow = engine.binding({"agent_id": AGENT, "binding": bound.digest})

        async def async_commit(db, row):
            return None

        async def prepare(identity, value):
            callback = async_commit if failure == "async" else lambda db, row: True
            return PreparedAdmission({"input": value}, callback)

        workflow.prepare = prepare
        assert (await session.post(uri + "/execute", json=body)).status_code == 503
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0


async def test_generic_fence_cannot_treat_false_as_reauthorization(tmp_path):
    async with host(tmp_path) as (_, app, _):

        async def denied():
            return False

        fence = AdmissionFence(denied, lambda db, row: None, lambda db, row: None)
        with pytest.raises(AgentError) as error:
            await app.state.engine.submit(
                await app.state.engine.identity("alice"),
                AGENT,
                "denied",
                {"number": 1},
                admission_fence=fence,
            )
        assert error.value.code == "admission_contract" and counts(app)["requests"] == 0


async def test_existing_unmapped_request_key_is_never_adopted(tmp_path):
    async with host(tmp_path) as (session, app, _):
        path, uri, body, record = await proposal(session)
        engine, executor = app.state.engine, app.state.intent_execution
        actor = await engine.identity("alice")
        bound = engine.registry.agent(AGENT)
        binding = engine.binding({"agent_id": AGENT, "binding": bound.digest})
        options = RunOptions(conversation_id="case")
        value = body["draft"]["task"]["input"]
        key = executor.request_key(path.rsplit("/", 1)[1], record["proposal_id"])
        engine.store.admit(
            actor,
            AGENT,
            key,
            engine.client_request_digest(binding, value, options),
            await binding.prepare(actor, value),
            bound.digest,
            options,
            time.time() + 10,
            "pause-review",
        )
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert counts(app)["requests"] == 1 and counts(app)["agent_intent_executions"] == 0


async def test_checkpoint_failure_retains_one_recoverable_admission(tmp_path, monkeypatch):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        engine = app.state.engine

        async def unavailable(*args, **kwargs):
            engine.available = False
            raise AgentError("checkpoint_unavailable", "synthetic outage", 503)

        with monkeypatch.context() as patch:
            patch.setattr(engine, "update_checkpoint", unavailable)
            assert (await session.post(uri + "/execute", json=body)).status_code == 503
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 1
        assert counts(app)["agent_tool_operations"] == 0
        engine.available = True
        queried = await session.get(uri + "/execution")
        assert (
            queried.status_code == 200
            and queried.json()["run"]["status"] == "initialization_incomplete"
        )
        again = await session.post(uri + "/execute", json=body)
        assert (
            again.status_code == 202
            and again.json()["run"]["run_id"] == queried.json()["run"]["run_id"]
        )
        assert counts(app)["requests"] == 1 and counts(app)["agent_tool_operations"] == 0


async def test_sqlite_connection_race_keeps_one_atomic_mapping(tmp_path):
    async with host(tmp_path) as (session, app, _):
        path, _, body, record = await proposal(session)
        engine, executor = app.state.engine, app.state.intent_execution
        actor = await engine.identity("alice")
        bound = engine.registry.agent(AGENT)
        options = RunOptions(conversation_id="case")
        binding = engine.binding({"agent_id": AGENT, "binding": bound.digest})
        with engine.store.connection() as db:
            credential = db.execute(
                "SELECT credential_digest FROM agent_intent_sessions"
            ).fetchone()[0]
        session_id = path.rsplit("/", 1)[1]
        choice = IntentConfirmation.model_validate(body)
        key = executor.request_key(session_id, record["proposal_id"])
        payload = await binding.prepare(actor, choice.draft.task.input)
        barrier = Barrier(2)

        def check(db, row, created):
            executor._admission(
                db,
                row,
                created=created,
                identity=actor,
                agent_id=AGENT,
                session_id=session_id,
                proposal_id=record["proposal_id"],
                credential=credential,
                body=choice,
            )

        def admit(_):
            barrier.wait(timeout=5)
            row, created, _ = engine.store.admit(
                actor,
                AGENT,
                key,
                engine.client_request_digest(binding, choice.draft.task.input, options),
                payload,
                bound.digest,
                options,
                time.time() + 10,
                "pause-review",
                admission_commit=lambda db, row: check(db, row, True),
                admission_replay=lambda db, row: check(db, row, False),
            )
            return row["run_id"], created

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(admit, range(2)))
        assert len({run for run, _ in results}) == 1 and sorted(
            created for _, created in results
        ) == [False, True]
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 1
        assert counts(app)["agent_tool_operations"] == 0


async def test_query_rechecks_authority_after_waiting_for_runtime_view(tmp_path, monkeypatch):
    allowed = [True]
    async with host(tmp_path, allowed=allowed) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        await session.post(uri + "/execute", json=body)
        await completed(session, uri)
        original = app.state.engine.view

        async def revoke_after_view(*args, **kwargs):
            result = await original(*args, **kwargs)
            allowed[0] = False
            return result

        monkeypatch.setattr(app.state.engine, "view", revoke_after_view)
        assert (await session.get(uri + "/execution")).status_code == 403


async def test_cancelled_http_waiter_after_admission_does_not_create_another_run(
    tmp_path, monkeypatch
):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        engine = app.state.engine
        entered, release = asyncio.Event(), asyncio.Event()
        original = engine.update_checkpoint

        async def wait_before_checkpoint(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(engine, "update_checkpoint", wait_before_checkpoint)
            task = asyncio.create_task(session.post(uri + "/execute", json=body))
            try:
                async with asyncio.timeout(5):
                    await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 1
        queried = await session.get(uri + "/execution")
        assert (
            queried.status_code == 200
            and queried.json()["run"]["status"] == "initialization_incomplete"
        )
        retried = await session.post(uri + "/execute", json=body)
        assert (
            retried.status_code == 202
            and retried.json()["run"]["run_id"] == queried.json()["run"]["run_id"]
        )
        assert counts(app)["requests"] == 1 and counts(app)["agent_tool_operations"] == 0


async def test_prepared_metadata_cannot_mask_a_different_stored_input(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        engine = app.state.engine
        bound = engine.registry.agent(AGENT)
        workflow = engine.binding({"agent_id": AGENT, "binding": bound.digest})

        async def prepare(identity, value):
            def mask_snapshot(db, row):
                row["payload"] = json.dumps({"input": value})

            return PreparedAdmission({"input": {"number": 99}}, mask_snapshot)

        workflow.prepare = prepare
        response = await session.post(uri + "/execute", json=body)
        assert response.status_code == 409
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0


@pytest.mark.parametrize("location", ["prepare", "commit"])
async def test_derived_task_arguments_must_match_confirmed_input(tmp_path, location):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        engine = app.state.engine
        bound = engine.registry.agent(AGENT)
        workflow = engine.binding({"agent_id": AGENT, "binding": bound.digest})
        original = workflow.prepare

        async def wrong_arguments(identity, value):
            payload = await original(identity, value)
            if location == "prepare":
                payload["task_arguments"]["double"]["number"] = 99
                return payload

            def change_persisted_arguments(db, row):
                stored = json.loads(
                    db.execute(
                        "SELECT payload FROM requests WHERE run_id=?", (row["run_id"],)
                    ).fetchone()[0]
                )
                assert stored["input"] == {"number": 7}
                stored["task_arguments"]["double"]["number"] = 99
                db.execute(
                    "UPDATE requests SET payload=? WHERE run_id=?",
                    (json.dumps(stored), row["run_id"]),
                )

            return PreparedAdmission(payload, change_persisted_arguments)

        workflow.prepare = wrong_arguments
        response = await session.post(uri + "/execute", json=body)
        if response.status_code == 202:
            result = await completed(session, uri)
            print(
                json.dumps(
                    {
                        "confirmed_input": body["draft"]["task"]["input"],
                        "actual_result": result["run"]["result"],
                        "counts": counts(app),
                    }
                )
            )
        assert response.status_code == 409
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0
        assert (await session.get(uri)).json()["status"] == "confirmed"


async def test_query_and_replay_reject_only_derived_argument_drift(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        assert (await session.post(uri + "/execute", json=body)).status_code == 202
        await completed(session, uri)
        with app.state.engine.store.connection() as db:
            row = db.execute("SELECT run_id,payload FROM requests").fetchone()
            payload = json.loads(row["payload"])
            assert payload["input"] == {"number": 7}
            payload["task_arguments"]["double"]["number"] = 99
            db.execute(
                "UPDATE requests SET payload=? WHERE run_id=?", (json.dumps(payload), row["run_id"])
            )
        queried = await session.get(uri + "/execution")
        assert queried.status_code == 503
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert (
            counts(app)["requests"]
            == counts(app)["agent_intent_executions"]
            == counts(app)["agent_tool_operations"]
            == 1
        )
