"""Authenticated proposal choices never execute or approve business operations."""

import asyncio
import copy
import json
import time
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest

from agentbase.application import create_agent_app
from agentbase.contracts import AgentError
from agentbase.intent_proposals import (
    IntentConfirmation,
    IntentProposalLimits,
    IntentProposalService,
    OfflineStructuredIntent,
)
from agentbase.runtime_contracts import RunOptions
from raglab.config import Settings
from tests.conftest import ROOT
from tests.test_agentbase_http import client, final
from tests.test_agentbase_intent_runtime import runtime

BASE = "/api/v1/agents/calculation/intent-sessions"
DRAFT = {
    "text": "创建一个数字计算提案",
    "action": "propose_task",
    "task": {"input": {"left": 1, "right": 2}},
    "target_run_id": None,
}


def business(app):
    with app.state.service.runtime.store.connection() as db:
        return {
            name: [tuple(r) for r in db.execute(f"SELECT * FROM {name}")]
            for name in ("requests", "agent_tool_operations", "agent_reviews", "agent_memories")
        }


async def opened(session, headers=None, conversation="intent"):
    response = await session.post(BASE, json={"conversation_id": conversation}, headers=headers)
    assert response.status_code == 201, response.text
    return BASE + "/" + response.json()["session_id"]


async def proposed(session, path, draft=None, headers=None):
    state = (await session.get(path, headers=headers)).json()
    response = await session.post(
        path + "/proposals",
        json={"expected_revision": state["revision"], "draft": draft or DRAFT},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    data = response.json()
    return path + "/proposals/" + data["proposal_id"], data


def confirmation(record):
    return {key: record[key] for key in ("revision", "request_digest", "decision_digest")} | {
        "draft": copy.deepcopy(record["payload"]["draft"])
    }


async def test_proposal_confirmation_and_withdrawal_have_no_business_effects(tmp_path):
    async with client(tmp_path) as (session, app, fixture):
        before = business(app)
        path = await opened(session)
        uri, record = await proposed(session, path)
        fetched = await session.get(uri)
        assert fetched.headers["cache-control"] == "no-store"
        assert fetched.headers["vary"] == "Authorization, Cookie"
        assert record["status"] == "pending" and not record["execution_connected"]
        result = await session.post(uri + "/confirm", json=confirmation(record))
        assert result.status_code == 200 and result.json()["status"] == "confirmed"
        assert result.json()["confirmed_at"] and result.json()["tool_approval"] is False
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 409
        result = await session.post(uri + "/revoke", json=confirmation(record))
        assert result.status_code == 200 and result.json()["status"] == "revoked"
        assert result.json()["confirmed_at"] and result.json()["revoked_at"]
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 409
        assert business(app) == before and fixture.calls == 0


@pytest.mark.parametrize("changed", ["revision", "request_digest", "decision_digest", "draft"])
async def test_confirmation_rejects_any_parameter_change(tmp_path, changed):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        body = confirmation(record)
        if changed == "revision":
            body[changed] += 1
        elif changed == "draft":
            body[changed]["task"]["input"]["left"] = 99
        else:
            body[changed] = "sha256:" + "0" * 64
        before = business(app)
        assert (await session.post(uri + "/confirm", json=body)).status_code == 409
        assert (await session.get(uri)).json()["status"] == "pending"
        assert business(app) == before


@pytest.mark.parametrize(
    "extra",
    [
        {"owner": "admin"},
        {"scope": "write"},
        {"approval": True},
        {"epoch": 999},
        {"origin": "authenticated-user"},
    ],
)
async def test_client_control_fields_cannot_grant_authority(tmp_path, extra):
    async with client(tmp_path) as (session, app, _):
        path = await opened(session)
        assert (
            await session.post(
                path + "/proposals", json={"expected_revision": 0, "draft": DRAFT | extra}
            )
        ).status_code == 422
        assert all(not rows for rows in business(app).values())


async def test_new_requirement_and_explicit_invalidation_reject_old_confirmation(tmp_path):
    async with client(tmp_path) as (session, app, _):
        path = await opened(session)
        old_uri, old = await proposed(session, path)
        new_uri, new = await proposed(session, path, DRAFT | {"text": "另一项需求"})
        assert (await session.post(old_uri + "/confirm", json=confirmation(old))).status_code == 409
        assert (await session.get(old_uri)).json()["status"] == "superseded"
        assert (await session.post(path + "/invalidate")).status_code == 200
        assert (await session.post(new_uri + "/confirm", json=confirmation(new))).status_code == 409
        assert all(not rows for rows in business(app).values())


async def test_expiry_is_server_enforced_and_does_not_create_execution(tmp_path):
    async with client(tmp_path) as (session, app, _):
        now = [time.time()]
        app.state.intents.clock = lambda: now[0]
        uri, record = await proposed(session, await opened(session))
        now[0] = record["expires_at"] + 0.01
        assert (await session.get(uri)).json()["status"] == "expired"
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 410
        assert all(not rows for rows in business(app).values())


async def test_confirmation_is_atomic_under_duplicate_clicks(tmp_path):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        responses = await asyncio.gather(
            *[session.post(uri + "/confirm", json=confirmation(record)) for _ in range(2)]
        )
        assert sorted(r.status_code for r in responses) == [200, 409]
        assert all(not rows for rows in business(app).values())


async def test_cross_session_and_cross_conversation_proposal_ids_are_rejected(tmp_path):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        for other in (await opened(session), await opened(session, conversation="other")):
            swapped = other + "/proposals/" + record["proposal_id"]
            assert (await session.get(swapped)).status_code == 404
            assert (
                await session.post(swapped + "/confirm", json=confirmation(record))
            ).status_code == 404
        assert all(not rows for rows in business(app).values())


async def test_authenticated_owner_and_credential_are_both_bound(tmp_path):
    a, a2, b = (
        "synthetic-token-alice-one-long",
        "synthetic-token-alice-two-long",
        "synthetic-token-bob-one-long",
    )
    settings = Settings(
        data_dir=tmp_path,
        knowledge_root=ROOT / "data/knowledge",
        auth_mode="bearer",
        bearer_principals={a: "alice", a2: "alice", b: "bob"},
    )
    async with client(tmp_path, settings=settings) as (session, app, _):
        headers = {"authorization": "Bearer " + a}
        path = await opened(session, headers)
        uri, record = await proposed(session, path, headers=headers)
        for token, status in ((b, 404), (a2, 403)):
            response = await session.post(
                uri + "/confirm",
                json=confirmation(record),
                headers={"authorization": "Bearer " + token},
            )
            assert response.status_code == status
        with app.state.service.runtime.store.connection() as db:
            text = str([tuple(r) for r in db.execute("SELECT * FROM agent_intent_sessions")])
        assert all(token not in text for token in (a, a2, b))
        assert all(not rows for rows in business(app).values())


@pytest.mark.parametrize("text", ["不要执行新任务，先解释", "Don't run a new task"])
async def test_negation_clarifies_and_cannot_be_confirmed(tmp_path, text):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session), DRAFT | {"text": text})
        assert record["status"] == "clarify"
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 409
        assert all(not rows for rows in business(app).values())


async def test_service_reconstruction_keeps_consumed_confirmation(tmp_path):
    async with client(tmp_path) as (session, app, _):
        path = await opened(session)
        uri, record = await proposed(session, path)
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 200
        service = IntentProposalService(app.state.service.runtime, OfflineStructuredIntent())
        service.install()
        with app.state.service.runtime.store.connection() as db:
            row = dict(db.execute("SELECT * FROM agent_intent_sessions").fetchone())
        actor = await app.state.service.runtime.identity("local")
        restored = service.view(
            actor, "calculation", row["id"], record["proposal_id"], row["credential_digest"]
        )
        assert restored["status"] == "confirmed" and restored["confirmed_at"]
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 409


async def test_retention_limits_fail_closed(tmp_path):
    async with client(
        tmp_path, intent_limits=IntentProposalLimits(max_sessions=1, max_proposals=1)
    ) as (session, app, _):
        path = await opened(session)
        assert (await session.post(BASE, json={"conversation_id": "other"})).status_code == 429
        await proposed(session, path)
        assert (
            await session.post(path + "/proposals", json={"expected_revision": 1, "draft": DRAFT})
        ).status_code == 429
        assert all(not rows for rows in business(app).values())


async def test_out_of_order_old_request_cannot_follow_invalidation(tmp_path):
    async with client(tmp_path) as (session, app, _):
        path = await opened(session)
        assert (await session.post(path + "/invalidate")).status_code == 200
        response = await session.post(
            path + "/proposals", json={"expected_revision": 0, "draft": DRAFT}
        )
        assert response.status_code == 409
        with app.state.service.runtime.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM agent_intent_proposals").fetchone()[0] == 0
        assert all(not rows for rows in business(app).values())


async def test_reopening_same_conversation_retires_old_session(tmp_path):
    async with client(tmp_path) as (session, _, _):
        uri, record = await proposed(session, await opened(session))
        await opened(session)
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 410


class GatedInterpreter(OfflineStructuredIntent):
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def propose(self, request, draft):
        self.entered.set()
        await self.release.wait()
        return await super().propose(request, draft)


async def test_edit_during_interpretation_rejects_late_response(tmp_path):
    interpreter = GatedInterpreter()
    async with client(tmp_path, intent_interpreter=interpreter) as (session, app, _):
        path = await opened(session)
        task = asyncio.create_task(
            session.post(path + "/proposals", json={"expected_revision": 0, "draft": DRAFT})
        )
        try:
            async with asyncio.timeout(5):
                await interpreter.entered.wait()
            assert (await session.post(path + "/invalidate")).status_code == 200
        finally:
            interpreter.release.set()
        assert (await task).status_code == 409
        with app.state.service.runtime.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM agent_intent_proposals").fetchone()[0] == 0
        assert all(not rows for rows in business(app).values())


async def test_agent_acl_revoked_while_interpreting_is_rechecked(tmp_path):
    interpreter, allowed = GatedInterpreter(), [True]

    async def authorize(identity, agent):
        return allowed[0]

    async with client(tmp_path, intent_interpreter=interpreter, authorize_agent=authorize) as (
        session,
        app,
        _,
    ):
        path = await opened(session)
        task = asyncio.create_task(
            session.post(path + "/proposals", json={"expected_revision": 0, "draft": DRAFT})
        )
        try:
            async with asyncio.timeout(5):
                await interpreter.entered.wait()
            allowed[0] = False
        finally:
            interpreter.release.set()
        assert (await task).status_code == 403
        with app.state.service.runtime.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM agent_intent_proposals").fetchone()[0] == 0
        assert all(not rows for rows in business(app).values())


async def test_acl_revocation_blocks_confirming_existing_record(tmp_path):
    allowed = [True]

    async def authorize(identity, agent):
        return allowed[0]

    async with client(tmp_path, authorize_agent=authorize) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        allowed[0] = False
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 403
        assert all(not rows for rows in business(app).values())


@pytest.mark.parametrize("change", ["cancel", "epoch", "input"])
async def test_changed_target_cannot_be_confirmed(tmp_path, change):
    async with client(tmp_path) as (session, app, _):
        engine = app.state.service.runtime
        actor = await engine.identity("local")
        run = await engine.submit(
            actor,
            "calculation",
            "target",
            {"left": 1, "right": 2},
            RunOptions(conversation_id="intent", pause_for_review=True),
        )
        await final(engine, "local", run["run_id"], "awaiting_review")
        draft = {
            "text": "取消这个目标任务",
            "action": "propose_cancel",
            "target_run_id": run["run_id"],
            "task": None,
        }
        uri, record = await proposed(session, await opened(session), draft)
        if change == "cancel":
            await engine.cancel("local", run["run_id"])
        elif change == "epoch":
            new = await engine.submit(
                actor,
                "calculation",
                "new",
                {"left": 2, "right": 3},
                RunOptions(conversation_id="intent", switch_topic=True, pause_for_review=True),
            )
            await final(engine, "local", new["run_id"], "awaiting_review")
        else:
            with engine.store.connection() as db:
                row = engine.store.request("local", run["run_id"])
                payload = json.loads(row["payload"])
                payload["input"]["left"] = 99
                db.execute(
                    "UPDATE requests SET payload=? WHERE run_id=?",
                    (json.dumps(payload), run["run_id"]),
                )
        before = business(app)
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 409
        assert business(app) == before


async def test_cancel_confirmation_does_not_cancel_the_target_run(tmp_path):
    async with client(tmp_path) as (session, app, _):
        engine = app.state.service.runtime
        actor = await engine.identity("local")
        run = await engine.submit(
            actor,
            "calculation",
            "target",
            {"left": 1, "right": 2},
            RunOptions(conversation_id="intent", pause_for_review=True),
        )
        await final(engine, "local", run["run_id"], "awaiting_review")
        draft = {
            "text": "取消这个目标任务",
            "action": "propose_cancel",
            "target_run_id": run["run_id"],
            "task": None,
        }
        uri, record = await proposed(session, await opened(session), draft)
        before = business(app)
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 200
        assert engine.store.request("local", run["run_id"])["cancel"] == 0
        assert business(app) == before


async def test_cross_owner_target_is_not_available(tmp_path):
    a, b = "synthetic-alice-token-long-value", "synthetic-bob-token-long-value"
    settings = Settings(
        data_dir=tmp_path,
        knowledge_root=ROOT / "data/knowledge",
        auth_mode="bearer",
        bearer_principals={a: "alice", b: "bob"},
    )
    async with client(tmp_path, settings=settings) as (session, app, _):
        ha, hb = {"authorization": "Bearer " + a}, {"authorization": "Bearer " + b}
        await opened(session, ha)
        actor = await app.state.service.runtime.identity("alice")
        run = await app.state.service.runtime.submit(
            actor,
            "calculation",
            "alice",
            {"left": 1, "right": 2},
            RunOptions(conversation_id="intent", pause_for_review=True),
        )
        await final(app.state.service.runtime, "alice", run["run_id"], "awaiting_review")
        path = await opened(session, hb)
        response = await session.post(
            path + "/proposals",
            headers=hb,
            json={
                "expected_revision": 0,
                "draft": {
                    "text": "取消指定任务",
                    "action": "propose_cancel",
                    "target_run_id": run["run_id"],
                },
            },
        )
        assert response.status_code == 404


@pytest.mark.parametrize("payload", ['{"tampered":true}', "not-json"])
async def test_record_integrity_fails_closed(tmp_path, payload):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        with app.state.service.runtime.store.connection() as db:
            db.execute(
                "UPDATE agent_intent_proposals SET payload=? WHERE id=?",
                (payload, record["proposal_id"]),
            )
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 503
        assert all(not rows for rows in business(app).values())


async def test_disabled_host_has_no_proposal_endpoint_or_tables(tmp_path):
    async with client(tmp_path, intent_interpreter=None) as (session, app, _):
        assert (await session.post(BASE, json={"conversation_id": "intent"})).status_code == 404
        with app.state.service.runtime.store.connection() as db:
            assert not db.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'agent_intent_%'"
            ).fetchall()


async def test_ui_module_is_served_as_browser_javascript(tmp_path):
    async with client(tmp_path) as (session, _, _):
        response = await session.get("/static/intent-ui.mjs")
        assert response.status_code == 200 and response.headers["content-type"].startswith(
            "application/javascript"
        )


async def test_two_sqlite_connections_consume_only_once(tmp_path):
    async with client(tmp_path) as (session, app, _):
        _, record = await proposed(session, await opened(session))
        service = app.state.intents
        actor = await app.state.service.runtime.identity("local")
        with app.state.service.runtime.store.connection() as db:
            row = dict(db.execute("SELECT * FROM agent_intent_sessions").fetchone())

        def consume():
            try:
                return service.decide(
                    actor,
                    "calculation",
                    row["id"],
                    record["proposal_id"],
                    IntentConfirmation.model_validate(confirmation(record)),
                    row["credential_digest"],
                )["status"]
            except AgentError as exc:
                return exc.code

        results = await asyncio.gather(asyncio.to_thread(consume), asyncio.to_thread(consume))
        assert sorted(results) == ["confirmed", "intent_not_pending"]
        assert all(not rows for rows in business(app).values())


async def test_interpreter_timeout_releases_slot_and_rejects_without_execution(tmp_path):
    interpreter = GatedInterpreter()
    limits = IntentProposalLimits(interpreter_seconds=0.01, max_inflight=1)
    async with client(tmp_path, intent_interpreter=interpreter, intent_limits=limits) as (
        session,
        app,
        _,
    ):
        path = await opened(session)
        response = await session.post(
            path + "/proposals", json={"expected_revision": 0, "draft": DRAFT}
        )
        assert response.status_code == 503 and response.json()["error"] == "intent_interpreter"
        interpreter.release.set()
        _, record = await proposed(session, path)
        assert record["status"] == "pending"
        assert all(not rows for rows in business(app).values())


async def test_inflight_bound_prevents_parallel_interpreter_overload(tmp_path):
    interpreter = GatedInterpreter()
    async with client(
        tmp_path, intent_interpreter=interpreter, intent_limits=IntentProposalLimits(max_inflight=1)
    ) as (session, app, _):
        first, second = await opened(session), await opened(session, conversation="other")
        task = asyncio.create_task(
            session.post(first + "/proposals", json={"expected_revision": 0, "draft": DRAFT})
        )
        try:
            async with asyncio.timeout(5):
                await interpreter.entered.wait()
            response = await session.post(
                second + "/proposals", json={"expected_revision": 0, "draft": DRAFT}
            )
            assert response.status_code == 429 and response.json()["error"] == "intent_busy"
        finally:
            interpreter.release.set()
        assert (await task).status_code == 201
        await proposed(session, second)
        assert all(not rows for rows in business(app).values())


async def test_versioned_store_rejects_unknown_schema(tmp_path):
    async with client(tmp_path) as (_, app, _):
        with app.state.service.runtime.store.connection() as db:
            db.execute("UPDATE agent_intent_meta SET version=999")
        with pytest.raises(AgentError) as error:
            app.state.intents.install()
        assert error.value.code == "intent_schema"


async def test_standalone_host_explicitly_enables_metadata_routes(tmp_path):
    async with runtime(tmp_path) as (engine, actor, effects, *_):

        @asynccontextmanager
        async def factory():
            yield engine

        async def authenticate(request):
            return actor

        app = create_agent_app(
            factory,
            authenticate=authenticate,
            refresh_identity=authenticate,
            intent_interpreter=OfflineStructuredIntent(),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as session:
                base = "/api/v1/agents/worker/intent-sessions"
                created = await session.post(base, json={"conversation_id": "same"})
                path = base + "/" + created.json()["session_id"]
                blocked = await session.post(
                    base,
                    json={"conversation_id": "same"},
                    headers={"origin": "https://untrusted.example"},
                )
                assert blocked.status_code == 403
                response = await session.post(
                    path + "/proposals",
                    json={"expected_revision": 0, "draft": DRAFT | {"task": {"input": {"x": 2}}}},
                )
                assert response.status_code == 201
                record = response.json()
                result = await session.post(
                    path + "/proposals/" + record["proposal_id"] + "/confirm",
                    json=confirmation(record),
                )
                assert result.status_code == 200 and result.json()["status"] == "confirmed"
                with engine.store.connection() as db:
                    assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
                assert effects == []


async def test_expiry_crossing_at_commit_does_not_record_confirmation(tmp_path):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        before = record["expires_at"] - 0.01
        times = iter([before, before, record["expires_at"]])
        app.state.intents.clock = lambda: next(times)
        response = await session.post(uri + "/confirm", json=confirmation(record))
        app.state.intents.clock = time.time
        assert response.status_code == 410
        with app.state.service.runtime.store.connection() as db:
            assert (
                db.execute("SELECT confirmed_at FROM agent_intent_proposals").fetchone()[0] is None
            )


async def test_server_identity_scope_change_invalidates_saved_session(tmp_path):
    async with client(tmp_path) as (session, app, _):
        _, record = await proposed(session, await opened(session))
        engine = app.state.service.runtime
        actor = await engine.identity("local")
        with engine.store.connection() as db:
            row = dict(db.execute("SELECT * FROM agent_intent_sessions").fetchone())
        for changed in (
            actor.model_copy(update={"scopes": frozenset()}),
            actor.model_copy(update={"permission_version": 2}),
        ):
            with pytest.raises(AgentError) as error:
                app.state.intents.view(
                    changed,
                    "calculation",
                    row["id"],
                    record["proposal_id"],
                    row["credential_digest"],
                )
            assert error.value.code == "intent_identity"


@pytest.mark.parametrize("mutation", ["owner", "task", "malformed"])
async def test_untrusted_interpreter_cannot_change_authority_or_requested_input(tmp_path, mutation):
    class Untrusted(OfflineStructuredIntent):
        async def propose(self, request, draft):
            raw = await super().propose(request, draft)
            if mutation == "owner":
                raw["owner"] = "admin"
            elif mutation == "task":
                raw["task"] = {"input": {"left": 999, "right": 999}}
            else:
                return "not a JSON object"
            return raw

    async with client(tmp_path, intent_interpreter=Untrusted()) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        assert record["status"] == "clarify"
        assert (await session.post(uri + "/confirm", json=confirmation(record))).status_code == 409
        assert all(not rows for rows in business(app).values())


async def test_agent_release_change_invalidates_confirmation(tmp_path, monkeypatch):
    async with client(tmp_path) as (session, app, _):
        uri, record = await proposed(session, await opened(session))
        registry = app.state.service.runtime.registry
        original = registry.agent
        with monkeypatch.context() as patch:
            patch.setattr(
                registry,
                "agent",
                lambda agent: replace(original(agent), digest="sha256:" + "0" * 64),
            )
            assert (
                await session.post(uri + "/confirm", json=confirmation(record))
            ).status_code == 409
        assert all(not rows for rows in business(app).values())


async def test_shutdown_during_interpretation_prevents_late_metadata_commit(tmp_path):
    interpreter = GatedInterpreter()
    async with client(tmp_path, intent_interpreter=interpreter) as (session, app, _):
        path = await opened(session)
        task = asyncio.create_task(
            session.post(path + "/proposals", json={"expected_revision": 0, "draft": DRAFT})
        )
        try:
            async with asyncio.timeout(5):
                await interpreter.entered.wait()
            await app.state.service.runtime.close()
        finally:
            interpreter.release.set()
        assert (await task).status_code == 503
        with app.state.service.runtime.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM agent_intent_proposals").fetchone()[0] == 0
        assert all(not rows for rows in business(app).values())
