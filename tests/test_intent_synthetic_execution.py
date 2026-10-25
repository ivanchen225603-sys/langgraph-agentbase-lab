"""Opted-in arithmetic only, disposable SQLite and in-process ASGI; no real IO tools."""

import asyncio
import copy
from contextlib import asynccontextmanager

import httpx
import pytest

from agentbase.application import create_agent_app, open_runtime
from agentbase.contracts import AgentError, Identity
from agentbase.intent_execution import AGENT, register_synthetic_intent_agent
from agentbase.intent_proposals import OfflineStructuredIntent
from agentbase.registry import ComponentRegistry
from agentbase.runtime_contracts import TERMINAL
from agentbase.workflows import register_workflows

BASE = f"/api/v1/agents/{AGENT}/intent-sessions"
DRAFT = {
    "text": "为数字生成本地只读提案",
    "action": "propose_task",
    "task": {"input": {"number": 7}},
}
TOKEN = "synthetic-intent-alice-one"


@asynccontextmanager
async def host(path, *, enabled=True, allowed=None, scopes=frozenset({"read"}), no_acl=False):
    registry = ComponentRegistry()
    register_workflows(registry)
    binding = register_synthetic_intent_agent(registry)
    actors = {
        TOKEN: Identity(owner="alice", scopes=scopes),
        "synthetic-intent-alice-two": Identity(owner="alice", scopes=scopes),
        "synthetic-intent-bob": Identity(owner="bob", scopes=scopes),
    }

    async def resolve(owner):
        return next(actor for actor in actors.values() if actor.owner == owner)

    async def authenticate(request):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in actors:
            raise AgentError("unauthorized", "synthetic bearer required", 401)
        return actors[token]

    async def authorize(identity, agent):
        return True if allowed is None else allowed[0]

    @asynccontextmanager
    async def factory():
        async with open_runtime(registry, data_dir=path, identity_resolver=resolve) as engine:
            yield engine

    app = create_agent_app(
        factory,
        authenticate=authenticate,
        refresh_identity=authenticate,
        authorize_agent=None if no_acl else authorize,
        intent_interpreter=OfflineStructuredIntent(),
        synthetic_intent=binding if enabled else None,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"authorization": "Bearer " + TOKEN},
        ) as session:
            yield session, app, actors


def counts(app):
    with app.state.engine.store.connection() as db:
        return {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "requests",
                "agent_tool_operations",
                "agent_reviews",
                "agent_memories",
                "agent_intent_executions",
            )
        }


async def proposal(session, *, confirm=True, draft=None):
    opened = await session.post(BASE, json={"conversation_id": "case"})
    assert opened.status_code == 201, opened.text
    path = BASE + "/" + opened.json()["session_id"]
    result = await session.post(
        path + "/proposals", json={"expected_revision": 0, "draft": draft or DRAFT}
    )
    assert result.status_code == 201, result.text
    record = result.json()
    uri = path + "/proposals/" + record["proposal_id"]
    body = {key: record[key] for key in ("revision", "request_digest", "decision_digest")} | {
        "draft": record["payload"]["draft"]
    }
    if confirm:
        result = await session.post(uri + "/confirm", json=body)
        assert result.status_code == 200, result.text
    return path, uri, body, record


async def completed(session, uri):
    async with asyncio.timeout(10):
        while True:
            response = await session.get(uri + "/execution")
            assert response.status_code == 200, response.text
            value = response.json()
            if value["run"] and value["run"]["status"] in TERMINAL:
                return value
            await asyncio.sleep(0.01)


async def test_separate_execute_creates_one_run_and_confirmation_grants_no_tool_approval(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, record = await proposal(session, confirm=False)
        assert record["execution_connected"] is True
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert (await session.post(uri + "/confirm", json=body)).status_code == 200
        assert all(value == 0 for value in counts(app).values())
        assert (await session.get(uri + "/execution")).json()["admitted"] is False
        executed = await session.post(uri + "/execute", json=body)
        assert executed.status_code == 202, executed.text
        assert executed.headers["cache-control"] == "no-store"
        result = await completed(session, uri)
        assert result["run"]["status"] == "succeeded", result
        assert result["run"]["result"]["double"]["value"] == {"doubled": 14}
        again = await session.post(uri + "/execute", json=body)
        assert again.status_code == 202 and again.json()["run"]["run_id"] == result["run"]["run_id"]
        assert counts(app) == {
            "requests": 1,
            "agent_tool_operations": 1,
            "agent_reviews": 0,
            "agent_memories": 0,
            "agent_intent_executions": 1,
        }
        assert (await session.post(uri + "/revoke", json=body)).status_code == 409
        assert (await session.get(uri)).json()["status"] == "executed"


async def test_default_does_not_mount_execution_routes_or_install_execution_tables(tmp_path):
    async with host(tmp_path, enabled=False) as (session, app, _):
        _, uri, body, record = await proposal(session)
        assert record["execution_connected"] is False
        assert (await session.post(uri + "/execute", json=body)).status_code == 404
        assert (await session.get(uri + "/execution")).status_code == 404
        with app.state.engine.store.connection() as db:
            assert not db.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'agent_intent_execution%'"
            ).fetchall()


@pytest.mark.parametrize("field", ["revision", "request_digest", "decision_digest", "draft"])
async def test_changed_execution_parameters_are_rejected_before_admission(tmp_path, field):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        changed = copy.deepcopy(body)
        if field == "draft":
            changed[field]["task"]["input"]["number"] = 99
        elif field == "revision":
            changed[field] += 1
        else:
            changed[field] = "sha256:" + "0" * 64
        assert (await session.post(uri + "/execute", json=changed)).status_code == 409
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0


async def test_concurrent_execution_replays_one_run(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        responses = await asyncio.gather(
            *[session.post(uri + "/execute", json=body) for _ in range(8)]
        )
        assert all(r.status_code == 202 for r in responses), [r.text for r in responses]
        assert len({r.json()["run"]["run_id"] for r in responses}) == 1
        assert (await completed(session, uri))["run"]["status"] == "succeeded"
        assert (
            counts(app)["requests"]
            == counts(app)["agent_tool_operations"]
            == counts(app)["agent_intent_executions"]
            == 1
        )


@pytest.mark.parametrize("change", ["invalidate", "reopen", "revoke", "expire"])
async def test_invalidated_or_expired_choices_cannot_admit(tmp_path, change):
    async with host(tmp_path) as (session, app, _):
        path, uri, body, record = await proposal(session)
        if change == "invalidate":
            await session.post(path + "/invalidate")
        elif change == "reopen":
            await session.post(BASE, json={"conversation_id": "case"})
        elif change == "revoke":
            await session.post(uri + "/revoke", json=body)
        else:
            app.state.intents.clock = lambda: record["expires_at"] + 1
        result = await session.post(uri + "/execute", json=body)
        assert result.status_code == (410 if change in {"reopen", "expire"} else 409)
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0


@pytest.mark.parametrize(
    "token,status", [("synthetic-intent-alice-two", 403), ("synthetic-intent-bob", 404)]
)
async def test_execution_and_queries_do_not_cross_credentials_or_owners(tmp_path, token, status):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        headers = {"authorization": "Bearer " + token}
        assert (
            await session.post(uri + "/execute", json=body, headers=headers)
        ).status_code == status
        assert (await session.get(uri + "/execution", headers=headers)).status_code == status
        assert counts(app)["requests"] == 0


async def test_metadata_cannot_grant_missing_read_scope(tmp_path):
    async with host(tmp_path, scopes=frozenset()) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        assert (await session.post(uri + "/execute", json=body)).status_code == 403
        assert counts(app)["requests"] == 0


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://untrusted.invalid"},
        {"sec-fetch-site": "cross-site"},
        {"x-owner": "admin"},
    ],
)
async def test_execution_preserves_origin_and_spoof_protection(tmp_path, headers):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        assert (await session.post(uri + "/execute", json=body, headers=headers)).status_code == 403
        assert counts(app)["requests"] == 0


@pytest.mark.parametrize(
    "extra", [{"approval": True}, {"owner": "admin"}, {"request_key": "chosen"}]
)
async def test_execution_control_parameters_are_host_owned(tmp_path, extra):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        assert (await session.post(uri + "/execute", json=body | extra)).status_code == 422
        assert counts(app)["requests"] == 0


@pytest.mark.parametrize(
    "change", ["invalidate", "revoke", "expire", "acl", "identity", "shutdown"]
)
async def test_admission_rechecks_after_await_and_rolls_back_late_requests(tmp_path, change):
    allowed = [True]
    async with host(tmp_path, allowed=allowed, no_acl=(change == "identity")) as (
        session,
        app,
        actors,
    ):
        path, uri, body, record = await proposal(session)
        engine = app.state.engine
        bound = engine.registry.agent(AGENT)
        workflow = engine.binding({"agent_id": AGENT, "binding": bound.digest})
        entered, release = asyncio.Event(), asyncio.Event()

        async def prepare(identity, value):
            entered.set()
            await release.wait()
            return {"input": value}

        workflow.prepare = prepare
        task = asyncio.create_task(session.post(uri + "/execute", json=body))
        try:
            async with asyncio.timeout(5):
                await entered.wait()
            if change == "invalidate":
                await session.post(path + "/invalidate")
            elif change == "revoke":
                await session.post(uri + "/revoke", json=body)
            elif change == "expire":
                app.state.intents.clock = lambda: record["expires_at"] + 1
            elif change == "acl":
                allowed[0] = False
            elif change == "identity":
                actors[TOKEN] = actors[TOKEN].model_copy(update={"permission_version": 2})
            else:
                await engine.close()
        finally:
            release.set()
        result = await task
        expected = {
            "invalidate": 409,
            "revoke": 409,
            "expire": 410,
            "acl": 403,
            "identity": 403,
            "shutdown": 503,
        }
        assert result.status_code == expected[change], result.text
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0


@pytest.mark.parametrize("failure", ["mapping", "consume"])
async def test_admission_metadata_failure_rolls_back_run_and_consumption(tmp_path, failure):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        with app.state.engine.store.connection() as db:
            if failure == "mapping":
                db.execute(
                    "CREATE TRIGGER synthetic_failure BEFORE INSERT ON agent_intent_executions BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
                )
            else:
                db.execute(
                    "CREATE TRIGGER synthetic_failure BEFORE UPDATE OF status ON agent_intent_proposals WHEN NEW.status='executed' BEGIN SELECT RAISE(IGNORE); END"
                )
        response = await session.post(uri + "/execute", json=body)
        assert response.status_code == (503 if failure == "mapping" else 409)
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 0
        assert (await session.get(uri)).json()["status"] == "confirmed"
        with app.state.engine.store.connection() as db:
            db.execute("DROP TRIGGER synthetic_failure")
        assert (await session.post(uri + "/execute", json=body)).status_code == 202


async def test_lost_execution_response_can_be_queried_and_retried_without_resubmission(
    tmp_path, monkeypatch
):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        original = session._transport.handle_async_request
        lost = [False]

        async def lose_response(request):
            response = await original(request)
            if request.url.path.endswith("/execute") and not lost[0]:
                lost[0] = True
                raise httpx.ReadError("synthetic response loss after commit")
            return response

        monkeypatch.setattr(session._transport, "handle_async_request", lose_response)
        with pytest.raises(httpx.ReadError):
            await session.post(uri + "/execute", json=body)
        queried = await completed(session, uri)
        again = await session.post(uri + "/execute", json=body)
        assert (
            again.status_code == 202 and again.json()["run"]["run_id"] == queried["run"]["run_id"]
        )
        assert (
            counts(app)["requests"]
            == counts(app)["agent_intent_executions"]
            == counts(app)["agent_tool_operations"]
            == 1
        )


async def test_execution_mapping_survives_host_reconstruction(tmp_path):
    async with host(tmp_path) as (session, _, _):
        _, uri, body, _ = await proposal(session)
        assert (await session.post(uri + "/execute", json=body)).status_code == 202
        first = await completed(session, uri)
    async with host(tmp_path) as (session, app, _):
        result = await session.post(uri + "/execute", json=body)
        assert (
            result.status_code == 202 and result.json()["run"]["run_id"] == first["run"]["run_id"]
        )
        assert counts(app)["requests"] == counts(app)["agent_tool_operations"] == 1


async def test_query_after_proposal_expiry_does_not_reexecute(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, record = await proposal(session)
        await session.post(uri + "/execute", json=body)
        first = await completed(session, uri)
        app.state.intents.clock = lambda: record["expires_at"] + 1
        assert (await session.post(uri + "/execute", json=body)).status_code == 410
        assert (await session.get(uri + "/execution")).json()["run"]["run_id"] == first["run"][
            "run_id"
        ]
        assert counts(app)["requests"] == 1


async def test_sql_replay_path_rechecks_metadata_even_when_lookup_raced(tmp_path, monkeypatch):
    async with host(tmp_path) as (session, app, _):
        path, uri, body, _ = await proposal(session)
        await session.post(uri + "/execute", json=body)
        await completed(session, uri)
        engine = app.state.engine
        monkeypatch.setattr(engine.store, "lookup", lambda *args: None)
        assert (await session.post(uri + "/execute", json=body)).status_code == 202
        original = engine.store.admit

        def changed_before_sql(*args, **kwargs):
            with engine.store.connection() as db:
                row = db.execute("SELECT * FROM agent_intent_sessions").fetchone()
            app.state.intents.invalidate(awaited_actor, AGENT, row["id"], row["credential_digest"])
            return original(*args, **kwargs)

        awaited_actor = await engine.identity("alice")
        monkeypatch.setattr(engine.store, "admit", changed_before_sql)
        assert (await session.post(uri + "/execute", json=body)).status_code == 409
        assert counts(app)["requests"] == counts(app)["agent_intent_executions"] == 1


async def test_mapping_corruption_is_not_returned_as_a_different_run(tmp_path):
    async with host(tmp_path) as (session, app, _):
        _, uri, body, _ = await proposal(session)
        await session.post(uri + "/execute", json=body)
        await completed(session, uri)
        with app.state.engine.store.connection() as db:
            db.execute("UPDATE agent_intent_executions SET proof='tampered'")
        response = await session.get(uri + "/execution")
        assert (
            response.status_code == 503 and response.json()["error"] == "intent_execution_storage"
        )
