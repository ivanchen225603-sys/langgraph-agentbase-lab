"""ASGI send gates prove replay authorization and cleanup under real buffering."""

import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI

from agentbase.artifacts import ArtifactOperation, ArtifactScope
from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Message
from agentbase.http import AgentHttpLimits, mount_agent_routes
from agentbase.runtime_contracts import RunOptions
from raglab.config import Settings
from tests.test_agentbase_http import client, final
from tests.test_agentbase_runtime import admit, package, system
from tests.test_agentbase_runtime import store as store


class Connection:
    def __init__(self, app, path, *, headers=None, send_hook=None):
        parsed = urlsplit(path)
        self.messages, self.disconnect, self.changed = [], asyncio.Event(), asyncio.Event()
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await self.disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            self.messages.append(dict(message))
            self.changed.set()
            if send_hook:
                await send_hook(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [
                (b"host", b"127.0.0.1:8001"),
                *[(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            ],
            "client": ("127.0.0.1", 2345),
            "server": ("127.0.0.1", 8001),
        }
        self.task = asyncio.create_task(app(scope, receive, send))

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate(self.messages):
                if self.task.done():
                    await self.task
                    raise AssertionError(self.messages)
                self.changed.clear()
                await self.changed.wait()

    async def close(self):
        self.disconnect.set()
        async with asyncio.timeout(5):
            await self.task

    @property
    def body(self):
        return b"".join(message.get("body", b"") for message in self.messages)


@asynccontextmanager
async def exposed(engine, identities, *, limits=None, buffer=False):
    app = FastAPI()
    if buffer:

        @app.middleware("http")
        async def buffered(request, call_next):
            return await call_next(request)

    async def actor(request):
        return identities["user"]

    mount_agent_routes(app, lambda: engine, actor, refresh_identity=actor, limits=limits)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 2345)),
        base_url="http://127.0.0.1:8001",
    ) as session:
        yield app, session


@pytest.mark.parametrize("after", [-1, True, 1.0, "0", 9223372036854775808])
def test_store_cursor_rejects_invalid_values(store, after):
    row = admit(store)
    with pytest.raises(AgentError) as error:
        store.events("user", row["run_id"], after)
    assert error.value.code == "event_cursor_invalid"


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, 513])
def test_store_event_limit_is_strict(store, limit):
    row = admit(store)
    with pytest.raises(AgentError) as error:
        store.events("user", row["run_id"], 0, limit)
    assert error.value.code == "event_limit"


def test_initial_retention_gap_and_exact_head(store):
    row = admit(store)
    assert store.events("user", row["run_id"], 0) == []
    with pytest.raises(AgentError) as error:
        store.events("user", row["run_id"], 1)
    assert error.value.code == "event_cursor_future"
    for _ in range(4):
        store.append_event(row, "tool.completed", {})
    for after in (0, 1):
        with pytest.raises(AgentError) as error:
            store.events("user", row["run_id"], after)
        assert error.value.code == "event_cursor_expired"
    assert [event.seq for event in store.events("user", row["run_id"], 2)] == [3, 4]
    assert store.events("user", row["run_id"], 4) == []
    with pytest.raises(AgentError) as error:
        store.events("user", row["run_id"], 5)
    assert error.value.code == "event_cursor_future"


@pytest.mark.parametrize(
    "query,header",
    [
        ("-1", None),
        ("01", None),
        ("1.2", None),
        ("9223372036854775808", None),
        ("9999", None),
        (None, "other:1"),
        ("0", "RUN:1"),
        ("0&after=1", None),
    ],
)
async def test_http_cursor_rejected_before_stream(tmp_path, query, header):
    async with client(tmp_path) as (session, app, _):
        created = (
            await session.post(
                "/api/v1/agents/calculation/runs",
                headers={"Idempotency-Key": "cursor"},
                json={"input": {"left": 2, "right": 3}},
            )
        ).json()
        await final(app.state.service.runtime, "local", created["run_id"])
        url = f"/api/v1/agents/calculation/runs/{created['run_id']}/events"
        response = await session.get(
            url + (f"?after={query}" if query is not None else ""),
            headers={"Last-Event-ID": header.replace("RUN", created["run_id"])} if header else {},
        )
        assert response.status_code in {400, 409}
        assert response.json()["error"].startswith("event_cursor")
        assert app.state.agent_streams.active == {}


async def test_sse_terminal_frames_keep_sequence_and_current_snapshot(tmp_path):
    async with client(tmp_path) as (session, app, _):
        created = (
            await session.post(
                "/api/v1/agents/calculation/runs",
                headers={"Idempotency-Key": "frames"},
                json={"input": {"left": 2, "right": 3}},
            )
        ).json()
        await final(app.state.service.runtime, "local", created["run_id"])
        response = await session.get(f"/api/v1/agents/calculation/runs/{created['run_id']}/events")
        assert (
            response.status_code == 200 and "text/event-stream" in response.headers["content-type"]
        )
        frames = response.text.strip().split("\n\n")
        events = [json.loads(frame.split("data: ", 1)[1]) for frame in frames]
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        assert events[-1]["kind"] == "run.final"
        assert events[-1]["data"]["value"] == {"status": "succeeded", "result_available": True}
        assert "result" not in events[-1]["data"]
        assert app.state.agent_streams.active == {}


@pytest.mark.parametrize("buffer", [False, True])
@pytest.mark.parametrize("source_kind", ["memory", "artifact"])
async def test_sse_reauthorizes_sources_at_actual_send(tmp_path, buffer, source_kind):
    data = package().model_dump(mode="json")
    data["agents"][0]["context"] = {"memory_scope": "agent"}
    hold, sending, release, prefetched = (asyncio.Event() for _ in range(4))
    async with system(
        tmp_path, ["first_sensitive"], gate=hold, config=AgentPackage.model_validate(data)
    ) as (engine, provider, contexts, identities):
        if source_kind == "memory":
            memory = contexts.create_memory("user", "agent0", "", "protected context")
            source = engine.contexts.memory_reference(memory)
        else:
            scope = ArtifactScope(owner="user", agent_id="agent0", namespace="default")
            artifact = engine.artifacts.create(
                scope,
                ArtifactOperation(
                    run_id="host",
                    key="create",
                    config_digest=engine.registry.agent("agent0").digest,
                ),
                "protected context",
                sources=[],
                guard=lambda db: True,
            )
            source = artifact.reference()
            contexts.append_turn(
                "user",
                "agent0",
                "default",
                0,
                "previous",
                [
                    Message(role="user", content="Previous"),
                    Message(role="assistant", content="Protected", sources=[source]),
                ],
                guard=lambda db: True,
            )
        created = await engine.submit(identities["user"], "agent0", "source", "Input")
        async with asyncio.timeout(5):
            for _ in range(1000):
                if any(
                    event.kind == "model.delta"
                    for event in await engine.events("user", created["run_id"])
                ):
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError("native delta was not published")
        first = next(
            event
            for event in await engine.events("user", created["run_id"])
            if event.kind == "model.delta"
        )
        engine.store.append_event(
            engine.store.request("user", created["run_id"]),
            "model.delta",
            {"text": "second_sensitive"},
            provisional=True,
            sources=[source],
        )
        original = engine.events

        async def observe(owner, run_id, after=0, *, limit=128):
            result = await original(owner, run_id, after, limit=limit)
            if after == first.seq and result:
                prefetched.set()
            return result

        engine.events = observe

        async def slow_send(message):
            if b"first_sensitive" in message.get("body", b""):
                sending.set()
                await release.wait()

        async with exposed(engine, identities, buffer=buffer) as (app, _):
            connection = Connection(
                app, f"/api/v1/agents/agent0/runs/{created['run_id']}/events", send_hook=slow_send
            )
            try:
                await asyncio.wait_for(sending.wait(), 5)
                if buffer:
                    await asyncio.wait_for(prefetched.wait(), 5)
                if source_kind == "memory":
                    contexts.update_memory(
                        "user", "agent0", "", memory.id, memory.version, enabled=False
                    )
                else:
                    engine.artifacts.revoke(
                        scope,
                        ArtifactOperation(
                            run_id="host",
                            key="revoke",
                            config_digest=engine.registry.agent("agent0").digest,
                        ),
                        artifact.id,
                        artifact.version,
                        guard=lambda db: True,
                    )
                release.set()
                await asyncio.wait_for(connection.task, 5)
                assert b"first_sensitive" in connection.body
                assert b"second_sensitive" not in connection.body
                assert b"event: stream.error" in connection.body
                error = connection.body.split(b"event: stream.error", 1)[1]
                assert b"id:" not in error and b"source_revoked" in error
                assert app.state.agent_streams.active == {}
                assert provider.requests and created["run_id"] in engine.active
            finally:
                release.set()
                await connection.close()
                hold.set()


@pytest.mark.parametrize("failure", ["timeout", "disconnect", "oserror", "cancel", "start"])
async def test_slow_send_quota_and_all_exit_paths_release(tmp_path, failure):
    hold, sending, release = (asyncio.Event() for _ in range(3))
    async with system(tmp_path, ["candidate"], gate=hold) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "quota", "Input")
        limits = AgentHttpLimits(connections=1, send_seconds=0.2, poll_seconds=0.005)
        async with exposed(engine, identities, limits=limits, buffer=True) as (app, session):

            async def blocked(message):
                if (
                    failure == "start" and message["type"] == "http.response.start"
                ) or b"candidate" in message.get("body", b""):
                    sending.set()
                    if failure == "oserror":
                        raise OSError("closed transport")
                    await release.wait()

            connection = Connection(
                app, f"/api/v1/agents/agent0/runs/{created['run_id']}/events", send_hook=blocked
            )
            try:
                await asyncio.wait_for(sending.wait(), 5)
                if failure not in {"oserror", "start"}:
                    rejected = await session.get(
                        f"/api/v1/agents/agent0/runs/{created['run_id']}/events"
                    )
                    assert (
                        rejected.status_code == 429
                        and rejected.json()["error"] == "stream_capacity"
                    )
                if failure == "disconnect":
                    connection.disconnect.set()
                if failure == "cancel":
                    connection.task.cancel()
                if failure == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await connection.task
                else:
                    await asyncio.wait_for(connection.task, 5)
                assert app.state.agent_streams.active == {}
                assert (
                    created["run_id"] in engine.active
                    and engine.store.request("user", created["run_id"])["cancel"] == 0
                )
                assert len(provider.requests) == 1
            finally:
                release.set()
                connection.disconnect.set()
                await asyncio.gather(connection.task, return_exceptions=True)
                hold.set()


async def test_original_token_is_rechecked_after_agent_acl_await(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        auth_mode="bearer",
        bearer_principals={"a" * 24: "alice", "b" * 24: "alice"},
        rate_per_minute=2,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    block = False

    async def access(actor, agent_id):
        if block:
            entered.set()
            await release.wait()
        return True

    async with client(tmp_path, settings=settings, authorize_agent=access) as (
        session,
        app,
        fixture,
    ):
        headers = {"Authorization": "Bearer " + "a" * 24}
        created = (
            await session.post(
                "/api/v1/agents/classification/runs",
                headers={**headers, "Idempotency-Key": "token"},
                json={"input": "question?"},
            )
        ).json()

        async def send_hook(message):
            nonlocal block
            if b"event: run.accepted" in message.get("body", b""):
                block = True

        connection = Connection(
            app,
            f"/api/v1/agents/classification/runs/{created['run_id']}/events",
            headers=headers,
            send_hook=send_hook,
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            settings.bearer_principals.pop("a" * 24)
            release.set()
            await asyncio.wait_for(connection.task, 5)
            assert b"event: model.delta" not in connection.body
            assert b"event: stream.error" in connection.body and b"unauthorized" in connection.body
            assert app.state.agent_streams.active == {}
            assert fixture.calls == 1
            assert (
                app.state.service.runtime.store.request("alice", created["run_id"])["cancel"] == 0
            )
        finally:
            release.set()
            await connection.close()


async def test_total_duration_bounds_slow_outer_authorization(tmp_path):
    hold, entered = asyncio.Event(), asyncio.Event()
    block = False
    async with system(tmp_path, ["candidate"], gate=hold) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "duration", "Input")
        app = FastAPI()

        @app.middleware("http")
        async def buffered(request, call_next):
            return await call_next(request)

        async def actor(request):
            return identities["user"]

        async def access(actor, agent_id):
            if block:
                entered.set()
                await asyncio.Event().wait()
            return True

        mount_agent_routes(
            app,
            lambda: engine,
            actor,
            refresh_identity=actor,
            authorize_agent=access,
            limits=AgentHttpLimits(duration_seconds=0.2, send_seconds=0.05, poll_seconds=0.005),
        )

        async def send_hook(message):
            nonlocal block
            if b"event: run.accepted" in message.get("body", b""):
                block = True

        connection = Connection(
            app, f"/api/v1/agents/agent0/runs/{created['run_id']}/events", send_hook=send_hook
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(connection.task, 1)
            assert app.state.agent_streams.active == {}
            assert b"candidate" not in connection.body
            assert b"stream_duration" in connection.body
            assert created["run_id"] in engine.active
        finally:
            await connection.close()
            hold.set()


@pytest.mark.parametrize("mode", ["cancel", "topic"])
async def test_old_candidate_rejected_but_terminal_facts_remain_queryable(tmp_path, mode):
    hold = asyncio.Event()
    async with system(tmp_path, ["old", "new"], gate=hold) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        ready = asyncio.Event()
        original = engine.emit

        async def observe(run_id, kind, data, **kwargs):
            result = await original(run_id, kind, data, **kwargs)
            if kind == "model.delta":
                ready.set()
            return result

        engine.emit = observe
        created = await engine.submit(identities["user"], "agent0", "old", "Input")
        await asyncio.wait_for(ready.wait(), 5)
        delta = next(
            event for event in await engine.events("user", created["run_id"]) if event.provisional
        )
        if mode == "cancel":
            await engine.cancel("user", created["run_id"])
        else:
            await engine.submit(
                identities["user"], "agent0", "new", "New", RunOptions(switch_topic=True)
            )
        with pytest.raises(AgentError) as error:
            await engine.events("user", created["run_id"], delta.seq - 1, limit=1)
        assert error.value.code == "event_provisional_unavailable"
        with engine.store.connection() as db:
            db.execute("UPDATE requests SET deadline=0 WHERE run_id=?", (created["run_id"],))
        assert all(
            not event.provisional
            for event in await engine.events("user", created["run_id"], delta.seq)
        )
        hold.set()


async def test_inner_middleware_cannot_rewrite_candidate_as_control_body(tmp_path):
    hold = asyncio.Event()
    async with system(tmp_path, ["sensitive"], gate=hold) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "rewrite", "Input")
        app = FastAPI()

        class Rewrite:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                async def rewritten(message):
                    if b"event: model.delta" in message.get("body", b""):
                        message = {**message, "body": b'data: {"private":"sensitive"}\n\n'}
                    await send(message)

                await self.app(scope, receive, rewritten)

        app.add_middleware(Rewrite)

        async def actor(request):
            return identities["user"]

        mount_agent_routes(app, lambda: engine, actor, refresh_identity=actor)
        connection = Connection(app, f"/api/v1/agents/agent0/runs/{created['run_id']}/events")
        try:
            await asyncio.wait_for(connection.task, 5)
            assert b"sensitive" not in connection.body
            assert b"event_store_conflict" in connection.body
            assert app.state.agent_streams.active == {}
            assert created["run_id"] in engine.active
        finally:
            await connection.close()
            hold.set()
