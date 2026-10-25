"""The public HTTP surface uses the actual host and its single durable engine."""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from agentbase import AgentPackage, load_package
from agentbase.contracts import AgentError
from agentbase.http import AgentHttpLimits
from agentbase.tools import ToolResult
from agentbase.workflows import register_workflows
from examples.agentbase_offline import OfflineFixture, number
from raglab.api import create_app
from raglab.config import Settings
from tests.conftest import ROOT


@asynccontextmanager
async def client(tmp_path, **kwargs):
    fixture = OfflineFixture()

    def configure(registry):
        register_workflows(registry)
        registry.register_provider("demo.fixture", lambda config: fixture)
        registry.register_handler("demo.number", number)
        registry.bind(load_package(ROOT / "examples/agentbase.json"))

    settings = kwargs.pop(
        "settings", Settings(data_dir=tmp_path, knowledge_root=ROOT / "data/knowledge")
    )
    app = create_app(
        settings, configure_registry=kwargs.pop("configure_registry", configure), **kwargs
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 2345)),
            base_url="http://127.0.0.1:8001",
        ) as session:
            try:
                yield session, app, fixture
            finally:
                fixture.release.set()


async def final(engine, owner, run_id, status="succeeded"):
    async with asyncio.timeout(10):
        while True:
            view = await engine.view(owner, run_id)
            if view["status"] == status:
                return view
            assert view["status"] not in {"failed", "timed_out"}, view
            await asyncio.sleep(0.01)


async def test_generic_http_uses_one_runtime_without_rag_index(tmp_path):
    async with client(tmp_path) as (session, app, fixture):
        service = app.state.service
        assert service.runtime is service.runtime_adapter.engine
        assert service.queue is service.runtime.queue
        assert service.workers is service.runtime.workers
        assert service.registry.slots is service.runtime.tools.slots
        assert service.store.active_index() is None
        assert (await session.get("/health/ready")).status_code == 503
        assert (await session.get("/health/ready/runtime")).status_code == 200
        module = await session.get("/static/sse.mjs")
        assert module.status_code == 200
        assert module.headers["content-type"].split(";")[0] == "application/javascript"
        entries = (await session.get("/api/v1/agents")).json()["agents"]
        for entry in entries:
            assert set(entry) == {
                "id",
                "version",
                "description",
                "digest",
                "input_schema",
                "output_schema",
            }
        assert {"calculation", "classification"} <= {entry["id"] for entry in entries}
        created = await session.post(
            "/api/v1/agents/calculation/runs",
            headers={"Idempotency-Key": "key"},
            json={"input": {"left": 2, "right": 3}},
        )
        assert created.status_code == 202, created.text
        view = await final(service.runtime, "local", created.json()["run_id"])
        assert view["result"]["total"]["value"] == {"value": 10}
        assert fixture.calls == 0
        assert len(service.runtime.actions.operations("local", view["run_id"])) == 3


async def test_http_binding_idempotency_and_input_rejection(tmp_path):
    async with client(tmp_path) as (session, app, _):
        path = "/api/v1/agents/calculation/runs"
        args = {"headers": {"Idempotency-Key": "key"}, "json": {"input": {"left": 1, "right": 2}}}
        first = (await session.post(path, **args)).json()
        await final(app.state.service.runtime, "local", first["run_id"])
        duplicate = await session.post(path, **args)
        assert duplicate.json()["run_id"] == first["run_id"]
        conflict = await session.post(
            path, headers=args["headers"], json={"input": {"left": 8, "right": 2}}
        )
        assert conflict.status_code == 409
        for suffix, method, body in [
            ("", "GET", None),
            ("/cancel", "POST", {}),
            ("/resume", "POST", {}),
        ]:
            response = await session.request(
                method, f"/api/v1/agents/classification/runs/{first['run_id']}{suffix}", json=body
            )
            assert response.status_code == 404
            assert "result" not in response.json()
        assert not app.state.service.runtime.store.request("local", first["run_id"])["cancel"]
        for body in [
            {"input": {"left": "1", "right": 2}},
            {"input": {"left": 1, "right": 2}, "owner": "evil"},
            {"input": {"left": 1, "right": 2}, "options": {"scopes": ["write"]}},
        ]:
            rejected = await session.post(path, headers={"Idempotency-Key": "bad"}, json=body)
            assert rejected.status_code in {400, 422}
        assert app.state.service.runtime.store.lookup("local", "calculation", "bad") is None
        missing = await session.post(path, json=args["json"])
        assert missing.status_code == 422


async def test_http_principal_and_agent_acl(tmp_path):
    tokens = {"a" * 24: "alice", "b" * 24: "bob"}
    settings = Settings(data_dir=tmp_path, auth_mode="bearer", bearer_principals=tokens)

    async def access(actor, agent_id):
        return actor.owner == "alice" or agent_id == "classification"

    async with client(tmp_path, settings=settings, authorize_agent=access) as (session, app, _):
        headers = {"Authorization": "Bearer " + "a" * 24, "Idempotency-Key": "key"}
        response = await session.post(
            "/api/v1/agents/calculation/runs",
            headers=headers,
            json={"input": {"left": 1, "right": 2}},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        await final(app.state.service.runtime, "alice", run_id)
        bob = {"Authorization": "Bearer " + "b" * 24}
        assert [
            entry["id"]
            for entry in (await session.get("/api/v1/agents", headers=bob)).json()["agents"]
        ] == ["classification"]
        for suffix, method in [("", "GET"), ("/cancel", "POST"), ("/resume", "POST")]:
            denied = await session.request(
                method,
                f"/api/v1/agents/calculation/runs/{run_id}{suffix}",
                headers=bob,
                json={} if method == "POST" else None,
            )
            assert denied.status_code == 404
        for headers in [{}, {"Authorization": "Bearer " + "c" * 24}, {**bob, "X-Scopes": "write"}]:
            assert (await session.get("/api/v1/agents", headers=headers)).status_code in {401, 403}


def action_package(*, write=True):
    return AgentPackage.model_validate(
        {
            "tools": [
                {
                    "name": "extension.action",
                    "handler": "extension.action",
                    "description": "Trusted test action",
                    "scope": "memory:write" if write else "read",
                    "read_only": not write,
                    "idempotency": "business-key" if write else "read-only",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                }
            ],
            "agents": [
                {
                    "id": "extension",
                    "description": "No-index extension",
                    "workflow": "fixed",
                    "prompt": {"system_template": "Run declared steps."},
                    "input_schema": {"type": "object"},
                    "tools": ["extension.action"],
                    "tasks": [{"id": "action", "tool": "extension.action", "arguments": {}}],
                }
            ],
        }
    )


async def test_http_resume_requires_exact_visible_review(tmp_path):
    calls = []

    async def handler(arguments, context):
        calls.append(context.operation_key)
        return {"value": 7}

    def configure(registry):
        register_workflows(registry)
        registry.register_handler("extension.action", handler)
        registry.bind(action_package())

    async with client(tmp_path, configure_registry=configure) as (session, app, _):
        created = (
            await session.post(
                "/api/v1/agents/extension/runs",
                headers={"Idempotency-Key": "write"},
                json={"input": {}},
            )
        ).json()
        engine = app.state.service.runtime
        review = await final(engine, "local", created["run_id"], "awaiting_review")
        before = engine.store.request("local", review["run_id"])
        challenge = review["interrupts"][0]
        url = f"/api/v1/agents/extension/runs/{review['run_id']}/resume"
        for body in [
            {},
            {"approve": True},
            {"review_id": challenge["review_id"]},
            {"review_id": "wrong", "digest": challenge["digest"]},
            {"review_id": challenge["review_id"], "digest": "sha256:" + "0" * 64},
        ]:
            response = await session.post(url, json=body)
            assert response.status_code in {409, 422}, response.text
            assert engine.store.request("local", review["run_id"])["deadline"] == before["deadline"]
            assert calls == [] and engine.actions.operations("local", review["run_id"]) == []
        assert (
            await session.post(url, json={key: challenge[key] for key in ("review_id", "digest")})
        ).status_code == 200
        await final(engine, "local", review["run_id"])
        assert len(calls) == 1
        assert engine.actions.operations("local", review["run_id"])[0]["attempts"] == 1


async def test_foreign_domain_sources_use_both_trusted_policies(tmp_path):
    allowed = True
    checked = []

    async def handler(arguments, context):
        return {"value": 7}

    async def result_policy(context, tool, value):
        return ToolResult(value=value, sources=[{"kind": "extension.resource", "id": "one"}])

    def configure(registry):
        register_workflows(registry)
        registry.register_handler("extension.action", handler)
        registry.bind(action_package(write=False))

    async def checker(actor, agent_id, source):
        checked.append((actor.owner, agent_id, source["id"]))
        return allowed

    def guard(db, row, source):
        assert db.execute("SELECT 1").fetchone()[0] == 1
        return allowed and row["owner"] == "local" and source["id"] == "one"

    async with client(
        tmp_path,
        configure_registry=configure,
        source_checker=checker,
        local_source_guard=guard,
        result_policy=result_policy,
    ) as (session, app, _):
        created = (
            await session.post(
                "/api/v1/agents/extension/runs",
                headers={"Idempotency-Key": "source"},
                json={"input": {}},
            )
        ).json()
        completed = await final(app.state.service.runtime, "local", created["run_id"])
        assert completed["result"]["action"]["value"] == {"value": 7} and checked
        allowed = False
        revoked = await session.get(f"/api/v1/agents/extension/runs/{created['run_id']}")
        assert revoked.json()["result"] is None and revoked.json()["result_visibility"] == "revoked"


async def test_registry_assembly_rejects_async_callback(tmp_path):
    async def invalid(registry):
        raise AssertionError("callback must not execute")

    app = create_app(Settings(data_dir=tmp_path), configure_registry=invalid)
    with pytest.raises(AgentError, match="synchronous"):
        async with app.router.lifespan_context(app):
            pass


async def test_generic_chunked_body_is_bounded_before_admission(tmp_path):
    settings = Settings(data_dir=tmp_path, agent_http=AgentHttpLimits(request_bytes=1024))
    async with client(tmp_path, settings=settings) as (session, app, _):

        async def chunks():
            for _ in range(3):
                yield b"x" * 512

        response = await session.post(
            "/api/v1/agents/calculation/runs",
            headers={"Idempotency-Key": "large"},
            content=chunks(),
        )
        assert response.status_code == 413
        assert app.state.service.runtime.store.lookup("local", "calculation", "large") is None
