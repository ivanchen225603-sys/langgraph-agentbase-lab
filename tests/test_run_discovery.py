"""Owner/Agent-scoped discovery uses the normal host, without executing a Run."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from agentbase.application import create_agent_app, open_runtime
from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Identity
from agentbase.registry import ComponentRegistry
from agentbase.runtime_contracts import TERMINAL
from agentbase.workflows import register_workflows

BASE = "/api/v1/agents/index-a/runs"


@asynccontextmanager
async def host(path):
    registry = ComponentRegistry()
    register_workflows(registry)
    state = {
        "actors": {
            "alice": Identity(owner="alice", scopes=frozenset({"read"})),
            "bob": Identity(owner="bob", scopes=frozenset({"read"})),
        },
        "tokens": {"fixture-alice": "alice", "fixture-bob": "bob"},
        "denied": set(),
        "calls": [],
        "invalid_refresh": False,
    }

    async def echo(arguments, context):
        state["calls"].append(context.run_id)
        return arguments

    registry.register_handler("index.echo", echo, version="1")
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "maxLength": 100}},
        "required": ["text"],
        "additionalProperties": False,
    }
    registry.bind(
        AgentPackage.model_validate(
            {
                "tools": [
                    {
                        "name": "index.echo",
                        "handler": "index.echo",
                        "handler_version": "1",
                        "description": "Pure synthetic echo",
                        "input_schema": schema,
                        "output_schema": schema,
                    }
                ],
                "agents": [
                    {
                        "id": agent,
                        "workflow": "fixed",
                        "description": "Synthetic Run index fixture",
                        "prompt": {"system_template": "Run only the declared echo."},
                        "input_schema": schema,
                        "tools": ["index.echo"],
                        "tasks": [
                            {
                                "id": "echo",
                                "tool": "index.echo",
                                "input_bindings": {"text": "/text"},
                            }
                        ],
                        "context": {"memory_scope": "none", "max_memories": 0},
                    }
                    for agent in ("index-a", "index-b")
                ],
            }
        )
    )

    async def resolve(owner):
        if owner not in state["actors"]:
            raise AgentError("identity", "unknown fixture identity", 403)
        return state["actors"][owner]

    async def authenticate(request):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in state["tokens"]:
            raise AgentError("unauthorized", "fixture credential required", 401)
        return await resolve(state["tokens"][token])

    async def refresh(request):
        return False if state["invalid_refresh"] else await authenticate(request)

    async def authorize(identity, agent_id):
        return (identity.owner, agent_id) not in state["denied"]

    @asynccontextmanager
    async def factory():
        async with open_runtime(registry, data_dir=path, identity_resolver=resolve) as engine:
            yield engine

    app = create_agent_app(
        factory, authenticate=authenticate, refresh_identity=refresh, authorize_agent=authorize
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer fixture-alice"},
        ) as session:
            yield session, app, state


async def seed(session, app, key, *, owner="alice", agent="index-a"):
    response = await session.post(
        f"/api/v1/agents/{agent}/runs",
        json={"input": {"text": "PRIVATE-INPUT-" + key}},
        headers={"Idempotency-Key": key, "Authorization": "Bearer fixture-" + owner},
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    async with asyncio.timeout(10):
        while True:
            view = await app.state.engine.view(owner, run_id)
            if view["status"] in TERMINAL:
                assert view["status"] == "succeeded"
                return run_id
            await asyncio.sleep(0.01)


def rows(app):
    with app.state.engine.store.connection() as db:
        return {
            table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in (
                "requests",
                "agent_run_controls",
                "agent_tool_operations",
                "agent_reviews",
                "agent_memories",
                "agent_model_reservations",
            )
        }


async def test_empty_index_is_bounded_uncached_and_does_not_create_a_run(tmp_path):
    async with host(tmp_path) as (session, app, state):
        before = rows(app)
        response = await session.get(BASE)
        assert response.status_code == 200
        assert response.json() == {"runs": [], "next_after": None}
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["vary"] == "Authorization, Cookie"
        assert rows(app) == before and state["calls"] == []


async def test_index_filters_owner_and_agent_and_never_loads_result_or_payload(
    tmp_path, monkeypatch
):
    async with host(tmp_path) as (session, app, state):
        own = [await seed(session, app, "a1"), await seed(session, app, "a2")]
        foreign = await seed(session, app, "b1", owner="bob")
        other = await seed(session, app, "a3", agent="index-b")
        before, calls = rows(app), list(state["calls"])

        async def forbidden_view(*args, **kwargs):
            raise AssertionError("Run discovery must not project or reconcile results")

        monkeypatch.setattr(app.state.engine, "view", forbidden_view)
        response = await session.get(BASE)
        assert response.status_code == 200
        assert response.json() == {
            "runs": [{"run_id": run_id, "agent_id": "index-a"} for run_id in sorted(own)],
            "next_after": None,
        }
        assert foreign not in response.text and other not in response.text
        assert "PRIVATE" not in response.text
        assert set(response.json()["runs"][0]) == {"run_id", "agent_id"}
        bob = await session.get(BASE, headers={"Authorization": "Bearer fixture-bob"})
        assert bob.json()["runs"] == [{"run_id": foreign, "agent_id": "index-a"}]
        other_page = await session.get(BASE.replace("index-a", "index-b"))
        assert other_page.json()["runs"] == [{"run_id": other, "agent_id": "index-b"}]
        assert rows(app) == before and state["calls"] == calls


@pytest.mark.parametrize("limit", [1, 2, 50])
async def test_keyset_pagination_is_complete_ordered_and_has_no_duplicates(tmp_path, limit):
    async with host(tmp_path) as (session, app, _):
        expected = sorted([await seed(session, app, f"page-{index}") for index in range(5)])
        seen, after = [], None
        for _ in range(6):
            query = {"limit": limit} | ({"after": after} if after else {})
            response = await session.get(BASE, params=query)
            assert response.status_code == 200
            page = response.json()
            assert 0 < len(page["runs"]) <= limit
            seen.extend(row["run_id"] for row in page["runs"])
            after = page["next_after"]
            if after is None:
                break
            assert after == page["runs"][-1]["run_id"]
        assert seen == expected and len(set(seen)) == len(expected)
        end = await session.get(BASE, params={"after": expected[-1]})
        assert end.json() == {"runs": [], "next_after": None}


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=51",
        "limit=true",
        "limit=1.5",
        "after=",
        "after=a/b",
        "after=" + "a" * 65,
        "limit=1&limit=2",
        "after=a&after=b",
        "owner=bob",
        "status=succeeded",
        "offset=0",
    ],
)
async def test_malformed_repeated_and_unknown_parameters_are_rejected(tmp_path, query):
    async with host(tmp_path) as (session, app, state):
        before = rows(app)
        response = await session.get(BASE + "?" + query)
        assert response.status_code in {400, 422}
        assert rows(app) == before and state["calls"] == []


async def test_foreign_cursor_is_only_a_boundary_and_never_grants_access(tmp_path):
    async with host(tmp_path) as (session, app, _):
        own = await seed(session, app, "own")
        foreign = await seed(session, app, "foreign", owner="bob")
        response = await session.get(BASE, params={"after": foreign})
        assert response.status_code == 200 and foreign not in response.text
        assert [row["run_id"] for row in response.json()["runs"]] == (
            [own] if own > foreign else []
        )


async def test_original_credential_current_acl_and_registered_agent_are_required(tmp_path):
    async with host(tmp_path) as (session, app, state):
        assert (
            await session.get(BASE, headers={"Authorization": "Bearer absent"})
        ).status_code == 401
        state["denied"].add(("alice", "index-a"))
        assert (await session.get(BASE)).status_code == 403
        state["denied"].clear()

        async def runtime_acl(identity, agent_id):
            return agent_id != "index-a"

        app.state.engine.add_agent_authorizer(runtime_acl)
        assert (await session.get(BASE)).status_code == 403
        assert (await session.get(BASE.replace("index-a", "missing"))).status_code == 400


@pytest.mark.parametrize("change", ["acl", "identity", "credential", "owner", "invalid_refresh"])
async def test_authority_change_during_selection_discards_the_entire_page(
    tmp_path, monkeypatch, change
):
    async with host(tmp_path) as (session, app, state):
        run_id = await seed(session, app, "private")
        original = app.state.engine.store.list_run_ids

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if change == "acl":
                state["denied"].add(("alice", "index-a"))
            elif change == "identity":
                state["actors"]["alice"] = state["actors"]["alice"].model_copy(
                    update={"permission_version": 2}
                )
            elif change == "credential":
                del state["tokens"]["fixture-alice"]
            elif change == "owner":
                state["tokens"]["fixture-alice"] = "bob"
            else:
                state["invalid_refresh"] = True
            return result

        monkeypatch.setattr(app.state.engine.store, "list_run_ids", changed)
        before = rows(app)
        response = await session.get(BASE)
        assert response.status_code == (401 if change == "credential" else 403)
        assert run_id not in response.text and "runs" not in response.json()
        assert rows(app) == before


async def test_identity_change_inside_async_acl_is_rechecked_before_selection(
    tmp_path, monkeypatch
):
    async with host(tmp_path) as (session, app, state):

        async def change_identity(identity, agent_id):
            state["actors"]["alice"] = identity.model_copy(update={"permission_version": 2})
            return True

        def forbidden_selection(*args, **kwargs):
            raise AssertionError("changed identity must be rejected before selection")

        app.state.engine.add_agent_authorizer(change_identity)
        monkeypatch.setattr(app.state.engine.store, "list_run_ids", forbidden_selection)
        assert (await session.get(BASE)).status_code == 403


async def test_index_survives_restart_without_reexecution_and_detail_remains_separate(tmp_path):
    async with host(tmp_path) as (session, app, _):
        run_id = await seed(session, app, "durable")
        original_rows = rows(app)
    async with host(tmp_path) as (session, app, state):
        response = await session.get(BASE)
        assert response.json()["runs"] == [{"run_id": run_id, "agent_id": "index-a"}]
        assert state["calls"] == [] and rows(app) == original_rows
        detail = await session.get(BASE + "/" + run_id)
        assert detail.status_code == 200 and detail.json()["status"] == "succeeded"
        assert detail.json()["result"]["echo"]["value"] == {"text": "PRIVATE-INPUT-durable"}
        assert state["calls"] == []


@pytest.mark.parametrize("limit", [True, 0, 51, 1.5, "2"])
async def test_store_index_rejects_unbounded_or_coerced_limits(tmp_path, limit):
    async with host(tmp_path) as (_, app, _):
        with pytest.raises(AgentError) as error:
            app.state.engine.store.list_run_ids("alice", "index-a", limit=limit)
        assert error.value.code == "run_list_query"


async def test_index_openapi_declares_query_bounds_and_admits_no_owner_parameter(tmp_path):
    async with host(tmp_path) as (session, _, _):
        response = await session.get("/openapi.json")
        operation = response.json()["paths"]["/api/v1/agents/{agent_id}/runs"]["get"]
        parameters = {item["name"]: item for item in operation["parameters"]}
        assert set(parameters) == {"agent_id", "limit", "after"}
        assert parameters["limit"]["schema"]["minimum"] == 1
        assert parameters["limit"]["schema"]["maximum"] == 50
        assert "owner" not in json.dumps(operation)
