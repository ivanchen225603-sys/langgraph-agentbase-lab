"""Real in-process HTTP sequences supporting the two UI integrations."""

import asyncio

import pytest

from agentbase.contracts import AgentError
from agentbase.runtime_contracts import TERMINAL
from tests.test_run_discovery import BASE, host, rows


async def incomplete(session, app, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise AgentError("checkpoint_unavailable", "synthetic initial checkpoint outage", 503)

    with monkeypatch.context() as patch:
        patch.setattr(app.state.engine, "update_checkpoint", unavailable)
        response = await session.post(
            BASE,
            json={"input": {"text": "synthetic recovery"}},
            headers={"Idempotency-Key": "recover"},
        )
    assert response.status_code == 503
    run_id = app.state.engine.store.lookup("alice", "index-a", "recover")["run_id"]
    return run_id


async def test_discover_incomplete_query_then_explicit_empty_resume_runs_one_operation(
    tmp_path, monkeypatch
):
    async with host(tmp_path) as (session, app, state):
        run_id = await incomplete(session, app, monkeypatch)
        before = rows(app)
        index = await session.get(BASE, params={"limit": 20})
        assert index.json() == {
            "runs": [{"run_id": run_id, "agent_id": "index-a"}],
            "next_after": None,
        }
        detail = await session.get(BASE + "/" + run_id)
        assert detail.status_code == 200
        assert detail.json()["recoverable"] is True and detail.json()["interrupts"] == []
        assert detail.json()["status"] == "initialization_incomplete"
        assert rows(app) == before and state["calls"] == []
        resumed = await session.post(BASE + "/" + run_id + "/resume", json={})
        assert resumed.status_code == 200
        async with asyncio.timeout(10):
            while True:
                view = (await session.get(BASE + "/" + run_id)).json()
                if view["status"] in TERMINAL:
                    break
                await asyncio.sleep(0.01)
        assert view["status"] == "succeeded"
        assert view["result"]["echo"]["value"] == {"text": "synthetic recovery"}
        final = rows(app)
        assert len(final["requests"]) == len(final["agent_tool_operations"]) == 1
        assert final["agent_reviews"] == [] and final["agent_memories"] == []
        assert state["calls"] == [run_id]
        assert (await session.post(BASE + "/" + run_id + "/resume", json={})).status_code == 409
        assert state["calls"] == [run_id]


@pytest.mark.parametrize("change", ["acl", "identity", "owner", "expired"])
async def test_current_server_state_rejects_resume_after_preflight_without_running_tools(
    tmp_path, monkeypatch, change
):
    async with host(tmp_path) as (session, app, state):
        run_id = await incomplete(session, app, monkeypatch)
        path = BASE + "/" + run_id
        assert (await session.get(path)).json()["recoverable"] is True
        if change == "acl":
            state["denied"].add(("alice", "index-a"))
        elif change == "identity":
            state["actors"]["alice"] = state["actors"]["alice"].model_copy(
                update={"permission_version": 2}
            )
        elif change == "owner":
            session.headers["Authorization"] = "Bearer fixture-bob"
        else:
            with app.state.engine.store.connection() as db:
                db.execute("UPDATE requests SET deadline=0 WHERE run_id=?", (run_id,))
        response = await session.post(path + "/resume", json={})
        assert response.status_code in {403, 404, 409}
        final = rows(app)
        assert len(final["requests"]) == 1 and final["agent_tool_operations"] == []
        assert final["agent_reviews"] == [] and state["calls"] == []


async def test_current_review_cannot_be_bypassed_by_empty_resume(tmp_path):
    async with host(tmp_path) as (session, app, state):
        created = await session.post(
            BASE,
            json={
                "input": {"text": "synthetic exact review"},
                "options": {"pause_for_review": True},
            },
            headers={"Idempotency-Key": "review"},
        )
        path = BASE + "/" + created.json()["run_id"]
        async with asyncio.timeout(10):
            while True:
                view = (await session.get(path)).json()
                if view["status"] == "awaiting_review":
                    break
                await asyncio.sleep(0.01)
        assert view["recoverable"] is True and len(view["interrupts"]) == 1
        pending = view["interrupts"][0]
        assert (await session.post(path + "/resume", json={})).status_code == 409
        assert (
            await session.post(
                path + "/resume",
                json={"review_id": pending["review_id"], "digest": "sha256:" + "0" * 64},
            )
        ).status_code == 409
        assert state["calls"] == []
        approved = await session.post(
            path + "/resume", json={key: pending[key] for key in ("review_id", "digest")}
        )
        assert approved.status_code == 200
        async with asyncio.timeout(10):
            while True:
                if (await session.get(path)).json()["status"] in TERMINAL:
                    break
                await asyncio.sleep(0.01)
        assert len(state["calls"]) == 1


async def test_actual_rag_host_serves_new_controls_and_shared_index_route(tmp_path):
    from tests.test_agentbase_http import client

    async with client(tmp_path) as (session, _, _):
        page = await session.get("/")
        assert page.status_code == 200
        for identifier in (
            "generic-run-list",
            "generic-run-refresh",
            "generic-run-next",
            "generic-run-open",
            "generic-resume",
        ):
            assert f'id="{identifier}"' in page.text
        source = await session.get("/static/agent-ui.js?v=0.3.0-discovery")
        assert source.status_code == 200 and "currentRun" in source.text
        index = await session.get("/api/v1/agents/calculation/runs?limit=20")
        assert index.status_code == 200 and index.json() == {"runs": [], "next_after": None}
