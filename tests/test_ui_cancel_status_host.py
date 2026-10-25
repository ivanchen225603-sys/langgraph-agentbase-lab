"""Synthetic HTTP host views consumed by the cancellation observer."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from agentbase.contracts import AgentError
from agentbase.http import mount_agent_routes
from tests.test_agentbase_cancel_admission import system, until


async def test_cancellation_module_has_browser_javascript_mime(tmp_path):
    from tests.test_agentbase_http import client

    async with client(tmp_path) as (session, _, _):
        response = await session.get("/static/cancel-status.mjs")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/javascript")
        assert "export class CancellationMonitor" in response.text


@pytest.mark.parametrize("failure", [False, True])
async def test_http_cleanup_pending_and_terminal_or_unknown(tmp_path, failure):
    entered, cleaning, release = [asyncio.Event() for _ in range(3)]

    async def handler(arguments, context):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async with system(tmp_path, handler, wait_seconds=5.0) as (engine, identity):
        app = FastAPI()

        async def authenticate(request):
            return identity

        mount_agent_routes(app, lambda: engine, authenticate, refresh_identity=authenticate)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            created = await client.post(
                "/api/v1/agents/worker/runs",
                json={"input": {"text": "old"}},
                headers={"Idempotency-Key": "ui-observe"},
            )
            run_id = created.json()["run_id"]
            await asyncio.wait_for(entered.wait(), 3)
            if failure:

                async def broken(*args):
                    raise AgentError("checkpoint_unavailable", "synthetic cleanup failure", 503)

                engine.terminal_checkpoint = broken
            cancelling = asyncio.create_task(
                client.post(f"/api/v1/agents/worker/runs/{run_id}/cancel", json={})
            )
            try:
                await asyncio.wait_for(cleaning.wait(), 2)
                pending = (await client.get(f"/api/v1/agents/worker/runs/{run_id}")).json()
                assert pending["status"] == "cancelling" and pending["cleanup_pending"] is True
                assert pending["cancel_requested"] and pending["result"] is None
            finally:
                release.set()
            response = await cancelling
            if failure:
                assert (
                    response.status_code == 503
                    and response.json()["error"] == "cancel_cleanup_failed"
                )
                unknown = (await client.get(f"/api/v1/agents/worker/runs/{run_id}")).json()
                assert unknown["status"] not in {"cancelled", "succeeded"}
                assert unknown["cleanup_pending"] is False and unknown["cancel_requested"]
                assert unknown["result"] is None and not unknown["recoverable"]
            else:
                final = await until(engine, run_id, "cancelled")
                assert response.status_code == 200
                assert final["cleanup_pending"] is False and final["result"] is None
