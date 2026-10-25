"""Explicit in-process arithmetic demo; no socket, model or external write service."""

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from agentbase.application import create_agent_app, open_runtime
from agentbase.contracts import AgentError, Identity
from agentbase.intent_execution import AGENT, register_synthetic_intent_agent
from agentbase.intent_proposals import OfflineStructuredIntent
from agentbase.registry import ComponentRegistry
from agentbase.runtime_contracts import TERMINAL
from agentbase.workflows import register_workflows


async def run_demo():
    registry = ComponentRegistry()
    register_workflows(registry)
    binding = register_synthetic_intent_agent(registry)
    actor = Identity(owner="synthetic-demo", scopes=frozenset({"read"}))

    async def identity(owner):
        if owner != actor.owner:
            raise AgentError("identity", "unknown synthetic identity", 403)
        return actor

    async def authenticate(request):
        if request.headers.get("authorization") != "Bearer synthetic-local-intent-fixture":
            raise AgentError("unauthorized", "synthetic fixture credential required", 401)
        return actor

    with tempfile.TemporaryDirectory(prefix="intent-synthetic-") as folder:

        @asynccontextmanager
        async def factory():
            async with open_runtime(
                registry, data_dir=Path(folder), identity_resolver=identity
            ) as engine:
                yield engine

        app = create_agent_app(
            factory,
            authenticate=authenticate,
            refresh_identity=authenticate,
            intent_interpreter=OfflineStructuredIntent(),
            synthetic_intent=binding,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"authorization": "Bearer synthetic-local-intent-fixture"},
            ) as client:
                base = f"/api/v1/agents/{AGENT}/intent-sessions"
                opened = await client.post(base, json={"conversation_id": "demo"})
                opened.raise_for_status()
                path = base + "/" + opened.json()["session_id"]
                response = await client.post(
                    path + "/proposals",
                    json={
                        "expected_revision": 0,
                        "draft": {
                            "text": "生成数字的本地只读提案",
                            "action": "propose_task",
                            "task": {"input": {"number": 7}},
                        },
                    },
                )
                response.raise_for_status()
                proposal = response.json()
                uri = path + "/proposals/" + proposal["proposal_id"]
                choice = {
                    key: proposal[key] for key in ("revision", "request_digest", "decision_digest")
                } | {"draft": proposal["payload"]["draft"]}
                confirmed = await client.post(uri + "/confirm", json=choice)
                confirmed.raise_for_status()
                with app.state.engine.store.connection() as db:
                    assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
                first = await client.post(uri + "/execute", json=choice)
                first.raise_for_status()
                again = await client.post(uri + "/execute", json=choice)
                again.raise_for_status()
                assert first.json()["run"]["run_id"] == again.json()["run"]["run_id"]
                async with asyncio.timeout(10):
                    while True:
                        query = await client.get(uri + "/execution")
                        query.raise_for_status()
                        result = query.json()["run"]
                        if result["status"] in TERMINAL:
                            break
                        await asyncio.sleep(0.01)
                assert result["status"] == "succeeded" and result["result"]["double"]["value"] == {
                    "doubled": 14
                }
                with app.state.engine.store.connection() as db:
                    counts = {
                        name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                        for name in (
                            "requests",
                            "agent_tool_operations",
                            "agent_reviews",
                            "agent_memories",
                            "agent_model_reservations",
                        )
                    }
                assert counts == {
                    "requests": 1,
                    "agent_tool_operations": 1,
                    "agent_reviews": 0,
                    "agent_memories": 0,
                    "agent_model_reservations": 0,
                }
                return {
                    "mode": "synthetic-local-read-only",
                    "result": {"doubled": 14},
                    "one_run_for_retries": True,
                    "confirmation_alone_created_no_run": True,
                    "counts": counts,
                    "scope": "temporary synthetic database and in-process ASGI only; default hosts remain disconnected",
                }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), ensure_ascii=False))
