"""Real loopback TCP proves first delta delivery and read-only reconnection."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager, nullcontext

import httpx
import pytest
import uvicorn

from agentbase import AgentPackage
from agentbase.contracts import ModelChunk, ToolCallDelta
from agentbase.workflows import register_workflows
from raglab.api import create_app
from raglab.config import Settings
from tests.test_agentbase_http import final


class TcpFixture:
    native_stream = True

    def __init__(self, use_tool):
        self.use_tool = use_tool
        self.requests, self.handler_calls = [], []
        self.next_delta, self.finish, self.second_sent = (asyncio.Event() for _ in range(3))
        self.finished = False

    async def complete(self, request):
        raise AssertionError("the transport must use native provider streaming")

    async def stream(self, request):
        self.requests.append(request)
        if self.use_tool and len(self.requests) == 1:
            yield ModelChunk(
                tool_call_deltas=[
                    ToolCallDelta(index=0, call_id="read", name="tcp.read", arguments_delta="{}")
                ]
            )
            yield ModelChunk(finish_reason="tool_calls")
            return
        yield ModelChunk(text_delta="候选\n第一段")
        await self.next_delta.wait()
        yield ModelChunk(text_delta="第二段")
        self.second_sent.set()
        await self.finish.wait()
        self.finished = True
        yield ModelChunk(finish_reason="stop")

    def count_input_tokens(self, request):
        return 1

    def count_output_tokens(self, text):
        return len(text)

    async def handler(self, arguments, context):
        self.handler_calls.append((context.operation_key, context.attempt_token))
        return {"checked": True}

    def configure(self, registry):
        register_workflows(registry)
        registry.register_provider("tcp.fixture", lambda config: self)
        registry.register_handler("tcp.read", self.handler)
        registry.bind(
            AgentPackage.model_validate(
                {
                    "models": [{"id": "fixture", "provider": "tcp.fixture", "model": "scripted"}],
                    "tools": [
                        {
                            "name": "tcp.read",
                            "handler": "tcp.read",
                            "description": "Synthetic read",
                            "scope": "read",
                            "input_schema": {"type": "object"},
                            "output_schema": {"type": "object"},
                        }
                    ],
                    "agents": [
                        {
                            "id": "tcp",
                            "description": "No-index TCP protocol fixture",
                            "model": "fixture",
                            "tools": ["tcp.read"] if self.use_tool else [],
                            "prompt": {"system_template": "Use declared tools."},
                            "input_schema": {"type": "string"},
                        }
                    ],
                }
            )
        )


class QuietServer(uvicorn.Server):
    def capture_signals(self):
        return nullcontext()


@asynccontextmanager
async def running(tmp_path, *, use_tool=False):
    provider = TcpFixture(use_tool)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    port = sock.getsockname()[1]
    app = create_app(
        Settings(mode="local", data_dir=tmp_path, port=port), configure_registry=provider.configure
    )
    server = QuietServer(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
            http="h11",
            ws="none",
            timeout_graceful_shutdown=2,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            for _ in range(1000):
                if server.started:
                    break
                if task.done():
                    await task
                    raise AssertionError("server exited before startup")
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("TCP server did not start")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as session:
            yield session, app, provider
    finally:
        provider.next_delta.set()
        provider.finish.set()
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 10)
        finally:
            sock.close()


async def next_event(lines):
    event_id = None
    async with asyncio.timeout(5):
        async for line in lines:
            if line.startswith("id: "):
                event_id = line[4:]
            if line.startswith("data: "):
                value = json.loads(line[6:])
                if "seq" in value:
                    return event_id, value
                raise AssertionError(value)
    raise AssertionError("stream ended before the event")


async def first_delta(lines):
    for _ in range(100):
        event_id, event = await next_event(lines)
        if event["kind"] == "model.delta":
            return event_id, event
    raise AssertionError("model delta was not received")


@pytest.mark.parametrize("use_tool", [False, True])
async def test_tcp_delta_before_provider_finish_and_reconnect_without_execution(tmp_path, use_tool):
    async with running(tmp_path, use_tool=use_tool) as (session, app, provider):
        created = await session.post(
            "/api/v1/agents/tcp/runs",
            headers={"Idempotency-Key": "network"},
            json={"input": "Input"},
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        engine = app.state.service.runtime
        url = f"/api/v1/agents/tcp/runs/{run_id}/events"
        async with session.stream("GET", url) as response:
            assert response.status_code == 200
            cursor, event = await first_delta(response.aiter_lines())
            assert event["provisional"] and event["data"]["value"]["text"] == "候选\n第一段"
            assert not provider.finished and not provider.next_delta.is_set()
            assert run_id in engine.active
            assert engine.store.final_receipt("local", run_id) is None
            assert engine.contexts.history.turns("local", "tcp", "default", 0, 10) == []
        original_deadline = engine.store.request("local", run_id)["deadline"]
        assert engine.store.request("local", run_id)["cancel"] == 0
        provider.next_delta.set()
        await asyncio.wait_for(provider.second_sent.wait(), 5)
        before = engine.actions.operations("local", run_id)
        with engine.store.connection() as db:
            reservations = db.execute(
                "SELECT COUNT(*) FROM agent_model_reservations WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        calls = len(provider.requests)
        assert calls == (2 if use_tool else 1)
        assert len(provider.handler_calls) == (1 if use_tool else 0)
        async with session.stream("GET", url, headers={"Last-Event-ID": cursor}) as response:
            assert response.status_code == 200
            _, second = await next_event(response.aiter_lines())
            assert second["seq"] == event["seq"] + 1
            assert second["data"]["value"]["text"] == "第二段"
            assert len(provider.requests) == calls and not provider.finished
            assert engine.actions.operations("local", run_id) == before
            assert engine.store.request("local", run_id)["deadline"] == original_deadline
            with engine.store.connection() as db:
                assert (
                    db.execute(
                        "SELECT COUNT(*) FROM agent_model_reservations WHERE run_id=?", (run_id,)
                    ).fetchone()[0]
                    == reservations
                )
        provider.finish.set()
        completed = await final(engine, "local", run_id)
        assert completed["result"] == "候选\n第一段第二段"
        assert len(provider.handler_calls) == (1 if use_tool else 0)
        assert all(operation["attempts"] == 1 for operation in before)
