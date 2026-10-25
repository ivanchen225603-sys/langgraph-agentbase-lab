"""Explicit loopback-only intent UI fixture; no model, business writes, or execution."""

import argparse
import html
import ipaddress
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from agentbase.configuration import AgentPackage
from agentbase.intent_proposals import IntentProposalLimits
from agentbase.workflows import register_workflows
from raglab.api import create_app
from raglab.config import Settings


def build_fixture(evidence: Path, port: int):
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "screenshots").mkdir()
    state = {"tool_calls": 0, "confirm_requests": 0, "revoke_requests": 0, "clock_offset": 0}

    async def never_execute(arguments, context):
        state["tool_calls"] += 1
        raise AssertionError("intent proposal/confirmation must never execute a tool")

    def configure(registry):
        register_workflows(registry)
        registry.register_handler("intent.fixture", never_execute)
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        registry.bind(
            AgentPackage.model_validate(
                {
                    "tools": [
                        {
                            "name": "intent.fixture",
                            "handler": "intent.fixture",
                            "description": "TEST ONLY: must never execute",
                            "input_schema": schema,
                            "output_schema": {"type": "null"},
                        }
                    ],
                    "agents": [
                        {
                            "id": "intent-demo",
                            "description": "TEST ONLY: proposal metadata, no execution",
                            "workflow": "fixed",
                            "prompt": {"system_template": "Synthetic fixture"},
                            "input_schema": schema,
                            "tools": ["intent.fixture"],
                            "tasks": [
                                {
                                    "id": "never",
                                    "tool": "intent.fixture",
                                    "input_bindings": {"text": "/text"},
                                }
                            ],
                        }
                    ],
                }
            )
        )

    app = create_app(
        Settings(mode="local", data_dir=evidence / "state", port=port),
        configure_registry=configure,
        intent_limits=IntentProposalLimits(proposal_seconds=120.0),
    )
    original_lifespan = app.router.lifespan_context

    def snapshot():
        with app.state.service.runtime.store.connection() as db:
            return {
                "fixture": dict(state),
                "business_counts": {
                    table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in (
                        "requests",
                        "agent_tool_operations",
                        "agent_reviews",
                        "agent_memories",
                    )
                },
                "proposals": [
                    dict(row)
                    for row in db.execute(
                        "SELECT id,revision,status,confirmed_at,revoked_at FROM agent_intent_proposals ORDER BY created_at"
                    )
                ],
            }

    @asynccontextmanager
    async def lifespan(current):
        async with original_lifespan(current):
            app.state.intents.clock = lambda: time.time() + state["clock_offset"]
            try:
                yield
            finally:
                (evidence / "final.json").write_text(
                    json.dumps(snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def record_choices(request, call_next):
        if request.method == "POST" and request.url.path.endswith("/confirm"):
            state["confirm_requests"] += 1
        if request.method == "POST" and request.url.path.endswith("/revoke"):
            state["revoke_requests"] += 1
        return await call_next(request)

    def local(request):
        if not request.client or not ipaddress.ip_address(request.client.host).is_loopback:
            raise HTTPException(403)
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403)

    @app.get("/__test", response_class=HTMLResponse)
    async def controls(request: Request):
        local(request)
        return (
            "<!doctype html><meta charset='utf-8'><title>TEST ONLY intent controls</title>"
            "<h1>TEST ONLY: intent metadata controls</h1>"
            f"<pre>{html.escape(json.dumps(snapshot(), ensure_ascii=False, indent=2))}</pre>"
            "<form method='post' action='/__test/expire'><button>Advance test clock by 121 seconds</button></form>"
            "<form method='post' action='/__test/shutdown'><button>Stop intent fixture</button></form>"
        )

    @app.post("/__test/expire")
    async def expire(request: Request):
        local(request)
        state["clock_offset"] += 121
        return RedirectResponse("/__test", status_code=303)

    @app.post("/__test/shutdown")
    async def shutdown(request: Request):
        local(request)
        app.state.fixture_server.should_exit = True
        return HTMLResponse("<h1>Intent fixture shutdown requested</h1>")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18085)
    args = parser.parse_args()
    app = build_fixture(args.evidence_dir.resolve(), args.port)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
    )
    app.state.fixture_server = server
    server.run()


if __name__ == "__main__":
    main()
