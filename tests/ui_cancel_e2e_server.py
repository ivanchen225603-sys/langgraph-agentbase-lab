"""Explicit, loopback-only UI acceptance fixture; never a production entry point.

Run with a NEW evidence directory. Uses the product UI without injected test scripts and one read-only
fake tool, no model/provider. /__test controls only release its cleanup gate.
"""

import argparse
import asyncio
import html
import ipaddress
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError
from agentbase.workflows import register_workflows
from raglab.api import create_app
from raglab.config import Settings

AGENT = "ui-cancel-fixture"


def build_fixture(evidence: Path, port: int, fail_cleanup=False):
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "screenshots").mkdir()
    release = asyncio.Event()
    state = {
        "phase": "starting",
        "events": [],
        "run_ids": [],
        "calls": [],
        "queries": {},
        "cancels": {},
    }

    def record(kind, **fields):
        event = {
            "seq": len(state["events"]) + 1,
            "utc": datetime.now(UTC).isoformat(),
            "kind": kind,
            **fields,
        }
        state["events"].append(event)
        with (evidence / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def slow(arguments, context):
        text = arguments["text"]
        state["calls"].append({"run_id": context.run_id, "text": text})
        record("handler.enter", run_id=context.run_id, text=text)
        if text.startswith("OLD"):
            release.clear()
            state["phase"] = "old_running"
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                state["phase"] = "old_cleanup_pending"
                record("cleanup.begin", run_id=context.run_id)
                try:
                    async with asyncio.timeout(90):
                        await release.wait()
                except TimeoutError:
                    record("cleanup.watchdog_release", run_id=context.run_id)
                state["phase"] = "old_cleanup_complete"
                record("cleanup.end", run_id=context.run_id)
                # Deliberately swallow cancellation and produce a late candidate.
                record("old.late_candidate", run_id=context.run_id)
                return {"marker": "OLD_LATE_RESULT_MUST_NOT_PUBLISH", "text": text}
        state["phase"] = "new_handler_complete"
        record("new.handler_complete", run_id=context.run_id)
        return {"marker": "NEW_RESULT_MARKER", "text": text}

    def configure(registry):
        register_workflows(registry)
        registry.register_handler("ui.fixture.slow", slow)
        registry.bind(
            AgentPackage.model_validate(
                {
                    "tools": [
                        {
                            "name": "ui.fixture",
                            "handler": "ui.fixture.slow",
                            "description": "TEST ONLY: synthetic read with a controlled cleanup gate",
                            "scope": "read",
                            "read_only": True,
                            "retry_safe": False,
                            "timeout_seconds": 120.0,
                            "input_schema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                            "output_schema": {
                                "type": "object",
                                "properties": {
                                    "marker": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "required": ["marker", "text"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                    "agents": [
                        {
                            "id": AGENT,
                            "description": "TEST ONLY: browser cancellation acceptance",
                            "workflow": "fixed",
                            "prompt": {"system_template": "No model is used."},
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "default": "OLD_REQUEST"}
                                },
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                            "tools": ["ui.fixture"],
                            "budget": {"timeout_seconds": 120.0},
                            "tasks": [
                                {
                                    "id": "read",
                                    "tool": "ui.fixture",
                                    "input_bindings": {"text": "/text"},
                                }
                            ],
                        }
                    ],
                }
            )
        )

    settings = Settings(
        _env_file=None,
        mode="local",
        auth_mode="local",
        host="127.0.0.1",
        port=port,
        openai_api_key=None,
        bearer_principals={},
        data_dir=evidence / "state",
        knowledge_root=Path(__file__).resolve().parents[1] / "data" / "knowledge",
        tools_concurrency=1,
        workflows=2,
        enable_simulation=False,
    )
    app = create_app(settings, configure_registry=configure)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            engine = app.state.service.runtime
            previous = engine.observer

            def observe(kind, row, data):
                if row["agent_id"] == AGENT:
                    if row["run_id"] not in state["run_ids"]:
                        state["run_ids"].append(row["run_id"])
                    record(kind, run_id=row["run_id"], data=data)
                if previous is not None:
                    previous(kind, row, data)

            engine.observer = observe
            if fail_cleanup:
                original_terminal = engine.terminal_checkpoint

                async def fail_terminal(row, status, code):
                    if row["agent_id"] == AGENT:
                        record("fixture.cleanup_failure", run_id=row["run_id"])
                        raise AgentError("checkpoint_unavailable", "TEST ONLY cleanup failure", 503)
                    return await original_terminal(row, status, code)

                engine.terminal_checkpoint = fail_terminal
            state["phase"] = "ready"
            record("fixture.ready", port=port)
            try:
                yield
            finally:
                release.set()
                views = [await engine.view("local", run_id) for run_id in state["run_ids"]]
                histories = {
                    str(epoch): [
                        [message.model_dump(mode="json") for message in turn]
                        for turn in engine.contexts.history.turns(
                            "local", AGENT, "ui-e2e", epoch, 20
                        )
                    ]
                    for epoch in range(4)
                }
                (evidence / "final.json").write_text(
                    json.dumps(
                        {"state": state, "views": views, "histories": histories},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                record("fixture.shutdown")

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def count_observations(request, call_next):
        prefix = f"/api/v1/agents/{AGENT}/runs/"
        path = request.url.path
        if path.startswith(prefix):
            tail = path[len(prefix) :]
            if request.method == "GET" and "/" not in tail:
                state["queries"][tail] = state["queries"].get(tail, 0) + 1
                record("ui.query", run_id=tail)
            if request.method == "POST" and tail.endswith("/cancel"):
                run_id = tail.split("/")[0]
                state["cancels"][run_id] = state["cancels"].get(run_id, 0) + 1
                record("ui.cancel", run_id=run_id)
        return await call_next(request)

    def local(request):
        if not request.client or not ipaddress.ip_address(request.client.host).is_loopback:
            raise ValueError("fixture controls are loopback-only")
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise ValueError("fixture controls require same origin")

    @app.get("/__test", response_class=HTMLResponse)
    async def controls(request: Request):
        local(request)
        summary = html.escape(
            json.dumps(
                {key: value for key, value in state.items() if key != "events"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return (
            "<!doctype html><meta charset='utf-8'><title>TEST ONLY cleanup controls</title>"
            "<h1>TEST ONLY: cleanup gate</h1><p>Product UI (without test scripts) at <a href='/'>/</a>.</p>"
            f"<pre>{summary}</pre><form method='post' action='/__test/release'>"
            "<button>Release old cleanup (test only)</button></form>"
            "<form method='post' action='/__test/shutdown'>"
            "<button>Stop fixture server (test only)</button></form>"
        )

    @app.post("/__test/release")
    async def release_cleanup(request: Request):
        local(request)
        record("control.release")
        release.set()
        return RedirectResponse("/__test", status_code=303)

    @app.post("/__test/shutdown")
    async def shutdown(request: Request):
        local(request)
        record("control.shutdown")
        app.state.fixture_server.should_exit = True
        return HTMLResponse("<h1>Fixture shutdown requested</h1>")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument(
        "--fail-cleanup", action="store_true", help="TEST ONLY: inject terminal storage failure"
    )
    args = parser.parse_args()
    app = build_fixture(args.evidence_dir.resolve(), args.port, args.fail_cleanup)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
    )
    app.state.fixture_server = server
    server.run()


if __name__ == "__main__":
    main()
