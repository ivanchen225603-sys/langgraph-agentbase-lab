"""Professional onboarding contracts exercise actual storage, tools and HTTP lifecycle."""

import asyncio
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, ValidationError

from agentbase import (
    AgentPackage,
    build_registry,
    create_agent_app,
    function_tool,
    langchain_tool,
    open_runtime,
)
from agentbase.contracts import AgentError, Identity, ModelChunk, RunContext, ToolCallDelta
from agentbase.evaluation import (
    EvalCase,
    diagnose_run,
    evaluate_cases,
    load_cases,
    result_failures,
    wait_for_outcome,
)
from agentbase.runtime import RunEngine
from agentbase.scaffold import TEMPLATES, scaffold


class Details(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int
    day: date
    category: Literal["a", "b"] = "a"


async def typed(details: Details, context: RunContext, suffix: str = "ok") -> Details:
    assert context.identity.owner == "alice"
    assert suffix == "ok"
    return details


def context():
    import time

    return RunContext(
        identity=Identity(owner="alice", scopes=frozenset({"read"})),
        agent_id="double",
        run_id="run1",
        conversation_id="default",
        config_digest="sha256:" + "0" * 64,
        deadline=time.time() + 30,
    )


async def test_typed_models_dates_defaults_and_context_injection():
    tool = function_tool(
        typed, name="typed", scope="read", read_only=True, context_parameter="context"
    )
    assert "context" not in tool.declaration.input_schema["properties"]
    assert tool.declaration.input_schema["additionalProperties"] is False
    result = await tool.handler({"details": {"count": 2, "day": "2026-10-09"}}, context())
    assert result == {"count": 2, "day": "2026-10-09", "category": "a"}
    for invalid in [
        {"details": {"count": "2", "day": "2026-10-09"}},
        {"details": {"count": 2, "day": "bad"}},
        {"details": {"count": 2, "day": "2026-10-09"}, "context": {"owner": "admin"}},
    ]:
        with pytest.raises(AgentError, match="declared types"):
            await tool.handler(invalid, context())


def test_adapters_require_async_types_and_explicit_effects():
    def sync(value: int) -> int:
        return value

    with pytest.raises(AgentError):
        function_tool(sync, name="sync", scope="write", read_only=False)
    with pytest.raises(AgentError):
        function_tool(typed, name="typed", scope="read", read_only=True)
    with pytest.raises(ValidationError):
        function_tool(
            typed, name="typed", scope="write", read_only=False, context_parameter="context"
        )

    async def untyped(value):
        return value

    with pytest.raises(AgentError):
        function_tool(untyped, name="untyped", scope="read", read_only=True)


async def test_return_validation_and_async_langchain():
    async def bad(value: int) -> int:
        return "secret-invalid-value"

    tool = function_tool(bad, name="bad", scope="read", read_only=True)
    with pytest.raises(AgentError) as invalid:
        await tool.handler({"value": 1}, context())
    assert "secret" not in str(invalid.value)

    async def double(value: int) -> int:
        """Double an integer."""
        return value * 2

    lc = StructuredTool.from_function(coroutine=double)
    tool = langchain_tool(lc, name="lc.double", output_type=int, scope="read", read_only=True)
    assert await tool.handler({"value": 3}, context()) == 6
    with pytest.raises(AgentError):
        await tool.handler({"value": "3"}, context())
    threaded = StructuredTool.from_function(
        func=lambda value: value, name="sync", description="Sync", args_schema=Details
    )
    with pytest.raises(AgentError):
        langchain_tool(threaded, name="sync", output_type=int, scope="read", read_only=True)


@pytest.mark.parametrize("policy", ["handle_tool_error", "handle_validation_error", "callbacks"])
def test_langchain_adapter_rejects_suppressed_failures_and_external_callbacks(policy):
    async def business(value: int) -> int:
        """Return a value."""
        return value

    options = {policy: [BaseCallbackHandler()] if policy == "callbacks" else True}
    tool = StructuredTool.from_function(coroutine=business, **options)
    with pytest.raises(AgentError, match="suppression and callbacks"):
        langchain_tool(tool, name="business", output_type=int, scope="read", read_only=True)


def registry(calls):
    async def double(value: int) -> int:
        calls.append(value)
        return value * 2

    tool = function_tool(double, name="double", scope="read", read_only=True)
    package = AgentPackage.model_validate(
        {
            "tools": [tool.declaration.model_dump(mode="json")],
            "agents": [
                {
                    "id": "double",
                    "description": "Fixed calculation",
                    "workflow": "fixed",
                    "prompt": {"system_template": "Calculate"},
                    "input_schema": tool.declaration.input_schema,
                    "tools": ["double"],
                    "tasks": [
                        {"id": "double", "tool": "double", "input_bindings": {"value": "/value"}}
                    ],
                }
            ],
        }
    )
    return build_registry(package, tools=[tool])


ACTOR = Identity(owner="alice", scopes=frozenset({"read"}))


async def identity(owner):
    return ACTOR if owner == ACTOR.owner else None


async def test_runtime_lock_lifecycle_real_run_and_replay(tmp_path):
    calls = []
    bound = registry(calls)
    async with open_runtime(bound, data_dir=tmp_path, identity_resolver=identity) as engine:
        async with asyncio.timeout(5):
            with pytest.raises(AgentError, match="acquired"):
                async with open_runtime(bound, data_dir=tmp_path, identity_resolver=identity):
                    pytest.fail("second runtime acquired the same directory")
        created = await engine.submit(ACTOR, "double", "key", {"value": 3})
        view = await wait_for_outcome(engine, "alice", created["run_id"])
        assert view["status"] == "succeeded" and view["result"]["double"]["value"] == 6
        assert (await engine.submit(ACTOR, "double", "key", {"value": 3}))["run_id"] == created[
            "run_id"
        ]
        assert calls == [3]
        with pytest.raises(AgentError):
            await engine.view("bob", created["run_id"])
        diagnosis = await diagnose_run(engine, "alice", created["run_id"])
        assert diagnosis["operation_phases"] == ["read_success"]
        assert diagnosis["applied_operations"] == 0
        assert "result" not in diagnosis
    assert not engine.workers and not engine.active
    async with open_runtime(bound, data_dir=tmp_path, identity_resolver=identity) as reopened:
        assert (await reopened.view("alice", created["run_id"]))["status"] == "succeeded"


async def test_runtime_rejects_symlink_lock_without_creating_databases(tmp_path, monkeypatch):
    from agentbase.application import RuntimeProcessLock

    lock = tmp_path / "owner.lock"
    lock.write_bytes(b"0")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == lock or original(path))
    assert RuntimeProcessLock(lock).path == lock
    with pytest.raises(AgentError, match="regular file"):
        async with open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity):
            pytest.fail("symlink lock accepted")
    assert not (tmp_path / "runs.sqlite").exists()
    assert not (tmp_path / "checkpoints.sqlite").exists()


async def test_runtime_rejects_directory_lock_without_creating_databases(tmp_path):
    (tmp_path / "owner.lock").mkdir()
    with pytest.raises(AgentError, match="regular file"):
        async with open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity):
            pytest.fail("directory lock accepted")
    assert not (tmp_path / "runs.sqlite").exists()


async def test_startup_failure_releases_workers_saver_and_lock(tmp_path, monkeypatch):
    instances = []
    original = RunEngine.start

    async def fail_after_start(self):
        instances.append(self)
        await original(self)
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(RunEngine, "start", fail_after_start)
    with pytest.raises(RuntimeError):
        async with open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity):
            pass
    assert instances and not instances[0].workers
    monkeypatch.setattr(RunEngine, "start", original)
    async with open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity):
        pass


async def test_shutdown_cancellation_keeps_lock_until_tasks_stop(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    manager = open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity)
    engine = await manager.__aenter__()
    original = engine.task_coordinator.close

    async def gated_close():
        entered.set()
        await release.wait()
        await original()

    engine.task_coordinator.close = gated_close
    exiting = asyncio.create_task(manager.__aexit__(None, None, None))
    await entered.wait()
    exiting.cancel()
    await asyncio.sleep(0.01)
    assert not exiting.done()
    with pytest.raises(AgentError):
        async with open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity):
            pass
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await exiting
    assert not engine.workers and not engine.active
    async with open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity):
        pass


@dataclass
class WrappedContext:
    context: RunContext


@pytest.mark.parametrize(
    "annotation", [RunContext | None, Annotated[RunContext, "injected"], WrappedContext]
)
def test_context_cannot_be_wrapped_into_model_input(annotation):
    async def business(payload) -> int:
        return 1

    business.__annotations__["payload"] = annotation
    with pytest.raises(AgentError, match="server injection"):
        function_tool(business, name="business", scope="read", read_only=True)


async def test_independent_http_auth_validation_and_lifecycle(tmp_path):
    async def authenticate(request):
        if request.headers.get("authorization") != "Bearer trusted":
            raise AgentError("identity", "unauthorized", 403)
        return ACTOR

    async def access(actor, agent_id):
        return agent_id == "double" and actor.owner == "alice"

    app = create_agent_app(
        lambda: open_runtime(registry([]), data_dir=tmp_path, identity_resolver=identity),
        authenticate=authenticate,
        refresh_identity=authenticate,
        authorize_agent=access,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        assert (await client.get("/health/ready/runtime")).status_code == 503
        async with app.router.lifespan_context(app):
            assert (await client.get("/health/ready/runtime")).status_code == 200
            assert (await client.get("/api/v1/agents")).status_code == 403
            headers = {"Authorization": "Bearer trusted", "Idempotency-Key": "http-key"}
            invalid = await client.post(
                "/api/v1/agents/double/runs",
                headers=headers,
                json={"input": {"value": 1}, "options": {"timeout_seconds": "secret-marker"}},
            )
            assert invalid.status_code == 422 and "secret-marker" not in invalid.text
            created = await client.post(
                "/api/v1/agents/double/runs", headers=headers, json={"input": {"value": 4}}
            )
            assert created.status_code == 202, created.text
            view = await wait_for_outcome(app.state.engine, "alice", created.json()["run_id"])
            assert view["result"]["double"]["value"] == 8
        assert app.state.engine is None
        assert (await client.get("/health/ready/runtime")).status_code == 503


@pytest.mark.parametrize(
    "status",
    [
        "queued",
        "running",
        "interrupted",
        "waiting_children",
        "initialization_incomplete",
        "finalization_incomplete",
        "failed",
        "partial",
    ],
)
def test_evaluator_does_not_pass_intermediate_or_wrong_terminal(status):
    case = EvalCase(id="case", input={}, checks=[{"pointer": "", "expected": "answer"}])
    assert "unexpected_status" in result_failures(case, {"status": status, "result": "answer"})


@pytest.mark.parametrize("visibility", ["revoked", "unverified"])
def test_evaluator_does_not_pass_revoked_results(visibility):
    case = EvalCase(id="case", input={}, checks=[{"expected": "answer"}])
    assert result_failures(
        case, {"status": "succeeded", "result": "answer", "result_visibility": visibility}
    )


def test_evaluator_checks_business_result_and_comparable_membership():
    case = EvalCase(id="case", input={}, checks=[{"pointer": "/payer", "expected": "甲"}])
    assert result_failures(case, {"status": "succeeded", "result": {"payer": "乙"}})
    case = EvalCase(id="case", input={}, checks=[{"operation": "not_contains", "expected": 123}])
    assert result_failures(case, {"status": "succeeded", "result": "secret"})
    with pytest.raises(ValidationError):
        EvalCase(id="empty", input={})
    number = EvalCase(id="number", input={}, checks=[{"expected": 120.0}])
    assert not result_failures(number, {"status": "succeeded", "result": 120})
    boolean = EvalCase(id="bool", input={}, checks=[{"expected": True}])
    assert result_failures(boolean, {"status": "succeeded", "result": 1})


def test_invalid_case_files_fail_closed(tmp_path):
    for content in [
        "",
        '{"id":"a","input":{},"checks":[{"expected":1}]}\n' * 2,
        '{"id":"a","input":{},"checks":[{"operation":"unknown","expected":1}]}',
    ]:
        path = tmp_path / "cases.jsonl"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(AgentError):
            load_cases(path)


def load_starter(template, directory):
    paths = scaffold(template, directory)
    assert len(paths) == 6
    with pytest.raises(FileExistsError):
        scaffold(template, directory)
    spec = importlib.util.spec_from_file_location(
        "starter_" + template.replace("-", "_"), directory / "agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text,label,enabled",
    [
        pytest.param("enabled: true", None, True, id="missing"),
        pytest.param("label:", None, None, id="empty"),
        pytest.param("label:   ", None, None, id="spaces"),
        pytest.param("label:\t", None, None, id="tab"),
        pytest.param("label: \u3000", None, None, id="unicode-space"),
        pytest.param("label: ; enabled: false", None, False, id="empty-before-semicolon"),
        pytest.param("label:\nenabled: true", None, True, id="lf"),
        pytest.param("label: \t\r\nenabled: true", None, True, id="crlf"),
        pytest.param("label:\renabled: false", None, False, id="cr"),
        pytest.param("label:\n \nenabled: false", None, False, id="blank-line"),
        pytest.param("label:\u2028enabled: true", None, True, id="unicode-line"),
        pytest.param("label:\u2029enabled: false", None, False, id="unicode-paragraph"),
        pytest.param("label\n: alpha\nenabled: true", None, True, id="colon-next-line"),
        pytest.param("label \t: alpha", "alpha", None, id="horizontal-spacing"),
        pytest.param("label: alpha; enabled: true", "alpha", True, id="semicolon"),
        pytest.param("label: alpha beta\nenabled: false", "alpha beta", False, id="multiline"),
        pytest.param("label： 示例\nenabled: true", "示例", True, id="unicode-value"),
        pytest.param("label: beta", "beta", None, id="label-only"),
    ],
)
async def test_extraction_label_stays_on_same_line_and_blank_is_null(
    tmp_path, text, label, enabled
):
    module = load_starter("extraction", tmp_path / "extraction")
    request = SimpleNamespace(
        messages=[SimpleNamespace(role="user", content=json.dumps({"text": text}))]
    )
    response = await module.offline_response(request)
    assert json.loads(response.text) == {"record_id": None, "label": label, "enabled": enabled}
    assert response.tool_calls == []


@pytest.mark.parametrize("template", ["extraction", "confirmed-action"])
async def test_starter_cli_wait_uses_bound_budget_for_submit_and_resume(
    tmp_path, monkeypatch, template
):
    directory = tmp_path / template
    module = load_starter(template, directory)
    monkeypatch.setitem(sys.modules, "agent", module)
    spec = importlib.util.spec_from_file_location("starter_runner", directory / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    waits = []

    async def scaled_wait(engine, owner, run_id, *, timeout_seconds=0.01):
        waits.append(timeout_seconds)
        assert timeout_seconds == 31
        return await wait_for_outcome(
            engine, owner, run_id, timeout_seconds=min(timeout_seconds, 2)
        )

    monkeypatch.setattr(runner, "wait_for_outcome", scaled_wait)
    if template == "extraction":
        response = module.offline_response

        async def delayed_response(request):
            await asyncio.sleep(0.05)
            return await response(request)

        monkeypatch.setattr(module, "offline_response", delayed_response)
    args = SimpleNamespace(
        mode="offline",
        model=None,
        data_dir=directory / ".runtime",
        cases=None,
        diagnose=None,
        resume=None,
        review_id=None,
        digest=None,
        input=directory / "sample.json",
        request_key="first",
    )
    outcome = await runner.run(args)
    if template == "confirmed-action":
        assert outcome["status"] == "awaiting_review"
        review = outcome["interrupts"][0]
        args.resume = outcome["run_id"]
        args.review_id, args.digest = review["review_id"], review["digest"]
        outcome = await runner.run(args)
        assert len(waits) == 2
    assert outcome["status"] == "succeeded", outcome


@pytest.mark.parametrize("template", TEMPLATES)
async def test_all_starters_cases_and_business_isolation(tmp_path, template):
    module = load_starter(template, tmp_path / template)
    if template == "extraction":
        assert module.AGENT_ID == "record-extraction"
        assert module.SCOPES == ()
        schema = module.Record.model_json_schema()
        assert set(schema["properties"]) == {"record_id", "label", "enabled"}
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

    class Fixture:
        native_stream = True

        async def complete(self, request):
            return await module.offline_response(request)

        async def stream(self, request):
            response = await self.complete(request)
            if response.text:
                yield ModelChunk(text_delta=response.text)
            for index, call in enumerate(response.tool_calls):
                yield ModelChunk(
                    tool_call_deltas=[
                        ToolCallDelta(
                            index=index,
                            call_id=call.call_id,
                            name=call.name,
                            arguments_delta=json.dumps(call.arguments),
                        )
                    ]
                )
            yield ModelChunk(finish_reason="tool_calls" if response.tool_calls else "stop")

        def count_input_tokens(self, request):
            return 1

        def count_output_tokens(self, text):
            return 1

    actor = Identity(owner="alice", scopes=frozenset(module.SCOPES))

    async def resolve(owner):
        return actor if owner == actor.owner else None

    data_dir = tmp_path / template / ".runtime"
    bound = module.create_registry(lambda config: Fixture(), "fixture", data_dir)
    async with open_runtime(bound, data_dir=data_dir, identity_resolver=resolve) as engine:
        if hasattr(module, "initialize"):
            module.initialize(engine)
        cases = load_cases(tmp_path / template / "cases.jsonl")
        assert len(cases) == 10
        report = await evaluate_cases(engine, actor, module.AGENT_ID, cases)
        assert report["failed"] == 0, report
        assert report["real_model_quality_verified"] is False
        if template != "confirmed-action":
            first = report["cases"][0]["diagnosis"]
            assert first["reported_usage"] is None and first["usage_coverage"]["unknown_calls"] > 0
        if template == "confirmed-action":
            with engine.store.connection() as db:
                assert db.execute("SELECT COUNT(*) FROM starter_notes").fetchone()[0] == 0
            for outcome in report["cases"]:
                if outcome["run_id"] is not None:
                    assert (await engine.view(actor.owner, outcome["run_id"]))[
                        "status"
                    ] == "cancelled"


async def test_confirmed_note_approval_exactness_receipts_and_cross_run_idempotency(tmp_path):
    module = load_starter("confirmed-action", tmp_path / "confirmed")
    actor = Identity(owner="alice", scopes=frozenset(module.SCOPES))

    async def resolve(owner):
        return actor if owner == actor.owner else None

    bound = module.create_registry(None, "unused", tmp_path)
    async with open_runtime(bound, data_dir=tmp_path, identity_resolver=resolve) as engine:
        module.initialize(engine)
        payload = {"business_key": "one", "text": "same content"}
        for key in ("first", "second"):
            created = await engine.submit(actor, module.AGENT_ID, key, payload)
            pending = await wait_for_outcome(engine, actor.owner, created["run_id"])
            assert pending["status"] == "awaiting_review"
            review = pending["interrupts"][0]
            with pytest.raises(AgentError):
                await engine.resume(
                    actor,
                    created["run_id"],
                    review_id=review["review_id"],
                    digest="sha256:" + "0" * 64,
                )
            with engine.store.connection() as db:
                assert db.execute("SELECT COUNT(*) FROM starter_notes").fetchone()[0] == (
                    0 if key == "first" else 1
                )
            await engine.resume(
                actor, created["run_id"], review_id=review["review_id"], digest=review["digest"]
            )
            final = await wait_for_outcome(engine, actor.owner, created["run_id"])
            assert final["status"] == "succeeded", final
            replay = await engine.submit(actor, module.AGENT_ID, key, payload)
            assert replay["run_id"] == created["run_id"]
        with engine.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM starter_notes").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM starter_note_receipts").fetchone()[0] == 2


async def test_evaluator_timeout_records_committed_admission_and_cleanup_failure(tmp_path):
    calls = []
    bound = registry(calls)
    async with open_runtime(bound, data_dir=tmp_path, identity_resolver=identity) as engine:
        original = engine.view

        async def slow(owner, run_id):
            await asyncio.sleep(1)
            return await original(owner, run_id)

        engine.view = slow
        case = EvalCase(
            id="timeout",
            input={"value": 2},
            timeout_seconds=0.05,
            checks=[{"pointer": "/double/value", "expected": 4}],
        )
        report = await evaluate_cases(engine, ACTOR, "double", [case])
        assert report["total"] == report["failed"] == 1
        assert report["cases"][0]["status"] == "incomplete"
        assert "evaluation_timeout" in report["cases"][0]["failures"]
        assert report["cases"][0]["run_id"] is not None
        assert "cleanup_failed" in report["cases"][0]["failures"]


@pytest.mark.parametrize("reason", ["other_owner", "cancelled", "expired"])
async def test_confirmed_note_rejects_unusable_approval_without_effect(tmp_path, reason):
    module = load_starter("confirmed-action", tmp_path / "confirmed")
    actors = {
        name: Identity(owner=name, scopes=frozenset(module.SCOPES)) for name in ("alice", "bob")
    }

    async def resolve(owner):
        return actors.get(owner)

    bound = module.create_registry(None, "unused", tmp_path)
    async with open_runtime(bound, data_dir=tmp_path, identity_resolver=resolve) as engine:
        module.initialize(engine)
        created = await engine.submit(
            actors["alice"], module.AGENT_ID, "first", {"business_key": "one", "text": "original"}
        )
        pending = await wait_for_outcome(engine, "alice", created["run_id"])
        assert pending["status"] == "awaiting_review"
        review = pending["interrupts"][0]
        if reason == "cancelled":
            await engine.cancel("alice", created["run_id"])
        elif reason == "expired":
            with engine.store.connection() as db:
                db.execute(
                    "UPDATE agent_reviews SET expires_at=0 WHERE id=?", (review["review_id"],)
                )
        with pytest.raises(AgentError):
            await engine.resume(
                actors["bob" if reason == "other_owner" else "alice"],
                created["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
        with engine.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM starter_notes").fetchone()[0] == 0
            assert db.execute("SELECT COUNT(*) FROM starter_note_receipts").fetchone()[0] == 0


async def test_confirmed_note_missing_scope_does_not_create_approval_or_effect(tmp_path):
    module = load_starter("confirmed-action", tmp_path / "confirmed")
    actor = Identity(owner="alice", scopes=frozenset())

    async def resolve(owner):
        return actor if owner == actor.owner else None

    async with open_runtime(
        module.create_registry(None, "unused", tmp_path),
        data_dir=tmp_path,
        identity_resolver=resolve,
    ) as engine:
        module.initialize(engine)
        with pytest.raises(AgentError) as rejected:
            await engine.submit(
                actor, module.AGENT_ID, "first", {"business_key": "one", "text": "original"}
            )
        assert rejected.value.code == "forbidden"
        with engine.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM starter_notes").fetchone()[0] == 0
            assert db.execute("SELECT COUNT(*) FROM agent_reviews").fetchone()[0] == 0


async def test_confirmed_note_conflict_and_owner_isolation(tmp_path):
    module = load_starter("confirmed-action", tmp_path / "confirmed")
    actors = {
        name: Identity(owner=name, scopes=frozenset(module.SCOPES)) for name in ("alice", "bob")
    }

    async def resolve(owner):
        return actors.get(owner)

    async with open_runtime(
        module.create_registry(None, "unused", tmp_path),
        data_dir=tmp_path,
        identity_resolver=resolve,
    ) as engine:
        module.initialize(engine)
        for owner, key, text in [
            ("alice", "first", "original"),
            ("alice", "conflict", "changed"),
            ("bob", "first", "bob's text"),
        ]:
            created = await engine.submit(
                actors[owner], module.AGENT_ID, key, {"business_key": "one", "text": text}
            )
            pending = await wait_for_outcome(engine, owner, created["run_id"])
            assert pending["status"] == "awaiting_review"
            review = pending["interrupts"][0]
            await engine.resume(
                actors[owner],
                created["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
            final = await wait_for_outcome(engine, owner, created["run_id"])
            if key == "conflict":
                assert final["status"] == "failed"
                assert final["result"]["save"]["error"] == "note_conflict"
            else:
                assert final["status"] == "succeeded"
        with engine.store.connection() as db:
            assert [
                tuple(row)
                for row in db.execute("SELECT owner,text FROM starter_notes ORDER BY owner")
            ] == [("alice", "original"), ("bob", "bob's text")]
            assert db.execute("SELECT COUNT(*) FROM starter_note_receipts").fetchone()[0] == 3
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM starter_note_receipts WHERE value='null'"
                ).fetchone()[0]
                == 1
            )
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM agent_tool_operations WHERE phase='not_applied'"
                ).fetchone()[0]
                == 1
            )
            operation = dict(
                db.execute(
                    "SELECT * FROM agent_tool_operations WHERE phase='applied' LIMIT 1"
                ).fetchone()
            )
            db.execute(
                "UPDATE starter_note_receipts SET fingerprint='tampered' WHERE operation_key=?",
                (operation["key"],),
            )
            with pytest.raises(AgentError, match="differs"):
                module.inspect_receipt(db, operation)


async def test_evaluator_cleanup_exception_preserves_batch_denominator(tmp_path):
    module = load_starter("confirmed-action", tmp_path / "confirmed")
    actor = Identity(owner="alice", scopes=frozenset(module.SCOPES))

    async def resolve(owner):
        return actor if owner == actor.owner else None

    async with open_runtime(
        module.create_registry(None, "unused", tmp_path),
        data_dir=tmp_path,
        identity_resolver=resolve,
    ) as engine:
        module.initialize(engine)
        original = engine.cancel
        attempts = 0

        async def flaky_cancel(owner, run_id):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected cleanup failure")
            return await original(owner, run_id)

        engine.cancel = flaky_cancel
        cases = [
            EvalCase(
                id=key,
                input={"business_key": key, "text": "candidate"},
                expected_status="awaiting_review",
                expected_applied_operations=0,
            )
            for key in ("first", "second")
        ]
        report = await evaluate_cases(engine, actor, module.AGENT_ID, cases)
        assert report["total"] == 2 and report["failed"] == report["passed"] == 1
        assert report["cases"][0]["failures"] == ["cleanup_failed"]
        assert (await engine.view("alice", report["cases"][1]["run_id"]))["status"] == "cancelled"
        await original("alice", report["cases"][0]["run_id"])
        with engine.store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM starter_notes").fetchone()[0] == 0
