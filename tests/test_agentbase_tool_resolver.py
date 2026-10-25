"""Trusted external catalogs still use the same durable execution controls."""

import asyncio
import time

import pytest

from agentbase import AgentPackage, ComponentRegistry, ResolvedTool, ToolDeclaration
from agentbase.contracts import AgentError, Identity, RunContext, ToolCall
from agentbase.effects import AppliedResultUnavailable, SqliteActionLedger
from agentbase.schema import content_digest
from agentbase.storage import SqliteDatabase
from agentbase.tools import ToolExecutor


def declaration(*, write=False):
    return ToolDeclaration.model_validate(
        {
            "name": "outside.task",
            "handler": "outside.handler",
            "handler_version": "1",
            "description": "External trusted tool catalog",
            "scope": "write" if write else "read",
            "read_only": not write,
            "retry_safe": not write,
            "idempotency": "business-key" if write else "read-only",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        }
    )


class Catalog:
    def __init__(self, handler, *, write=False):
        self.selected = ResolvedTool(
            declaration(write=write), handler, content_digest({"release": 1})
        )

    def __call__(self, context, call):
        if context.run_id != "run1" or call.name != "outside.task":
            raise AgentError("tool_not_allowed", "tool is not in the admitted catalog", 403)
        return self.selected


@pytest.fixture
def system(tmp_path):
    database = SqliteDatabase(tmp_path / "business.sqlite")
    ledger = SqliteActionLedger(database.connection)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        ledger.install(db)
    registry = ComponentRegistry()
    registry.register_workflow("outside", lambda: None)
    registry.bind(
        AgentPackage.model_validate(
            {
                "agents": [
                    {
                        "id": "alpha",
                        "workflow": "outside",
                        "description": "Host-owned catalog fixture",
                        "prompt": {"system_template": "Use trusted host workflow"},
                        "input_schema": {"type": "string"},
                        "tools": [],
                        "budget": {"max_tool_calls": 2},
                    }
                ]
            }
        )
    )
    context = RunContext(
        identity=Identity(owner="user", scopes=frozenset({"read", "write"})),
        agent_id="alpha",
        run_id="run1",
        conversation_id="default",
        config_digest=registry.agent("alpha").digest,
        deadline=time.time() + 10,
    )
    return registry, ledger, context


def call(call_id="one", x=1):
    return ToolCall(call_id=call_id, name="outside.task", arguments={"x": x})


async def blocked(predicate):
    async with asyncio.timeout(2):
        while not predicate():  # noqa: ASYNC110 -- observe the real semaphore waiter, bounded above
            await asyncio.sleep(0)


async def test_external_catalog_has_no_default_fallback_and_keeps_all_execution_limits(system):
    registry, ledger, context = system
    invoked = []

    async def handler(args, ctx):
        invoked.append((args, ctx.operation_key, ctx.attempt_token))
        return {"value": args["x"]}

    catalog = Catalog(handler)
    with pytest.raises(AgentError) as denied:
        await ToolExecutor(registry, ledger).execute(context, call())
    assert denied.value.code == "tool_not_allowed"
    executor = ToolExecutor(registry, ledger, resolver=catalog)
    for wrong_context, wrong_call in (
        (context.model_copy(update={"identity": Identity(owner="user")}), call()),
        (context, call().model_copy(update={"arguments": {"x": "wrong"}})),
        (context, call().model_copy(update={"name": "outside.missing"})),
        (context.model_copy(update={"config_digest": "sha256:" + "0" * 64}), call()),
        (context.model_copy(update={"deadline": time.time() - 1}), call()),
    ):
        with pytest.raises(AgentError):
            await executor.execute(wrong_context, wrong_call)
    assert not invoked and ledger.operations("user", "run1") == []
    challenge = executor.challenge(context, call())
    assert challenge != ToolExecutor.action_digest(context, catalog.selected.declaration, call())
    assert ledger.operations("user", "run1") == []
    first = await executor.execute(context, call())
    assert first.succeeded and first.value == {"value": 1}
    assert await executor.execute(context, call()) == first
    assert len(invoked) == 1 and invoked[0][1] and invoked[0][2]
    assert (await executor.execute(context, call("two", 2))).succeeded
    with pytest.raises(AgentError) as budget:
        await executor.execute(context, call("three", 3))
    assert budget.value.code == "tool_call_budget"
    assert [row["attempts"] for row in ledger.operations("user", "run1")] == [1, 1]


@pytest.mark.parametrize("change", ["handler", "declaration", "manifest"])
async def test_catalog_change_during_real_slot_wait_never_calls_replacement(system, change):
    registry, ledger, context = system
    invoked = []

    async def handler(args, ctx):
        invoked.append("original")
        return {"value": 1}

    async def replacement(args, ctx):
        invoked.append("replacement")
        return {"value": 2}

    catalog = Catalog(handler)
    executor = ToolExecutor(registry, ledger, resolver=catalog, concurrency=1)
    await executor.slots.acquire()
    task = asyncio.create_task(executor.execute(context, call()))
    try:
        await blocked(lambda: bool(executor.slots._waiters))
        original = catalog.selected
        catalog.selected = ResolvedTool(
            original.declaration.model_copy(update={"description": "changed"})
            if change == "declaration"
            else original.declaration,
            replacement if change == "handler" else original.handler,
            content_digest({"release": 2}) if change == "manifest" else original.manifest_digest,
        )
        executor.slots.release()
        observed = await task
        assert not observed.succeeded and observed.error == "tool_binding_changed"
        assert invoked == []
        assert ledger.operations("user", "run1")[0]["attempts"] == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_same_manifest_cannot_replace_handler_between_calls_or_reuse_cached_old_action(
    system,
):
    registry, ledger, context = system
    invoked = []

    async def first(args, ctx):
        invoked.append("v1")
        return {"value": 1}

    async def second(args, ctx):
        invoked.append("v2")
        return {"value": 2}

    catalog = Catalog(first)
    executor = ToolExecutor(registry, ledger, resolver=catalog)
    assert (await executor.execute(context, call())).succeeded
    original = catalog.selected
    catalog.selected = ResolvedTool(original.declaration, second, original.manifest_digest)
    with pytest.raises(AgentError) as invalid:
        await executor.execute(context, call("two"))
    assert invalid.value.code == "tool_binding_changed"
    catalog.selected = ResolvedTool(original.declaration, second, content_digest({"release": 2}))
    with pytest.raises(AgentError) as conflict:
        await executor.execute(context, call())
    assert conflict.value.code == "operation_conflict"
    assert (await executor.execute(context, call("two"))).value == {"value": 2}
    assert invoked == ["v1", "v2"]


async def test_trusted_callbacks_cannot_mutate_captured_arguments_schema_or_dependencies(system):
    registry, ledger, context = system
    invoked = []

    async def handler(args, ctx):
        invoked.append((args.copy(), ctx.dependency_results.copy()))
        args["x"] = 99
        return {"value": 1}

    catalog = Catalog(handler)

    def resolver(ctx, action):
        action.arguments["x"] = 99
        ctx.dependency_results["seed"] = 99
        return catalog.selected

    async def authorizer(ctx, tool, action):
        action.arguments["x"] = 99
        tool.input_schema["properties"]["x"]["type"] = "string"
        ctx.dependency_results["seed"] = 99
        await asyncio.sleep(0)
        return True

    executor = ToolExecutor(registry, ledger, resolver=resolver, authorizer=authorizer)
    context = context.model_copy(update={"dependency_results": {"seed": 7}}, deep=True)
    action = call()
    assert (await executor.execute(context, action)).succeeded
    assert invoked == [({"x": 1}, {"seed": 7})]
    assert action.arguments == {"x": 1} and context.dependency_results == {"seed": 7}
    assert catalog.selected.declaration.input_schema["properties"]["x"]["type"] == "integer"


@pytest.mark.parametrize("stage", ["authorization", "approval"])
async def test_binding_changed_while_policy_waits_is_rejected_before_admission(system, stage):
    registry, ledger, context = system
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def handler(args, ctx):
        effects.append(args)
        return {"value": 1}

    catalog = Catalog(handler, write=stage == "approval")

    async def wait(*args):
        entered.set()
        await release.wait()
        return True

    executor = ToolExecutor(
        registry,
        ledger,
        resolver=catalog,
        authorizer=wait if stage == "authorization" else None,
        approval_checker=wait if stage == "approval" else None,
    )
    task = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(entered.wait(), 2)
    catalog.selected = ResolvedTool(
        catalog.selected.declaration, handler, content_digest({"release": 2})
    )
    release.set()
    with pytest.raises(AgentError) as denied:
        await task
    assert denied.value.code == "tool_binding_changed"
    assert effects == [] and ledger.operations("user", "run1") == []


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("change", ["manifest", "handler"])
async def test_cached_result_policy_change_cannot_publish_or_replay_known_effect(
    system, write, change
):
    registry, ledger, context = system
    entered, release = asyncio.Event(), asyncio.Event()
    invoked = []

    async def handler(args, ctx):
        invoked.append(1)
        return {"value": 1}

    async def replacement(args, ctx):
        invoked.append(2)
        return {"value": 2}

    async def approved(*args):
        return True

    catalog = Catalog(handler, write=write)
    executor = ToolExecutor(registry, ledger, resolver=catalog, approval_checker=approved)
    assert (await executor.execute(context, call())).succeeded

    async def policy(ctx, tool, value):
        entered.set()
        await release.wait()
        return value

    executor.result_policy = policy
    task = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(entered.wait(), 2)
    original = catalog.selected
    catalog.selected = ResolvedTool(
        original.declaration,
        replacement if change == "handler" else handler,
        content_digest({"release": 2}) if change == "manifest" else original.manifest_digest,
    )
    release.set()
    with pytest.raises(AppliedResultUnavailable if write else AgentError):
        await task
    assert invoked == [1] and not ledger.uncertain("run1")
    row = ledger.operations("user", "run1")[0]
    assert row["phase"] == ("applied" if write else "read_success") and row["attempts"] == 1


async def test_raw_receipt_is_frozen_during_result_policy_await(system):
    registry, ledger, context = system
    entered, release = asyncio.Event(), asyncio.Event()
    raw = {"value": 1}

    async def handler(args, ctx):
        return raw

    async def policy(ctx, tool, value):
        entered.set()
        await release.wait()
        tool.output_schema["required"] = ["private"]
        return value

    executor = ToolExecutor(registry, ledger, resolver=Catalog(handler), result_policy=policy)
    task = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(entered.wait(), 2)
    raw["value"] = 99
    release.set()
    assert (await task).value == {"value": 1}
    assert ledger.previous_value(ledger.operations("user", "run1")[0]) == {"value": 1}


async def test_write_confirmation_binds_manifest_arguments_and_dependency_values(system):
    registry, ledger, context = system
    effects = []

    async def handler(args, ctx):
        effects.append(args["x"])
        return {"value": args["x"]}

    catalog = Catalog(handler, write=True)
    executor = ToolExecutor(registry, ledger, resolver=catalog)
    context = context.model_copy(update={"dependency_results": {"previous": 1}}, deep=True)
    expected = executor.challenge(context, call())
    assert not effects and ledger.operations("user", "run1") == []

    async def approved(ctx, tool, action, challenge):
        action.arguments["x"] = 99
        tool.input_schema["required"] = ["private"]
        return challenge == expected

    executor.approval_checker = approved
    original = catalog.selected
    catalog.selected = ResolvedTool(original.declaration, handler, content_digest({"release": 2}))
    with pytest.raises(AgentError) as denied:
        await executor.execute(context, call())
    assert denied.value.code == "approval_required"
    catalog.selected = original
    for actor, action in (
        (context.model_copy(update={"dependency_results": {"previous": 2}}), call()),
        (context, call(x=2)),
    ):
        with pytest.raises(AgentError) as denied:
            await executor.execute(actor, action)
        assert denied.value.code == "approval_required"
    assert effects == [] and ledger.operations("user", "run1") == []
    assert (await executor.execute(context, call())).succeeded
    assert (await executor.execute(context, call())).succeeded and effects == [1]
    assert ledger.operations("user", "run1")[0]["phase"] == "applied"


@pytest.mark.parametrize(
    "invalid",
    ["none_manifest", "manifest", "declaration", "name", "handler", "coroutine", "exception"],
)
async def test_invalid_resolvers_fail_closed_without_coroutine_warning_or_private_error(
    system, invalid
):
    registry, ledger, context = system

    async def handler(args, ctx):
        raise AssertionError("must not execute")

    selected = Catalog(handler).selected

    async def async_result():
        return selected

    def resolver(ctx, action):
        if invalid == "coroutine":
            return async_result()
        if invalid == "exception":
            raise RuntimeError("private SQL credentials")
        return ResolvedTool(
            selected.declaration.model_copy(update={"timeout_seconds": -1})
            if invalid == "declaration"
            else selected.declaration.model_copy(update={"name": "wrong"})
            if invalid == "name"
            else selected.declaration,
            None if invalid == "handler" else handler,
            None
            if invalid == "none_manifest"
            else "wrong"
            if invalid == "manifest"
            else selected.manifest_digest,
        )

    executor = ToolExecutor(registry, ledger, resolver=resolver)
    with pytest.raises(AgentError) as denied:
        await executor.execute(context, call())
    assert denied.value.code == "tool_resolution" and "private" not in str(denied.value)
    assert ledger.operations("user", "run1") == []


def test_custom_resolution_capacity_is_bounded_and_does_not_evict_original(system):
    registry, ledger, context = system

    async def handler(args, ctx):
        return {"value": 1}

    catalog = Catalog(handler)
    executor = ToolExecutor(registry, ledger, resolver=catalog, max_resolutions=1)
    original = catalog.selected
    executor.challenge(context, call())
    catalog.selected = ResolvedTool(original.declaration, handler, content_digest({"release": 2}))
    with pytest.raises(AgentError) as denied:
        executor.challenge(context, call())
    assert denied.value.code == "tool_binding_capacity"
    catalog.selected = original
    executor.challenge(context, call())
    assert ledger.operations("user", "run1") == []


async def test_different_agents_share_actual_executor_slots_and_durable_attempts(system):
    registry, ledger, context = system
    registry.bind(
        AgentPackage.model_validate(
            {
                "agents": [
                    {
                        "id": "beta",
                        "workflow": "outside",
                        "description": "Second external Agent",
                        "prompt": {"system_template": "Use trusted host workflow"},
                        "input_schema": {"type": "string"},
                        "budget": {"max_tool_calls": 1},
                    }
                ]
            }
        )
    )
    beta = context.model_copy(
        update={
            "agent_id": "beta",
            "run_id": "run2",
            "config_digest": registry.agent("beta").digest,
        }
    )
    entered, release = asyncio.Event(), asyncio.Event()
    active = maximum = 0
    invoked = []

    async def handler(args, ctx):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        invoked.append(ctx.agent_id)
        if ctx.agent_id == "alpha":
            entered.set()
            await release.wait()
        active -= 1
        return {"value": args["x"]}

    selected = Catalog(handler).selected

    def resolver(ctx, action):
        if (ctx.agent_id, ctx.run_id) not in {("alpha", "run1"), ("beta", "run2")}:
            raise AgentError("tool_not_allowed", "run is not admitted", 403)
        return selected

    executor = ToolExecutor(registry, ledger, resolver=resolver, concurrency=1)
    first = asyncio.create_task(executor.execute(context, call()))
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(executor.execute(beta, call()))
    try:
        await blocked(lambda: bool(executor.slots._waiters))
        assert invoked == ["alpha"] and maximum == 1
        release.set()
        results = await asyncio.gather(first, second)
        assert all(value.succeeded for value in results)
        assert invoked == ["alpha", "beta"] and maximum == 1
        assert ledger.operations("user", "run1")[0]["attempts"] == 1
        assert ledger.operations("user", "run2")[0]["attempts"] == 1
    finally:
        release.set()
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_external_catalog_example_runs_real_graph_original_sql_and_public_ledger():
    from examples.agentbase_catalog import run_demo

    result = await run_demo()
    assert result == {
        "result": {"text": "HELLO AGENT"},
        "handler_calls": 1,
        "logical_actions": 1,
        "attempts": 1,
        "uses_public_executor": True,
        "model_calls": 0,
    }
