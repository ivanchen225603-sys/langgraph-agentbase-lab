"""Actual opt-in planning over the existing one-worker engine and SQL receipts."""

import asyncio
import copy
import json

import pytest

from agentbase import artifact_read_tool
from agentbase.artifacts import ArtifactOperation, ArtifactScope
from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Message, ModelResponse
from agentbase.planning import PlanningStore
from agentbase.schema import canonical_json
from tests.test_agentbase_task_tree import package, system, wait


class PlanProvider:
    native_stream = False

    def __init__(self, proposals):
        self.proposals = copy.deepcopy(proposals)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if request.output_schema and "summary" in request.output_schema.get("properties", {}):
            return ModelResponse(text='{"summary":"bounded prior context"}', finish_reason="stop")
        proposal = self.proposals.pop(0)
        return ModelResponse(text=canonical_json(proposal).decode(), finish_reason="stop")

    def count_input_tokens(self, request):
        return len(canonical_json(request.model_dump(mode="json"))) // 4 + 1

    def count_output_tokens(self, text):
        return len(text.encode()) // 4 + 1


def task(task_id, value, dependencies=()):
    return {
        "id": task_id,
        "agent_id": "child",
        "input": {"x": value},
        "depends_on": list(dependencies),
    }


def proposal(tasks, *, finish=False, result=None):
    return {"action": "finish" if finish else "execute", "tasks": tasks, "result": result}


def configuration(*, write=False):
    data = package(write=write).model_dump(mode="json")
    data["models"] = [{"id": "planner", "provider": "fixture.plan", "model": "neutral"}]
    data["agents"][0].update(
        workflow="planning",
        model="planner",
        planning={"max_tasks": 8, "max_revisions": 4},
        budget={"max_steps": 32, "max_model_calls": 8},
    )
    data["agents"][1]["input_schema"] = data["tools"][0]["input_schema"]
    return AgentPackage.model_validate(data)


@pytest.mark.parametrize("valid", [True, False])
async def test_final_output_schema_reaches_model_and_invalid_finish_is_not_accepted(
    tmp_path, valid
):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    data = configuration().model_dump(mode="json")
    data["agents"][0]["output_schema"] = schema
    provider = PlanProvider(
        [proposal([], finish=True, result={"value": 7 if valid else "invalid"})]
    )

    async def echo(arguments, context):
        raise AssertionError("empty finish must not dispatch a child")

    async with system(
        tmp_path, echo, configuration=AgentPackage.model_validate(data), provider=provider
    ) as (engine, identity):
        run = await engine.submit(identity, "parent", "final-schema", {})
        view = await wait(engine, run["run_id"])
        assert any(
            message.role == "system" and canonical_json(schema).decode() in message.content
            for message in provider.requests[0].messages
        )
        if valid:
            assert view["status"] == "succeeded", view
            assert view["result"] == {"value": 7}
        else:
            assert view["status"] == "failed" and view["error"] == "agent_output", view
            assert PlanningStore(engine).revision(run["run_id"], 0) is None
            assert engine.store.final_receipt("user", run["run_id"]) is None


async def test_actual_two_wave_plan_revises_unattempted_task_and_charges_calls(tmp_path):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    initial = [task("a", 1), task("b", 2, ["a"])]
    revised = [task("a", 1), task("b", 9, ["a"])]
    provider = PlanProvider(
        [
            proposal(initial),
            proposal(revised),
            proposal(revised, finish=True, result={"done": True}),
        ]
    )
    async with system(tmp_path, echo, configuration=configuration(), provider=provider) as (
        engine,
        identity,
    ):
        run = await engine.submit(identity, "parent", "plan", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "succeeded", view
        assert view["result"] == {"done": True}
        assert calls == [1, 9]
        assert len(provider.requests) == 3
        assert all(request.tools == [] for request in provider.requests)
        assert all(request.output_schema["properties"]["action"] for request in provider.requests)
        assert engine.model_caller.ledger.snapshot(run["run_id"])["calls"] == 3
        assert [item["status"] for item in view["plan_tasks"]] == ["succeeded", "succeeded"]
        final = engine.store.final_receipt("user", run["run_id"])
        assert (
            len([source for source in final["sources"] if source["kind"] == "child_receipt"]) == 2
        )
        with engine.store.connection() as db:
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM agent_plan_attempts WHERE run_id=?", (run["run_id"],)
                ).fetchone()[0]
                == 2
            )


@pytest.mark.parametrize(
    "invalid",
    [
        {**proposal([task("a", 1)]), "status": "succeeded"},
        proposal([{**task("a", 1), "agent_id": "parent"}]),
        proposal([{**task("a", 1), "input": {"x": "invalid"}}]),
        proposal([task("a", 1, ["missing"])]),
        proposal([task("a", 1, ["b"]), task("b", 2, ["a"])]),
        proposal([task("a", 1), task("a", 2)]),
        proposal([task("a", 1)], finish=True, result="forged completion"),
    ],
)
async def test_invalid_model_proposals_never_dispatch(tmp_path, invalid):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments)
        return {"value": arguments["x"]}

    provider = PlanProvider([invalid])
    async with system(tmp_path, echo, configuration=configuration(), provider=provider) as (
        engine,
        identity,
    ):
        run = await engine.submit(identity, "parent", "invalid", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "failed", view
        assert not calls
        assert not engine.task_coordinator.children(run["run_id"])
        assert engine.store.final_receipt("user", run["run_id"]) is None


@pytest.mark.parametrize("change", ["input", "depends_on", "remove"])
async def test_replanning_cannot_mutate_attempted_declaration(tmp_path, change):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    initial = [task("a", 1), task("b", 2, ["a"])]
    changed = copy.deepcopy(initial)
    if change == "remove":
        changed = [task("b", 2)]
    else:
        changed[0][change] = {"x": 7} if change == "input" else ["b"]
    provider = PlanProvider([proposal(initial), proposal(changed)])
    async with system(tmp_path, echo, configuration=configuration(), provider=provider) as (
        engine,
        identity,
    ):
        run = await engine.submit(identity, "parent", "mutate", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "failed", view
        assert view["error"] == "planning_immutable", view
        assert calls == [1]
        assert len(engine.task_coordinator.children(run["run_id"])) == 1


async def test_failed_dependency_is_visible_and_never_runs_downstream(tmp_path):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        raise AgentError("fixture_failed", "deterministic read failure", 502)

    tasks = [task("a", 1), task("b", 2, ["a"])]
    provider = PlanProvider([proposal(tasks), proposal(tasks, finish=True, result="failed")])
    async with system(tmp_path, echo, configuration=configuration(), provider=provider) as (
        engine,
        identity,
    ):
        run = await engine.submit(identity, "parent", "failed_dependency", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "failed", view
        assert all(value == 1 for value in calls)
        assert [item["status"] for item in view["plan_tasks"]] == ["failed", "blocked"]
        assert len(engine.task_coordinator.children(run["run_id"])) == 1
        runtime_data = json.loads(provider.requests[1].messages[-1].content)[
            "untrusted_planning_data"
        ]
        assert runtime_data["runtime_results"][0]["status"] == "failed"


async def test_saved_revision_survives_ack_loss_without_second_model_call(tmp_path, monkeypatch):
    tasks = [task("a", 1)]
    provider = PlanProvider([proposal(tasks), proposal(tasks, finish=True, result="done")])
    original = PlanningStore.record
    failed = False

    def record_then_fail(self, *args, **kwargs):
        nonlocal failed
        result = original(self, *args, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("ack lost after plan commit")
        return result

    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    monkeypatch.setattr(PlanningStore, "record", record_then_fail)
    async with system(tmp_path, echo, configuration=configuration(), provider=provider) as (
        engine,
        identity,
    ):
        run = await engine.submit(identity, "parent", "ack_loss", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "succeeded", view
        assert PlanningStore(engine).revision(run["run_id"], 0) is not None
        assert len(provider.requests) == 2
        assert calls == [1]
        assert view["plan_revision"] == 1
        assert view["plan_tasks"][0]["status"] == "succeeded"


def test_disabled_planning_preserves_serialization_and_digest():
    original = package()
    data = original.model_dump(mode="json")
    assert "planning" not in data["agents"][0]
    data["agents"][0]["planning"] = None
    restored = AgentPackage.model_validate(data)
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")
    assert restored.digest == original.digest


async def test_child_review_and_plan_survive_restart_without_duplicate_model_or_effect(tmp_path):
    tasks = [task("a", 1)]
    provider = PlanProvider([proposal(tasks), proposal(tasks, finish=True, result="done")])
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    config = configuration(write=True)
    async with system(tmp_path, echo, configuration=config, provider=provider) as (
        engine,
        identity,
    ):
        run = await engine.submit(identity, "parent", "restart", {})
        for _ in range(200):
            children = engine.task_coordinator.children(run["run_id"])
            if children and children[0]["state"] == "awaiting_review":
                break
            await asyncio.sleep(0.01)
        assert children[0]["state"] == "awaiting_review"
        child_id = children[0]["run_id"]
        before = await wait(engine, child_id)
        review = before["interrupts"][0]
        assert len(provider.requests) == 1 and not calls
    async with system(tmp_path, echo, configuration=config, provider=provider) as (
        engine,
        identity,
    ):
        before = await wait(engine, child_id)
        assert before["interrupts"][0] == review
        await engine.resume(
            identity, child_id, review_id=review["review_id"], digest=review["digest"]
        )
        view = await wait(engine, run["run_id"])
        assert view["status"] == "succeeded", view
        assert calls == [1]
        assert len(provider.requests) == 2
        assert PlanningStore(engine).revision(run["run_id"], 0).proposal.tasks[0].input == {"x": 1}


@pytest.mark.parametrize("stage", ["locked", "admitted", "tool_authorizer"])
async def test_revoked_plan_input_cannot_cross_child_tool_boundary(tmp_path, monkeypatch, stage):
    tasks = [task("a", 7)]
    provider = PlanProvider([proposal(tasks), proposal(tasks, finish=True, result="done")])
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    async with system(tmp_path, echo, configuration=configuration(), provider=provider) as (
        engine,
        identity,
    ):
        bound = engine.registry.agent("parent")
        scope = ArtifactScope(owner="user", agent_id="parent", namespace="default")
        artifact = engine.artifacts.create(
            scope,
            ArtifactOperation(run_id="seed", key="source", config_digest=bound.digest),
            {"value": 7},
            sources=[],
            guard=lambda db: True,
        )
        engine.contexts.history.append_turn(
            "user",
            "parent",
            "default",
            0,
            "seed",
            [
                Message(role="user", content="Use this input", sources=[artifact.reference()]),
                Message(role="assistant", content="Retained"),
            ],
            guard=lambda db: True,
        )
        revoked = False

        def revoke():
            nonlocal revoked
            if not revoked:
                revoked = True
                engine.artifacts.revoke(
                    scope,
                    ArtifactOperation(run_id="seed", key="revoke", config_digest=bound.digest),
                    artifact.id,
                    artifact.version,
                    guard=lambda db: True,
                )

        if stage == "locked":
            original = PlanningStore.lock_ready

            def locked_then_revoke(self, *args):
                result = original(self, *args)
                revoke()
                return result

            monkeypatch.setattr(PlanningStore, "lock_ready", locked_then_revoke)
        elif stage == "admitted":
            original = engine.dispatch_children

            async def admitted_then_revoke(*args, **kwargs):
                result = await original(*args, **kwargs)
                revoke()
                return result

            monkeypatch.setattr(engine, "dispatch_children", admitted_then_revoke)
        else:

            async def authorize(context, declaration, call):
                revoke()
                return True

            engine.tool_authorizer = authorize
        run = await engine.submit(identity, "parent", "revoke", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "failed", view
        assert revoked
        assert not calls
        assert len(provider.requests) == 1
        assert engine.store.final_receipt("user", run["run_id"]) is None


async def test_all_four_enhancements_run_together_and_preserve_provenance(tmp_path):
    tasks = [task("a", 7)]
    provider = PlanProvider([proposal(tasks), proposal(tasks, finish=True, result="done")])
    data = configuration().model_dump(mode="json")
    reader = artifact_read_tool().model_dump(mode="json")
    data["tools"].append(reader)
    parent = data["agents"][0]
    parent["tools"] = [reader["name"]]
    parent["context"]["management"] = {
        "retain_recent_units": 1,
        "summary_chars": 128,
        "offload_bytes": 1024,
        "preview_chars": 64,
    }
    parent["budget"].update(
        max_input_chars=6500, max_input_tokens=2500, max_run_tokens=100000, max_model_calls=16
    )
    config = AgentPackage.model_validate(data)
    calls = []

    async def echo(arguments, context):
        calls.append(arguments["x"])
        return {"value": arguments["x"]}

    async with system(tmp_path, echo, configuration=config, provider=provider) as (
        engine,
        identity,
    ):
        for index in range(2):
            engine.contexts.history.append_turn(
                "user",
                "parent",
                "default",
                0,
                f"history{index}",
                [
                    Message(role="user", content="old input " + "x" * 2000),
                    Message(role="assistant", content="old output " + "y" * 2000),
                ],
                guard=lambda db: True,
            )
        run = await engine.submit(identity, "parent", "all_four", {})
        view = await wait(engine, run["run_id"])
        assert view["status"] == "succeeded", view
        assert calls == [7]
        assert len(provider.requests) > 2
        assert all(
            len(canonical_json(request.model_dump(mode="json")).decode()) <= 6500
            for request in provider.requests
        )
        artifacts = engine.artifacts.list(
            ArtifactScope(owner="user", agent_id="parent", namespace="default")
        )
        assert artifacts
        assert any(
            source["kind"] == "artifact"
            for source in PlanningStore(engine).revision(run["run_id"], 0).sources
        )
        child = engine.task_coordinator.children(run["run_id"])[0]
        assert any(
            source["kind"] == "artifact"
            for source in engine.task_coordinator.metadata(child["run_id"]).input_sources
        )
        final = engine.store.final_receipt("user", child["run_id"])
        assert any(source["kind"] == "task_input" for source in final["sources"])
        history = engine.contexts.history.turns(
            "user", "child", engine.store.control(child["run_id"])["conversation_id"], 0, 10
        )
        assert all(
            any(source["kind"] == "task_input" for source in message.sources)
            for turn in history
            for message in turn
        )
        assert engine.model_caller.ledger.snapshot(run["run_id"])["calls"] == len(provider.requests)
