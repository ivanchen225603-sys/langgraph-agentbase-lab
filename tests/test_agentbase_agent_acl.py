"""Host Agent ACL applies equally to entry points, delegation and derived output."""

import asyncio
from functools import partial

import pytest

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Message, ModelRequest, ModelResponse
from agentbase.runtime import RunEngine
from tests.test_agentbase_http import client
from tests.test_agentbase_task_tree import package, parent_factory, system, wait, wait_event


async def test_host_policy_is_installed_before_first_http_request(tmp_path):
    async def allowed(identity, agent_id):
        return agent_id == "classification"

    async with client(tmp_path, authorize_agent=allowed) as (_, app, _):
        engine = app.state.service.runtime
        identity = await engine.identity("local")
        assert not await engine.agent_allowed(identity, "calculation")
        with pytest.raises(AgentError) as error:
            await engine.submit(identity, "calculation", "denied", {"left": 1, "right": 2})
        assert error.value.code == "agent_forbidden"
        assert engine.store.lookup("local", "calculation", "denied") is None


async def test_denied_child_never_reaches_prepare_or_handler(tmp_path):
    prepared, calls = [], []

    async def echo(arguments, context):
        calls.append(arguments)
        return {"value": arguments["x"]}

    async def allowed(identity, agent_id):
        return agent_id == "parent"

    async with system(tmp_path, echo, agent_authorizer=allowed) as (engine, identity):
        binding = engine.binding(
            {"agent_id": "child", "binding": engine.registry.agent("child").digest}
        )
        original = binding.prepare

        async def prepare(actor, value):
            prepared.append(value)
            return await original(actor, value)

        binding.prepare = prepare
        parent = await engine.submit(identity, "parent", "denied-child", {})
        result = await wait(engine, parent["run_id"])
        assert result["status"] == "failed"
        assert not prepared and not calls
        assert not engine.task_coordinator.children(parent["run_id"])


async def test_child_acl_change_during_prepare_prevents_batch_admission(tmp_path):
    allowed_agents = {"parent", "child"}
    calls = []

    async def allowed(identity, agent_id):
        return agent_id in allowed_agents

    async def echo(arguments, context):
        calls.append(arguments)
        return {"value": arguments["x"]}

    async with system(tmp_path, echo, agent_authorizer=allowed) as (engine, identity):
        binding = engine.binding(
            {"agent_id": "child", "binding": engine.registry.agent("child").digest}
        )
        original = binding.prepare

        async def prepare(actor, value):
            result = await original(actor, value)
            allowed_agents.discard("child")
            return result

        binding.prepare = prepare
        parent = await engine.submit(identity, "parent", "changed-child", {})
        result = await wait(engine, parent["run_id"])
        assert result["status"] == "failed"
        assert not calls and not engine.task_coordinator.children(parent["run_id"])


async def test_agent_acl_receives_narrowed_child_scopes(tmp_path):
    observed = []

    async def allowed(identity, agent_id):
        observed.append((agent_id, identity.scopes))
        return agent_id == "parent" or identity.scopes == frozenset({"read"})

    async def echo(arguments, context):
        assert context.effective_scopes == frozenset({"read"})
        return {"value": arguments["x"]}

    configuration = package().model_dump(mode="json")
    configuration["agents"][0]["delegation"]["targets"][0]["scopes"] = ["read"]
    async with system(
        tmp_path,
        echo,
        configuration=AgentPackage.model_validate(configuration),
        agent_authorizer=allowed,
    ) as (engine, identity):
        parent = await engine.submit(identity, "parent", "narrowed", {})
        result = await wait(engine, parent["run_id"])
        assert result["status"] == "succeeded", result
        assert all(scopes == frozenset({"read"}) for agent, scopes in observed if agent == "child")


@pytest.mark.parametrize("policy_kind", ["agent", "frame"])
async def test_source_free_child_acl_revocation_blocks_derived_reuse(tmp_path, policy_kind):
    consumed, release = asyncio.Event(), asyncio.Event()
    allowed_agents = {"parent", "child"}

    async def hold():
        consumed.set()
        await release.wait()

    async def allowed(identity, agent_id):
        return policy_kind != "agent" or agent_id in allowed_agents

    async def frame_allowed(frame):
        return policy_kind != "frame" or frame.agent_id in allowed_agents

    async def echo(arguments, context):
        return {"value": arguments["x"]}

    class Provider:
        native_stream = False
        calls = 0

        def count_input_tokens(self, request):
            return 1

        def count_output_tokens(self, value):
            return 1

        async def complete(self, request):
            self.calls += 1
            return ModelResponse(text="unused")

    async with system(
        tmp_path,
        echo,
        workflow_factory=partial(parent_factory, after_consumption=hold),
        agent_authorizer=allowed,
        frame_authorizer=frame_allowed,
    ) as (engine, identity):
        parent = await engine.submit(identity, "parent", "source-free", {})
        await wait_event(engine, identity, parent["run_id"], consumed)
        for child in engine.task_coordinator.children(parent["run_id"]):
            assert engine.store.final_receipt("user", child["run_id"])["sources"] == []
        event = await engine.emit(parent["run_id"], "result.projection", {"text": "derived"})
        allowed_agents.discard("child")
        with pytest.raises(AgentError):
            await engine.events("user", parent["run_id"], event.seq - 1)
        provider = Provider()
        current = await engine.context(parent["run_id"])
        budget = engine.budget_policy(current)
        with pytest.raises(AgentError):
            await engine.model_caller.call(
                provider,
                current,
                budget,
                ModelRequest(
                    messages=[Message(role="user", content="use results")],
                    max_output_tokens=budget.max_output_tokens,
                ),
                streaming=False,
            )
        assert provider.calls == 0
        release.set()
        result = await wait(engine, parent["run_id"])
        assert result["status"] == "failed"
        assert engine.store.final_receipt("user", parent["run_id"]) is None


async def test_host_frame_policy_and_agent_policies_intersect(tmp_path):
    frames = []
    calls = []

    async def frame_allowed(frame):
        frames.append(frame.agent_id)
        return frame.agent_id != "child"

    async def agent_allowed(identity, agent_id):
        return True

    async def echo(arguments, context):
        calls.append(arguments)
        return {"value": arguments["x"]}

    async with system(
        tmp_path, echo, agent_authorizer=agent_allowed, frame_authorizer=frame_allowed
    ) as (engine, identity):
        parent = await engine.submit(identity, "parent", "host-frames", {})
        result = await wait(engine, parent["run_id"])
        assert "child" in frames
        assert not calls
        assert all(
            item["status"] == "failed" and item["value"] is None for item in result["result"]
        )


async def test_reusing_context_provider_replaces_runtime_policy_and_retains_host_policy(tmp_path):
    frames = []

    async def host(frame):
        frames.append(frame.agent_id)
        return True

    async def old_policy(identity, agent_id):
        return False

    async def new_policy(identity, agent_id):
        return True

    async def echo(arguments, context):
        return {"value": arguments["x"]}

    async with system(tmp_path, echo, agent_authorizer=old_policy, frame_authorizer=host) as (
        engine,
        identity,
    ):
        replacement = RunEngine(
            engine.registry,
            engine.store,
            engine.saver,
            engine.contexts,
            engine.identity_resolver,
            agent_authorizer=new_policy,
        )
        assert not await engine.agent_allowed(identity, "parent")
        assert await replacement.agent_allowed(identity, "parent")
        assert await replacement.contexts.authorized_sources(identity, "parent", [])
        assert frames == ["parent", "parent"]


async def test_authorization_stall_expires_worker_and_preserves_terminal_control(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    request_task = asyncio.current_task()

    async def allowed(identity, agent_id):
        if calls and asyncio.current_task() is not request_task:
            entered.set()
            await release.wait()
        return True

    async def echo(arguments, context):
        calls.append(arguments)
        return {"value": arguments["x"]}

    async with system(
        tmp_path, echo, configuration=package(timeout=0.4), agent_authorizer=allowed
    ) as (engine, identity):
        # Arm only after admission so it is the first worker control check that stalls.
        admit = engine.store.admit

        def admitted(*args, **kwargs):
            result = admit(*args, **kwargs)
            calls.append("armed")
            return result

        engine.store.admit = admitted
        parent = await engine.submit(identity, "parent", "stalled", {})
        await asyncio.wait_for(entered.wait(), 1)
        try:
            result = await wait(engine, parent["run_id"], ("timed_out",))
            assert result["status"] == "timed_out"
            assert engine.available
            assert calls == ["armed"]
            assert not engine.active
        finally:
            release.set()


@pytest.mark.parametrize("dispatch_state", ["terminal", "ready"])
async def test_restart_repairs_status_only_checkpoint_from_committed_terminal_outcome(
    tmp_path, dispatch_state
):
    calls = []

    async def echo(arguments, context):
        calls.append(arguments)
        return {"value": arguments["x"]}

    async with system(tmp_path, echo) as (engine, identity):
        await engine.task_coordinator.close()
        parent = await engine.submit(identity, "parent", "legacy-terminal", {})
        row = engine.store.internal_request(parent["run_id"])
        original = engine.update_checkpoint

        async def old_terminal(record, values, *, as_node):
            # Reproduce the old pre-initialization terminal writer; SQL outcome commits
            # normally while its checkpoint contains only status.
            if isinstance(values, dict):
                values = {"status": values["status"]}
            return await original(record, values, as_node=as_node)

        engine.update_checkpoint = old_terminal
        await engine.terminal_checkpoint(row, "timed_out", "original_deadline_expired")
        assert (await engine.checkpoint_state(row)).values == {"status": "timed_out"}
        assert engine.task_coordinator.outcome(row["run_id"])["status"] == "timed_out"
        with engine.store.connection() as db:
            db.execute(
                "UPDATE agent_task_dispatch SET state=? WHERE run_id=?",
                (dispatch_state, row["run_id"]),
            )

    async with system(tmp_path, echo) as (replacement, identity):
        result = await wait(replacement, parent["run_id"], ("timed_out",))
        snapshot = await replacement.checkpoint_state(row)
        replacement._validate_snapshot(row, snapshot)
        assert result["status"] == "timed_out"
        assert not calls
