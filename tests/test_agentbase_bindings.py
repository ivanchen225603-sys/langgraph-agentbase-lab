"""Historical execution refs, release activation and real checkpoint compatibility.

Engine reconstruction below is in-process, not evidence of killing a process.
"""

import asyncio
import copy

import pytest

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, Identity
from agentbase.fixed_workflow import fixed_factory
from agentbase.registry import ComponentRegistry, RegistrationSelection
from agentbase.runtime import RunEngine
from agentbase.schema import content_digest
from agentbase.workflows import conversation_factory
from raglab.models import RunInput
from tests.test_agentbase_configuration import handler, package_data, registry
from tests.test_agentbase_runtime import (
    ScriptProvider,
    declaration,
    fixed_package,
    package,
    system,
    wait,
)


def pinned_release(version="2"):
    data = package_data()
    data["models"][0]["provider_version"] = version
    data["tools"][0]["handler_version"] = version
    for agent in data["agents"]:
        agent.update(version=version, workflow_version=version)
    return AgentPackage.model_validate(data)


def register_second(components):
    components.register_handler("external.calculate", handler, version="2")
    components.register_provider("test", lambda config: None, version="2")
    components.register_workflow("conversation", lambda context: None, version="2")


def test_old_serialization_and_binding_digests_are_unchanged():
    config = AgentPackage.model_validate(package_data())
    components = registry()
    components.bind(config)
    assert components.agent("calculator").digest == (
        "sha256:b030f3f06523c16d010ecb981f1dcb965e96960c3f6b29720641c9e3035c9794"
    )
    assert components.agent("extract").digest == (
        "sha256:5f498176fec625ce85c55e0785ae4f97de060b4ae5b67254d5be403ff722ea99"
    )
    assert "provider_version" not in config.models[0].model_dump(mode="json")
    assert "handler_version" not in config.tools[0].model_dump(mode="json")
    assert "compensation_version" not in config.tools[0].model_dump(mode="json")
    assert "workflow_version" not in config.agents[0].model_dump(mode="json")
    legacy = ComponentRegistry()
    legacy.register_workflow("legacy.rag", lambda: None, version="2")
    legacy.bind(
        AgentPackage.model_validate(
            {
                "agents": [
                    {
                        "id": "rag",
                        "version": "2",
                        "description": "Legacy RAG API and source-aware result projection",
                        "workflow": "legacy.rag",
                        "prompt": {"system_template": "Use the trusted RAG workflow."},
                        "input_schema": RunInput.model_json_schema(),
                        "budget": {"timeout_seconds": 120.0, "review_seconds": 120.0},
                    }
                ]
            }
        )
    )
    assert legacy.agent("rag").digest == (
        "sha256:2b04d9c3a05c117a97ac9c9fec501ce2c9723838383b5cbf358dc0d5d2eff5e9"
    )


def test_activation_is_atomic_and_old_actual_refs_are_retained():
    components = registry()
    original = AgentPackage.model_validate(package_data())
    components.bind(original)
    old = components.agent("calculator")
    register_second(components)
    with pytest.raises(AgentError) as error:
        components.activate(original)
    assert error.value.code == "component_version_required"
    bad = pinned_release().model_dump(mode="json")
    bad["agents"][1]["workflow"] = "missing"
    with pytest.raises(AgentError):
        components.activate(AgentPackage.model_validate(bad))
    assert components.agent("extract").definition.version == "1"
    assert components.agent("calculator").digest == old.digest
    components.activate(pinned_release())
    assert components.agent("calculator").definition.version == "2"
    restored = components.resolve("calculator", old.digest)
    assert components.handler(restored.tools[0], bound=restored) is handler
    assert restored.workflow_registration is old.workflow_registration
    assert restored.provider_registration is old.provider_registration
    restored.definition.tools.clear()
    restored.tools[0].input_schema.clear()
    assert components.resolve("calculator", old.digest).definition.tools == ["calculate"]
    assert components.resolve("calculator", old.digest).tools[0].input_schema["type"] == "object"
    with pytest.raises(AgentError) as error:
        components.authorized_tool(
            "calculator", "calculate", Identity(owner="u"), digest=old.digest
        )
    assert error.value.code == "forbidden"
    with pytest.raises(AgentError):
        components.register_handler("external.calculate", lambda: None, version="1")


def test_restore_sidecar_is_exact_and_does_not_change_active_release():
    components = registry()
    register_second(components)
    components.bind(pinned_release())
    active = components.agent("calculator").digest
    selection = RegistrationSelection(
        workflows={"conversation": "1"},
        providers={"test": "1"},
        handlers={"external.calculate": "1"},
    )
    original = AgentPackage.model_validate(package_data())
    reference = registry()
    reference.bind(original)
    expected = {item["id"]: item["digest"] for item in reference.agents()}
    with pytest.raises(AgentError):
        components.resolve("calculator", expected["calculator"])
    with pytest.raises(AgentError):
        components.restore(original, versions=selection, expected_digests={"calculator": "wrong"})
    with pytest.raises(AgentError):
        components.resolve("calculator", expected["calculator"])
    components.restore(original, versions=selection, expected_digests=expected)
    assert components.agent("calculator").digest == active
    assert components.resolve("calculator", expected["calculator"]).definition.version == "1"
    components.activate(original, versions=selection)
    assert components.agent("calculator").digest == expected["calculator"]
    with pytest.raises(AgentError) as error:
        components.activate(pinned_release(), versions=selection)
    assert error.value.code == "component_version_conflict"


def test_capacity_and_same_agent_version_changes_cannot_partially_publish():
    components = ComponentRegistry(max_bindings=2)
    components.register_handler("external.calculate", handler)
    components.register_provider("test", lambda config: None)
    components.register_workflow("conversation", lambda context: None)
    components.bind(AgentPackage.model_validate(package_data()))
    before = components.agents()
    altered = package_data()
    altered["agents"][0]["prompt"]["system_template"] = "Changed"
    with pytest.raises(AgentError) as error:
        components.activate(AgentPackage.model_validate(altered))
    assert error.value.code == "agent_version_conflict"
    register_second(components)
    with pytest.raises(AgentError) as error:
        components.activate(pinned_release())
    assert error.value.code == "binding_capacity"
    assert components.agents() == before
    with pytest.raises(AgentError) as error:
        components.activate(
            pinned_release(),
            versions=RegistrationSelection(workflows={"unused": "1"}),
        )
    assert error.value.code == "unused_component_selector"
    assert len(components._snapshots) == 2 and components.agents() == before


async def test_compensation_resolves_original_executable_reference():
    called = []

    async def original(args, context):
        called.append("old")
        return 1

    async def newer(args, context):
        called.append("new")
        return 2

    components = registry()
    components.register_handler("undo", original)
    data = package_data()
    data["tools"][0]["compensation"] = "undo"
    components.bind(AgentPackage.model_validate(data))
    old = components.agent("calculator")
    components.register_handler("undo", newer, version="2")
    data["tools"][0]["compensation_version"] = "2"
    for agent in data["agents"]:
        agent["version"] = "2"
    components.activate(AgentPackage.model_validate(data))
    new = components.agent("calculator")
    assert await components.compensation_handler(old.tools[0], bound=old)({}, None) == 1
    assert await components.compensation_handler(new.tools[0], bound=new)({}, None) == 2
    assert called == ["old", "new"]  # Reference resolution, not business undo validation.


async def test_cold_conversation_resume_uses_old_provider_tool_schema_and_memory(tmp_path):
    calls, factories = [], []

    async def original(args, context):
        calls.append((context.config_digest, args))
        return {"value": args["x"]}

    async def wrong(args, context):
        raise AssertionError("new handler must not execute the old approved action")

    data = package(tools=[declaration(True)]).model_dump(mode="json")
    data["agents"][0].update(
        output_schema={
            "type": "object",
            "properties": {"old": {"type": "integer"}},
            "required": ["old"],
            "additionalProperties": False,
        },
        context={"memory_scope": "agent"},
        budget={"max_model_calls": 2, "max_tool_calls": 1, "review_seconds": 120},
    )
    config = AgentPackage.model_validate(data)
    async with system(
        tmp_path, [[("custom", {"x": 4})], '{"old":4}'], config=config, handler=original
    ) as (
        engine,
        old_provider,
        contexts,
        identities,
    ):
        contexts.create_memory("user", "agent0", "", "Remember the original policy")
        created = await engine.submit(identities["user"], "agent0", "old", "Input")
        paused = await wait(engine, created["run_id"])
        assert paused["status"] == "awaiting_review" and calls == []
        review = paused["interrupts"][0]
        row = engine.store.request("user", created["run_id"])
        await engine.close()
        newer = copy.deepcopy(data)
        newer["agents"][0].update(
            version="2",
            workflow_version="2",
            tools=[],
            input_schema={"type": "object"},
            output_schema={
                "type": "object",
                "properties": {"new": {"type": "integer"}},
                "required": ["new"],
                "additionalProperties": False,
            },
            context={"memory_scope": "none"},
            budget={"max_model_calls": 1, "max_tool_calls": 0},
        )
        new_provider = ScriptProvider(['{"new":9}'])
        engine.registry.register_provider("offline", lambda config: new_provider, version="2")
        engine.registry.register_handler("custom.handler", wrong, version="2")

        def new_factory(runtime, saver, bound):
            factories.append("v2")
            return conversation_factory(runtime, saver, bound)

        engine.registry.register_workflow("conversation", new_factory, version="2")
        engine.registry.activate(
            AgentPackage.model_validate(newer),
            versions=RegistrationSelection(providers={"offline": "2"}),
        )
        restarted = RunEngine(
            engine.registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        await restarted.start()
        try:
            new = await restarted.submit(identities["user"], "agent0", "new", {})
            assert (await wait(restarted, new["run_id"]))["result"] == {"new": 9}
            await restarted.resume(
                identities["user"],
                row["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
            final = await wait(restarted, row["run_id"])
            assert final["status"] == "succeeded" and final["result"] == {"old": 4}, final
            assert calls == [(row["binding"], {"x": 4})]
            assert len(old_provider.requests) == 2 and len(new_provider.requests) == 1
            assert factories == ["v2"]
            assert engine.store.request("user", row["run_id"])["binding"] == row["binding"]
            assert old_provider.requests[1].output_schema["required"] == ["old"]
            assert any(message.sources for message in old_provider.requests[1].messages)
            assert (
                restarted.registry.resolve(
                    "agent0", row["binding"]
                ).definition.budget.max_tool_calls
                == 1
            )
            assert restarted.actions.operations("user", row["run_id"])[0]["phase"] == "applied"
        finally:
            await restarted.close()


async def test_cold_fixed_resume_and_prepare_use_original_tool_contract(tmp_path):
    calls = []

    async def original(args, context):
        calls.append(args)
        return {"value": args["x"]}

    async def newer(args, context):
        raise AssertionError("old fixed task cannot use the new handler")

    config = fixed_package([{"id": "one", "tool": "custom", "input_bindings": {"x": "/a"}}])
    async with system(tmp_path, [], config=config, handler=original) as (engine, _, _, identities):
        from agentbase.runtime_contracts import RunOptions

        created = await engine.submit(
            identities["user"], "agent0", "old", {"a": 4}, RunOptions(pause_for_review=True)
        )
        paused = await wait(engine, created["run_id"])
        row = engine.store.request("user", created["run_id"])
        await engine.close()
        data = config.model_dump(mode="json")
        data["agents"][0].update(version="2", budget={"max_tool_calls": 0})
        data["agents"][0]["tasks"][0]["input_bindings"] = {"y": "/a"}
        data["tools"][0]["handler_version"] = "2"
        data["tools"][0]["input_schema"] = {
            "type": "object",
            "properties": {"y": {"type": "integer"}},
            "required": ["y"],
        }
        engine.registry.register_handler("custom.handler", newer, version="2")
        engine.registry.activate(AgentPackage.model_validate(data))
        restarted = RunEngine(
            engine.registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        await restarted.start()
        try:
            old_binding = restarted.binding(row)
            assert await old_binding.prepare(identities["user"], {"a": 4}) == {
                "input": {"a": 4},
                "task_arguments": {"one": {"x": 4}},
            }
            review = paused["interrupts"][0]
            await restarted.resume(
                identities["user"],
                row["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
            final = await wait(restarted, row["run_id"])
            assert final["status"] == "succeeded" and final["result"]["one"]["value"] == {
                "value": 4
            }
            assert calls == [{"x": 4}]
        finally:
            await restarted.close()


@pytest.mark.parametrize("race", [False, True])
async def test_existing_submission_uses_original_client_contract_even_after_activation(
    tmp_path, race
):
    entered, release = asyncio.Event(), asyncio.Event()
    factory_calls = []

    def factory(version):
        def bind(engine, saver, bound):
            factory_calls.append(version)
            binding = fixed_factory(engine, saver, bound)
            binding.client_digest = lambda value, options: content_digest(
                {"version": version, "input": value if version == "1" else value["text"]}
            )
            if version == "1" and race:
                prepare = binding.prepare

                async def gate(identity, value):
                    entered.set()
                    await release.wait()
                    return await prepare(identity, value)

                binding.prepare = gate
            return binding

        return bind

    config = fixed_package([]).model_dump(mode="json")
    config["tools"] = []
    config["agents"][0]["tools"] = []
    config["agents"][0].update(workflow="idempotency", input_schema={"type": "string"})
    async with system(
        tmp_path,
        [],
        config=AgentPackage.model_validate(
            {**config, "agents": [{**config["agents"][0], "workflow": "fixed"}]}
        ),
    ) as (engine, _, _, identities):
        engine.registry.register_workflow("idempotency", factory("1"))
        config["agents"][0]["version"] = "idem-1"
        engine.registry.activate(AgentPackage.model_validate(config))
        first = asyncio.create_task(engine.submit(identities["user"], "agent0", "key", "original"))
        if race:
            await asyncio.wait_for(entered.wait(), 2)
        else:
            await first
        next_config = copy.deepcopy(config)
        next_config["agents"][0].update(version="idem-2", workflow_version="2")
        next_config["agents"][0]["input_schema"] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        engine.registry.register_workflow("idempotency", factory("2"), version="2")
        engine.registry.activate(AgentPackage.model_validate(next_config))
        second = asyncio.create_task(engine.submit(identities["user"], "agent0", "key", "original"))
        if race:
            # Await until the second request really queues behind first admission.
            for _ in range(100):
                if engine.admission_lock._waiters:
                    break
                await asyncio.sleep(0.001)
            else:
                raise AssertionError("second request never reached the admission lock")
            release.set()
        left, right = await asyncio.gather(first, second)
        assert left["run_id"] == right["run_id"]
        assert factory_calls == ["1"]
        with engine.store.connection() as db:
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM requests WHERE owner=? AND agent_id=?", ("user", "agent0")
                ).fetchone()[0]
                == 1
            )
        with pytest.raises(AgentError) as error:
            await engine.submit(identities["user"], "agent0", "key", "changed")
        assert error.value.code == "idempotency_conflict"
        row = engine.store.request("user", left["run_id"])
        assert engine.store.control(row["run_id"])["client_digest"] == content_digest(
            {"version": "1", "input": "original"}
        )
        newer = await engine.submit(identities["user"], "agent0", "new-key", {"text": "new"})
        assert newer["run_id"] != row["run_id"] and factory_calls == ["1", "2"]


async def test_current_rag_binding_is_preserved_on_reopen(service):
    from raglab.service import LabService
    from tests.test_legacy_agentbase import LOCAL, paused_write, wait_run

    paused = await paused_write(service)
    before = service.store.request("local", paused["run_id"])
    review = paused["interrupts"][0]
    service.runtime.store.accept_review(
        "local", before["run_id"], review["review_id"], review["digest"]
    )
    await service.close()
    reopened = await LabService(service.settings).start()
    try:
        await reopened.resume(LOCAL, before["run_id"], approve=True)
        assert (await wait_run(reopened, before["run_id"]))["status"] == "succeeded"
        after = reopened.store.request("local", before["run_id"])
        for field in ("binding", "thread", "deadline", "payload", "epoch", "digest"):
            assert after[field] == before[field]
    finally:
        await reopened.close()


async def test_missing_history_refuses_execution_until_trusted_restore(tmp_path):
    config = package()
    async with system(tmp_path, ["old-result"], config=config) as (
        engine,
        old_provider,
        _,
        identities,
    ):
        from agentbase.runtime_contracts import RunOptions

        created = await engine.submit(
            identities["user"], "agent0", "old", "Input", RunOptions(pause_for_review=True)
        )
        paused = await wait(engine, created["run_id"])
        row = engine.store.request("user", created["run_id"])
        await engine.close()
        fresh_registry = ComponentRegistry()
        new_provider = ScriptProvider(["new-result"])
        fresh_registry.register_provider("offline", lambda config: new_provider, version="2")
        fresh_registry.register_workflow("conversation", conversation_factory, version="2")
        newer = config.model_dump(mode="json")
        newer["models"][0]["provider_version"] = "2"
        newer["agents"][0].update(version="2", workflow_version="2")
        fresh_registry.bind(AgentPackage.model_validate(newer))
        restarted = RunEngine(
            fresh_registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        await restarted.start()
        try:
            review = paused["interrupts"][0]
            with pytest.raises(AgentError) as error:
                await restarted.resume(
                    identities["user"],
                    row["run_id"],
                    review_id=review["review_id"],
                    digest=review["digest"],
                )
            assert error.value.code == "binding_changed"
            assert new_provider.requests == [] and old_provider.requests == []
            assert restarted.available
            fresh_registry.register_provider("offline", lambda config: old_provider)
            fresh_registry.register_workflow("conversation", conversation_factory)
            fresh_registry.restore(
                config,
                versions=RegistrationSelection(
                    workflows={"conversation": "1"}, providers={"offline": "1"}
                ),
                expected_digests={"agent0": row["binding"]},
            )
            assert fresh_registry.agent("agent0").definition.version == "2"
            await restarted.resume(
                identities["user"],
                row["run_id"],
                review_id=review["review_id"],
                digest=review["digest"],
            )
            assert (await wait(restarted, row["run_id"]))["result"] == "old-result"
            assert len(old_provider.requests) == 1 and new_provider.requests == []
        finally:
            await restarted.close()


async def test_old_run_creating_new_action_review_uses_original_review_policy(tmp_path):
    from agentbase.runtime_contracts import RunOptions

    async def original(args, context):
        return {"value": args["x"]}

    config = package(tools=[declaration(True)], budget={"review_seconds": 120})
    async with system(
        tmp_path, [[("custom", {"x": 4})], "done"], config=config, handler=original
    ) as (engine, _, _, identities):
        created = await engine.submit(
            identities["user"], "agent0", "old", "Input", RunOptions(pause_for_review=True)
        )
        paused = await wait(engine, created["run_id"])
        newer = config.model_dump(mode="json")
        newer["agents"][0].update(version="2", budget={"review_seconds": 300}, tools=[])
        engine.registry.activate(AgentPackage.model_validate(newer))
        review = paused["interrupts"][0]
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=review["review_id"],
            digest=review["digest"],
        )
        action = await wait(engine, created["run_id"])
        assert action["status"] == "awaiting_review"
        review = action["interrupts"][0]
        assert review["review_id"] != paused["interrupts"][0]["review_id"]
        record = engine.store.review("user", created["run_id"], review["review_id"])
        assert record["expires_at"] - record["paused_at"] == pytest.approx(120)
        await engine.resume(
            identities["user"],
            created["run_id"],
            review_id=review["review_id"],
            digest=review["digest"],
        )
        assert (await wait(engine, created["run_id"]))["result"] == "done"
