"""Actual conversation requests, durable artifacts, summaries and bounded readback."""

import copy
import json

import pytest

from agentbase import RuntimeContextManager, artifact_read_tool
from agentbase.artifacts import ArtifactOperation, ArtifactScope
from agentbase.configuration import AgentPackage, RuntimeContextPolicy
from agentbase.contracts import AgentError, Message, ToolCall
from agentbase.providers import Utf8BudgetEstimator
from agentbase.runtime import RunEngine
from agentbase.schema import canonical_json, content_digest
from tests.test_agentbase_runtime import declaration, package, system, wait


def configuration(*, management=None, budget=None, custom=False):
    tools = [artifact_read_tool().model_dump(mode="json")]
    if custom:
        tool = declaration()
        tool["output_schema"] = {"type": "string"}
        tools.append(tool)
    data = package(tools=tools, budget=budget).model_dump(mode="json")
    data["agents"][0]["context"]["management"] = management or {
        "retain_recent_units": 1,
        "summary_chars": 256,
        "offload_bytes": 1024,
        "preview_chars": 64,
    }
    return AgentPackage.model_validate(data)


def scope(context):
    return ArtifactScope(
        owner=context.identity.owner, agent_id=context.agent_id, namespace=context.conversation_id
    )


def seed(contexts, *, sources=None):
    for index in range(2):
        assert contexts.append_turn(
            "user",
            "agent0",
            "default",
            0,
            f"old{index}",
            [
                Message(
                    role="user", content=f"old-input-{index}" + "x" * 1900, sources=sources or []
                ),
                Message(
                    role="assistant",
                    content=f"old-output-{index}" + "y" * 1900,
                    sources=sources or [],
                ),
            ],
            guard=lambda db: True,
        )


def test_optional_policy_preserves_old_serialization_and_digests():
    original = package()
    data = original.model_dump(mode="json")
    assert "management" not in data["agents"][0]["context"]
    data["agents"][0]["context"]["management"] = None
    restored = AgentPackage.model_validate(data)
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")
    assert restored.digest == original.digest
    with pytest.raises(ValueError):
        RuntimeContextPolicy(offload_bytes=512, preview_chars=128)


async def test_actual_tool_result_offload_preserves_raw_ledger_and_history(tmp_path):
    original = "private-original-" + "中🙂" * 2200

    async def handler(args, context):
        return original

    async with system(
        tmp_path,
        [[("custom", {"x": 1})], "done"],
        config=configuration(custom=True),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "offload", "current-input")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "succeeded", result
        observation = next(m for m in provider.requests[1].messages if m.role == "tool")
        envelope = json.loads(observation.content)
        assert original not in observation.content
        assert envelope["read_tool"] == "artifact.read"
        ref = envelope["artifact"]
        assert ref in observation.sources
        context = await engine.context(created["run_id"])
        record = engine.artifacts.read(scope(context), ref["id"])
        assert json.loads(record.value["content"])["value"] == original
        history = contexts.turns("user", "agent0", "default", 0, 10)[0]
        assert json.loads(next(m.content for m in history if m.role == "tool"))["value"] == original
        operation = engine.actions.operations("user", created["run_id"])[0]
        assert engine.actions.previous_value(operation) == original
        await RuntimeContextManager(engine).prepare(
            engine.registry.resolve("agent0", context.config_digest),
            context,
            [
                *provider.requests[0].messages,
                next(m for m in history if m.role == "assistant" and m.tool_calls),
                next(m for m in history if m.role == "tool"),
            ],
            current_index=len(provider.requests[0].messages) - 1,
        )
        assert len(engine.artifacts.list(scope(context))) == 1
        engine.artifacts.revoke(
            scope(context),
            ArtifactOperation(
                run_id=context.run_id, key="revoke", config_digest=context.config_digest
            ),
            record.id,
            record.version,
            guard=lambda db: True,
        )
        assert (await engine.view("user", context.run_id))["result"] is None
        assert engine.store.final_receipt("user", context.run_id) is not None


async def test_actual_model_readback_uses_normal_tool_budget_and_unicode_pages(tmp_path):
    original = "中🙂" * 2000

    async def handler(args, context):
        return original

    async with system(
        tmp_path,
        [[("custom", {"x": 1})], [], "done"],
        config=configuration(custom=True),
        handler=handler,
    ) as (engine, provider, contexts, identities):
        original_stream = provider.stream

        async def stream(request):
            if len(provider.requests) == 1:
                ref = json.loads(next(m.content for m in request.messages if m.role == "tool"))[
                    "artifact"
                ]
                provider.scripts[1] = [
                    (
                        "artifact.read",
                        {
                            "id": ref["id"],
                            "version": ref["version"],
                            "digest": ref["digest"],
                            "offset": 12,
                            "limit": 64,
                        },
                    )
                ]
            async for chunk in original_stream(request):
                yield chunk

        provider.stream = stream
        created = await engine.submit(identities["user"], "agent0", "readback", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "succeeded", result
        assert len(provider.requests) == 3
        reader_message = [m for m in provider.requests[2].messages if m.role == "tool"][-1]
        page = json.loads(reader_message.content)["value"]
        record = engine.artifacts.read(
            ArtifactScope(owner="user", agent_id="agent0", namespace="default"),
            page["reference"]["id"],
        )
        assert page["text"] == canonical_json(record.value).decode()[12:76]
        assert page["next_offset"] == 76 and len(page["text"]) == 64
        assert page["reference"] in reader_message.sources
        assert len(engine.actions.operations("user", created["run_id"])) == 2
        assert (
            len(
                engine.artifacts.list(
                    ArtifactScope(owner="user", agent_id="agent0", namespace="default")
                )
            )
            == 1
        )


async def test_budgeted_summaries_run_in_actual_workflow_and_survive_new_engine(tmp_path):
    config = configuration(budget={"max_input_chars": 6000, "max_model_calls": 4})
    async with system(
        tmp_path,
        [
            json.dumps({"summary": "retained-data"}),
            json.dumps({"summary": "retained-data-2"}),
            "done",
        ],
        config=config,
    ) as (engine, provider, contexts, identities):
        seed(contexts)
        created = await engine.submit(
            identities["user"], "agent0", "summaries", "protected-current-input"
        )
        result = await wait(engine, created["run_id"])
        assert result["status"] == "succeeded", result
        summary_requests = [
            r
            for r in provider.requests
            if r.output_schema and "summary" in r.output_schema.get("properties", {})
        ]
        assert len(summary_requests) == 2 and len(provider.requests) == 3
        main = provider.requests[-1]
        assert any(m.content == "protected-current-input" for m in main.messages)
        assert any("untrusted_context_summary" in m.content for m in main.messages)
        assert all("x" * 100 not in m.content for m in main.messages)
        assert (
            sum(
                len(m.content)
                for turn in contexts.turns("user", "agent0", "default", 0, 10)
                for m in turn
            )
            > 7600
        )
        assert engine.model_caller.ledger.snapshot(created["run_id"])["calls"] == 3
        context = await engine.context(created["run_id"])
        bound = engine.registry.resolve("agent0", context.config_digest)
        original = [
            Message(role="system", content=bound.definition.prompt.system_template),
            *[m for turn in contexts.turns("user", "agent0", "default", 0, 10)[:2] for m in turn],
            Message(role="user", content="protected-current-input"),
        ]
        reopened = RunEngine(
            engine.registry, engine.store, engine.saver, engine.contexts, engine.identity_resolver
        )
        prepared = await RuntimeContextManager(reopened).prepare(
            bound, context, original, current_index=len(original) - 1
        )
        assert prepared == main
        assert engine.model_caller.ledger.snapshot(created["run_id"])["calls"] == 3
        assert len(provider.requests) == 3
        assert engine.store.final_receipt("user", context.run_id) is not None


async def test_summary_and_main_share_call_cap_without_resetting_policy(tmp_path):
    config = configuration(budget={"max_input_chars": 6000, "max_model_calls": 1})
    async with system(tmp_path, [json.dumps({"summary": "retained-data"})], config=config) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        seed(contexts)
        created = await engine.submit(identities["user"], "agent0", "cap", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "failed" and result["error"] == "model_run_budget"
        assert len(provider.requests) == 1
        assert engine.model_caller.ledger.snapshot(created["run_id"])["calls"] == 1
        assert engine.store.final_receipt("user", created["run_id"]) is None


@pytest.mark.parametrize("summary", [{"summary": "z" * 257}, {"summary": "ok", "sources": []}])
async def test_invalid_summary_is_not_sent_as_main_context(tmp_path, summary):
    async with system(
        tmp_path, [json.dumps(summary)], config=configuration(budget={"max_input_chars": 6000})
    ) as (engine, provider, contexts, identities):
        seed(contexts)
        created = await engine.submit(identities["user"], "agent0", "invalid", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "failed" and len(provider.requests) == 1
        assert engine.store.final_receipt("user", created["run_id"]) is None


async def test_current_input_and_tool_schema_cannot_be_hidden_by_summary(tmp_path):
    config = configuration(budget={"max_input_chars": 2100})
    async with system(tmp_path, [], config=config) as (engine, provider, contexts, identities):
        created = await engine.submit(identities["user"], "agent0", "large-input", "current" * 500)
        result = await wait(engine, created["run_id"])
        assert result["status"] == "failed" and result["error"] == "context_limit"
        assert (
            provider.requests == []
            and engine.model_caller.ledger.snapshot(created["run_id"])["calls"] == 0
        )


async def test_revoke_in_summary_completion_blocks_main_provider_request(tmp_path):
    async with system(
        tmp_path,
        [json.dumps({"summary": "retained-data"})],
        config=configuration(budget={"max_input_chars": 6000}),
    ) as (engine, provider, contexts, identities):
        artifact = engine.artifacts.create(
            ArtifactScope(owner="user", agent_id="agent0", namespace="default"),
            ArtifactOperation(run_id="seed", key="source", config_digest="sha256:" + "0" * 64),
            "source",
            sources=[],
            guard=lambda db: True,
        )
        seed(contexts, sources=[artifact.reference()])
        original_stream = provider.stream

        async def stream(request):
            async for chunk in original_stream(request):
                if chunk.finish_reason == "stop":
                    engine.artifacts.revoke(
                        ArtifactScope(owner="user", agent_id="agent0", namespace="default"),
                        ArtifactOperation(
                            run_id="seed", key="revoke", config_digest="sha256:" + "0" * 64
                        ),
                        artifact.id,
                        1,
                        guard=lambda db: True,
                    )
                yield chunk

        provider.stream = stream
        created = await engine.submit(identities["user"], "agent0", "revoke", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "failed" and result["resume_reason"] == "source_revoked", result
        snapshot = await engine.checkpoint_state(engine.store.internal_request(created["run_id"]))
        assert snapshot.values["error"] == "source_revoked"
        assert (
            len(provider.requests) == 1
            and engine.store.final_receipt("user", created["run_id"]) is None
        )


@pytest.mark.parametrize("field,value", [("owner", "other"), ("limit", 4097), ("offset", -1)])
async def test_reader_arguments_use_tool_schema_before_handler(tmp_path, field, value):
    async with system(tmp_path, ["done"], config=configuration()) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "run", "current")
        await wait(engine, created["run_id"])
        context = await engine.context(created["run_id"])
        arguments = {
            "id": "unknown",
            "version": 1,
            "digest": "sha256:" + "0" * 64,
            "offset": 0,
            "limit": 1,
        }
        arguments[field] = value
        with pytest.raises(AgentError):
            await engine.tools.execute(
                context, ToolCall(call_id="invalid", name="artifact.read", arguments=arguments)
            )
        assert engine.actions.operations("user", context.run_id) == []


async def test_context_management_requires_explicit_reader_allowlist(tmp_path):
    data = configuration().model_dump(mode="json")
    data["agents"][0]["tools"] = []
    with pytest.raises(AgentError, match="bounded runtime artifact reader"):
        async with system(tmp_path, [], config=AgentPackage.model_validate(data)):
            pass


async def test_reader_cached_result_is_reauthorized_after_revoke(tmp_path):
    async with system(tmp_path, ["done"], config=configuration()) as (
        engine,
        provider,
        contexts,
        identities,
    ):
        created = await engine.submit(identities["user"], "agent0", "run", "current")
        await wait(engine, created["run_id"])
        context = await engine.context(created["run_id"])
        record = engine.artifacts.create(
            scope(context),
            ArtifactOperation(
                run_id=context.run_id, key="create", config_digest=context.config_digest
            ),
            "original",
            sources=[],
            guard=lambda db: True,
        )
        args = {k: record.reference()[k] for k in ("id", "version", "digest")}
        call = ToolCall(
            call_id="read", name="artifact.read", arguments={**args, "offset": 0, "limit": 8}
        )
        result = await engine.tools.execute(context, call)
        assert result.succeeded and result.sources == [record.reference()]
        engine.artifacts.revoke(
            scope(context),
            ArtifactOperation(
                run_id=context.run_id, key="revoke", config_digest=context.config_digest
            ),
            record.id,
            1,
            guard=lambda db: True,
        )
        with pytest.raises(AgentError):
            await engine.tools.execute(context, call)
        assert len(engine.actions.operations("user", context.run_id)) == 1


def test_management_does_not_accept_embedded_business_fields():
    data = configuration().model_dump(mode="json")
    candidate = copy.deepcopy(data)
    candidate["agents"][0]["context"]["management"]["document_type"] = "anything"
    with pytest.raises(ValueError):
        AgentPackage.model_validate(candidate)


async def test_separate_large_units_are_summarized_without_recompressing_new_summary(tmp_path):
    config = configuration(
        management={"retain_recent_units": 1, "summary_chars": 64, "trigger_ratio": 1.0},
        budget={"max_input_chars": 6000},
    )
    async with system(
        tmp_path,
        [json.dumps({"summary": "summary-A"}), json.dumps({"summary": "summary-B"}), "done"],
        config=config,
    ) as (engine, provider, contexts, identities):
        contexts.append_turn(
            "user",
            "agent0",
            "default",
            0,
            "earlier",
            [
                Message(role="user", content="OLD-A-" + "x" * 4700),
                Message(role="assistant", content="OLD-B-" + "y" * 4700),
            ],
            guard=lambda db: True,
        )
        created = await engine.submit(identities["user"], "agent0", "separate", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "succeeded", result
        assert len(provider.requests) == 3
        assert "OLD-A-" in provider.requests[0].messages[-1].content
        assert "OLD-B-" in provider.requests[1].messages[-1].content
        assert "untrusted_context_summary" not in provider.requests[1].messages[-1].content
        assert (
            sum("untrusted_context_summary" in m.content for m in provider.requests[-1].messages)
            == 2
        )


async def test_multibyte_candidate_over_final_json_limit_reaches_summary_in_actual_workflow(
    tmp_path,
):
    config = configuration(
        budget={"max_input_chars": 65536, "max_input_tokens": 131072, "max_run_tokens": 600000}
    )
    async with system(
        tmp_path,
        [
            json.dumps({"summary": "one"}),
            json.dumps({"summary": "two"}),
            json.dumps({"summary": "three"}),
            "done",
        ],
        config=config,
    ) as (engine, provider, contexts, identities):
        provider.count_input_tokens = Utf8BudgetEstimator.input_tokens
        for index in range(2):
            contexts.append_turn(
                "user",
                "agent0",
                "default",
                0,
                f"old{index}",
                [
                    Message(role="user", content=f"user{index}" + "中" * 30000),
                    Message(role="assistant", content=f"assistant{index}" + "中" * 30000),
                ],
                guard=lambda db: True,
            )
        created = await engine.submit(identities["user"], "agent0", "unicode", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "succeeded", result
        assert len(provider.requests) == 4
        for request in provider.requests:
            assert len(canonical_json(request.model_dump(mode="json"))) <= 262144
            assert Utf8BudgetEstimator.input_tokens(request) <= 131072
        assert (
            sum(
                len(m.content)
                for turn in contexts.turns("user", "agent0", "default", 0, 10)
                for m in turn
            )
            >= 120000
        )


async def test_completed_groups_may_reuse_call_ids_without_changing_offload_type(tmp_path):
    async def handler(args, context):
        return "unused"

    async with system(
        tmp_path, [[], "done"], config=configuration(custom=True), handler=handler
    ) as (engine, provider, contexts, identities):
        artifact = engine.artifacts.create(
            ArtifactScope(owner="user", agent_id="agent0", namespace="default"),
            ArtifactOperation(run_id="seed", key="source", config_digest="sha256:" + "0" * 64),
            "readable",
            sources=[],
            guard=lambda db: True,
        )
        ref = artifact.reference()
        provider.scripts[0] = [
            (
                "artifact.read",
                {
                    "id": ref["id"],
                    "version": ref["version"],
                    "digest": ref["digest"],
                    "offset": 0,
                    "limit": 8,
                },
            )
        ]
        repeated = "c1_" + content_digest("call0")[7:39]
        original = "OLD-CUSTOM-" + "x" * 5000
        contexts.append_turn(
            "user",
            "agent0",
            "default",
            0,
            "older",
            [
                Message(role="user", content="old"),
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(call_id=repeated, name="custom", arguments={"x": 1})],
                ),
                Message(role="tool", tool_call_id=repeated, content=original),
            ],
            guard=lambda db: True,
        )
        created = await engine.submit(identities["user"], "agent0", "same-call", "current")
        result = await wait(engine, created["run_id"])
        assert result["status"] == "succeeded", result
        first, second = provider.requests
        first_old = next(m for m in first.messages if m.role == "tool")
        second_old = next(m for m in second.messages if m.role == "tool")
        assert (
            json.loads(first_old.content)["artifact"] == json.loads(second_old.content)["artifact"]
        )
        assert original not in second_old.content
        assert (
            len(
                engine.artifacts.list(
                    ArtifactScope(owner="user", agent_id="agent0", namespace="default")
                )
            )
            == 2
        )
