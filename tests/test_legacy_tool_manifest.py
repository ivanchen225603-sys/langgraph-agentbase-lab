"""Real legacy checkpoint tool releases and atomic admission metadata.

The checked-in schema2 SQL came from the pre-bridge source in a separate process.
It contains actual completed/paused requests and approvals/effects; its one history
marker is synthetic. Engine reopen tests below are in-process, not process kills.
"""

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import NamedTuple

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model, model_validator
from typing_extensions import TypedDict

from agentbase.contracts import AgentError
from agentbase.runtime_contracts import PreparedAdmission
from agentbase.schema import content_digest
from raglab.config import Settings
from raglab.errors import LabError
from raglab.models import RunInput, TaskSpec
from raglab.persistence import Store
from raglab.service import LabService
from raglab.tools import ReadArgs, ToolSpec
from tests.conftest import LOCAL, ROOT, wait_run


def release(function, *, version="1", raw=None):
    return ToolSpec(
        "test.release",
        StructuredTool.from_function(
            coroutine=function,
            name="test.release",
            description="Trusted versioned release test",
            args_schema=ReadArgs,
            metadata={"nested": {"value": "old"}},
        ),
        version=version,
        raw_output_schema=raw,
    )


def request(*, pause=True, question="original", **kwargs):
    return RunInput(
        tasks=[TaskSpec(id="task", tool="test.release", arguments={"question": question})],
        pause_for_review=pause,
        **kwargs,
    )


async def paused(service, key="old"):
    run = await service.submit(LOCAL, key, request())
    return await wait_run(service, run["run_id"], {"awaiting_review"})


async def test_manifest_and_request_committed_together_preserve_legacy_contract(service):
    async def old(question):
        return {"answer": question}

    service.registry.register(release(old))
    run = await paused(service)
    row = service.store.request("local", run["run_id"])
    with service.store.connection() as db:
        manifest = dict(
            db.execute(
                "SELECT * FROM legacy_tool_manifests WHERE run_id=?", (row["run_id"],)
            ).fetchone()
        )
        assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == "4"
        assert (
            db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
            == "8"
        )
    recorded = json.loads(manifest["payload"])
    assert manifest["binding"] == row["binding"]
    assert manifest["digest"] == content_digest(recorded)
    assert set(recorded["tools"]) == {"test.release"}
    entry = recorded["tools"]["test.release"]
    assert entry["legacy"] == json.loads(row["payload"])["tool_contracts"]["test.release"]
    assert "task_id" in entry["legacy"]["output_schema"]["properties"]
    assert entry["declaration"]["output_schema"] == {"type": "object"}
    original = "sha256:2b04d9c3a05c117a97ac9c9fec501ce2c9723838383b5cbf358dc0d5d2eff5e9"
    assert row["binding"] == service.runtime.registry.agent("rag").digest != original
    assert service.runtime.registry.resolve("rag", original).definition.version == "2"


async def test_failed_admission_rolls_back_manifest_controls_topic_and_victim(service):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def running(question):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    service.registry.register(release(running))
    old = await service.submit(LOCAL, "running", request(pause=False, conversation_id="topic"))
    await asyncio.wait_for(entered.wait(), 3)
    binding = service.runtime.binding(service.store.request("local", old["run_id"]))
    prepare = binding.prepare
    commits = []

    async def failing(actor, value):
        prepared = await prepare(actor, value)
        assert isinstance(prepared, PreparedAdmission)

        def commit(db, row):
            prepared.commit(db, row)
            assert db.in_transaction
            assert (
                db.execute(
                    "SELECT cancel FROM requests WHERE run_id=?", (old["run_id"],)
                ).fetchone()[0]
                == 1
            )
            assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 2
            assert db.execute("SELECT COUNT(*) FROM legacy_tool_manifests").fetchone()[0] == 2
            commits.append(row["run_id"])
            raise AgentError("injected", "atomic admission failure", 503)

        return PreparedAdmission(prepared.payload, commit)

    binding.prepare = failing
    try:
        with pytest.raises(LabError, match="atomic admission failure"):
            await service.submit(
                LOCAL, "failed", request(conversation_id="topic", switch_topic=True)
            )
        assert len(commits) == 1
        assert not cancelled.is_set() and old["run_id"] in service.active
        assert service.store.epoch("local", "topic") == 0
        with service.store.connection() as db:
            for table in ("requests", "agent_run_controls", "legacy_tool_manifests"):
                assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
            assert db.execute("SELECT cancel FROM requests").fetchone()[0] == 0
        assert service.telemetry.counts["runs_submitted"] == 1
        assert service.queue.empty()
    finally:
        binding.prepare = prepare


async def test_admission_rejects_async_commit_and_rolls_back_without_warning(service):
    binding = service.runtime.binding(
        {"agent_id": "rag", "binding": service.runtime.registry.agent("rag").digest}
    )
    prepare = binding.prepare

    async def invalid(actor, value):
        prepared = await prepare(actor, value)

        async def commit(db, row):
            raise AssertionError("an asynchronous SQL hook must never run")

        return PreparedAdmission(prepared.payload, commit)

    binding.prepare = invalid
    try:
        with pytest.raises(LabError, match="synchronous"):
            await service.submit(
                LOCAL,
                "async-hook",
                RunInput(
                    tasks=[TaskSpec(id="a", tool="memory.search", arguments={"question": "q"})]
                ),
            )
        with service.store.connection() as db:
            for table in ("requests", "agent_run_controls", "legacy_tool_manifests"):
                assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert service.queue.empty()
    finally:
        binding.prepare = prepare


async def test_old_paused_run_uses_actual_v1_new_run_uses_v2_and_retransmit_skips_prepare(service):
    called = []

    async def old(question):
        called.append("v1")
        return {"release": "v1"}

    async def newer(question):
        called.append("v2")
        return {"release": "v2"}

    service.registry.register(release(old))
    run = await paused(service)
    before = service.store.request("local", run["run_id"])
    frozen = service.registry.catalog.for_run(before)
    service.registry.register(release(newer, version="2"))
    assert (await service.submit(LOCAL, "old", request()))["run_id"] == run["run_id"]
    assert service.registry.catalog.for_run(before).digest == frozen.digest
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["task"]["value"] == {"release": "v1"}
    next_run = await service.submit(LOCAL, "new", request(pause=False))
    assert (await wait_run(service, next_run["run_id"], {"succeeded"}))["results"]["task"][
        "value"
    ] == {"release": "v2"}
    assert called == ["v1", "v2"]
    after = service.store.request("local", run["run_id"])
    for field in ("payload", "binding", "digest", "thread", "deadline", "epoch"):
        assert after[field] == before[field]


async def test_mutating_original_structured_tool_and_schema_does_not_change_old_run(service):
    called = []

    async def old(question, config: RunnableConfig):
        called.append(("v1", config["configurable"]["tool_context"]["operation_key"]))
        return {"answer": question}

    async def replacement(question):
        called.append("replacement")
        return {"wrong": True}

    class OtherArgs(BaseModel):
        different: str

    raw = {"type": "object", "required": ["answer"]}
    spec = release(old, raw=raw)
    service.registry.register(spec)
    run = await paused(service)
    spec.tool.coroutine = replacement
    spec.tool.args_schema = OtherArgs
    spec.tool.description = "Changed declaration"
    spec.tool.metadata["nested"]["value"] = "changed"
    raw["required"].append("wrong")
    row = service.store.request("local", run["run_id"])
    frozen = service.registry.catalog.for_run(row).tools["test.release"]
    assert frozen.spec.tool.metadata == {"nested": {"value": "old"}}
    assert frozen.declaration.output_schema == {"type": "object", "required": ["answer"]}
    with pytest.raises(LabError) as error:
        await service.submit(LOCAL, "changed", request())
    assert error.value.code == "tool_version_conflict"
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["task"]["value"] == {"answer": "original"}
    assert called == [("v1", f"{run['run_id']}:task")]
    assert len(service.store.requests()) == 1


@pytest.mark.parametrize("change", ["function", "raw_schema", "input_schema"])
async def test_same_version_release_changes_are_rejected_before_new_admission(service, change):
    async def old(question):
        return {"answer": question}

    async def different(question):
        return {"answer": "changed"}

    class OtherArgs(BaseModel):
        different: str

    first = release(old)
    service.registry.register(first)
    run = await paused(service)
    candidate = release(different) if change == "function" else replace(first)
    if change == "raw_schema":
        candidate = replace(candidate, raw_output_schema={"type": "object", "required": ["new"]})
    if change == "input_schema":
        candidate = replace(
            candidate, tool=first.tool.model_copy(update={"args_schema": OtherArgs})
        )
    # Exercise the old direct specs escape hatch as well as register().
    service.registry.specs[candidate.name] = candidate
    with pytest.raises(LabError) as error:
        await service.submit(LOCAL, "changed", request())
    assert error.value.code == "tool_version_conflict"
    await service.resume(LOCAL, run["run_id"], True)
    assert (await wait_run(service, run["run_id"], {"succeeded"}))["results"]["task"]["value"] == {
        "answer": "original"
    }
    assert len(service.store.requests()) == 1


async def test_cold_engine_requires_matching_trusted_historical_release(service):
    called = []

    async def original(question):
        called.append("v1")
        return {"version": "v1"}

    async def newer(question):
        called.append("v2")
        return {"version": "v2"}

    original_spec = release(original)
    service.registry.register(original_spec)
    run = await paused(service)
    before = service.store.request("local", run["run_id"])
    await service.close()
    restarted = await LabService(service.settings).start()
    try:
        restarted.registry.register(release(newer, version="2"))
        view = await restarted.view("local", run["run_id"])
        assert not view["recoverable"] and view["resume_reason"] == "tool_version_mismatch"
        with pytest.raises(LabError):
            await restarted.resume(LOCAL, run["run_id"], True)
        assert called == []
        restarted.registry.register(
            replace(original_spec, raw_output_schema={"type": "object", "required": ["wrong"]}),
            activate=False,
        )
        assert (await restarted.view("local", run["run_id"]))[
            "resume_reason"
        ] == "tool_version_mismatch"
        restarted.registry.register(original_spec, activate=False)
        await restarted.resume(LOCAL, run["run_id"], True)
        assert (await wait_run(restarted, run["run_id"], {"succeeded"}))["results"]["task"][
            "value"
        ] == {"version": "v1"}
        assert called == ["v1"]
        assert restarted.registry.specs["test.release"].version == "2"
        after = restarted.store.request("local", run["run_id"])
        for field in ("payload", "binding", "digest", "thread", "deadline", "epoch"):
            assert after[field] == before[field]
    finally:
        await restarted.close()


async def test_private_parameter_class_preserves_actual_argument_type_after_model_rebuild(service):
    observed = []

    class Arguments(BaseModel):
        question: str

    async def original(question):
        observed.append((question, type(question)))
        return {"answer": question}

    spec = release(original)
    spec = replace(spec, tool=spec.tool.model_copy(update={"args_schema": Arguments}))
    service.registry.register(spec)
    run = await service.submit(LOCAL, "class", request(question="123"))
    await wait_run(service, run["run_id"], {"awaiting_review"})
    Arguments.model_fields["question"].annotation = int
    Arguments.model_rebuild(force=True)
    assert Arguments.model_validate({"question": "123"}).question == 123
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["task"]["value"] == {"answer": "123"}
    assert observed == [("123", str)]
    with pytest.raises(LabError) as error:
        await service.submit(LOCAL, "changed-class", request(question="123"))
    assert error.value.code == "tool_version_conflict"


async def test_explicit_history_same_version_is_used_instead_of_incompatible_active(service):
    observed = []

    async def original(question):
        observed.append("old")
        return {"version": "original"}

    original_spec = release(original)
    service.registry.register(original_spec)
    run = await paused(service)
    await service.close()
    restarted = await LabService(service.settings).start()
    try:
        active = replace(original_spec, raw_output_schema={"type": "object", "required": ["new"]})
        restarted.registry.register(active)
        assert (await restarted.view("local", run["run_id"]))[
            "resume_reason"
        ] == "tool_version_mismatch"
        restarted.registry.register(original_spec, activate=False)
        await restarted.resume(LOCAL, run["run_id"], True)
        assert (await wait_run(restarted, run["run_id"], {"succeeded"}))["results"]["task"][
            "value"
        ] == {"version": "original"}
        assert observed == ["old"]
        assert restarted.registry.specs["test.release"] is active
        with pytest.raises(LabError) as error:
            await restarted.submit(LOCAL, "invalid-active", request())
        assert error.value.code == "tool_version_conflict"
    finally:
        await restarted.close()


@pytest.mark.parametrize("change", ["default", "parse_method", "dump_method"])
async def test_private_parser_defaults_and_entry_points_are_not_dynamic_inheritance(
    service, change
):
    observed = []

    class Arguments(BaseModel):
        question: str
        words: list[str] = ["old"]

        @model_validator(mode="after")
        def expand(self):
            self.question += " " + ",".join(self.words)
            return self

    async def original(question, words=None):
        observed.append(question)
        return {"answer": question}

    spec = release(original)
    spec = replace(spec, tool=spec.tool.model_copy(update={"args_schema": Arguments}))
    service.registry.register(spec)
    run = await paused(service)
    if change == "default":
        Arguments.model_fields["words"].default.append("changed")
    elif change == "parse_method":
        Arguments.model_validate = classmethod(
            lambda cls, obj: cls.model_construct(question="changed")
        )
    else:
        Arguments.model_dump = lambda self, **kwargs: {}
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["task"]["value"] == {"answer": "original old"}
    assert observed == ["original old"]
    with pytest.raises(LabError) as error:
        await service.submit(LOCAL, "mutated-parser", request())
    assert error.value.code == "tool_version_conflict"


@pytest.mark.parametrize("container", ["direct", "list", "optional", "dict"])
@pytest.mark.parametrize("change", ["default", "parse_method"])
async def test_nested_argument_models_preserve_actual_handler_values(service, container, change):
    observed = []

    class Inner(BaseModel):
        words: list[str] = ["old"]

    annotations = {
        "direct": Inner,
        "list": list[Inner],
        "optional": Inner | None,
        "dict": dict[str, Inner],
    }
    Arguments = create_model(
        "NestedArguments", question=(str, ...), inner=(annotations[container], ...)
    )

    async def original(question, inner):
        item = inner[0] if container == "list" else inner["item"] if container == "dict" else inner
        observed.append(list(item.words))
        return {"words": list(item.words)}

    spec = release(original)
    service.registry.register(
        replace(spec, tool=spec.tool.model_copy(update={"args_schema": Arguments}))
    )
    inner = [{}] if container == "list" else {"item": {}} if container == "dict" else {}
    run = await service.submit(
        LOCAL,
        "nested",
        RunInput(
            tasks=[
                TaskSpec(
                    id="task", tool="test.release", arguments={"question": "q", "inner": inner}
                )
            ],
            pause_for_review=True,
        ),
    )
    await wait_run(service, run["run_id"], {"awaiting_review"})
    if change == "default":
        Inner.model_fields["words"].default.append("changed")
    else:
        Inner.model_validate = classmethod(lambda cls, obj: cls.model_construct(words=["changed"]))
    await service.resume(LOCAL, run["run_id"], True)
    final = await wait_run(service, run["run_id"], {"succeeded"})
    assert final["results"]["task"]["value"] == {"words": ["old"]}
    assert observed == [["old"]]
    with pytest.raises(LabError) as error:
        await service.submit(
            LOCAL,
            "changed-nested",
            RunInput(
                tasks=[
                    TaskSpec(
                        id="task", tool="test.release", arguments={"question": "q", "inner": inner}
                    )
                ]
            ),
        )
    assert error.value.code == "tool_version_conflict"


@pytest.mark.parametrize("kind", ["model_default", "factory", "recursive"])
async def test_unsupported_python_argument_models_are_rejected_before_admission(service, kind):
    class Inner(BaseModel):
        words: list[str] = ["old"]

    class Recursive(BaseModel):
        question: str
        child: "Recursive | None" = None

    defaults = {"model_default": Inner(), "factory": Field(default_factory=Inner)}
    Arguments = (
        Recursive
        if kind == "recursive"
        else create_model("ComplexArguments", question=(str, ...), inner=(Inner, defaults[kind]))
    )

    async def original(question, inner=None, child=None):
        raise AssertionError("unsupported freeze must never invoke handler")

    spec = release(original)
    service.registry.register(
        replace(spec, tool=spec.tool.model_copy(update={"args_schema": Arguments}))
    )
    with pytest.raises(LabError) as error:
        await service.submit(LOCAL, "unsupported", request())
    assert error.value.code == "tool_schema_freeze"
    assert not service.store.requests() and service.queue.empty()


@pytest.mark.parametrize("structure", ["typed_dict", "dataclass", "named_tuple"])
async def test_unhandled_structures_cannot_hide_shared_nested_models(service, structure):
    class Inner(BaseModel):
        words: list[str] = ["old"]

    class Holder(TypedDict):
        item: Inner

    @dataclass
    class DataHolder:
        item: Inner = dataclass_field(default_factory=Inner)

    class TupleHolder(NamedTuple):
        item: Inner

    choices = {"typed_dict": Holder, "dataclass": DataHolder, "named_tuple": TupleHolder}
    Arguments = create_model(
        "StructuredArguments", question=(str, ...), holder=(choices[structure], ...)
    )

    async def original(question, holder):
        raise AssertionError("unsupported structure must never reach handler")

    spec = release(original)
    service.registry.register(
        replace(spec, tool=spec.tool.model_copy(update={"args_schema": Arguments}))
    )
    with pytest.raises(LabError) as error:
        await service.submit(
            LOCAL,
            "hidden-model",
            RunInput(
                tasks=[
                    TaskSpec(
                        id="task",
                        tool="test.release",
                        arguments={"question": "q", "holder": {"item": {}}},
                    )
                ]
            ),
        )
    assert error.value.code == "tool_schema_freeze"
    assert not service.store.requests()


async def test_prepare_captures_release_before_active_changes_and_sql_commit(service):
    observed = []

    async def original(question):
        observed.append("original")
        return {"version": "original"}

    async def newer(question):
        observed.append("new")
        return {"version": "new"}

    service.registry.register(release(original))
    binding = service.runtime.binding(
        {"agent_id": "rag", "binding": service.runtime.registry.agent("rag").digest}
    )
    prepare = binding.prepare

    async def switch(actor, value):
        prepared = await prepare(actor, value)
        service.registry.register(release(newer, version="2"))
        return prepared

    binding.prepare = switch
    try:
        run = await service.submit(LOCAL, "captured", request(pause=False))
    finally:
        binding.prepare = prepare
    assert (await wait_run(service, run["run_id"], {"succeeded"}))["results"]["task"]["value"] == {
        "version": "original"
    }
    assert observed == ["original"]
    row = service.store.request("local", run["run_id"])
    assert json.loads(row["payload"])["tool_versions"] == {"test.release": "1"}
    assert service.registry.specs["test.release"].version == "2"


async def test_checkpoint_initialization_failure_keeps_manifest_and_original_review_policy(
    service, monkeypatch
):
    observed = []

    async def write(question, config: RunnableConfig):
        observed.append("original-write")
        context = config["configurable"]["tool_context"]
        key = context["operation_key"]
        assert service.store.operation(key, context["principal"], context["run_id"]) is None
        result = {"effect": "synthetic"}
        service.store.receipt(key, result)
        return result

    async def read(question):
        observed.append("new-read")
        return {"answer": question}

    service.registry.register(
        replace(release(write), read_only=False, retry_safe=False, scope="memory:write")
    )
    binding = service.runtime.binding(
        {"agent_id": "rag", "binding": service.runtime.registry.agent("rag").digest}
    )

    async def failing(*args, **kwargs):
        raise OSError("checkpoint initialization failure")

    with monkeypatch.context() as failures:
        failures.setattr(binding.graph, "aupdate_state", failing)
        with pytest.raises(LabError) as error:
            await service.submit(LOCAL, "initial", request(pause=False))
        assert error.value.code == "checkpoint_unavailable"
    row = service.store.requests()[0]
    before = dict(row)
    assert (
        service.registry.catalog.for_run(row).tools["test.release"].declaration.read_only is False
    )
    service.registry.register(release(read, version="2"))
    service.available = True
    await service.resume(LOCAL, row["run_id"])
    await wait_run(service, row["run_id"], {"awaiting_review"})
    assert observed == []
    await service.resume(LOCAL, row["run_id"], True)
    await wait_run(service, row["run_id"], {"succeeded"})
    assert observed == ["original-write"]
    after = service.store.request("local", row["run_id"])
    for field in ("payload", "binding", "digest", "thread", "deadline", "epoch"):
        assert after[field] == before[field]


def test_catalog_capacity_and_failed_capture_keep_frozen_release(service):
    async def old(question):
        return {"answer": question}

    async def changed(question):
        return {"changed": question}

    catalog = service.registry.catalog
    catalog.capacity = len(catalog._registered) + 1
    service.registry.register(release(old))
    captured = catalog.capture(["test.release"])
    with pytest.raises(LabError) as error:
        service.registry.register(release(changed, version="2"))
    assert error.value.code == "tool_binding_capacity"
    with pytest.raises(LabError) as error:
        service.registry.register(release(changed))
    assert error.value.code == "tool_version_conflict"
    assert captured.tools["test.release"].spec.tool.coroutine is old


@pytest.mark.parametrize("tamper", ["binding", "digest", "declaration", "tool_set"])
async def test_manifest_inconsistency_cannot_restore_or_execute(service, tamper):
    called = []

    async def old(question):
        called.append(True)
        return {"answer": question}

    service.registry.register(release(old))
    run = await paused(service)
    with service.store.connection() as db:
        recorded = dict(db.execute("SELECT * FROM legacy_tool_manifests").fetchone())
        payload = json.loads(recorded["payload"])
        if tamper == "binding":
            recorded["binding"] = "wrong"
        elif tamper == "digest":
            recorded["digest"] = "wrong"
        else:
            if tamper == "tool_set":
                payload["tools"].clear()
            else:
                payload["tools"]["test.release"]["declaration"]["scope"] = "other"
            recorded["payload"] = json.dumps(payload)
            recorded["digest"] = content_digest(payload)
        db.execute(
            "UPDATE legacy_tool_manifests SET binding=?,digest=?,payload=?",
            (recorded["binding"], recorded["digest"], recorded["payload"]),
        )
    view = await service.view("local", run["run_id"])
    assert not view["recoverable"] and view["resume_reason"] == "tool_manifest_conflict"
    with pytest.raises(LabError):
        await service.resume(LOCAL, run["run_id"], True)
    assert called == []


def old_fixture(path):
    with sqlite3.connect(path) as db:
        db.executescript((ROOT / "tests/fixtures/legacy_schema2.sql").read_text(encoding="utf-8"))


def original_records(path):
    with sqlite3.connect(path) as db:
        tables = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            if row[0]
            not in {
                "legacy_tool_manifests",
                "legacy_execution_protocol",
                "legacy_action_links",
                "legacy_write_actions",
                "legacy_plan_grants",
                "legacy_plan_actions",
                "agent_artifact_heads",
                "agent_artifact_revisions",
                "agent_artifact_mutations",
                "agent_artifact_meta",
                "agent_task_runs",
                "agent_task_closure",
                "agent_task_dispatch",
                "agent_task_waits",
                "agent_task_outcomes",
                "agent_task_consumptions",
                "agent_plan_revisions",
                "agent_plan_attempts",
                "agent_memory_audit",
            }
        ]
        records = {
            table: db.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            for table in tables
        }
        if "meta" in records:
            records["meta"] = [row for row in records["meta"] if row[0] != "schema"]
        if "agent_runtime_meta" in records:
            records["agent_runtime_meta"] = [
                row for row in records["agent_runtime_meta"] if row[0] != "schema"
            ]
        if "agent_context_meta" in records:
            records["agent_context_meta"] = [
                row for row in records["agent_context_meta"] if row[0] != "schema"
            ]
        return records


async def test_true_prebridge_schema_two_upgrade_preserves_all_records_and_query_only(
    tmp_path, monkeypatch
):
    path = tmp_path / "business.sqlite"
    old_fixture(path)
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()[0]
            == "1"
        )
    before = original_records(path)
    checkpoint = tmp_path / "checkpoints.sqlite"
    with sqlite3.connect(checkpoint) as db:
        db.executescript(
            (ROOT / "tests/fixtures/legacy_schema2_checkpoints.sql").read_text(encoding="utf-8")
        )
    before_checkpoint = original_records(checkpoint)
    lab = await LabService(Settings(data_dir=tmp_path, enable_simulation=True)).start()
    try:
        assert original_records(path) == before
        assert original_records(checkpoint) == before_checkpoint
        with lab.store.connection() as db:
            assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == "4"
            assert (
                db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()[0]
                == "8"
            )
            assert db.execute("SELECT COUNT(*) FROM legacy_tool_manifests").fetchone()[0] == 0
            assert (
                db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()[0]
                == "2"
            )
            assert db.execute("SELECT COUNT(*) FROM agent_memory_audit").fetchone()[0] == 0
            assert (
                db.execute("SELECT value FROM agent_artifact_meta WHERE key='schema'").fetchone()[0]
                == "2"
            )
            for table in (
                "legacy_execution_protocol",
                "legacy_action_links",
                "legacy_write_actions",
                "legacy_plan_grants",
                "legacy_plan_actions",
                "agent_artifact_heads",
                "agent_artifact_revisions",
                "agent_artifact_mutations",
            ):
                assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        row = next(row for row in lab.store.requests() if row["key"] == "old-paused")
        completed = next(row for row in lab.store.requests() if row["key"] == "old-done")
        completed_view = await lab.view("local", completed["run_id"])
        assert completed_view["status"] == "succeeded"
        assert completed_view["results"]["write"]["value"]["value"] == "old effect"
        assert completed_view["results"]["write"]["value"]["effect"] == "applied"
        view = await lab.view("local", row["run_id"])
        assert not view["recoverable"]
        assert view["resume_reason"] == "legacy_tool_execution_requires_matching_release"
        # Keep the captured deadline byte-for-byte, and isolate version rejection
        # from this fixture's necessarily aging wall clock.
        with monkeypatch.context() as clocks:
            clocks.setattr("time.time", lambda: row["deadline"] - 1)
            with pytest.raises(LabError, match="matching_release"):
                await lab.resume(LOCAL, row["run_id"], True)
        assert lab.queue.empty() and not lab.active
        assert original_records(path) == before
        assert original_records(checkpoint) == before_checkpoint
    finally:
        await lab.close()


def test_schema_two_fixtures_match_captured_provenance():
    fixtures = ROOT / "tests/fixtures"
    provenance = json.loads(
        (fixtures / "legacy_schema2_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["capture"]["schema"] == 2
    assert provenance["checkpoint_export_included"] is True
    assert provenance["history_marker_synthetic"] is True
    for name, record in provenance["fixtures"].items():
        data = (fixtures / name).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]


def test_schema_two_failed_migration_rolls_back_manifest_table_and_original_records(
    tmp_path, monkeypatch
):
    from agentbase.run_store import SqliteRunStore

    path = tmp_path / "business.sqlite"
    old_fixture(path)
    before = original_records(path)
    install = SqliteRunStore.install

    def failing(db, **kwargs):
        install(db, **kwargs)
        raise AgentError("injected", "migration failure")

    monkeypatch.setattr(SqliteRunStore, "install", staticmethod(failing))
    with pytest.raises(AgentError):
        Store(path)
    assert original_records(path) == before
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == "2"
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='legacy_tool_manifests'"
        ).fetchone()
        assert (
            db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()[0]
            == "1"
        )
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='agent_memory_audit'"
        ).fetchone()
