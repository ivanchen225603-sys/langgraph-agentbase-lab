"""共享 Engine 上的 RAG 策略层；不含队列、worker，也不镜像执行状态。"""

import copy
import inspect
import json
import time
from contextlib import contextmanager

from langsmith import tracing_context

from agentbase.configuration import AgentPackage, ToolDeclaration
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, RunContext, ToolCall
from agentbase.registry import ComponentRegistry
from agentbase.run_store import SqliteRunStore
from agentbase.runtime import RunEngine
from agentbase.runtime_contracts import (
    EngineSettings,
    FinalProjection,
    GraphBinding,
    PreparedAdmission,
    RunOptions,
)
from agentbase.schema import canonical_json, content_digest
from agentbase.storage import SqliteContextStore

from .budget import provider_scope
from .errors import LabError
from .models import Principal, RunInput, TaskSpec, simple_plan
from .persistence import LEGACY_AGENT_ID, digest, encode
from .tool_bridge import PROTOCOL


def identity(principal: Principal) -> Identity:
    return Identity(
        owner=principal.owner,
        scopes=frozenset(principal.scopes),
        permission_version=principal.permission_version,
    )


def principal(actor: Identity) -> Principal:
    return Principal(
        owner=actor.owner, scopes=set(actor.scopes), permission_version=actor.permission_version
    )


class RagRuntimeAdapter:
    def __init__(self, service, saver):
        self.service = service
        self.identities: dict[str, Principal] = {}
        self.observed: set[tuple[str, str]] = set()
        registry = ComponentRegistry()
        registry.register_workflow("legacy.rag", self.factory, version="2")
        registry.bind(
            AgentPackage.model_validate(
                {
                    "agents": [
                        {
                            "id": LEGACY_AGENT_ID,
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
        # 保留原始的 version-2 描述符/digest，供已授权的旧查询使用。
        old = registry.agent(LEGACY_AGENT_ID).definition.model_dump(mode="json")
        self.legacy_query_binding = registry.agent(LEGACY_AGENT_ID).digest
        registry.register_workflow("legacy.rag", self.factory, version="3")
        registry.activate(
            AgentPackage.model_validate(
                {"agents": [{**old, "version": "3", "workflow_version": "3"}]}
            )
        )
        legacy_active = registry.agent(LEGACY_AGENT_ID).digest
        if service.configure_registry is not None:
            configured = service.configure_registry(registry)
            if inspect.isawaitable(configured):
                if inspect.iscoroutine(configured):
                    configured.close()
                raise AgentError(
                    "registry_configuration", "registry assembly must be synchronous", 503
                )
            if configured is not None or registry.agent(LEGACY_AGENT_ID).digest != legacy_active:
                raise AgentError(
                    "registry_configuration",
                    "assembly cannot replace the legacy Agent binding",
                    503,
                )
        settings = EngineSettings(
            workers=service.settings.workflows,
            queue_size=service.settings.queue_size,
            max_runs=service.settings.max_runs,
            tools_concurrency=service.settings.tools_concurrency,
        )
        store = SqliteRunStore(service.store.connection, settings)
        contexts = SqliteContextStore(service.store.connection)
        self.engine = RunEngine(
            registry,
            store,
            saver,
            ContextProvider(contexts, contexts, self.source_checker),
            self.resolve_identity,
            settings=settings,
            local_source_guard=self.source_guard,
            observer=self.observe,
            tool_resolver=service.registry.bridge.resolve,
            tool_authorizer=service.registry.bridge.authorize,
            result_policy=service.registry.bridge.result_policy,
            agent_authorizer=service.extension_agent_authorizer,
        )

    def remember(self, actor: Principal):
        # 这是可信的 Python/服务端鉴权入口，绝不接受客户端自述的身份。
        self.identities[actor.owner] = actor.model_copy(deep=True)

    def observe(self, kind, row, data):
        if row["agent_id"] != LEGACY_AGENT_ID or (row["run_id"], kind) in self.observed:
            return
        self.observed.add((row["run_id"], kind))
        counters = self.service.telemetry.counts
        if kind == "run.accepted":
            counters["runs_submitted"] += 1
        elif kind == "run.final":
            counters["runs_completed"] += 1
        elif kind == "run.terminal":
            status = data["status"]
            if status == "cancelled" or data["error"].startswith("user_cancel"):
                counters["runs_cancelled"] += 1
            if status == "timed_out":
                counters["timeouts"] += 1
            if status in {"failed", "needs_reconciliation", "effect_applied_result_unavailable"}:
                counters["errors"] += 1

    async def resolve_identity(self, owner):
        if self.service.settings.auth_mode == "bearer":
            if owner not in self.service.settings.bearer_principals.values():
                raise AgentError("identity", "server principal is no longer configured", 403)
            return identity(Principal(owner=owner, scopes={"read", "memory:write"}))
        return identity(self.identities.get(owner, Principal(owner=owner)))

    async def adopt_for_query(self):
        """保留旧记录，但绝不让旧 checkpoint 获得新的执行权限。"""
        bound = self.engine.registry.resolve(LEGACY_AGENT_ID, self.legacy_query_binding)
        with self.service.store.connection() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM requests WHERE agent_id=? AND binding=''", (LEGACY_AGENT_ID,)
                )
            ]
        for row in rows:
            payload = json.loads(row["payload"])
            try:
                actor = await self.resolve_identity(row["owner"])
            except AgentError:
                # 已移除的 principal 不得阻塞其他 Agent 启动。
                continue
            options = RunOptions(
                conversation_id=payload["conversation_id"],
                switch_topic=payload["switch_topic"],
                pause_for_review=payload["pause_for_review"],
                timeout_seconds=float(payload["timeout_seconds"]),
            )
            with self.service.store.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("INSERT OR IGNORE INTO legacy_unadopted VALUES(?)", (row["run_id"],))
                db.execute(
                    "UPDATE requests SET binding=? WHERE run_id=? AND binding=''",
                    (bound.digest, row["run_id"]),
                )
                db.execute(
                    "INSERT OR IGNORE INTO agent_run_controls(run_id,identity_version,conversation_id,execution_policy,original_deadline,options,client_digest,review_id,identity_digest) VALUES(?,?,?,'fixed-deadline',?,?,?,NULL,?)",
                    (
                        row["run_id"],
                        payload["permission_version"],
                        payload["conversation_id"],
                        row["deadline"],
                        options.model_dump_json(),
                        payload["client_digest"],
                        self.engine.identity_digest(actor),
                    ),
                )

    async def prepare(self, actor, value):
        request = RunInput.model_validate(value)
        try:
            tasks = request.tasks or simple_plan(request.message)
        except ValueError as exc:
            raise LabError(
                "clarification_required",
                "provide 1..8 explicit questions or a validated task plan",
                422,
            ) from exc
        captured = self.service.registry.catalog.capture(task.tool for task in tasks)
        self.service.registry.validate(tasks, principal(actor), manifest=captured)
        version = (
            self.service.index_version(actor.owner)
            if any(t.tool == "rag.answer" for t in tasks)
            else ""
        )
        payload = request.model_dump()
        payload.update(
            tasks=[task.model_dump() for task in tasks],
            permission_version=actor.permission_version,
            tool_versions={name: tool.declaration.version for name, tool in captured.tools.items()},
            index_version=version,
            state_version=self.service.settings.state_version,
            graph_version=self.service.settings.graph_version,
            client_digest=digest(request.model_dump()),
            tool_contracts={
                name: copy.deepcopy(tool.legacy) for name, tool in captured.tools.items()
            },
        )

        def commit(db, row):
            self.service.registry.catalog.commit(db, row, captured)
            db.execute(
                "INSERT INTO legacy_execution_protocol VALUES(?,?,?,?)",
                (row["run_id"], row["binding"], captured.digest, PROTOCOL),
            )

        return PreparedAdmission(payload, commit)

    def recovery_reason(self, row, actor):
        with self.service.store.connection() as db:
            if db.execute(
                "SELECT 1 FROM legacy_unadopted WHERE run_id=?", (row["run_id"],)
            ).fetchone():
                return "legacy_checkpoint_requires_matching_release"
        try:
            manifest = self.service.registry.catalog.for_run(row)
            with self.service.store.connection() as db:
                self.service.registry.bridge.stored_manifest(db, row)
            if any(tool.declaration.scope not in actor.scopes for tool in manifest.tools.values()):
                return "permission_changed"
        except (LabError, AgentError) as exc:
            return exc.code
        payload = json.loads(row["payload"])
        if (
            payload["graph_version"] != self.service.settings.graph_version
            or payload["state_version"] != self.service.settings.state_version
        ):
            return "checkpoint_version_mismatch"
        return None

    def validate_snapshot(self, row, snapshot):
        payload = json.loads(row["payload"])
        for name in (
            "conversation_id",
            "tasks",
            "params",
            "index_version",
            "permission_version",
            "state_version",
            "graph_version",
        ):
            if snapshot.values.get(name) != payload[name]:
                raise AgentError(
                    "checkpoint_control", "RAG checkpoint differs from admitted plan", 409
                )
        if "tool_contracts" in payload:
            required = payload["pause_for_review"] or any(
                not tool["read_only"] for tool in payload["tool_contracts"].values()
            )
            if snapshot.values.get("approval_required") != required:
                raise AgentError(
                    "checkpoint_control", "review requirement differs from admitted plan", 409
                )

    def validate_task(self, state, config):
        run_id = (config or {}).get("configurable", {}).get("runtime_run_id")
        if not run_id:
            raise AgentError("checkpoint_control", "task has no authoritative runtime run", 409)
        row = self.engine.store.internal_request(run_id)
        payload = json.loads(row["payload"])
        for field, expected in {
            "run_id": run_id,
            "principal": row["owner"],
            "deadline": row["deadline"],
            "topic_epoch": row["epoch"],
            "conversation_id": payload["conversation_id"],
            "tasks": payload["tasks"],
            "params": payload["params"],
            "index_version": payload["index_version"],
            "permission_version": payload["permission_version"],
        }.items():
            if state.get(field) != expected:
                raise AgentError(
                    "checkpoint_control", "pending task control differs from admitted plan", 409
                )
        task = state.get("task")
        if task not in payload["tasks"]:
            raise AgentError("checkpoint_control", "pending task differs from approved plan", 409)
        return row

    def review_descriptor(self, row, snapshot):
        interrupts = [
            (task.id, entry.id, entry.value) for task in snapshot.tasks for entry in task.interrupts
        ]
        if len(interrupts) != 1 or not isinstance(interrupts[0][2], dict):
            raise AgentError(
                "review_contract", "legacy review interrupt has no trusted anchor", 409
            )
        raw = interrupts[0][2]
        if raw.get("kind") == "action_review":
            review = self.engine.store.review(row["owner"], row["run_id"], raw.get("review_id", ""))
            preview = raw.get("preview")
            if (
                not review
                or not isinstance(preview, dict)
                or preview.get("kind") != "tool_action"
                or review["digest"] != raw.get("digest")
                or json.loads(review["preview"]) != preview
                or review["binding"] != row["binding"]
                or review["epoch"] != row["epoch"]
                or not self.approved(row)
            ):
                raise AgentError(
                    "review_contract", "native interrupt differs from durable action", 409
                )
            call = ToolCall.model_validate(preview.get("call"))
            task = self.service.registry.bridge.task(row, call)
            manifest = self.service.registry.catalog.for_run(row)
            selected = manifest.tools[task.tool]
            with self.service.store.connection() as db:
                dependencies = self.service.registry.bridge.dependencies(db, row, task)
            context = self.static_context(row).model_copy(
                update={"dependency_results": dependencies if task.use_dependencies else {}}
            )
            expected_digest = self.engine.tools.action_digest(
                context, selected.declaration, call, manifest_digest=manifest.digest
            )
            if (
                selected.declaration.read_only
                or not task.use_dependencies
                or raw["digest"] != expected_digest
                or canonical_json(preview.get("dependencies"))
                != canonical_json(context.dependency_results)
                or preview.get("manifest") != manifest.digest
            ):
                raise AgentError(
                    "review_contract", "native action differs from current original receipts", 409
                )
            return [{"digest": expected_digest, "preview": preview}]
        if raw.get("kind") != "review" or raw.get("run_id") != row["run_id"]:
            raise AgentError("review_contract", "invalid legacy plan interrupt", 409)
        payload = json.loads(row["payload"])
        expected = [{"id": task["id"], "tool": task["tool"]} for task in payload["tasks"]]
        if interrupts[0][2].get("tasks") != expected:
            raise AgentError("review_contract", "legacy review plan differs from request", 409)
        preview = {
            "owner": row["owner"],
            "run_id": row["run_id"],
            "agent_id": row["agent_id"],
            "binding": row["binding"],
            "epoch": row["epoch"],
            "plan": payload,
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
            "interrupt": {"task_id": interrupts[0][0], "id": interrupts[0][1]},
        }
        with self.service.store.connection() as db:
            protocol = db.execute(
                "SELECT manifest FROM legacy_execution_protocol WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            if protocol:
                # 即使执行凭证已失效，仍展示原始计划。
                # 创建/恢复路径由 recovery_reason 单独把关。
                preview.update(kind="legacy_plan", manifest=protocol[0])
        return [{"digest": content_digest(preview), "preview": preview}]

    @staticmethod
    def static_context(row):
        payload = json.loads(row["payload"])
        return RunContext(
            identity=Identity(owner=row["owner"], permission_version=payload["permission_version"]),
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            conversation_id=payload["conversation_id"],
            epoch=row["epoch"],
            deadline=row["deadline"],
            config_digest=row["binding"],
        )

    def bind_plan(self, row, visible):
        preview = visible["preview"]
        with self.service.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            manifest, frozen = self.service.registry.bridge.stored_manifest(db, row)
            review = db.execute(
                "SELECT * FROM agent_reviews WHERE id=? AND run_id=? AND owner=?",
                (visible["review_id"], row["run_id"], row["owner"]),
            ).fetchone()
            if (
                not review
                or review["digest"] != visible["digest"]
                or review["binding"] != row["binding"]
                or review["epoch"] != row["epoch"]
                or review["approved_at"] is None
                or review["expires_at"] <= time.time()
                or json.loads(review["preview"]) != preview
                or preview.get("kind") != "legacy_plan"
                or preview.get("manifest") != manifest
                or preview.get("plan") != json.loads(row["payload"])
                or db.execute(
                    "SELECT 1 FROM agent_review_visibility WHERE review_id=?", (review["id"],)
                ).fetchone()
                is None
            ):
                raise AgentError("review_mismatch", "original plan approval is unavailable", 409)
            values = (
                row["run_id"],
                row["owner"],
                row["binding"],
                row["epoch"],
                manifest,
                content_digest(json.loads(row["payload"])),
                review["id"],
                review["digest"],
            )
            db.execute("INSERT OR IGNORE INTO legacy_plan_grants VALUES(?,?,?,?,?,?,?,?)", values)
            saved = db.execute(
                "SELECT * FROM legacy_plan_grants WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            if tuple(saved) != values:
                raise AgentError("review_mismatch", "original plan grant changed", 409)
            for raw in json.loads(row["payload"])["tasks"]:
                task = TaskSpec.model_validate(raw)
                declaration = ToolDeclaration.model_validate(
                    frozen["tools"][task.tool]["declaration"]
                )
                if declaration.read_only or task.use_dependencies:
                    continue
                call = ToolCall(call_id=task.id, name=task.tool, arguments=task.arguments)
                action = self.engine.tools.action_digest(
                    self.static_context(row), declaration, call, manifest_digest=manifest
                )
                db.execute(
                    "INSERT OR IGNORE INTO legacy_plan_actions VALUES(?,?,?)",
                    (row["run_id"], task.id, action),
                )
                if (
                    db.execute(
                        "SELECT digest FROM legacy_plan_actions WHERE run_id=? AND call_id=?",
                        (row["run_id"], task.id),
                    ).fetchone()[0]
                    != action
                ):
                    raise AgentError("review_mismatch", "static plan action changed", 409)

    def approved(self, row):
        with self.service.store.connection() as db:
            grant = db.execute(
                "SELECT * FROM legacy_plan_grants WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            if grant is None:
                return False
            manifest, _ = self.service.registry.bridge.stored_manifest(db, row)
            review = db.execute(
                "SELECT * FROM agent_reviews WHERE id=?", (grant["review_id"],)
            ).fetchone()
            preview = json.loads(review["preview"]) if review else {}
            return bool(
                review
                and grant["owner"] == row["owner"]
                and grant["binding"] == row["binding"]
                and grant["epoch"] == row["epoch"]
                and grant["manifest"] == manifest
                and grant["plan_digest"] == content_digest(json.loads(row["payload"]))
                and review["owner"] == row["owner"]
                and review["run_id"] == row["run_id"]
                and review["binding"] == row["binding"]
                and review["epoch"] == row["epoch"]
                and review["digest"] == grant["digest"]
                and review["approved_at"] is not None
                and review["expires_at"] > time.time()
                and content_digest(preview) == grant["digest"]
                and preview.get("kind") == "legacy_plan"
                and preview.get("manifest") == manifest
                and all(
                    preview.get(field) == row[field]
                    for field in ("owner", "run_id", "agent_id", "binding", "epoch")
                )
                and preview.get("plan") == json.loads(row["payload"])
                and db.execute(
                    "SELECT 1 FROM agent_review_visibility WHERE review_id=?", (review["id"],)
                ).fetchone()
            )

    async def action_approved(self, context, tool, call, challenge):
        row = self.engine.store.internal_request(context.run_id)
        task = self.service.registry.bridge.task(row, call)
        if not self.approved(row):
            return False
        if task.use_dependencies:
            return self.engine.store.approval(row, challenge)
        if context.dependency_results:
            return False
        with self.service.store.connection() as db:
            saved = db.execute(
                "SELECT digest FROM legacy_plan_actions WHERE run_id=? AND call_id=?",
                (row["run_id"], task.id),
            ).fetchone()
            return saved is not None and saved[0] == challenge

    def resume_value(self, row, review):
        if review["preview"].get("kind") == "legacy_plan":
            self.bind_plan(row, review)
            return True
        if not self.approved(row):
            raise AgentError("review_mismatch", "legacy graph requires durable plan approval", 409)
        if review["preview"].get("kind") != "tool_action":
            raise AgentError("review_mismatch", "unknown approval contract", 409)
        return {"review_id": review["review_id"], "digest": review["digest"]}

    def results(self, row, values):
        results = copy.deepcopy(values.get("results", {}))
        operations = {
            operation["key"]: operation
            for operation in self.service.store.operations(row["owner"], row["run_id"])
        }
        tasks = {
            raw["id"]: TaskSpec.model_validate(raw) for raw in json.loads(row["payload"])["tasks"]
        }
        with self.service.store.connection() as db:
            bridged = self.requires_bridge(row)
            for task_id, result in results.items():
                operation = operations.get(result.get("operation_key"))
                if operation and operation["phase"] in ("compensating", "compensated"):
                    result["value"] = {
                        "effect": operation["phase"],
                        "operation_key": operation["key"],
                    }
                elif bridged and result.get("status") == "succeeded":
                    try:
                        result["value"] = self.service.registry.bridge.publish_value(
                            db, row, tasks[task_id], result
                        )
                    except (AgentError, LabError, TypeError, ValueError, KeyError):
                        result.update(status="failed", value={}, error="result_unavailable")
                elif result["tool"] in {"rag.answer", "memory.search"}:
                    result["value"] = self.service.sanitize_value(
                        row["owner"], result["value"], db=db
                    )
        return results

    def results_guard(self, db, row, results):
        if not self.requires_bridge(row):
            return True
        try:
            self.service.registry.bridge.stored_manifest(db, row)
        except AgentError:
            return False
        tasks = {
            raw["id"]: TaskSpec.model_validate(raw) for raw in json.loads(row["payload"])["tasks"]
        }
        try:
            for task_id, result in results.items():
                if result.get("status") == "succeeded":
                    key = result.get("operation_key")
                    operation = (
                        db.execute(
                            "SELECT * FROM operations WHERE key=? AND owner=? AND run_id=?",
                            (key, row["owner"], row["run_id"]),
                        ).fetchone()
                        if key
                        else None
                    )
                    if operation is not None and operation["phase"] in {
                        "compensated",
                        "compensating",
                    }:
                        public = db.execute(
                            "SELECT * FROM agent_tool_operations WHERE run_id=? AND call_id=? AND owner=? AND agent_id=? AND binding=?",
                            (row["run_id"], task_id, row["owner"], row["agent_id"], row["binding"]),
                        ).fetchone()
                        receipt = (
                            self.service.registry.bridge.inspect(db, dict(public))
                            if public
                            else None
                        )
                        if receipt is None or receipt.phase != (
                            "unknown" if operation["phase"] == "compensating" else "compensated"
                        ):
                            return False
                        if result.get("value") != {
                            "effect": operation["phase"],
                            "operation_key": key,
                        }:
                            return False
                    else:
                        self.service.registry.bridge.publish_value(db, row, tasks[task_id], result)
            return True
        except (AgentError, LabError, TypeError, ValueError, KeyError):
            return False

    def requires_bridge(self, row):
        # 契约由最初发布的版本决定。凭证丢失不能把新准入的 Run 降级回旧的只读查询适配器。
        return (
            self.engine.registry.resolve(
                row["agent_id"], row["binding"]
            ).workflow_registration.version
            != "2"
        )

    async def project(self, row, values, control):
        payload, results = json.loads(row["payload"]), self.results(row, values)
        return {
            "thread_id": row["thread"],
            "checkpoint_id": control["checkpoint_id"],
            "topic_epoch": row["epoch"],
            "tasks": payload["tasks"],
            "results": results,
            "unfinished": [
                task["id"]
                for task in payload["tasks"]
                if results.get(task["id"], {}).get("status") != "succeeded"
            ],
            "decisions": list(values.get("events", {}).values()),
            "result": {"results": results} if control["result"] is not None else None,
        }

    @staticmethod
    def local_sources(db, owner, documents, memories):
        for document in documents:
            record = db.execute("SELECT * FROM documents WHERE id=?", (document,)).fetchone()
            if (
                not record
                or not record["enabled"]
                or record["owner"] not in (owner, "public")
                or (record["valid_until"] is not None and record["valid_until"] <= time.time())
            ):
                return False
        for memory_id, version in memories:
            record = db.execute(
                "SELECT payload FROM memories WHERE owner=? AND id=?", (owner, memory_id)
            ).fetchone()
            if not record:
                return False
            value = json.loads(record[0])
            if (
                value["version"] != version
                or not value["enabled"]
                or (value["valid_until"] is not None and value["valid_until"] <= time.time())
            ):
                return False
        return True

    def sources(self, row, values):
        documents: set[str] = set()
        memories: set[tuple[str, int]] = set()
        for result in values.get("results", {}).values():
            value = result["value"]
            if result["tool"] == "rag.answer":
                documents.update(c["document_id"] for c in value.get("citations", []))
                documents.update(value.get("context_documents", []))
                documents.update(hit["chunk"]["document_id"] for hit in value.get("evidence", []))
            elif result["tool"] == "memory.search":
                memories.update(
                    (memory["memory_id"], memory["version"]) for memory in value.get("memories", [])
                )
        return [{"kind": "rag.document", "id": document} for document in sorted(documents)] + [
            {"kind": "rag.memory", "id": memory, "version": version}
            for memory, version in sorted(memories)
        ]

    def source_guard(self, db, row, source):
        if row["agent_id"] != LEGACY_AGENT_ID:
            policy = self.service.extension_source_guard
            return (
                policy is not None and policy(db, copy.deepcopy(row), copy.deepcopy(source)) is True
            )
        if source.get("kind") == "rag.document":
            return self.local_sources(db, row["owner"], [source.get("id")], [])
        if source.get("kind") == "rag.memory":
            return self.local_sources(
                db, row["owner"], [], [(source.get("id"), source.get("version"))]
            )
        return False

    async def source_checker(self, actor, agent_id, source):
        if agent_id != LEGACY_AGENT_ID:
            policy = self.service.extension_source_checker
            return (
                policy is not None
                and await policy(actor.model_copy(deep=True), agent_id, copy.deepcopy(source))
                is True
            )
        with self.service.store.connection() as db:
            return self.source_guard(db, {"agent_id": agent_id, "owner": actor.owner}, source)

    async def complete(self, row, values):
        results, histories = self.results(row, values), []
        if any(
            original.get("status") == "succeeded" and results[task_id].get("status") != "succeeded"
            for task_id, original in values.get("results", {}).items()
        ):
            raise AgentError(
                "result_unavailable", "final checkpoint has an unavailable original result", 409
            )
        documents: set[str] = set()
        memories: set[tuple[str, int]] = set()
        payload = json.loads(row["payload"])
        for task_id, result in results.items():
            value = result["value"]
            if result["tool"] not in {"rag.answer", "memory.search"}:
                continue
            documents.update(c["document_id"] for c in value.get("citations", []))
            documents.update(value.get("context_documents", []))
            documents.update(hit["chunk"]["document_id"] for hit in value.get("evidence", []))
            memories.update(
                (memory["memory_id"], memory["version"]) for memory in value.get("memories", [])
            )
            if (
                result["tool"] == "rag.answer"
                and result["status"] == "succeeded"
                and value.get("status") == "answered"
            ):
                histories.append(
                    (
                        f"{row['run_id']}:{task_id}",
                        {
                            "question": value["resolved_question"][:1000],
                            "answer": value["answer"][:1200],
                            "source_run": row["run_id"],
                            "documents": list(
                                dict.fromkeys(
                                    [c["document_id"] for c in value["citations"]]
                                    + value.get("context_documents", [])
                                )
                            ),
                        },
                    )
                )

        def commit(db):
            for turn_id, history in histories:
                db.execute(
                    "INSERT OR IGNORE INTO history VALUES(?,?,?,?,?,?)",
                    (
                        turn_id,
                        row["owner"],
                        payload["conversation_id"],
                        row["epoch"],
                        encode(history),
                        time.time(),
                    ),
                )
            db.execute(
                "DELETE FROM history WHERE owner=? AND conversation=? AND run_id NOT IN (SELECT run_id FROM history WHERE owner=? AND conversation=? ORDER BY created DESC LIMIT 20)",
                (
                    row["owner"],
                    payload["conversation_id"],
                    row["owner"],
                    payload["conversation_id"],
                ),
            )

        return FinalProjection(
            value={"results": results},
            status=values["status"],
            sources=self.sources(row, {"results": results}),
            local_guard=lambda db: (
                self.local_sources(db, row["owner"], documents, memories)
                and self.results_guard(db, row, results)
            ),
            extra_commit=commit,
        )

    @contextmanager
    def execution_scope(self, row):
        with (
            tracing_context(enabled=False),
            provider_scope(row["run_id"], self.service.settings.max_model_calls, row["deadline"]),
        ):
            yield

    def factory(self, engine, saver, bound):
        return GraphBinding(
            graph=self.service.agent.graph,
            initial=self.service.initial,
            project=self.project,
            projection_guard=lambda db, row, projection: self.results_guard(
                db, row, projection.get("results", {})
            ),
            on_complete=self.complete,
            prepare=self.prepare,
            recovery_reason=self.recovery_reason,
            visible_sources=self.sources,
            execution_policy="fixed-deadline",
            execution_scope=self.execution_scope,
            recursion_limit=64,
            effect_guard=self.service.registry.bridge.effect_guard,
            action_approval=self.action_approved,
            receipt_inspector=self.service.registry.bridge.inspect,
            read_attempt_recovery=self.service.registry.bridge.recover_read,
            control_fields={
                "run_id": "run_id",
                "principal": "owner",
                "topic_epoch": "epoch",
                "deadline": "deadline",
            },
            validate_snapshot=self.validate_snapshot,
            client_digest=lambda value, options: digest(value),
            review_descriptor=self.review_descriptor,
            resume_value=self.resume_value,
            applied_effects=lambda row: [
                {"operation_key": operation["key"], "phase": "applied"}
                for operation in self.service.store.operations(row["owner"], row["run_id"])
                if operation["phase"] == "done"
            ],
            exception_code=lambda exc: exc.code if isinstance(exc, LabError) else None,
            terminal_values=lambda status, code: {"status": status, "events": {"terminal": code}},
        )
