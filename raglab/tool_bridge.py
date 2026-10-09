"""公共 tool 执行器背后的原始 RAG 发布版本与业务收据（receipt）。

SQL 证明已准入的动作；可信注册表提供其实现。
本适配器绝不从请求中导入 handler，也绝不挑选某个"当前生效"的兜底实现。
"""

import copy
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass

from agentbase.contracts import AgentError
from agentbase.effects import ActionReceipt, AttemptCommit, AttemptEvidence, NeedsReconciliation
from agentbase.schema import SchemaContract, canonical_json, content_digest
from agentbase.tools import ResolvedTool, ToolResult, TransientToolError

from .effect_scope import EffectOwner, effect_owner
from .errors import LabError, UnknownEffect
from .models import TaskSpec
from .persistence import LEGACY_AGENT_ID
from .resilience import transient
from .tool_catalog import LegacyToolCatalog

PROTOCOL = "public-tool-v1"


@dataclass(frozen=True)
class LegacyInvocation:
    state: dict
    dependencies: dict
    config: dict


invocation: ContextVar[LegacyInvocation | None] = ContextVar("legacy_invocation", default=None)


class LegacyToolBridge:
    def __init__(self, registry):
        self.registry, self.service = registry, registry.service
        self._resolved: dict[tuple[str, str], tuple[object, ResolvedTool]] = {}

    @staticmethod
    def stored_manifest(db, row):
        stored = db.execute(
            "SELECT * FROM legacy_tool_manifests WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if stored is None:
            raise AgentError(
                "legacy_tool_execution_requires_matching_release", "missing original manifest", 409
            )
        try:
            payload = json.loads(stored["payload"])
            digest = content_digest(payload)
        except (TypeError, ValueError, AgentError) as exc:
            raise AgentError("tool_manifest_conflict", "original manifest is invalid", 409) from exc
        if stored["binding"] != row["binding"] or digest != stored["digest"]:
            raise AgentError("tool_manifest_conflict", "original manifest differs", 409)
        try:
            LegacyToolCatalog.validate_payload(row, payload)
        except LabError as exc:
            raise AgentError(exc.code, exc.message, exc.status) from exc
        except (TypeError, ValueError, KeyError) as exc:
            raise AgentError(
                "tool_manifest_conflict", "original declaration is invalid", 409
            ) from exc
        protocol = db.execute(
            "SELECT * FROM legacy_execution_protocol WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if protocol is None:
            raise AgentError(
                "legacy_tool_protocol_requires_matching_release",
                "old execution protocol is query only",
                409,
            )
        if (
            protocol["binding"] != row["binding"]
            or protocol["manifest"] != stored["digest"]
            or protocol["protocol"] != PROTOCOL
        ):
            raise AgentError("tool_manifest_conflict", "execution protocol differs", 409)
        return stored["digest"], payload

    @staticmethod
    def task(row, call):
        matches = [t for t in json.loads(row["payload"])["tasks"] if t["id"] == call.call_id]
        if (
            len(matches) != 1
            or matches[0]["tool"] != call.name
            or canonical_json(matches[0]["arguments"]) != canonical_json(call.arguments)
        ):
            raise AgentError("checkpoint_control", "tool call differs from original task", 409)
        return TaskSpec.model_validate(matches[0])

    def verified_link(self, db, operation):
        record = db.execute(
            "SELECT * FROM requests WHERE run_id=?", (operation["run_id"],)
        ).fetchone()
        if record is None:
            raise AgentError("receipt_conflict", "original request is missing", 409)
        row = dict(record)
        manifest, payload = self.stored_manifest(db, row)
        linked = db.execute(
            "SELECT * FROM legacy_action_links WHERE public_key=? AND attempt_token=?",
            (operation["key"], operation["attempt_token"]),
        ).fetchone()
        if linked is None:
            raise AgentError("receipt_conflict", "attempt link is missing", 409)
        link = dict(linked)
        try:
            evidence = AttemptEvidence.model_validate_json(link["evidence"])
        except (TypeError, ValueError) as exc:
            raise AgentError(
                "receipt_conflict", "original attempt evidence is invalid", 409
            ) from exc
        self.service.runtime.actions._check_evidence(operation, evidence)
        context, call = evidence.context, evidence.call
        task = self.task(row, call)
        expected = {
            "public_key": operation["key"],
            "attempt_token": operation["attempt_token"],
            "owner": operation["owner"],
            "agent_id": operation["agent_id"],
            "run_id": operation["run_id"],
            "call_id": operation["call_id"],
            "binding": operation["binding"],
            "manifest": manifest,
            "fingerprint": operation["fingerprint"],
            "attempt": operation["attempts"],
            "domain_key": f"{row['run_id']}:{call.call_id}",
        }
        admitted = json.loads(row["payload"])
        control = db.execute(
            "SELECT * FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if (
            any(link[name] != value for name, value in expected.items())
            or row["agent_id"] != LEGACY_AGENT_ID
            or any(
                row[name] != operation[name] for name in ("owner", "agent_id", "run_id", "binding")
            )
            or context.operation_key != operation["key"]
            or context.attempt_token != operation["attempt_token"]
            or context.conversation_id != admitted["conversation_id"]
            or context.epoch != row["epoch"]
            or context.deadline != row["deadline"]
            or context.capabilities
            or control is None
            or control["identity_digest"] != self.service.runtime.identity_digest(context.identity)
            or evidence.manifest_digest != manifest
            or evidence.tool.model_dump(mode="json") != payload["tools"][call.name]["declaration"]
            or (not task.use_dependencies and context.dependency_results)
            or (task.use_dependencies and set(context.dependency_results) != set(task.depends_on))
        ):
            raise AgentError(
                "receipt_conflict", "attempt link differs from original authority", 409
            )
        return row, link, evidence

    def resolve(self, context, call):
        if context.agent_id != LEGACY_AGENT_ID:
            return None
        row = self.service.runtime.store.internal_request(context.run_id)
        task = self.task(row, call)
        with self.service.store.connection() as db:
            digest, _ = self.stored_manifest(db, row)
        selected = self.registry.catalog.validate_task(row, task, context.identity)
        key = (selected.spec.name, selected.spec.version)
        previous = self._resolved.get(key)
        if previous is None:

            async def handler(arguments, current):
                scope = invocation.get()
                if scope is None or scope.state["run_id"] != current.run_id:
                    raise AgentError("checkpoint_control", "missing trusted invocation", 409)
                with self.service.store.connection() as db:
                    op = db.execute(
                        "SELECT * FROM agent_tool_operations WHERE key=?", (current.operation_key,)
                    ).fetchone()
                    if op is None:
                        raise AgentError("receipt_conflict", "claimed action is missing", 409)
                    current_row, link, proof = self.verified_link(db, dict(op))
                    current_task = self.task(current_row, proof.call)
                state = scope.state
                tool_context = {
                    k: copy.deepcopy(state[k])
                    for k in (
                        "principal",
                        "run_id",
                        "deadline",
                        "history",
                        "index_version",
                        "params",
                    )
                }
                tool_context.update(
                    dependencies=copy.deepcopy(scope.dependencies)
                    if current_task.use_dependencies
                    else {},
                    use_dependencies=current_task.use_dependencies,
                    operation_key=link["domain_key"],
                    public_operation_key=current.operation_key,
                    attempt_token=current.attempt_token,
                )
                config = {
                    **scope.config,
                    "configurable": {
                        **scope.config.get("configurable", {}),
                        "tool_context": tool_context,
                    },
                }
                owner = effect_owner.set(
                    EffectOwner(
                        link["domain_key"], op["key"], op["attempt_token"], op["fingerprint"]
                    )
                )
                try:
                    return await selected.spec.tool.ainvoke(arguments, config=config)
                except UnknownEffect as exc:
                    raise NeedsReconciliation() from exc
                except Exception as exc:
                    if transient(exc):
                        raise TransientToolError() from exc
                    if isinstance(exc, LabError):
                        raise AgentError(exc.code, exc.message, exc.status) from exc
                    raise
                finally:
                    effect_owner.reset(owner)

            hook = AttemptCommit(
                frozenset({"legacy_action_links", "legacy_write_actions", "operations", "audit"}),
                self.claim,
            )
            self._resolved[key] = (
                selected,
                ResolvedTool(selected.declaration, handler, digest, hook),
            )
            previous = self._resolved[key]
        if previous[0] is not selected:
            raise AgentError("tool_binding_changed", "original frozen implementation changed", 409)
        frozen = previous[1]
        # tool 的 wrapper/hook 是稳定的；每个已准入的计划各自拥有独立的 manifest。
        return ResolvedTool(frozen.declaration, frozen.handler, digest, frozen.attempt_commit)

    def claim(self, db, operation, evidence):
        record = db.execute(
            "SELECT * FROM requests WHERE run_id=?", (operation["run_id"],)
        ).fetchone()
        if record is None:
            raise AgentError("attempt_commit", "original request is missing", 409)
        row = dict(record)
        digest, payload = self.stored_manifest(db, row)
        task = self.task(row, evidence.call)
        if (
            evidence.manifest_digest != digest
            or evidence.tool.model_dump(mode="json") != payload["tools"][task.tool]["declaration"]
        ):
            raise AgentError("attempt_commit", "claim declaration differs", 409)
        admitted = json.loads(row["payload"])
        epoch = db.execute(
            "SELECT epoch FROM agent_conversations WHERE owner=? AND agent_id=? AND conversation_id=?",
            (row["owner"], row["agent_id"], admitted["conversation_id"]),
        ).fetchone()
        if (
            row["cancel"]
            or (epoch[0] if epoch else 0) != row["epoch"]
            or time.time() >= row["deadline"]
        ):
            raise AgentError("attempt_commit", "claim control is no longer valid", 409)
        self.dependencies(db, row, task, expected=evidence.context.dependency_results)
        domain_key = f"{row['run_id']}:{task.id}"
        db.execute(
            "INSERT INTO legacy_action_links VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                operation["key"],
                operation["attempt_token"],
                row["owner"],
                row["agent_id"],
                row["run_id"],
                task.id,
                row["binding"],
                digest,
                operation["fingerprint"],
                operation["attempts"],
                domain_key,
                evidence.model_dump_json(),
            ),
        )
        self.verified_link(db, operation)
        if not evidence.tool.read_only:
            db.execute(
                "INSERT INTO legacy_write_actions VALUES(?,?)", (domain_key, operation["key"])
            )
            if db.execute("SELECT 1 FROM operations WHERE key=?", (domain_key,)).fetchone():
                raise AgentError(
                    "operation_conflict", "unlinked business operation already exists", 409
                )
            db.execute(
                "INSERT INTO operations VALUES(?,?,?,'intent',NULL)",
                (domain_key, row["owner"], row["run_id"]),
            )
            self.service.store.audit(db, row["owner"], "operation.intent", domain_key)

    def dependencies(self, db, row, task, *, expected=None, trail=()):
        if task.id in trail or len(trail) >= 8:
            raise AgentError("dependency_conflict", "invalid dependency receipt graph", 409)
        payload = json.loads(row["payload"])
        by_id = {t["id"]: TaskSpec.model_validate(t) for t in payload["tasks"]}
        values = {}
        for dep in task.depends_on:
            record = db.execute(
                "SELECT * FROM agent_tool_operations WHERE run_id=? AND call_id=? AND owner=? AND agent_id=? AND binding=?",
                (row["run_id"], dep, row["owner"], row["agent_id"], row["binding"]),
            ).fetchone()
            if (
                record is None
                or record["phase"] not in {"read_success", "applied"}
                or record["result"] is None
                or record["error"] == "effect_applied_result_unavailable"
            ):
                raise AgentError(
                    "dependency_conflict", "dependency has no successful original receipt", 409
                )
            _, _, proof = self.verified_link(db, dict(record))
            if not proof.tool.read_only:
                receipt = self.inspect(db, dict(record))
                if (
                    receipt is None
                    or receipt.phase != "applied"
                    or not receipt.result_available
                    or canonical_json(receipt.value) != canonical_json(json.loads(record["result"]))
                ):
                    raise AgentError(
                        "dependency_conflict",
                        "dependency write receipt is no longer publishable",
                        409,
                    )
            self.dependencies(
                db,
                row,
                by_id[dep],
                expected=proof.context.dependency_results,
                trail=(*trail, task.id),
            )
            value = json.loads(record["result"])
            SchemaContract(proof.tool.output_schema).validate(
                value, max_bytes=proof.tool.result_bytes
            )
            if by_id[dep].tool in {"rag.answer", "memory.search"}:
                value = self.service.sanitize_value(row["owner"], value, db=db)
            values[dep] = value
        consumed = values if task.use_dependencies else {}
        if expected is not None and canonical_json(expected) != canonical_json(consumed):
            raise AgentError(
                "dependency_conflict", "consumed dependency differs from current receipt", 409
            )
        return values

    def publish_value(self, db, row, task, dto):
        """checkpoint 中的 DTO 永远不能替代当前 SQL 里的原始收据。"""
        operation = db.execute(
            "SELECT * FROM agent_tool_operations WHERE run_id=? AND call_id=? AND owner=? AND agent_id=? AND binding=?",
            (row["run_id"], task.id, row["owner"], row["agent_id"], row["binding"]),
        ).fetchone()
        if (
            operation is None
            or operation["phase"] not in {"read_success", "applied"}
            or operation["result"] is None
            or operation["error"] == "effect_applied_result_unavailable"
        ):
            raise AgentError("result_unavailable", "original result is no longer publishable", 409)
        _, _, proof = self.verified_link(db, dict(operation))
        self.dependencies(db, row, task, expected=proof.context.dependency_results)
        value = json.loads(operation["result"])
        SchemaContract(proof.tool.output_schema).validate(value, max_bytes=proof.tool.result_bytes)
        if not proof.tool.read_only:
            receipt = self.inspect(db, dict(operation))
            if (
                receipt is None
                or receipt.phase != "applied"
                or not receipt.result_available
                or canonical_json(receipt.value) != canonical_json(value)
            ):
                raise AgentError(
                    "result_unavailable", "write receipt is no longer publishable", 409
                )
        if task.tool in {"rag.answer", "memory.search"}:
            value = self.service.sanitize_value(row["owner"], value, db=db)
        if (
            dto.get("task_id") != task.id
            or dto.get("tool") != task.tool
            or canonical_json(dto.get("value")) != canonical_json(value)
        ):
            raise AgentError(
                "result_conflict", "checkpoint result differs from original receipt", 409
            )
        return value

    async def action_context(self, row, task, dependencies):
        context = await self.service.runtime.context(row["run_id"])
        with self.service.store.connection() as db:
            values = self.dependencies(db, row, task)
        if set(dependencies) != set(task.depends_on):
            raise AgentError("dependency_conflict", "dependency DTO set differs", 409)
        payload = json.loads(row["payload"])
        names = {t["id"]: t["tool"] for t in payload["tasks"]}
        for dep, value in values.items():
            dto = dependencies[dep]
            if (
                dto.get("task_id") != dep
                or dto.get("tool") != names[dep]
                or dto.get("status") != "succeeded"
                or canonical_json(dto.get("value")) != canonical_json(value)
            ):
                raise AgentError(
                    "dependency_conflict", "checkpoint dependency is stale or forged", 409
                )
        return context.model_copy(
            update={"dependency_results": values if task.use_dependencies else {}}, deep=True
        )

    async def authorize(self, context, tool, call):
        if context.agent_id != LEGACY_AGENT_ID:
            policy = self.service.extension_tool_authorizer
            return (
                policy is None
                or await policy(
                    context.model_copy(deep=True),
                    tool.model_copy(deep=True),
                    call.model_copy(deep=True),
                )
                is True
            )
        row = self.service.runtime.store.internal_request(context.run_id)
        task = self.task(row, call)
        with self.service.store.connection() as db:
            self.dependencies(db, row, task, expected=context.dependency_results)
        return True

    async def result_policy(self, context, tool, value):
        if context.agent_id != LEGACY_AGENT_ID:
            policy = self.service.extension_result_policy
            return (
                value
                if policy is None
                else await policy(
                    context.model_copy(deep=True), tool.model_copy(deep=True), copy.deepcopy(value)
                )
            )
        scope = invocation.get()
        if scope is None or scope.state["run_id"] != context.run_id:
            raise AgentError("legacy_invocation", "result has no original graph invocation", 409)
        task = TaskSpec.model_validate(scope.state["task"])
        with self.service.store.connection() as db:
            operation = db.execute(
                "SELECT * FROM agent_tool_operations WHERE run_id=? AND call_id=? AND owner=? AND agent_id=? AND binding=?",
                (
                    context.run_id,
                    task.id,
                    context.identity.owner,
                    context.agent_id,
                    context.config_digest,
                ),
            ).fetchone()
            if operation is None:
                raise AgentError("receipt_conflict", "result has no claimed original action", 409)
            row, _, proof = self.verified_link(db, dict(operation))
            if proof.call.name != tool.name:
                raise AgentError(
                    "receipt_conflict", "result tool differs from original action", 409
                )
            self.dependencies(db, row, task, expected=context.dependency_results)
            if tool.name not in {"rag.answer", "memory.search"}:
                return value
            result = self.service.sanitize_value(context.identity.owner, value, db=db)
        row = self.service.runtime.store.internal_request(context.run_id)
        sources = self.service.runtime_adapter.sources(
            row, {"results": {"value": {"tool": tool.name, "value": result}}}
        )
        return ToolResult(value=result, sources=sources)

    def inspect(self, db, operation):
        if operation["attempts"] == 0:
            return None
        row, link, evidence = self.verified_link(db, operation)
        mapping = db.execute(
            "SELECT public_key FROM legacy_write_actions WHERE domain_key=?", (link["domain_key"],)
        ).fetchone()
        record = db.execute(
            "SELECT * FROM operations WHERE key=?", (link["domain_key"],)
        ).fetchone()
        if (
            mapping is None
            or mapping[0] != operation["key"]
            or record is None
            or record["owner"] != row["owner"]
            or record["run_id"] != row["run_id"]
            or evidence.tool.read_only
        ):
            raise AgentError(
                "receipt_conflict", "business receipt is not linked to this action", 409
            )
        if record["phase"] == "done":
            try:
                value = json.loads(record["payload"])
                SchemaContract(evidence.tool.output_schema).validate(
                    value, max_bytes=evidence.tool.result_bytes
                )
            except (TypeError, ValueError, AgentError):
                return ActionReceipt(phase="applied", result_available=False)
            return ActionReceipt(phase="applied", value=value)
        if record["phase"] == "intent":
            if operation["effect_applied"]:
                return ActionReceipt(phase="unknown")
            return None
        if record["phase"] in {"unknown", "compensating"}:
            return ActionReceipt(phase="unknown")
        if record["phase"] in {"not_applied", "compensated"}:
            return ActionReceipt(phase=record["phase"])
        raise AgentError("receipt_conflict", "unknown business receipt phase", 409)

    def recover_read(self, db, operation):
        if (
            not db.execute(
                "SELECT 1 FROM legacy_execution_protocol WHERE run_id=?", (operation["run_id"],)
            ).fetchone()
            or not db.execute(
                "SELECT 1 FROM legacy_action_links WHERE public_key=? AND attempt_token=?",
                (operation["key"], operation["attempt_token"]),
            ).fetchone()
            or not db.execute(
                "SELECT 1 FROM legacy_tool_manifests WHERE run_id=?", (operation["run_id"],)
            ).fetchone()
        ):
            return False
        _, _, evidence = self.verified_link(db, operation)
        return evidence.tool.read_only

    def settle_unstarted(self, row):
        with self.service.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for record in db.execute(
                "SELECT * FROM agent_tool_operations WHERE run_id=? AND read_only=0 AND phase='not_applied' AND effect_applied=0",
                (row["run_id"],),
            ).fetchall():
                _, link, _ = self.verified_link(db, dict(record))
                db.execute(
                    "UPDATE operations SET phase='not_applied' WHERE key=? AND owner=? AND run_id=? AND phase='intent'",
                    (link["domain_key"], row["owner"], row["run_id"]),
                )

    def effect_guard(self, row, owned):
        with self.service.store.connection() as db:
            for record in db.execute(
                "SELECT * FROM operations WHERE run_id=? AND phase IN ('intent','unknown','compensating')",
                (row["run_id"],),
            ).fetchall():
                if record["owner"] != row["owner"] or record["phase"] != "intent" or owned is None:
                    return True
                public = db.execute(
                    "SELECT * FROM agent_tool_operations WHERE key=? AND attempt_token=? AND phase='intent'",
                    owned,
                ).fetchone()
                if public is None:
                    return True
                _, link, _ = self.verified_link(db, dict(public))
                if link["domain_key"] != record["key"]:
                    return True
        return False
