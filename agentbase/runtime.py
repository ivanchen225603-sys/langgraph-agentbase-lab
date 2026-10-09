"""所有已注册 LangGraph workflow 的唯一生命周期所有者。

应用自身负责数据库迁移、进程独占锁、saver 生命周期与鉴权。Graph 工厂只提供
领域策略，绝不另起 worker。
"""

import asyncio
import copy
import json
import time
from contextlib import nullcontext
from typing import Any

from langgraph.graph import END, START
from langgraph.types import Command, interrupt
from pydantic import JsonValue, TypeAdapter

from . import task_tree
from .artifact_tools import ArtifactReader, artifact_read_registration
from .artifacts import ArtifactReference, ArtifactScope, SqliteArtifactStore
from .budgets import SqliteBudgetLedger
from .configuration import BudgetPolicy
from .context import ContextProvider, SourceExpansion, SourceFrame
from .contracts import (
    AgentError,
    Identifier,
    Identity,
    Message,
    ModelProvider,
    RunContext,
    RunEvent,
)
from .effects import AppliedResultUnavailable, NeedsReconciliation, SqliteActionLedger
from .model_calls import ModelCaller
from .registry import BoundAgent, ComponentRegistry
from .run_store import SqliteRunStore
from .runtime_contracts import (
    AgentAuthorizer,
    EngineSettings,
    FinalProjection,
    GraphBinding,
    IdentityResolver,
    PreparedAdmission,
    RunOptions,
)
from .schema import SchemaContract, canonical_json, content_digest
from .task_execution import TaskCoordinator, TaskTicket
from .task_tree import ChildReceiptReference, ChildTask, TaskInputReference
from .tools import ResultPolicy, ToolAuthorizer, ToolExecutor, ToolResolver, ToolResult


class RunEngine:
    def __init__(
        self,
        registry: ComponentRegistry,
        store: SqliteRunStore,
        saver: Any,
        contexts: ContextProvider,
        identity_resolver: IdentityResolver,
        *,
        settings: EngineSettings | None = None,
        tool_authorizer: ToolAuthorizer | None = None,
        agent_authorizer: AgentAuthorizer | None = None,
        tool_resolver: ToolResolver | None = None,
        result_policy: ResultPolicy | None = None,
        local_source_guard=None,
        observer=None,
    ):
        self.registry, self.store, self.saver, self.contexts = registry, store, saver, contexts
        if contexts.artifacts is None:
            contexts.artifacts = SqliteArtifactStore(store.connection)
        if (
            not isinstance(contexts.artifacts, SqliteArtifactStore)
            or contexts.artifacts.connection != store.connection
        ):
            raise AgentError(
                "artifact_binding",
                "runtime artifacts require the RunStore transaction connection",
                503,
            )
        self.artifacts = contexts.artifacts
        contexts.source_expander = self._expand_child_source
        self.agent_authorizers: list[AgentAuthorizer] = []
        self.add_agent_authorizer(agent_authorizer)
        contexts._runtime_frame_authorizer = self._authorize_source_frame
        self.identity_resolver = identity_resolver
        self.settings = settings or store.settings
        self.local_source_guard = local_source_guard
        self.tool_authorizer = tool_authorizer
        self.observer, self.observer_errors = observer, 0
        self.queue: asyncio.Queue = asyncio.Queue(self.settings.queue_size)
        self.workers: list[asyncio.Task] = []
        self._ever_started = False
        self.active: dict[str, asyncio.Task] = {}
        self.pending: set[str] = set()
        self.admission_lock = asyncio.Lock()
        self.available, self.closing = True, False
        self._bindings: dict[tuple[str, str], GraphBinding] = {}
        self._providers: dict[str, ModelProvider] = {}
        self.model_caller = ModelCaller(
            SqliteBudgetLedger(store.connection),
            control_guard=lambda context: self.model_request_current(context, []),
            authorization_guard=self._authorize_model_request,
            request_control_guard=lambda context, request: self.model_request_current(
                context, [source for message in request.messages for source in message.sources]
            ),
        )
        self.actions = SqliteActionLedger(store.connection)
        self.task_coordinator = TaskCoordinator(self)
        self.artifact_reader = ArtifactReader(self)
        self._application_result_policy = result_policy
        self.tools = ToolExecutor(
            registry,
            self.actions,
            concurrency=self.settings.tools_concurrency,
            approval_checker=self._approved_action,
            authorizer=self._authorize_tool,
            result_policy=self._project_tool_result,
            resolver=tool_resolver,
            receipt_inspector=self._inspect_tool_receipt,
            handler_binder=lambda handler: (
                self.artifact_reader if handler is artifact_read_registration else handler
            ),
            local_source_guard=self._tool_sources_current_in,
        )

    async def _project_tool_result(self, context, declaration, value):
        bound = self.registry.resolve(context.agent_id, context.config_digest)
        entry = dict(bound.handler_registrations).get(declaration.name)
        sources = (
            await self.artifact_reader.sources(context, value)
            if entry is not None and entry.implementation is artifact_read_registration
            else []
        )
        projected = (
            await self._application_result_policy(context, declaration, value)
            if self._application_result_policy is not None
            else value
        )
        result = projected if isinstance(projected, ToolResult) else ToolResult(value=projected)
        unique = {canonical_json(source): source for source in [*sources, *result.sources]}
        return ToolResult(value=result.value, sources=list(unique.values()))

    def binding(self, row: dict) -> GraphBinding:
        bound = self.registry.resolve(row["agent_id"], row["binding"])
        key = (row["agent_id"], row["binding"])
        if key not in self._bindings:
            result = self.registry.workflow_factory(bound.definition, bound=bound)(
                self, self.saver, bound
            )
            if not isinstance(result, GraphBinding):
                raise AgentError("workflow_contract", "workflow must return a GraphBinding", 503)
            self._bindings[key] = result
        return self._bindings[key]

    def provider(self, bound: BoundAgent) -> ModelProvider:
        if bound.model is None:
            raise AgentError("model_binding", "Agent has no model binding")
        key = content_digest(
            {"binding": bound.digest, "model": bound.model.model_dump(mode="json")}
        )
        if key not in self._providers:
            self._providers[key] = self.registry.provider_factory(bound.model, bound=bound)(
                bound.model.model_dump(mode="json")
            )
        return self._providers[key]

    async def identity(self, owner: str) -> Identity:
        identity = await self.identity_resolver(owner)
        if not isinstance(identity, Identity) or identity.owner != owner:
            raise AgentError("identity", "current server identity is unavailable", 403)
        return identity

    @staticmethod
    def identity_digest(identity: Identity) -> str:
        return content_digest(
            {
                "owner": identity.owner,
                "scopes": sorted(identity.scopes),
                "version": identity.permission_version,
            }
        )

    def _control(self, row: dict) -> dict:
        control = self.store.control(row["run_id"])
        if control is None:
            raise AgentError("legacy_adapter_required", "run requires its original adapter", 409)
        return control

    def _inspect_tool_receipt(self, db, operation):
        record = db.execute(
            "SELECT * FROM requests WHERE run_id=?", (operation["run_id"],)
        ).fetchone()
        if record is None or any(
            record[field] != operation[field] for field in ("owner", "agent_id", "binding")
        ):
            raise AgentError(
                "effect_receipt_binding", "receipt action does not match its original run", 409
            )
        inspector = self.binding(dict(record)).receipt_inspector
        return inspector(db, operation) if inspector is not None else None

    def _reconcile_actions(self, row, binding=None):
        binding = binding or self.binding(row)
        if binding.receipt_inspector is not None:
            self.actions.reconcile(
                row["owner"], row["agent_id"], row["run_id"], self._inspect_tool_receipt
            )

    def _reason(
        self, row: dict, identity: Identity, *, allow_review=False, final=False, owned_intent=None
    ) -> str | None:
        control = self._control(row)
        try:
            binding = self.binding(row)
        except AgentError as exc:
            return exc.code
        # 副作用事实在过期、取消或权限丢失之后仍然是事实。
        # 但要发布它们的取值，仍需满足全部原有控制策略。
        self._reconcile_actions(row, binding)
        if (
            identity.owner != row["owner"]
            or identity.permission_version != control["identity_version"]
            or control["identity_digest"] != self.identity_digest(identity)
        ):
            return "permission_changed"
        if row["cancel"]:
            return "cancelled_or_old_topic"
        with self.store.connection() as db:
            db.execute("BEGIN")
            if (
                self.store.epoch(db, row["owner"], row["agent_id"], control["conversation_id"])
                != row["epoch"]
            ):
                return "cancelled_or_old_topic"
            node = task_tree.metadata_in(db, row["run_id"])
            if node is not None:
                try:
                    task_tree.assert_ancestors_in(
                        db,
                        RunContext(
                            identity=identity,
                            agent_id=row["agent_id"],
                            run_id=row["run_id"],
                            conversation_id=control["conversation_id"],
                            epoch=row["epoch"],
                            deadline=row["deadline"],
                            config_digest=row["binding"],
                            scope_ceiling=node.scope_ceiling,
                        ),
                        execution=not final,
                    )
                except AgentError as exc:
                    return (
                        "original_deadline_expired"
                        if row["deadline"] <= time.time() and not final
                        else exc.code
                    )
        domain_uncertain = (
            binding.effect_guard(copy.deepcopy(row), owned_intent)
            if binding.effect_guard is not None
            else binding.uncertain is not None and binding.uncertain(row["run_id"])
        )
        if self.actions.uncertain(row["run_id"], owned_intent=owned_intent) or domain_uncertain:
            return "unknown_effect_requires_reconciliation"
        if any(
            self._own_uncertain(child)
            for child in self._descendant_rows(row)
            if child["run_id"] != row["run_id"]
        ):
            return "unknown_effect_requires_reconciliation"
        if binding.recovery_reason is not None:
            reason = binding.recovery_reason(row, identity)
            if reason:
                return reason
        if row["deadline"] <= time.time() and not final:
            if not (allow_review and control["execution_policy"] == "pause-review"):
                return "original_deadline_expired"
        return None

    async def context(self, run_id: str, *, _owned_intent=None) -> RunContext:
        row = self.store.internal_request(run_id)
        identity = await self.identity(row["owner"])
        row = self.store.internal_request(run_id)
        reason = self._reason(row, identity, owned_intent=_owned_intent)
        if reason:
            raise AgentError(reason, "current run control disallows execution", 409)
        control = self._control(row)
        with self.store.connection() as db:
            db.execute("BEGIN")
            node = task_tree.metadata_in(db, run_id)
            ancestors = task_tree.ancestry_in(db, run_id)
        if self.agent_authorizers:
            for ancestor in ancestors:
                if not await self.agent_allowed(
                    identity.model_copy(
                        update={"scopes": identity.scopes & ancestor.scope_ceiling}
                    ),
                    ancestor.agent_id,
                ):
                    raise AgentError("agent_forbidden", "task Agent access is not authorized", 403)
            if not ancestors and not await self.agent_allowed(identity, row["agent_id"]):
                raise AgentError("agent_forbidden", "Agent access is not authorized", 403)
            if await self.identity(row["owner"]) != identity:
                raise AgentError(
                    "permission_changed", "identity changed during Agent authorization", 403
                )
            reason = self._reason(
                self.store.internal_request(run_id), identity, owned_intent=_owned_intent
            )
            if reason:
                raise AgentError(reason, "run control changed during Agent authorization", 409)
        return RunContext(
            identity=identity,
            agent_id=row["agent_id"],
            run_id=run_id,
            conversation_id=control["conversation_id"],
            epoch=row["epoch"],
            deadline=row["deadline"],
            config_digest=row["binding"],
            scope_ceiling=node.scope_ceiling if node is not None else None,
        )

    def budget_policy(self, context: RunContext) -> BudgetPolicy:
        with self.store.connection() as db:
            db.execute("BEGIN")
            node = task_tree.metadata_in(db, context.run_id)
        return (
            node.budget
            if node is not None
            else self.registry.resolve(context.agent_id, context.config_digest).definition.budget
        )

    def add_agent_authorizer(self, policy: AgentAuthorizer | None) -> None:
        """Intersect trusted host policies; mounting routes cannot replace runtime policy."""
        if policy is not None and all(
            existing is not policy for existing in self.agent_authorizers
        ):
            if not callable(policy) or len(self.agent_authorizers) >= 16:
                raise AgentError(
                    "agent_policy", "Agent authorization policy is invalid or full", 503
                )
            self.agent_authorizers.append(policy)

    async def agent_allowed(self, identity: Identity, agent_id: str) -> bool:
        for policy in tuple(self.agent_authorizers):
            if await policy(identity.model_copy(deep=True), agent_id) is not True:
                return False
        return True

    async def _authorize_source_frame(self, frame: SourceFrame) -> bool:
        return await self.agent_allowed(frame.identity, frame.agent_id)

    @staticmethod
    def config(row: dict):
        return {"configurable": {"thread_id": row["thread"], "runtime_run_id": row["run_id"]}}

    async def start(self):
        if self.workers:
            return
        self.closing = False
        self._ever_started = True
        self.workers = [asyncio.create_task(self.worker()) for _ in range(self.settings.workers)]
        await self.task_coordinator.start()

    def recover_startup_actions(self) -> int:
        """宿主持有其 OS 锁，并已证明上一个执行器已停止。"""
        if (
            self._ever_started
            or self.workers
            or self.active
            or self.pending
            or not self.queue.empty()
            or self.admission_lock.locked()
        ):
            raise AgentError(
                "read_recovery_state", "recovery requires an unused stopped engine", 409
            )

        def proof(db, operation):
            saved = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (operation["run_id"],)
            ).fetchone()
            if saved is None or any(
                saved[field] != operation[field] for field in ("owner", "agent_id", "binding")
            ):
                raise AgentError(
                    "read_recovery_conflict", "attempt differs from original request", 409
                )
            try:
                binding = self.binding(dict(saved))
            except AgentError as exc:
                if exc.code == "binding_changed":
                    return False
                raise
            return (
                binding.read_attempt_recovery(db, operation)
                if binding.read_attempt_recovery
                else False
            )

        return self.actions.recover_read_attempts(proof)

    async def close(self):
        self.closing = True
        await self.task_coordinator.close()
        # 先让执行完取消清理，再停止其等待中的 worker。
        remaining = list(self.active.values())
        for task in remaining:
            self.cancel_once(task)
        await asyncio.gather(*remaining, return_exceptions=True)
        for task in self.workers:
            self.cancel_once(task)
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        self.active.clear()
        self.pending.clear()
        # 队列中尚未取出的条目没有持久状态，其 checkpoint 可被显式恢复。
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

    def _accepting(self):
        if not self.available or self.closing:
            raise AgentError("not_accepting", "runtime is not accepting new work", 503)

    @staticmethod
    def cancel_once(task):
        if not task.done() and not task.cancelling():
            task.cancel()

    def enqueue(self, row: dict, command=None):
        if row["run_id"] in self.pending or row["run_id"] in self.active:
            return
        self._accepting()
        try:
            self.queue.put_nowait((row, command))
        except asyncio.QueueFull as exc:
            raise AgentError("queue_full", "workflow queue is full", 429) from exc
        self.pending.add(row["run_id"])

    def _observe(self, kind, row, data):
        if self.observer is not None:
            try:
                self.observer(kind, copy.deepcopy(row), copy.deepcopy(data))
            except Exception:
                self.observer_errors += 1

    @staticmethod
    def client_request_digest(binding: GraphBinding, value, options: RunOptions):
        return (
            binding.client_digest(value, options)
            if binding.client_digest is not None
            else content_digest({"input": value, "options": options.model_dump(mode="json")})
        )

    async def _idempotent_view(self, existing, value, options):
        # 即使当前 Schema 已变更，请求契约仍归原始发布版本所有。
        digest = self.client_request_digest(self.binding(existing), value, options)
        if self._control(existing)["client_digest"] != digest:
            raise AgentError("idempotency_conflict", "same key has different request content", 409)
        return await self.view(existing["owner"], existing["run_id"])

    async def submit(
        self,
        identity: Identity,
        agent_id: str,
        key: str,
        value: JsonValue,
        options: RunOptions | None = None,
    ) -> dict:
        options = options or RunOptions()
        TypeAdapter(Identifier).validate_python(key)
        if self.agent_authorizers:
            if (
                await self.identity(identity.owner) != identity
                or not await self.agent_allowed(identity, agent_id)
                or await self.identity(identity.owner) != identity
            ):
                raise AgentError("agent_forbidden", "Agent access is not authorized", 403)
        # 决定 idempotency 的是原始客户端输入，而非由版本推导出的 payload。
        existing = self.store.lookup(identity.owner, agent_id, key)
        if existing:
            return await self._idempotent_view(existing, value, options)
        bound = self.registry.agent(agent_id)
        current_identity = await self.identity(identity.owner)
        if current_identity != identity:
            raise AgentError("identity", "request identity changed before admission", 403)
        if not await self.agent_allowed(identity, agent_id):
            raise AgentError("agent_forbidden", "Agent access is not authorized", 403)
        timeout = min(
            options.timeout_seconds or bound.definition.budget.timeout_seconds,
            bound.definition.budget.timeout_seconds,
        )
        deadline = time.time() + timeout
        try:
            async with asyncio.timeout(max(0.0, deadline - time.time())), self.admission_lock:
                # 队列/容量检查绝不阻塞已有的幂等请求。
                existing = self.store.lookup(identity.owner, agent_id, key)
                if existing:
                    return await self._idempotent_view(existing, value, options)
                # 前一次准入可能仍在准备同一个 key。只有真正的新请求才会按所选
                # 发布版本解释。
                SchemaContract(bound.definition.input_schema).validate(value, code="agent_input")
                binding = self.binding({"agent_id": agent_id, "binding": bound.digest})
                digest = self.client_request_digest(binding, value, options)
                self._accepting()
                if self.queue.full() and bound.definition.delegation is None:
                    raise AgentError("queue_full", "workflow queue is full", 429)
                payload = {"input": copy.deepcopy(value)}
                admission_commit = None
                if binding.prepare is not None:
                    prepared = await binding.prepare(identity, copy.deepcopy(value))
                    if isinstance(prepared, PreparedAdmission):
                        payload, admission_commit = prepared.payload, prepared.commit
                    else:
                        payload = prepared
                payload = copy.deepcopy(payload)
                if self.agent_authorizers:
                    if not await self.agent_allowed(identity, agent_id):
                        raise AgentError(
                            "agent_forbidden", "Agent access changed during preparation", 403
                        )
                    if await self.identity(identity.owner) != identity:
                        raise AgentError(
                            "permission_changed", "identity changed during Agent authorization", 403
                        )
                if not isinstance(payload, dict):
                    raise AgentError(
                        "admission_contract", "prepared payload must be a JSON object", 503
                    )
                canonical_json(payload)
                payload["_identity_digest"] = self.identity_digest(identity)
                managed = bound.definition.delegation is not None
                if managed:
                    payload["_task_tree"] = True
                    admission_commit = self.task_coordinator.root_commit(
                        bound, identity, deadline, admission_commit
                    )
                row, created, victims = self.store.admit(
                    identity,
                    agent_id,
                    key,
                    digest,
                    payload,
                    bound.digest,
                    options,
                    deadline,
                    "fixed-deadline" if managed else binding.execution_policy,
                    admission_commit=admission_commit,
                )
                if created:
                    self._observe("run.accepted", row, {})
                for victim in victims:
                    self.pending.discard(victim)
                    if victim in self.active:
                        self.cancel_once(self.active[victim])
                    else:
                        victim_row = self.store.internal_request(victim)
                        await self.terminal_checkpoint(
                            victim_row,
                            self._cancel_status(victim_row),
                            "user_cancel_or_topic_switch",
                        )
                if not managed:
                    await self.update_checkpoint(row, binding.initial(row), as_node=START)
                self.store.append_event(row, "run.accepted", {"epoch": row["epoch"]})
                if managed:
                    self.task_coordinator.wake.set()
                else:
                    self.enqueue(row)
        except TimeoutError as exc:
            raise AgentError(
                "admission_timeout", "execution deadline expired during admission", 504
            ) from exc
        return await self.view(identity.owner, row["run_id"])

    async def checkpoint_state(self, row):
        try:
            return await self.binding(row).graph.aget_state(self.config(row))
        except AgentError:
            raise
        except Exception as exc:
            self.available = False
            raise AgentError(
                "checkpoint_unavailable", "execution state storage unavailable", 503
            ) from exc

    async def update_checkpoint(self, row, values, *, as_node):
        try:
            return await self.binding(row).graph.aupdate_state(
                self.config(row), values, as_node=as_node
            )
        except AgentError:
            raise
        except Exception as exc:
            self.available = False
            raise AgentError(
                "checkpoint_unavailable", "execution state persistence failed", 503
            ) from exc

    @staticmethod
    def interrupts(snapshot) -> list[dict]:
        return [i.value for task in snapshot.tasks for i in task.interrupts]

    def visible_reviews(self, row, snapshot, *, create=False) -> list[dict]:
        """创建兼容性挑战要求存在真实的 checkpoint interrupt。"""
        raw = self.interrupts(snapshot)
        binding = self.binding(row)
        if not raw or binding.review_descriptor is None:
            return raw
        descriptors = binding.review_descriptor(row, snapshot)
        if len(descriptors) != len(raw) or len(descriptors) != 1:
            raise AgentError("review_contract", "compatibility review must bind one interrupt", 409)
        result = []
        for descriptor in descriptors:
            digest, preview = descriptor["digest"], descriptor["preview"]
            with self.store.connection() as db:
                found = db.execute(
                    "SELECT * FROM agent_reviews WHERE run_id=? AND owner=? AND digest=?",
                    (row["run_id"], row["owner"], digest),
                ).fetchone()
            review = dict(found) if found else None
            if review is None and create:
                review = self.store.prepare_review(
                    row,
                    digest,
                    preview,
                    self.registry.resolve(
                        row["agent_id"], row["binding"]
                    ).definition.budget.review_seconds,
                )
            if review is None:
                result.append({"kind": "action_review", "digest": digest, "preview": preview})
                continue
            if create:
                self.store.seal_review(row, review["id"], digest)
            result.append(
                {
                    "kind": "action_review",
                    "review_id": review["id"],
                    "digest": digest,
                    "preview": preview,
                }
            )
        return result

    def _validate_snapshot(self, row, snapshot):
        if not snapshot.values:
            return
        for name, column in self.binding(row).control_fields.items():
            if snapshot.values.get(name) != row[column]:
                raise AgentError(
                    "checkpoint_control", "checkpoint control binding differs from run ledger", 409
                )
        policy = self.binding(row).validate_snapshot
        if policy is not None:
            policy(row, snapshot)

    async def worker(self):
        while not self.closing:
            row, command = await self.queue.get()
            if self.closing:
                self.queue.task_done()
                return
            run_id = row["run_id"]
            self.pending.discard(run_id)
            try:
                if isinstance(command, TaskTicket) and not self.task_coordinator.claim(
                    row, command
                ):
                    self.queue.task_done()
                    self.task_coordinator.wake.set()
                    continue
            except Exception:
                self.available = False
                self.failure_code = "task_claim_failure"
                self.queue.task_done()
                self.task_coordinator.wake.set()
                return
            task = asyncio.create_task(self.execute(row, command))
            self.active[run_id] = task
            try:
                await task
            except asyncio.CancelledError:
                if self.closing:
                    raise
            except Exception:
                self.available = False
            finally:
                self.active.pop(run_id, None)
                self.queue.task_done()
                self.task_coordinator.wake.set()

    async def terminal_checkpoint(self, row, status, code):
        previous = self.task_coordinator.outcome(row["run_id"])
        if previous is not None:
            status, code = previous["status"], previous["error"] or code
        if (
            self.task_coordinator.metadata(row["run_id"]) is not None
            and self.store.final_receipt(row["owner"], row["run_id"]) is None
        ):
            await self._stop_task_tree(row)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute(
                "SELECT payload FROM agent_final_receipts WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            task_tree.outcome_in(
                db,
                row["run_id"],
                status,
                code,
                content_digest(json.loads(receipt[0])) if receipt else None,
            )
        self.task_coordinator.wake.set()
        snapshot = await self.checkpoint_state(row)
        binding = self.binding(row)
        required = {name: row[column] for name, column in binding.control_fields.items()}
        if any(
            name in snapshot.values and snapshot.values[name] != value
            for name, value in required.items()
        ):
            raise AgentError(
                "checkpoint_control", "terminal checkpoint belongs to different control", 409
            )
        complete = all(snapshot.values.get(name) == value for name, value in required.items())
        if not snapshot.next and snapshot.values.get("status") == status and complete:
            return
        await self.update_checkpoint(row, None, as_node=END)
        terminal = {**required, **binding.terminal_values(status, code)}
        if not snapshot.values:
            terminal = {**binding.initial(row), **terminal}
        await self.update_checkpoint(row, terminal, as_node=binding.terminal_node)
        self.store.append_event(row, "run.terminal", {"status": status, "error": code})
        self._observe("run.terminal", row, {"status": status, "error": code})

    async def execute(self, row, command=None):
        row = self.store.internal_request(row["run_id"])
        try:
            outcome = self.task_coordinator.outcome(row["run_id"])
            if outcome is not None:
                await self.terminal_checkpoint(
                    row, outcome["status"], outcome["error"] or "task_outcome_recovered"
                )
                return
            receipt = self.store.final_receipt(row["owner"], row["run_id"])
            if receipt is not None:
                # SQL 终态提交已赢得竞态：只补齐其 checkpoint，绝不重跑副作用。
                identity = await self.identity(row["owner"])
                reason = self._reason(row, identity, final=True)
                if reason:
                    raise AgentError(reason, "final checkpoint cannot use a different binding", 409)
                await self.terminal_checkpoint(row, receipt["status"], "final_receipt_recovered")
                return
            async with asyncio.timeout(max(0.0, row["deadline"] - time.time())):
                await self.context(row["run_id"])
                snapshot = await self.checkpoint_state(row)
                self._validate_snapshot(row, snapshot)
                binding = self.binding(row)
                managed = self.task_coordinator.metadata(row["run_id"]) is not None
                if managed:
                    if not snapshot.values:
                        await self.update_checkpoint(row, binding.initial(row), as_node=START)
                        snapshot = await self.checkpoint_state(row)
                    command = self.task_coordinator.command(row, snapshot)
                    if self.interrupts(snapshot) and command is None:
                        if not self.task_coordinator.park(row, snapshot):
                            for visible in self.visible_reviews(row, snapshot, create=True):
                                self.store.seal_review(row, visible["review_id"], visible["digest"])
                            self.task_coordinator.awaiting_review(row["run_id"])
                        return
            # LangGraph 的后台 gate 同样包裹 checkpoint I/O。取消一个尚未启动的
            # gate 会让其内层协程变成孤儿。业务并发由 worker、tool 槽位与
            # workflow fan-out 共同限制。
            config = self.config(row) | {"recursion_limit": binding.recursion_limit}
            async with asyncio.timeout(max(0.0, row["deadline"] - time.time())):
                with binding.execution_scope(row) if binding.execution_scope else nullcontext():
                    result = await binding.graph.ainvoke(command, config, durability="sync")
                snapshot = await self.checkpoint_state(row)
                self._validate_snapshot(row, snapshot)
                if not snapshot.next and binding.on_complete is not None:
                    projection = await binding.on_complete(row, result)
                    await self.commit_final(row["run_id"], projection)
            if self.interrupts(snapshot):
                if managed and self.task_coordinator.park(row, snapshot):
                    return
                for visible in self.visible_reviews(row, snapshot, create=True):
                    self.store.seal_review(row, visible["review_id"], visible["digest"])
                self.store.append_event(row, "run.awaiting_review", {"review_available": True})
                if managed:
                    self.task_coordinator.awaiting_review(row["run_id"])
            elif managed and self.task_coordinator.outcome(row["run_id"]) is None:
                raise AgentError(
                    "task_final_receipt", "completed managed graph has no durable outcome", 409
                )
        except asyncio.CancelledError:
            if not self.closing:
                await self.terminal_checkpoint(
                    row,
                    self._cancel_status(row),
                    "user_cancel_or_topic_switch",
                )
            raise
        except Exception as exc:
            # 即便 graph/checkpoint 的确认失败，receipt 仍是权威依据。
            if self.store.final_receipt(row["owner"], row["run_id"]) is not None:
                self.available = False
                return
            policy_code = self.binding(row).exception_code
            code = (
                exc.code
                if isinstance(exc, AgentError)
                else "deadline_exceeded"
                if isinstance(exc, TimeoutError)
                else (policy_code(exc) if policy_code else None) or "execution_failure"
            )
            status = (
                "needs_reconciliation"
                if isinstance(exc, NeedsReconciliation)
                or code == "needs_reconciliation"
                or self._uncertain(row)
                else "effect_applied_result_unavailable"
                if isinstance(exc, AppliedResultUnavailable) or self._applied_effects(row)
                else "cancelled"
                if code == "cancelled_or_old_topic"
                else "timed_out"
                if isinstance(exc, TimeoutError)
                or code in {"original_deadline_expired", "model_timeout", "deadline_exceeded"}
                else "failed"
            )
            try:
                await self.terminal_checkpoint(row, status, code)
            except Exception:
                self.available = False

    async def view(self, owner: str, run_id: str) -> dict:
        row = self.store.request(owner, run_id)
        identity = await self.identity(owner)
        snapshot = await self.checkpoint_state(row)
        self._validate_snapshot(row, snapshot)
        values, interrupts = copy.deepcopy(snapshot.values), self.interrupts(snapshot)
        receipt = self.store.final_receipt(owner, run_id)
        proof_error = None
        try:
            uncertain = self._uncertain(row)
        except AgentError as exc:
            # 当原始执行凭据损坏时，查询元数据仍然可用。执行/恢复路径依旧保持
            # 它们的严格校验。
            uncertain, proof_error = True, exc.code
        status = (
            "running"
            if run_id in self.active
            else "queued"
            if run_id in self.pending
            else "needs_reconciliation"
            if uncertain
            else "initialization_incomplete"
            if not values
            else "awaiting_review"
            if interrupts and snapshot.next
            else "interrupted"
            if snapshot.next
            else values.get("status", "failed")
        )
        dispatch = self.task_coordinator.state(run_id)
        if dispatch is not None and dispatch["state"] != "terminal":
            status = {
                "ready": "queued",
                "running": "running",
                "waiting_children": "waiting_children",
                "awaiting_review": "awaiting_review",
            }[dispatch["state"]]
        if receipt is not None:
            status = (
                "needs_reconciliation"
                if uncertain
                else "finalization_incomplete"
                if snapshot.next or values.get("status") != receipt["status"]
                else receipt["status"]
            )
        reason = proof_error
        if proof_error is None:
            try:
                reason = self._reason(
                    row,
                    identity,
                    allow_review=status == "awaiting_review",
                    final=receipt is not None,
                )
                if status != "waiting_children":
                    interrupts = self.visible_reviews(
                        row, snapshot, create=status == "awaiting_review" and reason is None
                    )
            except AgentError as exc:
                reason, proof_error = exc.code, exc.code
        if status == "awaiting_review" and interrupts and reason is None:
            for pending in interrupts:
                review = (
                    self.store.review(owner, run_id, pending.get("review_id", ""))
                    if isinstance(pending, dict)
                    else None
                )
                if (
                    review is None
                    or review["digest"] != pending.get("digest")
                    or review["expires_at"] <= time.time()
                ):
                    reason = "review_unavailable_or_expired"
                elif self.store.visibility(review["id"]) is None and row["deadline"] <= time.time():
                    reason = "review_visibility_unsealed_expired"
        info = {
            "run_id": run_id,
            "thread_id": row["thread"],
            "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id")
            if snapshot.config
            else None,
            "agent_id": row["agent_id"],
            "conversation_id": self._control(row)["conversation_id"],
            "epoch": row["epoch"],
            "status": status,
            "deadline": row["deadline"],
            "recoverable": status
            in {
                "interrupted",
                "awaiting_review",
                "initialization_incomplete",
                "finalization_incomplete",
            }
            and reason is None,
            "resume_reason": reason,
            "interrupts": interrupts,
            "result": receipt["value"] if receipt else None,
        }
        if dispatch is not None:
            info["children"] = self.task_coordinator.children(run_id)
        if proof_error is not None:
            info.update(
                result=None, interrupts=[], recoverable=False, result_visibility="unverified"
            )
            return info
        sources = [
            *(receipt["sources"] if receipt else []),
            *self.binding(row).visible_sources(row, values),
            *self.task_coordinator.runtime_sources(run_id),
        ]
        if (
            reason == "permission_changed"
            or not await self.authorized_sources(row, identity, sources)
            or await self.identity(owner) != identity
        ):
            info["result"], info["interrupts"] = None, []
            info["result_visibility"] = "revoked"
            info["recoverable"] = False
            info["resume_reason"] = reason or "source_revoked"
            return info
        # 只有在可见性策略通过之后，投影器才获得权威控制信息。
        info["effects"] = self._applied_effects(row)
        projection = await self.binding(row).project(row, values, copy.deepcopy(info))
        if (
            not await self.authorized_sources(row, identity, sources)
            or await self.identity(owner) != identity
        ):
            info["result"], info["interrupts"] = None, []
            info["result_visibility"], info["recoverable"] = "revoked", False
            info["resume_reason"] = "permission_changed_or_source_revoked"
            return info
        with self.store.connection() as db:
            db.execute("BEGIN")
            if not self._local_sources(
                db,
                row,
                list(
                    {
                        canonical_json(source): source
                        for source in [
                            *sources,
                            *self.task_coordinator.runtime_sources_in(db, run_id),
                        ]
                    }.values()
                ),
                frame=self._source_frame_in(db, row, identity),
            ):
                info["result"], info["interrupts"] = None, []
                info["result_visibility"], info["recoverable"] = "revoked", False
                info["resume_reason"] = "source_revoked"
                return info
            guard = self.binding(row).projection_guard
            if (
                guard is not None
                and guard(db, copy.deepcopy(row), copy.deepcopy(projection)) is not True
            ):
                info.update(
                    result=None,
                    interrupts=[],
                    recoverable=False,
                    result_visibility="unverified",
                    resume_reason=reason or "result_unavailable",
                )
                return info
        if "result" in projection:
            info["result"] = projection["result"]
        info = {**projection, **info}
        return info

    def _applied_effects(self, row):
        return [
            effect
            for record in self._descendant_rows(row)
            for effect in self._own_applied_effects(record)
        ]

    def _own_applied_effects(self, row):
        self._reconcile_actions(row)
        policy = self.binding(row).applied_effects
        return [
            {"operation_key": operation["key"], "phase": operation["phase"]}
            for operation in self.actions.operations(row["owner"], row["run_id"])
            if not operation["read_only"] and operation["phase"] == "applied"
        ] + (policy(row) if policy else [])

    async def cancel(self, owner: str, run_id: str, *, _timeout=False) -> dict:
        row = self.store.request(owner, run_id)
        if self.task_coordinator.metadata(run_id) is not None:
            outcome = self.task_coordinator.outcome(run_id)
            if outcome is not None:
                return await self.view(owner, run_id)
            await self._stop_task_tree(row, include_self=True)
            status = self._cancel_status(row)
            if _timeout and status == "cancelled":
                status = "timed_out"
            if self.store.final_receipt(owner, run_id) is None:
                await self.terminal_checkpoint(
                    row,
                    status,
                    "original_deadline_expired" if _timeout else "user_cancel_or_topic_switch",
                )
            return await self.view(owner, run_id)
        async with self.admission_lock:
            if not self.store.cancel(owner, run_id):
                return await self.view(owner, run_id)
            self.pending.discard(run_id)
            task = self.active.get(run_id)
            if task:
                self.cancel_once(task)
                await asyncio.gather(task, return_exceptions=True)
            else:
                await self.terminal_checkpoint(
                    row,
                    self._cancel_status(row),
                    "user_cancel_or_topic_switch",
                )
        if not self.available:
            raise AgentError(
                "cancel_cleanup_failed", "cancel recorded; state storage needs repair", 503
            )
        return await self.view(owner, run_id)

    def _uncertain(self, row):
        descendants = self._descendant_rows(row)
        return any(self._own_uncertain(record) for record in descendants)

    def _own_uncertain(self, row):
        self._reconcile_actions(row)
        binding = self.binding(row)
        extra = (
            binding.effect_guard(copy.deepcopy(row), None)
            if binding.effect_guard is not None
            else binding.uncertain is not None and binding.uncertain(row["run_id"])
        )
        return self.actions.uncertain(row["run_id"]) or extra

    def _descendant_rows(self, row):
        with self.store.connection() as db:
            db.execute("BEGIN")
            if task_tree.metadata_in(db, row["run_id"]) is None:
                return [row]
            return [
                dict(record)
                for record in db.execute(
                    "SELECT r.* FROM agent_task_closure c JOIN requests r ON r.run_id=c.descendant_run_id WHERE c.ancestor_run_id=? ORDER BY c.distance DESC",
                    (row["run_id"],),
                )
            ]

    async def _stop_task_tree(self, row, *, include_self=False):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            task_tree.ancestry_in(db, row["run_id"])
            records = [
                dict(record)
                for record in db.execute(
                    "SELECT r.* FROM agent_task_closure c JOIN requests r ON r.run_id=c.descendant_run_id WHERE c.ancestor_run_id=? ORDER BY c.distance DESC",
                    (row["run_id"],),
                )
            ]
            db.executemany(
                "UPDATE requests SET cancel=1 WHERE run_id=? AND NOT EXISTS(SELECT 1 FROM agent_task_outcomes o WHERE o.run_id=requests.run_id)",
                [(record["run_id"],) for record in records],
            )
        if not include_self:
            records = [record for record in records if record["run_id"] != row["run_id"]]
        tasks = [
            self.active[record["run_id"]]
            for record in records
            if record["run_id"] in self.active
            and self.active[record["run_id"]] is not asyncio.current_task()
        ]
        for task in tasks:
            self.cancel_once(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        for record in records:
            if (
                record["run_id"] == row["run_id"]
                or self.task_coordinator.outcome(record["run_id"]) is not None
            ):
                continue
            self.pending.discard(record["run_id"])
            await self.terminal_checkpoint(
                record, self._cancel_status(record), "ancestor_task_stopped"
            )

    def _cancel_status(self, row):
        return (
            "needs_reconciliation"
            if self._uncertain(row)
            else "effect_applied_result_unavailable"
            if self._applied_effects(row)
            else "cancelled"
        )

    async def resume(
        self,
        identity: Identity,
        run_id: str,
        *,
        review_id: str | None = None,
        digest: str | None = None,
    ) -> dict:
        original = self.store.request(identity.owner, run_id)
        # 审查拥有自身不可变的失效时间；普通恢复则使用原始 deadline。
        snapshot = await self.checkpoint_state(original)
        pending_reviews = self.interrupts(snapshot)
        expiry = original["deadline"]
        if self.store.final_receipt(identity.owner, run_id) is not None:
            # 有界的管理式 checkpoint 修复，绝不构成新的执行许可。
            expiry = time.time() + 30.0
        if pending_reviews:
            review = self.store.review(identity.owner, run_id, review_id or "")
            if review is not None:
                expiry = review["expires_at"]
                if self.store.visibility(review["id"]) is None:
                    expiry = min(expiry, original["deadline"])
        try:
            async with asyncio.timeout(max(0.0, expiry - time.time())), self.admission_lock:
                row = self.store.request(identity.owner, run_id)
                current_identity = await self.identity(identity.owner)
                if current_identity != identity:
                    raise AgentError("identity", "current request identity changed", 403)
                view = await self.view(identity.owner, run_id)
                if not view["recoverable"]:
                    raise AgentError(
                        "not_recoverable", view["resume_reason"] or "run cannot be resumed", 409
                    )
                self._accepting()
                managed = self.task_coordinator.metadata(run_id) is not None
                if self.queue.full() and not managed:
                    raise AgentError("queue_full", "workflow queue is full", 429)
                command: Any = None
                if view["status"] == "initialization_incomplete":
                    await self.update_checkpoint(row, self.binding(row).initial(row), as_node=START)
                if view["status"] == "awaiting_review":
                    visible = view["interrupts"]
                    if (
                        len(visible) != 1
                        or visible[0].get("review_id") != review_id
                        or visible[0].get("digest") != digest
                    ):
                        raise AgentError(
                            "approval_required",
                            "approve the current visible review and its digest",
                            409,
                        )
                    self.store.seal_review(row, review_id, digest)
                    policy = self.binding(row).resume_value
                    value: JsonValue = {"review_id": review_id, "digest": digest}
                    callback = (
                        self.task_coordinator.resume_commit(
                            await self.context(run_id), snapshot, value, digest
                        )
                        if managed
                        else None
                    )
                    self.store.accept_review(
                        identity.owner, run_id, review_id, digest, resume_commit=callback
                    )
                    row = self.store.request(identity.owner, run_id)
                    if not managed and policy is not None:
                        value = policy(row, visible[0])
                    command = Command(resume=value)
                if managed:
                    if self.task_coordinator.outcome(run_id) is not None:
                        self.task_coordinator.repair(run_id)
                    self.task_coordinator.wake.set()
                else:
                    self.enqueue(row, command)
        except TimeoutError as exc:
            raise AgentError(
                "recovery_expired", "recovery or review deadline expired", 409
            ) from exc
        return await self.view(identity.owner, run_id)

    async def _approved_action(self, context, tool, call, digest) -> bool:
        owned = self._owned_action(context, call)
        fresh = await self.context(context.run_id, _owned_intent=owned)
        if not self._same_control(fresh, context):
            return False
        row = self.store.internal_request(context.run_id)
        policy = self.binding(row).action_approval
        approved = (
            await policy(
                context.model_copy(deep=True),
                tool.model_copy(deep=True),
                call.model_copy(deep=True),
                digest,
            )
            if policy is not None
            else self.store.approval(row, digest)
        )
        return bool(approved) and self._same_control(
            await self.context(context.run_id, _owned_intent=owned), context
        )

    def _owned_action(self, context, call):
        if (
            context.operation_key
            and context.attempt_token
            and context.operation_key == self.actions.key(context, call)
        ):
            return (context.operation_key, context.attempt_token)
        return None

    @staticmethod
    def _same_control(left, right):
        return all(
            getattr(left, field) == getattr(right, field)
            for field in (
                "identity",
                "agent_id",
                "run_id",
                "conversation_id",
                "epoch",
                "deadline",
                "config_digest",
                "scope_ceiling",
            )
        )

    async def _authorize_tool(self, context, tool, call) -> bool:
        owned_intent = self._owned_action(context, call)
        if not self._same_control(
            await self.context(context.run_id, _owned_intent=owned_intent), context
        ):
            return False
        if tool.scope not in context.effective_scopes:
            return False
        if self.tool_authorizer is not None and not await self.tool_authorizer(context, tool, call):
            return False
        if not await self.authorized_sources(
            self.store.internal_request(context.run_id), context.identity, []
        ):
            return False
        return self._same_control(
            await self.context(context.run_id, _owned_intent=owned_intent), context
        )

    def _tool_sources_current_in(self, db, context):
        row = db.execute("SELECT * FROM requests WHERE run_id=?", (context.run_id,)).fetchone()
        return row is not None and self._local_sources(
            db,
            dict(row),
            self.task_coordinator.runtime_sources_in(db, context.run_id),
            frame=self._source_frame_in(db, dict(row), context.access_identity),
        )

    async def require_review(self, run_id: str, digest: str, preview: dict):
        row = self.store.internal_request(run_id)
        context = await self.context(run_id)
        bound = self.registry.resolve(context.agent_id, context.config_digest)
        review = self.store.prepare_review(
            row, digest, preview, bound.definition.budget.review_seconds
        )
        resumed = interrupt(
            {
                "kind": "action_review",
                "review_id": review["id"],
                "digest": digest,
                "preview": preview,
            }
        )
        if resumed != {"review_id": review["id"], "digest": digest} or not self.store.approval(
            row, digest
        ):
            raise AgentError(
                "review_mismatch", "graph resume lacks the bound durable approval", 409
            )
        await self.context(run_id)

    async def authorized_sources(self, row, identity, sources) -> bool:
        sources = list(
            {
                canonical_json(source): source
                for source in [*sources, *self.task_coordinator.runtime_sources(row["run_id"])]
            }.values()
        )
        policy = self.registry.resolve(row["agent_id"], row["binding"]).definition.context
        node = self.task_coordinator.metadata(row["run_id"])
        access_identity = (
            identity
            if node is None
            else identity.model_copy(update={"scopes": identity.scopes & node.scope_ceiling})
        )
        valid = await self.contexts.authorized_sources(
            access_identity,
            row["agent_id"],
            sources,
            memory_agents=frozenset([row["agent_id"], *policy.shared_memory_agents])
            if policy.memory_scope != "none"
            else frozenset(),
            memory_conversation=self._control(row)["conversation_id"]
            if policy.memory_scope == "conversation"
            else "",
            artifact_namespace=self._control(row)["conversation_id"],
            run_id=row["run_id"],
        )
        if not valid:
            return False
        with self.store.connection() as db:
            db.execute("BEGIN")
            return self._local_sources(
                db, row, sources, frame=self._source_frame_in(db, row, access_identity)
            )

    def model_request_current(self, context: RunContext, sources: list[dict]) -> bool:
        """Final synchronous local guard after all awaits, immediately before provider use."""
        with self.store.connection() as db:
            db.execute("BEGIN")
            return self.model_request_current_in(db, context, sources)

    async def _authorize_model_request(self, context, request) -> bool:
        current = await self.context(context.run_id)
        if not self._same_control(current, context):
            return False
        sources = [source for message in request.messages for source in message.sources]
        if not await self.authorized_sources(
            self.store.internal_request(context.run_id), context.identity, sources
        ):
            return False
        return self._same_control(await self.context(context.run_id), context)

    def model_request_current_in(self, db, context: RunContext, sources: list[dict]) -> bool:
        """Same-connection guard for provider views and artifact mutation transactions."""
        if not db.in_transaction:
            raise AgentError(
                "runtime_transaction", "model authorization requires a transaction", 503
            )
        task_tree.assert_ancestors_in(db, context)
        if db.execute(
            "SELECT 1 FROM agent_task_closure c JOIN agent_tool_operations o ON o.run_id=c.descendant_run_id WHERE c.ancestor_run_id=? AND o.read_only=0 AND o.phase IN ('intent','unknown')",
            (context.run_id,),
        ).fetchone():
            return False
        row = db.execute("SELECT * FROM requests WHERE run_id=?", (context.run_id,)).fetchone()
        control = db.execute(
            "SELECT * FROM agent_run_controls WHERE run_id=?", (context.run_id,)
        ).fetchone()
        if (
            row is None
            or control is None
            or row["cancel"]
            or (row["owner"], row["agent_id"], row["binding"], row["epoch"], row["deadline"])
            != (
                context.identity.owner,
                context.agent_id,
                context.config_digest,
                context.epoch,
                context.deadline,
            )
            or row["deadline"] <= time.time()
            or control["conversation_id"] != context.conversation_id
            or control["identity_version"] != context.identity.permission_version
            or control["identity_digest"] != self.identity_digest(context.identity)
            or self.store.epoch(
                db, context.identity.owner, context.agent_id, context.conversation_id
            )
            != context.epoch
        ):
            return False
        return self._local_sources(
            db,
            dict(row),
            list(
                {
                    canonical_json(source): source
                    for source in [
                        *sources,
                        *self.task_coordinator.runtime_sources_in(db, context.run_id),
                    ]
                }.values()
            ),
            frame=self._source_frame_in(db, dict(row), context.access_identity),
        )

    def _source_frame_in(self, db, row, identity) -> SourceFrame:
        control = db.execute(
            "SELECT * FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if control is None:
            raise AgentError("source_frame", "original source frame is unavailable", 409)
        policy = self.registry.resolve(row["agent_id"], row["binding"]).definition.context
        node = task_tree.metadata_in(db, row["run_id"])
        access = (
            identity
            if node is None
            else identity.model_copy(update={"scopes": identity.scopes & node.scope_ceiling})
        )
        return SourceFrame(
            access,
            row["agent_id"],
            frozenset([row["agent_id"], *policy.shared_memory_agents])
            if policy.memory_scope != "none"
            else frozenset(),
            control["conversation_id"] if policy.memory_scope == "conversation" else "",
            control["conversation_id"],
            row["run_id"],
        )

    def _expand_child_source(self, frame: SourceFrame, source: dict) -> SourceExpansion | None:
        with self.store.connection() as db:
            db.execute("BEGIN")
            return self._expand_child_source_in(db, frame, source)

    def _expand_child_source_in(
        self, db, frame: SourceFrame, source: dict
    ) -> SourceExpansion | None:
        if source.get("kind") == "task_input":
            return self._expand_task_input_in(db, frame, source)
        try:
            reference = ChildReceiptReference.model_validate(source)
            child = task_tree.metadata_in(db, reference.child_run_id)
            if (
                child is None
                or child.parent_run_id is None
                or child.parent_run_id != reference.parent_run_id
            ):
                return None
            issuer = task_tree.metadata_in(db, child.parent_run_id)
            task_tree.ancestry_in(db, child.run_id)
            consumer_row = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (frame.run_id,)
            ).fetchone()
            child_row = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (child.run_id,)
            ).fetchone()
            issuer_row = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (child.parent_run_id,)
            ).fetchone()
            rows = (consumer_row, issuer_row, child_row)
            if issuer is None or any(row is None for row in rows):
                return None
            consumer_control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (frame.run_id,)
            ).fetchone()
            issuer_control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (issuer.run_id,)
            ).fetchone()
            child_control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (child.run_id,)
            ).fetchone()
            controls = (consumer_control, issuer_control, child_control)
            if any(control is None for control in controls):
                return None
            if (
                any(row["owner"] != frame.identity.owner for row in rows)
                or consumer_row["agent_id"] != frame.agent_id
                or issuer.agent_id != frame.agent_id
                or issuer_row["agent_id"] != issuer.agent_id
                or issuer_row["binding"] != issuer.binding
                or child_row["agent_id"] != child.agent_id
                or child_row["binding"] != child.binding
                or child.agent_id != reference.agent_id
                or child.binding != reference.binding
                or consumer_control["conversation_id"] != frame.artifact_namespace
                or issuer_control["conversation_id"] != frame.artifact_namespace
                or consumer_row["epoch"] != issuer_row["epoch"]
                or self.store.epoch(
                    db, issuer.owner, issuer.agent_id, issuer_control["conversation_id"]
                )
                != issuer_row["epoch"]
                or self.store.epoch(
                    db, child.owner, child.agent_id, child_control["conversation_id"]
                )
                != child_row["epoch"]
                or any(
                    control["identity_version"] != frame.identity.permission_version
                    for control in controls
                )
            ):
                return None
            # Resolve both original bindings; a reference never selects a new release.
            self.registry.resolve(issuer.agent_id, issuer.binding)
            receipt = db.execute(
                "SELECT * FROM agent_final_receipts WHERE run_id=?", (child.run_id,)
            ).fetchone()
            outcome = db.execute(
                "SELECT * FROM agent_task_outcomes WHERE run_id=?", (child.run_id,)
            ).fetchone()
            if (
                receipt is None
                or outcome is None
                or receipt["owner"] != child.owner
                or receipt["binding"] != child.binding
            ):
                return None
            value = json.loads(receipt["payload"])
            if (
                content_digest(value) != reference.digest
                or outcome["final_receipt_digest"] != reference.digest
                or outcome["status"] != value["status"]
            ):
                return None
            child_frame = self._source_frame_in(db, dict(child_row), frame.identity)
            stamp = content_digest(
                {
                    "requests": [
                        {
                            key: row[key]
                            for key in ("run_id", "owner", "agent_id", "binding", "digest", "epoch")
                        }
                        for row in rows
                    ],
                    "controls": [dict(control) for control in controls],
                    "issuer": issuer.model_dump(mode="json"),
                    "child": child.model_dump(mode="json"),
                    "outcome": dict(outcome),
                    "receipt": reference.digest,
                }
            )
            return SourceExpansion(
                ("child_receipt", child.owner, child.run_id), child_frame, value["sources"], stamp
            )
        except (AgentError, ValueError, KeyError, TypeError):
            return None

    def _expand_task_input_in(self, db, frame: SourceFrame, source: dict) -> SourceExpansion | None:
        try:
            reference = TaskInputReference.model_validate(source)
            child = task_tree.metadata_in(db, reference.run_id)
            if (
                child is None
                or child.parent_run_id is None
                or child.parent_run_id != reference.parent_run_id
                or frame.agent_id != child.agent_id
                or child.owner != frame.identity.owner
                or content_digest(child.input_sources) != reference.digest
            ):
                return None
            task_tree.ancestry_in(db, child.run_id)
            parent = task_tree.metadata_in(db, child.parent_run_id)
            consumer = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (frame.run_id,)
            ).fetchone()
            consumer_control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (frame.run_id,)
            ).fetchone()
            rows = [
                db.execute("SELECT * FROM requests WHERE run_id=?", (run_id,)).fetchone()
                for run_id in (child.run_id, child.parent_run_id)
            ]
            controls = [
                db.execute("SELECT * FROM agent_run_controls WHERE run_id=?", (run_id,)).fetchone()
                for run_id in (child.run_id, child.parent_run_id)
            ]
            if (
                parent is None
                or consumer is None
                or consumer_control is None
                or any(row is None for row in rows)
                or any(control is None for control in controls)
            ):
                return None
            if (
                controls[0]["conversation_id"] != frame.artifact_namespace
                or consumer_control["conversation_id"] != frame.artifact_namespace
                or consumer["owner"] != frame.identity.owner
                or consumer["agent_id"] != frame.agent_id
                or consumer["epoch"] != rows[0]["epoch"]
                or consumer_control["identity_version"] != frame.identity.permission_version
            ):
                return None
            for node, row, control in zip((child, parent), rows, controls, strict=True):
                if (
                    row["owner"] != frame.identity.owner
                    or row["agent_id"] != node.agent_id
                    or row["binding"] != node.binding
                    or control["identity_version"] != frame.identity.permission_version
                    or self.store.epoch(
                        db, row["owner"], row["agent_id"], control["conversation_id"]
                    )
                    != row["epoch"]
                ):
                    return None
                self.registry.resolve(node.agent_id, node.binding)
            child_frame = self._source_frame_in(db, dict(rows[0]), frame.identity)
            parent_frame = self._source_frame_in(db, dict(rows[1]), child_frame.identity)
            stamp = content_digest(
                {
                    "child": child.model_dump(mode="json"),
                    "parent": parent.model_dump(mode="json"),
                    "controls": [dict(control) for control in controls],
                    "consumer": {
                        key: consumer[key]
                        for key in ("run_id", "owner", "agent_id", "binding", "epoch")
                    },
                    "consumer_control": dict(consumer_control),
                    "requests": [
                        {
                            key: row[key]
                            for key in ("run_id", "owner", "agent_id", "binding", "epoch")
                        }
                        for row in rows
                    ],
                }
            )
            return SourceExpansion(
                ("task_input", child.owner, child.run_id), parent_frame, child.input_sources, stamp
            )
        except (AgentError, ValueError, KeyError, TypeError):
            return None

    async def emit(
        self, run_id: str, kind: str, data, *, provisional=False, sources=None
    ) -> RunEvent:
        row = self.store.internal_request(run_id)
        sources = list(
            {
                canonical_json(source): source
                for source in [*(sources or []), *self.task_coordinator.runtime_sources(run_id)]
            }.values()
        )
        context = await self.context(run_id)
        if not await self.authorized_sources(row, context.identity, sources or []):
            raise AgentError("source_revoked", "event sources are no longer authorized", 403)
        fresh = await self.context(run_id)
        if not self._same_control(fresh, context):
            raise AgentError(
                "permission_changed", "event authorization changed during publication", 403
            )
        # 消费速度相互独立：发布只写入有界的持久日志。
        return self.store.append_event(
            row,
            kind,
            data,
            provisional=provisional,
            sources=sources,
            guard=lambda db: self.model_request_current_in(db, fresh, sources),
        )

    async def events(
        self, owner: str, run_id: str, after: int = 0, *, limit: int = 128
    ) -> list[RunEvent]:
        row = self.store.request(owner, run_id)
        identity = await self.identity(owner)
        if identity.permission_version != self._control(row)["identity_version"] or self._control(
            row
        )["identity_digest"] != self.identity_digest(identity):
            raise AgentError("permission_changed", "event authorization changed", 403)
        result = self.store.events(owner, run_id, after, limit)
        for event in result:
            data = event.data
            sources = data.get("sources", []) if isinstance(data, dict) else []
            if not await self.authorized_sources(row, identity, sources):
                raise AgentError("source_revoked", "event sources are no longer authorized", 403)
        if await self.identity(owner) != identity:
            raise AgentError("permission_changed", "event authorization changed during replay", 403)
        with self.store.connection() as db:
            db.execute("BEGIN")
            self._event_replay_guard(db, row, identity, result)
        return result

    def _event_replay_guard(self, db, row, identity, result):
        current = db.execute("SELECT * FROM requests WHERE run_id=?", (row["run_id"],)).fetchone()
        control = db.execute(
            "SELECT * FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if (
            current is None
            or control is None
            or any(current[key] != row[key] for key in ("owner", "agent_id", "binding", "epoch"))
            or control["identity_version"] != identity.permission_version
            or control["identity_digest"] != self.identity_digest(identity)
        ):
            raise AgentError("permission_changed", "event replay control changed", 403)
        if any(event.provisional for event in result) and (
            current["cancel"]
            or self.store.epoch(db, row["owner"], row["agent_id"], control["conversation_id"])
            != row["epoch"]
        ):
            raise AgentError(
                "event_provisional_unavailable",
                "candidate content is unavailable; query snapshot",
                409,
            )
        if any(
            not self._local_sources(
                db,
                row,
                event.data.get("sources", []) if isinstance(event.data, dict) else [],
                frame=self._source_frame_in(db, row, identity),
            )
            for event in result
        ):
            raise AgentError("source_revoked", "local event sources changed during replay", 403)

    def check_event_frame(self, identity: Identity, event: RunEvent) -> None:
        """最后一次异步传输策略检查之后的仅 SQL 帧校验。

        历史回放绝不执行、对账或恢复某个 Run。这无法撤销传输层已接受的字节。
        """
        row = self.store.request(identity.owner, event.run_id)
        if row["agent_id"] != event.agent_id:
            raise AgentError("event_store_conflict", "event Agent differs from original run", 409)
        with self.store.connection() as db:
            db.execute("BEGIN")
            stored = db.execute(
                "SELECT payload,created_at FROM agent_run_events WHERE run_id=? AND seq=?",
                (event.run_id, event.seq),
            ).fetchone()
            if stored is None or stored[1] < time.time() - self.settings.event_seconds:
                raise AgentError(
                    "event_cursor_expired", "event is no longer retained; query snapshot", 409
                )
            if stored[0] != canonical_json(event.model_dump(mode="json")).decode():
                raise AgentError("event_store_conflict", "stored event differs from frame", 409)
            self._event_replay_guard(db, row, identity, [event])

    def _local_sources(
        self, db, row, sources, depth=0, seen=frozenset(), checks=None, *, frame=None
    ) -> bool:
        if depth > 16 or not isinstance(sources, list):
            return False
        checks = [0] if checks is None else checks
        policy = self.registry.resolve(row["agent_id"], row["binding"]).definition.context
        allowed = (
            {row["agent_id"], *policy.shared_memory_agents}
            if policy.memory_scope != "none"
            else set()
        )
        control = db.execute(
            "SELECT conversation_id FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if control is None:
            return False
        conversation = control[0] if policy.memory_scope == "conversation" else ""
        for source in sources:
            checks[0] += 1
            if checks[0] > 1024 or not isinstance(source, dict):
                return False
            if source.get("kind") in {"child_receipt", "task_input"}:
                current_control = db.execute(
                    "SELECT identity_version FROM agent_run_controls WHERE run_id=?",
                    (row["run_id"],),
                ).fetchone()
                node = task_tree.metadata_in(db, row["run_id"])
                if current_control is None:
                    return False
                child_frame = frame or self._source_frame_in(
                    db,
                    row,
                    Identity(
                        owner=row["owner"],
                        scopes=node.scope_ceiling if node is not None else frozenset(),
                        permission_version=current_control[0],
                    ),
                )
                expansion = self._expand_child_source_in(db, child_frame, source)
                if expansion is None or expansion.key in seen:
                    return False
                child_row = db.execute(
                    "SELECT * FROM requests WHERE run_id=?", (expansion.frame.run_id,)
                ).fetchone()
                if child_row is None or (
                    expansion.sources
                    and not self._local_sources(
                        db,
                        dict(child_row),
                        expansion.sources,
                        depth + 1,
                        seen | {expansion.key},
                        checks,
                        frame=expansion.frame,
                    )
                ):
                    return False
            elif source.get("kind") == "memory":
                key = (source.get("agent_id"), source.get("conversation_id", ""), source.get("id"))
                if not all(isinstance(part, str) for part in key):
                    return False
                ancestor_key = ("memory", row["owner"], *key)
                if ancestor_key in seen or key[0] not in allowed or key[1] != conversation:
                    return False
                record = db.execute(
                    "SELECT * FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND id=?",
                    (row["owner"], *key),
                ).fetchone()
                if (
                    record is None
                    or record["version"] != source.get("version")
                    or not record["enabled"]
                    or (record["valid_until"] is not None and record["valid_until"] <= time.time())
                ):
                    return False
                try:
                    payload = json.loads(record["payload"])
                except (ValueError, TypeError, RecursionError):
                    return False
                if not isinstance(payload, dict):
                    return False
                inherited = payload.get("sources", [])
                if not isinstance(inherited, list) or (
                    inherited
                    and not self._local_sources(
                        db,
                        row,
                        inherited,
                        depth + 1,
                        seen | {ancestor_key},
                        checks,
                        frame=frame,
                    )
                ):
                    return False
            elif source.get("kind") == "artifact":
                try:
                    reference = ArtifactReference.model_validate(source)
                    scope = ArtifactScope(
                        owner=row["owner"], agent_id=row["agent_id"], namespace=control[0]
                    )
                    ancestor_key = ("artifact", *scope.key(), reference.id)
                    if (
                        reference.agent_id != scope.agent_id
                        or reference.namespace != scope.namespace
                        or ancestor_key in seen
                    ):
                        return False
                    artifact = self.artifacts.read_in(db, scope, reference.id)
                except (AgentError, ValueError):
                    return False
                if (
                    not artifact.enabled
                    or artifact.version != reference.version
                    or artifact.digest != reference.digest
                    or (artifact.valid_until is not None and artifact.valid_until <= time.time())
                    or (
                        artifact.sources
                        and not self._local_sources(
                            db,
                            row,
                            artifact.sources,
                            depth + 1,
                            seen | {ancestor_key},
                            checks,
                            frame=frame,
                        )
                    )
                ):
                    return False
            elif self.local_source_guard is not None and not self.local_source_guard(
                db, row, source
            ):
                return False
        return True

    async def commit_final(self, run_id: str, projection: FinalProjection):
        row = self.store.internal_request(run_id)
        context = await self.context(run_id)
        bound = self.registry.resolve(row["agent_id"], row["binding"])
        if bound.definition.output_schema is not None:
            SchemaContract(bound.definition.output_schema).validate(
                projection.value, code="agent_output"
            )
        if projection.messages and not self.contexts._valid_message_sequence(projection.messages):
            raise AgentError("history_messages", "final history contains invalid message roles")
        sources = [
            *projection.sources,
            *self.task_coordinator.runtime_sources(run_id),
            *(source for message in projection.messages for source in message.sources),
        ]
        if not await self.authorized_sources(row, context.identity, sources):
            raise AgentError("source_revoked", "final sources are no longer authorized", 403)
        # 在异步 source 策略之后重新解析当前身份。
        if await self.identity(row["owner"]) != context.identity:
            raise AgentError("permission_changed", "identity changed before final publication", 403)
        original_guard = projection.local_guard
        sources = list({canonical_json(source): source for source in sources}.values())
        inherited = self.task_coordinator.runtime_sources(run_id)
        messages = []
        for message in projection.messages:
            references = list(
                {
                    canonical_json(source): source for source in [*message.sources, *inherited]
                }.values()
            )
            if len(references) > 100:
                raise AgentError("source_limit", "history provenance exceeds the limit", 409)
            messages.append(
                Message.model_validate({**message.model_dump(mode="json"), "sources": references})
            )
        source_keys = {canonical_json(source) for source in sources}
        committed = FinalProjection(
            value=projection.value,
            status=projection.status,
            messages=messages,
            sources=sources,
            local_guard=lambda db: (
                self.model_request_current_in(db, context, sources)
                and all(
                    canonical_json(source) in source_keys
                    for source in self.task_coordinator.runtime_sources_in(db, run_id)
                )
                and (original_guard is None or original_guard(db))
            ),
            extra_commit=projection.extra_commit,
        )
        if not self.store.commit_final(row, committed):
            raise AgentError(
                "final_control", "final publication lost a cancellation, epoch or source race", 409
            )
        self._observe("run.final", row, {"status": committed.status})
        self.task_coordinator.wake.set()

    async def dispatch_children(
        self, parent_run_id: str, tasks: list[ChildTask], *, sources=None
    ) -> list[str]:
        return await self.task_coordinator.dispatch(parent_run_id, tasks, sources=sources)

    async def wait_children(
        self, parent_run_id: str, wait_key: str, children: list[str]
    ) -> list[dict]:
        return await self.task_coordinator.wait(parent_run_id, wait_key, children)
