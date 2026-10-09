"""为已注册的通用 Tool Handler 提供统一的 Schema/授权/尝试路径。"""

import asyncio
import copy
import inspect
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field, JsonValue, ValidationError

from . import task_tree
from .configuration import ToolDeclaration
from .contracts import AgentError, BaseContract, RunContext, ToolCall, ToolHandler
from .effects import (
    ActionReceiptInspector,
    AppliedResultUnavailable,
    AttemptCommit,
    AttemptEvidence,
    NeedsReconciliation,
    SqliteActionLedger,
)
from .registry import ComponentRegistry
from .schema import canonical_json, content_digest
from .tool_control import ToolCircuit


class TransientToolError(AgentError):
    def __init__(self):
        super().__init__("tool_transient", "read tool temporarily unavailable", 503)


class ToolObservation(BaseContract):
    call_id: str
    name: str
    value: JsonValue = None
    error: str | None = None
    succeeded: bool = False
    operation_key: str | None = None
    sources: list[dict[str, JsonValue]] = Field(default_factory=list)


ApprovalChecker = Callable[[RunContext, ToolDeclaration, ToolCall, str], Awaitable[bool]]
ToolAuthorizer = Callable[[RunContext, ToolDeclaration, ToolCall], Awaitable[bool]]


class ToolResult(BaseContract):
    value: JsonValue
    sources: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=100)


ResultPolicy = Callable[[RunContext, ToolDeclaration, JsonValue], Awaitable[JsonValue | ToolResult]]


@dataclass(frozen=True)
class ResolvedTool:
    """可信实现的选择结果，绝不是模型提供的权限。

    自定义 resolver 必须从准入 Run 的原始清单中挑选实现，并在各版本之间保持
    这些实现不变。resolver 可以显式返回 None，以委托给最初配置的注册表；
    但出错时绝不委托。
    """

    declaration: ToolDeclaration
    handler: ToolHandler
    manifest_digest: str | None = None
    attempt_commit: AttemptCommit | None = None


class ToolResolver(Protocol):
    def __call__(self, context: RunContext, call: ToolCall) -> ResolvedTool | None: ...


class ToolExecutor:
    def __init__(
        self,
        registry: ComponentRegistry,
        ledger: SqliteActionLedger,
        *,
        concurrency: int = 4,
        approval_checker: ApprovalChecker | None = None,
        authorizer: ToolAuthorizer | None = None,
        result_policy: ResultPolicy | None = None,
        resolver: ToolResolver | None = None,
        receipt_inspector: ActionReceiptInspector | None = None,
        max_resolutions: int = 4096,
        handler_binder: Callable[[ToolHandler], ToolHandler] | None = None,
        local_source_guard: Callable[[Any, RunContext], bool] | None = None,
    ):
        if not 1 <= concurrency <= 64:
            raise AgentError("tool_concurrency", "tool concurrency must be 1..64")
        if type(max_resolutions) is not int or not 1 <= max_resolutions <= 65536:
            raise AgentError("tool_binding_capacity", "resolution capacity must be 1..65536")
        self.registry, self.ledger = registry, ledger
        self.resolver, self.max_resolutions = resolver, max_resolutions
        self.handler_binder = handler_binder
        self.local_source_guard = local_source_guard
        self.receipt_inspector = receipt_inspector
        self._resolutions: dict[tuple[str, str, str, str], ResolvedTool] = {}
        self._circuits: dict[tuple[str, str, str | None, str], ToolCircuit] = {}
        self.slots = asyncio.Semaphore(concurrency)
        self.approval_checker, self.authorizer, self.result_policy = (
            approval_checker,
            authorizer,
            result_policy,
        )

    @staticmethod
    def _snapshot(resolved) -> ResolvedTool:
        if not isinstance(resolved, ResolvedTool) or not callable(resolved.handler):
            raise AgentError("tool_resolution", "resolver must return a trusted tool binding", 503)
        try:
            tool = ToolDeclaration.model_validate(resolved.declaration.model_dump(mode="json"))
        except (AttributeError, ValidationError, TypeError, ValueError) as exc:
            raise AgentError(
                "tool_resolution", "resolver tool declaration is invalid", 503
            ) from exc
        if resolved.manifest_digest is not None and (
            not isinstance(resolved.manifest_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", resolved.manifest_digest) is None
        ):
            raise AgentError("tool_resolution", "resolver manifest digest is invalid", 503)
        if resolved.attempt_commit is not None and not isinstance(
            resolved.attempt_commit, AttemptCommit
        ):
            raise AgentError("tool_resolution", "resolver attempt registration is invalid", 503)
        return ResolvedTool(
            tool.model_copy(deep=True),
            resolved.handler,
            resolved.manifest_digest,
            resolved.attempt_commit,
        )

    @staticmethod
    def _same(left: ResolvedTool, right: ResolvedTool) -> bool:
        return (
            left.declaration == right.declaration
            and left.manifest_digest == right.manifest_digest
            and left.handler is right.handler
            and (
                left.attempt_commit is right.attempt_commit
                or left.attempt_commit is not None
                and right.attempt_commit is not None
                and left.attempt_commit.tables == right.attempt_commit.tables
                and left.attempt_commit.callback is right.attempt_commit.callback
            )
        )

    def resolve(self, context: RunContext, call: ToolCall) -> ResolvedTool:
        return self._resolve(context, call, check_control=True)

    def _resolve(self, context: RunContext, call: ToolCall, *, check_control: bool) -> ResolvedTool:
        # 自定义目录既不能提供也不能替换原始的 Agent 绑定。
        bound = self.registry.resolve(context.agent_id, context.config_digest)
        selected = None
        if self.resolver is not None:
            try:
                selected = self.resolver(context.model_copy(deep=True), call.model_copy(deep=True))
            except AgentError:
                raise
            except Exception as exc:
                raise AgentError("tool_resolution", "trusted tool resolver failed", 503) from exc
            if inspect.isawaitable(selected):
                if inspect.iscoroutine(selected):
                    selected.close()
                raise AgentError("tool_resolution", "tool resolver must be synchronous", 503)
        selected_custom = selected is not None
        if not selected_custom:
            tool = next((item for item in bound.tools if item.name == call.name), None)
            if tool is None:
                raise AgentError("tool_not_allowed", "tool is not allowed for this Agent", 403)
            entry = self.registry.handler_registration(tool, bound=bound)
            handler = (
                self.handler_binder(entry.implementation)
                if self.handler_binder is not None
                else entry.implementation
            )
            selected = ResolvedTool(tool, handler, attempt_commit=entry.attempt_commit)
        resolved = self._snapshot(selected)
        tool = resolved.declaration
        if tool.name != call.name:
            raise AgentError("tool_resolution", "resolver returned a different tool", 503)
        if check_control and tool.scope not in context.effective_scopes:
            raise AgentError("forbidden", "tool scope is not authorized", 403)
        self.registry.validate_arguments(tool, call.arguments)
        if check_control and time.time() >= context.deadline:
            raise AgentError("deadline", "tool execution deadline expired", 409)
        if selected_custom:
            if resolved.manifest_digest is None:
                raise AgentError("tool_resolution", "custom resolver requires a manifest", 503)
            key = (context.agent_id, context.config_digest, resolved.manifest_digest, tool.name)
            previous = self._resolutions.get(key)
            if previous is not None and not self._same(previous, resolved):
                raise AgentError(
                    "tool_binding_changed", "manifest tool implementation changed", 409
                )
            if previous is None:
                if len(self._resolutions) >= self.max_resolutions:
                    raise AgentError(
                        "tool_binding_capacity", "tool resolution capacity reached", 503
                    )
                self._resolutions[key] = self._snapshot(resolved)
        return resolved

    def declaration(self, context: RunContext, call: ToolCall) -> ToolDeclaration:
        return self.resolve(context, call).declaration

    def _unchanged(self, context, call, selected):
        if not self._same(selected, self.resolve(context, call)):
            raise AgentError("tool_binding_changed", "tool action binding changed", 409)
        with self.ledger.connection() as db:
            db.execute("BEGIN")
            task_tree.assert_ancestors_in(db, context, required_scope=selected.declaration.scope)
            if self.local_source_guard is not None and not self.local_source_guard(db, context):
                raise AgentError("source_revoked", "tool input sources changed", 403)

    def _circuit(self, context: RunContext, selected: ResolvedTool) -> ToolCircuit | None:
        policy = selected.declaration.execution
        if policy is None or policy.circuit_failures is None:
            return None
        key = (
            context.agent_id,
            context.config_digest,
            selected.manifest_digest,
            selected.declaration.name,
        )
        circuit = self._circuits.get(key)
        if circuit is None:
            if len(self._circuits) >= self.max_resolutions:
                raise AgentError("tool_binding_capacity", "tool circuit capacity reached", 503)
            circuit = ToolCircuit(policy.circuit_failures, policy.circuit_cooldown_seconds)
            self._circuits[key] = circuit
        return circuit

    @staticmethod
    def _execution_deadline(deadline: float):
        # 同步的可信回调可能占住事件循环越过 asyncio.timeout 的边界，
        # 而不向其取消回调让出执行权。
        if time.time() >= deadline:
            raise AgentError("tool_timeout", "tool execution deadline expired", 409)

    @staticmethod
    def action_digest(
        context: RunContext,
        tool: ToolDeclaration,
        call: ToolCall,
        *,
        manifest_digest: str | None = None,
    ) -> str:
        descriptor = {
            "owner": context.identity.owner,
            "agent": context.agent_id,
            "run": context.run_id,
            "binding": context.config_digest,
            "tool": tool.model_dump(mode="json"),
            "call": call.model_dump(mode="json"),
            "dependencies": context.dependency_results,
        }
        if manifest_digest is not None:
            descriptor["manifest"] = manifest_digest
        return content_digest(descriptor)

    def challenge(self, context: RunContext, call: ToolCall) -> str:
        selected = self.resolve(context, call)
        return self.action_digest(
            context, selected.declaration, call, manifest_digest=selected.manifest_digest
        )

    async def _authorization(self, context, tool, call):
        if self.authorizer is not None and not await self.authorizer(
            context.model_copy(deep=True), tool.model_copy(deep=True), call.model_copy(deep=True)
        ):
            raise AgentError("forbidden", "tool resource is not authorized", 403)

    async def _approval(self, context, tool, call, selected):
        if tool.read_only:
            return
        challenge = self.action_digest(
            context, tool, call, manifest_digest=selected.manifest_digest
        )
        if self.approval_checker is None or not await self.approval_checker(
            context.model_copy(deep=True),
            tool.model_copy(deep=True),
            call.model_copy(deep=True),
            challenge,
        ):
            raise AgentError("approval_required", "exact tool action requires review", 409)

    async def _value(self, context, tool, value):
        self.registry.validate_result(tool, value)
        value = copy.deepcopy(value)
        if self.result_policy is not None:
            value = await self.result_policy(
                context.model_copy(deep=True), tool.model_copy(deep=True), value
            )
        result = value if isinstance(value, ToolResult) else ToolResult(value=value)
        self.registry.validate_result(tool, result.value)
        canonical_json(result.model_dump(mode="json"))
        return result.model_copy(deep=True)

    def _reconcile(self, context):
        if self.receipt_inspector is not None:
            self.ledger.reconcile(
                context.identity.owner,
                context.agent_id,
                context.run_id,
                self.receipt_inspector,
            )

    def _matching_operation(self, context, call, selected):
        try:
            previous = self.ledger.operation(self.ledger.key(context, call))
        except AgentError as exc:
            if exc.code == "operation_missing":
                return None
            raise
        if previous["binding"] != context.config_digest or previous[
            "fingerprint"
        ] != self.ledger.fingerprint(
            context, selected.declaration, call, manifest_digest=selected.manifest_digest
        ):
            raise AgentError(
                "operation_conflict", "same action has different parameters or binding", 409
            )
        return previous

    @staticmethod
    def _receipt_phase(tool, current):
        if not tool.read_only:
            if current["phase"] in {"intent", "unknown"}:
                raise NeedsReconciliation()
            if current["phase"] in {"not_applied", "compensated"}:
                raise AgentError(
                    "effect_" + current["phase"], "terminal effect cannot be published", 409
                )
            if current["phase"] == "applied" and (
                current["result"] is None or current["error"] == "effect_applied_result_unavailable"
            ):
                raise AppliedResultUnavailable()

    async def _cached_value(self, context, call, selected, key, *, execution_deadline=None):
        tool = selected.declaration
        current = self.ledger.operation(key)
        deadline = (
            min(context.deadline, execution_deadline)
            if execution_deadline is not None
            else context.deadline
        )
        try:
            async with asyncio.timeout(max(0.0, deadline - time.time())):
                value = await self._value(context, tool, self.ledger.previous_value(current))
                self._execution_deadline(deadline)
                await self._authorization(context, tool, call)
                await self._approval(context, tool, call, selected)
                self._execution_deadline(deadline)
                self._unchanged(context, call, selected)
                self._reconcile(context)
                self._receipt_phase(tool, self.ledger.operation(key))
                self._execution_deadline(deadline)
                return value
        except asyncio.CancelledError:
            self._reconcile(context)
            raise
        except Exception as exc:
            # 截止时间到期或取消可能恰好与补偿动作或未知判定同时发生。
            # 当前事实必须优先于该条目缓存的成功结果。
            self._reconcile(context)
            current = self.ledger.operation(key)
            self._receipt_phase(tool, current)
            if not tool.read_only and current["phase"] == "applied":
                raise AppliedResultUnavailable() from exc
            if isinstance(exc, TimeoutError):
                raise AgentError("tool_timeout", "tool execution deadline expired", 409) from exc
            raise

    async def execute(self, context: RunContext, call: ToolCall) -> ToolObservation:
        # 在 await 策略期间，调用方或模型无法改动已批准的快照。
        call, context = call.model_copy(deep=True), context.model_copy(deep=True)
        self._reconcile(context)
        # 先选定原始证据，再做当前的权限与截止时间检查。
        # 这些检查通过之前，不会创建任何 intent、attempt 或 handler。
        selected = self._resolve(context, call, check_control=False)
        tool = selected.declaration
        self._matching_operation(context, call, selected)
        try:
            self._unchanged(context, call, selected)
            await self._authorization(context, tool, call)
            self._unchanged(context, call, selected)
            if not tool.read_only:
                await self._approval(context, tool, call, selected)
                # 复核可能等待或撤销权限，因此在创建 intent 之前重新检查。
                self._unchanged(context, call, selected)
                await self._authorization(context, tool, call)
                self._unchanged(context, call, selected)
        except asyncio.CancelledError:
            self._reconcile(context)
            raise
        except AgentError as exc:
            self._reconcile(context)
            known = self._matching_operation(context, call, selected)
            if known is not None and known["phase"] == "unknown" and not tool.read_only:
                raise NeedsReconciliation() from exc
            if known is not None and known["phase"] == "applied" and not tool.read_only:
                # 只有完全一致的原始绑定/指纹才能给已生效的动作定性；
                # 上面的冲突情况绝不被吞掉。
                raise AppliedResultUnavailable() from exc
            raise
        bound = self.registry.resolve(context.agent_id, context.config_digest)
        limit = bound.definition.budget.max_tool_calls
        with self.ledger.connection() as db:
            db.execute("BEGIN")
            node = task_tree.metadata_in(db, context.run_id)
            if node is not None:
                limit = node.budget.max_tool_calls
        if selected.manifest_digest is None:
            previous = self.ledger.begin(context, tool, call, limit)
        else:
            previous = self.ledger.begin(
                context, tool, call, limit, manifest_digest=selected.manifest_digest
            )
        key = previous["key"]
        self._reconcile(context)
        previous = self.ledger.operation(key)
        if previous["phase"] in {"applied", "read_success"}:
            value = await self._cached_value(context, call, selected, key)
            return ToolObservation(
                call_id=call.call_id,
                name=call.name,
                value=value.value,
                sources=value.sources,
                succeeded=True,
                operation_key=None if tool.read_only else key,
            )
        if not tool.read_only and previous["phase"] in {"intent", "unknown"}:
            raise NeedsReconciliation()
        if not tool.read_only and previous["phase"] in {"not_applied", "compensated"}:
            return ToolObservation(
                call_id=call.call_id,
                name=call.name,
                error=previous["error"] or "effect_" + previous["phase"],
                operation_key=key,
            )
        policy = tool.execution
        attempts = (
            (policy.max_attempts if policy else 2) if tool.read_only and tool.retry_safe else 1
        )
        backoff = policy.backoff_seconds if policy else 0.0
        handler_started = False
        attempt_token = None
        circuit = None
        ticket = None
        circuit_outcome = "neutral"
        deadline = None
        try:
            async with asyncio.timeout(max(0.0, context.deadline - time.time())):
                async with self.slots:
                    self._unchanged(context, call, selected)
                    current = self.ledger.operation(key)
                    if current["phase"] == "executing" and tool.read_only:
                        raise AgentError(
                            "operation_in_progress", "same tool action is already executing", 409
                        )
                    if current["attempts"] >= attempts:
                        raise AgentError(
                            "tool_attempt_budget", "tool attempt budget exhausted", 409
                        )
                    deadline = current["call_deadline"]
                    if deadline is None:
                        if current["attempts"]:
                            raise AgentError(
                                "tool_deadline_missing",
                                "original tool deadline is unavailable",
                                409,
                            )
                        deadline = min(context.deadline, time.time() + tool.timeout_seconds)
                    if time.time() >= deadline:
                        raise AgentError("tool_timeout", "tool execution deadline expired", 409)
                    circuit = self._circuit(context, selected)
                    ticket = circuit.acquire() if circuit is not None else None
                    async with asyncio.timeout(max(0.0, deadline - time.time())):
                        for attempt in range(attempts):
                            # 下一次认领失败时绝不能沿用上一次的令牌。
                            attempt_token = None
                            handler_started = False
                            self._unchanged(context, call, selected)
                            await self._authorization(context, tool, call)
                            await self._approval(context, tool, call, selected)
                            self._unchanged(context, call, selected)
                            self._execution_deadline(deadline)
                            if selected.attempt_commit is None:
                                attempt_token = self.ledger.claim_attempt(
                                    key,
                                    attempts,
                                    call_deadline=deadline,
                                    evidence=AttemptEvidence(
                                        context=context.model_copy(deep=True),
                                        tool=tool.model_copy(deep=True),
                                        call=call.model_copy(deep=True),
                                        manifest_digest=selected.manifest_digest,
                                    ),
                                )
                            else:
                                attempt_token = self.ledger.claim_attempt(
                                    key,
                                    attempts,
                                    call_deadline=deadline,
                                    commit=selected.attempt_commit,
                                    evidence=AttemptEvidence(
                                        context=context.model_copy(deep=True),
                                        tool=tool.model_copy(deep=True),
                                        call=call.model_copy(deep=True),
                                        manifest_digest=selected.manifest_digest,
                                    ),
                                )
                            self._unchanged(context, call, selected)
                            self._execution_deadline(deadline)
                            handler_context = context.model_copy(
                                update={"operation_key": key, "attempt_token": attempt_token},
                                deep=True,
                            )
                            await self._authorization(handler_context, tool, call)
                            await self._approval(handler_context, tool, call, selected)
                            self._unchanged(context, call, selected)
                            self._execution_deadline(deadline)
                            handler_started = True
                            try:
                                value = await selected.handler(
                                    copy.deepcopy(call.arguments), handler_context
                                )
                            except TransientToolError:
                                # 只有依赖侧的 handler 真正失败才驱动重试。
                                self.ledger.failed(
                                    key,
                                    "tool_transient",
                                    handler_started=True,
                                    attempt_token=attempt_token,
                                )
                                if (
                                    not tool.read_only
                                    or not tool.retry_safe
                                    or self.ledger.operation(key)["attempts"] >= attempts
                                ):
                                    circuit_outcome = "failure"
                                    raise
                                await asyncio.sleep(min(1.0, backoff * (2**attempt)))
                                continue
                            # 将原始证据与异步策略检查分离开冻结。
                            value = copy.deepcopy(value)
                            self._reconcile(context)
                            current = self.ledger.operation(key)
                            if not tool.read_only:
                                if current["phase"] == "unknown":
                                    raise NeedsReconciliation()
                                if current["phase"] in {"not_applied", "compensated"}:
                                    raise AgentError(
                                        "effect_" + current["phase"],
                                        "terminal effect cannot be published as success",
                                        409,
                                    )
                                if current["phase"] == "applied":
                                    value = self.ledger.previous_value(current)
                            self._execution_deadline(deadline)
                            self._unchanged(context, call, selected)
                            projected = await self._value(context, tool, value)
                            self._execution_deadline(deadline)
                            await self._authorization(handler_context, tool, call)
                            await self._approval(handler_context, tool, call, selected)
                            self._unchanged(context, call, selected)
                            self._execution_deadline(deadline)
                            self._reconcile(context)
                            current = self.ledger.operation(key)
                            if not tool.read_only and current["phase"] == "unknown":
                                raise NeedsReconciliation()
                            if not tool.read_only and current["phase"] in {
                                "not_applied",
                                "compensated",
                            }:
                                raise AgentError(
                                    "effect_" + current["phase"],
                                    "terminal effect cannot be published as success",
                                    409,
                                )
                            if (
                                not tool.read_only
                                and current["phase"] == "applied"
                                and canonical_json(self.ledger.previous_value(current))
                                != canonical_json(value)
                            ):
                                # 第一份权威回执可能在策略 await 期间到达。
                                # 此处投影的是它的原始取值，而非 handler 先前给出的候选值。
                                projected = await self._cached_value(
                                    context, call, selected, key, execution_deadline=deadline
                                )
                            self._execution_deadline(deadline)
                            if current["phase"] not in {"applied", "read_success"}:
                                self.ledger.complete(key, value, attempt_token)
                            self._execution_deadline(deadline)
                            circuit_outcome = "success"
                            return ToolObservation(
                                call_id=call.call_id,
                                name=call.name,
                                value=projected.value,
                                sources=projected.sources,
                                succeeded=True,
                                operation_key=None if tool.read_only else key,
                            )
        except asyncio.CancelledError:
            if attempt_token is not None:
                self._reconcile(context)
                self.ledger.failed(
                    key, "cancelled", handler_started=handler_started, attempt_token=attempt_token
                )
            raise
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, AgentError)
                else "tool_timeout"
                if isinstance(exc, TimeoutError)
                else "tool_failure"
            )
            if attempt_token is not None:
                # 回执提交可能已经成功，尽管其确认环节失败了。
                # 可信回执优先于异常分类。
                try:
                    self._reconcile(context)
                    current = self.ledger.operation(key)
                except AgentError as read_error:
                    if not tool.read_only:
                        raise NeedsReconciliation() from read_error
                    raise
                if current["phase"] in {"applied", "read_success"}:
                    if deadline is not None and time.time() >= deadline:
                        if not tool.read_only:
                            raise AppliedResultUnavailable() from exc
                        return ToolObservation(
                            call_id=call.call_id, name=call.name, error="tool_timeout"
                        )
                    value = await self._cached_value(
                        context, call, selected, key, execution_deadline=deadline
                    )
                    if deadline is not None and time.time() >= deadline:
                        if not tool.read_only:
                            raise AppliedResultUnavailable() from exc
                        return ToolObservation(
                            call_id=call.call_id, name=call.name, error="tool_timeout"
                        )
                    circuit_outcome = "success"
                    return ToolObservation(
                        call_id=call.call_id,
                        name=call.name,
                        value=value.value,
                        sources=value.sources,
                        succeeded=True,
                        operation_key=None if tool.read_only else key,
                    )
                self.ledger.failed(
                    key, code, handler_started=handler_started, attempt_token=attempt_token
                )
            current = self.ledger.operation(key)
            if not tool.read_only and current["phase"] in {"intent", "unknown"}:
                raise NeedsReconciliation() from exc
            if isinstance(exc, NeedsReconciliation):
                raise
            return ToolObservation(
                call_id=call.call_id,
                name=call.name,
                error=code,
                operation_key=None if tool.read_only else key,
            )
        finally:
            if circuit is not None and ticket is not None:
                circuit.finish(ticket, circuit_outcome)
        raise AgentError("tool_execution", "tool execution did not produce an observation", 500)
