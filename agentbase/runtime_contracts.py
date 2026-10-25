"""Graph 工厂只提供策略；共享引擎独占所有 Run 生命周期。"""

import inspect
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field, JsonValue

from .contracts import AgentError, BaseContract, Identifier, Identity, Message
from .effects import ActionReceiptInspector, ReadAttemptRecoveryInspector

TERMINAL = frozenset(
    {
        "succeeded",
        "partial",
        "failed",
        "cancelled",
        "timed_out",
        "needs_reconciliation",
        "effect_applied_result_unavailable",
    }
)


class RunOptions(BaseContract):
    conversation_id: Identifier = "default"
    switch_topic: bool = False
    pause_for_review: bool = False
    timeout_seconds: float | None = Field(None, gt=0, le=120)


class EngineSettings(BaseContract):
    workers: int = Field(2, ge=1, le=32)
    queue_size: int = Field(8, ge=1, le=1024)
    max_runs: int = Field(1000, ge=1, le=100000)
    tools_concurrency: int = Field(4, ge=1, le=64)
    cancel_wait_seconds: float = Field(1.0, ge=0, le=5)
    event_count: int = Field(512, ge=2, le=10000)
    event_bytes: int = Field(1048576, ge=65536, le=16777216)
    event_seconds: float = Field(86400.0, gt=0, le=2592000)


@dataclass
class FinalProjection:
    value: JsonValue
    status: Literal["succeeded", "partial", "failed"] = "succeeded"
    messages: list[Message] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    # 受信任的同步本地策略，在最终业务事务内执行。
    local_guard: Callable[[sqlite3.Connection], bool] | None = None
    extra_commit: Callable[[sqlite3.Connection], None] | None = None


@dataclass
class PreparedAdmission:
    """受信任的预处理器为本次确切准入捕获仅含 SQL 的元数据。

    commit 对新准入的请求只执行一次，且在请求事务内部执行。
    它不得 commit/rollback、不得 await、不得另开连接，也不得产生副作用。
    抛出异常将把请求、控制项与话题切换一并回滚。
    """

    payload: dict
    commit: Callable[[sqlite3.Connection, dict], None]


def checked_admission_metadata(callback, db: sqlite3.Connection, row: dict):
    """Run only trusted SQL metadata callbacks, never await inside admission."""
    returned = callback(db, row)
    if inspect.isawaitable(returned):
        if inspect.iscoroutine(returned):
            returned.close()
        raise AgentError("admission_contract", "admission metadata commit must be synchronous", 503)
    if returned is not None:
        raise AgentError("admission_contract", "admission metadata commit must return None", 503)


@dataclass(frozen=True)
class AdmissionFence:
    """Optional trusted host boundary; never accepted from serialized RunOptions.

    reauthorize runs after admission awaits. commit/replay use the same SQL
    transaction as new/existing request admission and may not commit or await.
    """

    reauthorize: Callable[[], Awaitable[object]]
    commit: Callable[[sqlite3.Connection, dict], None]
    replay: Callable[[sqlite3.Connection, dict], None]

    async def refresh(self):
        if await self.reauthorize() is not None:
            raise AgentError(
                "admission_contract", "admission reauthorization must return None", 503
            )


@dataclass
class GraphBinding:
    graph: Any
    initial: Callable[[dict], dict]
    project: Callable[[dict, dict, dict], Awaitable[dict]]
    on_complete: Callable[[dict, dict], Awaitable[FinalProjection]] | None = None
    recovery_reason: Callable[[dict, Identity], str | None] | None = None
    visible_sources: Callable[[dict, dict], list[dict]] = lambda row, values: []
    prepare: Callable[[Identity, JsonValue], Awaitable[dict | PreparedAdmission]] | None = None
    terminal_node: str = "finish"
    terminal_values: Callable[[str, str], dict] = lambda status, code: {
        "status": status,
        "error": code,
    }
    execution_policy: Literal["fixed-deadline", "pause-review"] = "pause-review"
    recursion_limit: int = 256
    uncertain: Callable[[str], bool] | None = None
    # 原始宿主策略可区分当前确切归属的意图。
    effect_guard: Callable[[dict, tuple[str, str] | None], bool] | None = None
    action_approval: Callable[..., Awaitable[bool]] | None = None
    execution_scope: Callable[[dict], Any] | None = None
    # 受信任的兼容性策略；它们绝不接受来自业务输入的权限。
    client_digest: Callable[[JsonValue, RunOptions], str] | None = None
    review_descriptor: Callable[[dict, Any], list[dict]] | None = None
    resume_value: Callable[[dict, dict], JsonValue] | None = None
    applied_effects: Callable[[dict], list[dict]] | None = None
    receipt_inspector: ActionReceiptInspector | None = None
    read_attempt_recovery: ReadAttemptRecoveryInspector | None = None
    exception_code: Callable[[Exception], str | None] | None = None
    validate_snapshot: Callable[[dict, Any], None] | None = None
    control_fields: dict[str, str] = field(
        default_factory=lambda: {
            "run_id": "run_id",
            "agent_id": "agent_id",
            "binding": "binding",
        }
    )
    # 最后一次异步可见性检查之后的仅 SQL 发布凭据。
    projection_guard: Callable[[sqlite3.Connection, dict, dict], bool] | None = None


IdentityResolver = Callable[[str], Awaitable[Identity]]
AgentAuthorizer = Callable[[Identity, str], Awaitable[bool]]
