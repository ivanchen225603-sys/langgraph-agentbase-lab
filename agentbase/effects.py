"""持久的逻辑动作与写入不确定性，独立于具体业务 Schema。"""

import copy
import inspect
import json
import re
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import JsonValue, ValidationError, model_validator

from . import task_tree
from .attempt_transaction import AttemptTransaction
from .configuration import ToolDeclaration
from .contracts import AgentError, BaseContract, RunContext, ToolCall
from .schema import canonical_json, content_digest
from .storage import ConnectionFactory


class NeedsReconciliation(AgentError):
    def __init__(self):
        super().__init__("needs_reconciliation", "tool effect requires trusted reconciliation", 409)


class AppliedResultUnavailable(AgentError):
    def __init__(self):
        super().__init__(
            "effect_applied_result_unavailable",
            "tool effect is applied but its result cannot be published",
            409,
        )


class ActionReceipt(BaseContract):
    """由宿主校验过的当前 effect 事实，而非来自 Tool/模型的主张。

    若记录不存在或仍被某个 intent 持有，则返回 None。Inspector 在返回事实之前会校验
    其原始 receipt 链接、fingerprint 与 attempt token。它只通过传入的事务读取，
    不产生任何副作用。
    """

    phase: Literal["applied", "not_applied", "unknown", "compensated"]
    value: JsonValue = None
    result_available: bool = True

    @model_validator(mode="after")
    def bounded_receipt(self):
        if self.phase != "applied" and self.value is not None:
            raise ValueError("only an applied receipt carries a value")
        if not self.result_available and (self.phase != "applied" or self.value is not None):
            raise ValueError("an unavailable applied result cannot carry a value")
        canonical_json(self.value)
        return self


class ActionReceiptInspector(Protocol):
    def __call__(self, db: sqlite3.Connection, operation: dict) -> ActionReceipt | None: ...


class ReadAttemptRecoveryInspector(Protocol):
    """在独占它的上一个进程停止之后，为读操作 attempt 提供的宿主凭证。"""

    def __call__(self, db: sqlite3.Connection, operation: dict) -> bool: ...


class AttemptEvidence(BaseContract):
    """被放行的 Tool 动作的私有快照，绝不包含可执行代码。"""

    context: RunContext
    tool: ToolDeclaration
    call: ToolCall
    manifest_digest: str | None = None


@dataclass(frozen=True)
class AttemptCommit:
    """在公开 claim 事务中执行的可信同步旁路 DML（sidecar）。

    这是一个 SQL 扩展边界，而非 Python 沙箱。宿主使用原始的 handler release 注册该
    hook，并且绝不在其中执行外部 effect。
    """

    tables: frozenset[str]
    callback: Callable[[AttemptTransaction, dict, AttemptEvidence], None]

    def __post_init__(self):
        if (
            type(self.tables) is not frozenset
            or not 1 <= len(self.tables) <= 16
            or not callable(self.callback)
            or any(
                not isinstance(table, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", table) is None
                or table.lower() == "requests"
                or table.lower().startswith(("agent_", "sqlite_"))
                for table in self.tables
            )
        ):
            raise AgentError("attempt_commit", "invalid trusted sidecar registration", 503)


class SqliteActionLedger:
    """框架的 receipt 并不等于外部的 exactly-once 保证。

    Handler 必须实现其声明的业务键/receipt 契约。若在外部 effect 与本 ledger 的
    receipt 之间发生崩溃，会留下一条持久 intent，从而阻止自动重放。
    安装是一个显式的 migration 步骤。
    """

    def __init__(self, connection: ConnectionFactory):
        self.connection = connection

    @staticmethod
    def install(
        db: sqlite3.Connection,
        *,
        allow_legacy_migration: bool = False,
        allow_deadline_migration: bool = False,
    ):
        if not db.in_transaction:
            raise AgentError("runtime_schema", "action installation requires a transaction", 503)
        db.execute(
            "CREATE TABLE IF NOT EXISTS agent_tool_operations(key TEXT PRIMARY KEY,owner TEXT NOT NULL,agent_id TEXT NOT NULL,run_id TEXT NOT NULL,call_id TEXT NOT NULL,binding TEXT NOT NULL,fingerprint TEXT NOT NULL,read_only INTEGER NOT NULL,phase TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,result TEXT,error TEXT,attempt_token TEXT,effect_applied INTEGER NOT NULL DEFAULT 0,effect_compensated INTEGER NOT NULL DEFAULT 0,call_deadline REAL,UNIQUE(run_id,call_id))"
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(agent_tool_operations)")}
        missing = {"effect_applied", "effect_compensated"} - columns
        if missing:
            if not allow_legacy_migration or len(missing) != 2:
                raise AgentError(
                    "runtime_schema", "action history needs an explicit migration", 503
                )
            for column in ("effect_applied", "effect_compensated"):
                db.execute(
                    f"ALTER TABLE agent_tool_operations ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                )
            db.execute(
                "UPDATE agent_tool_operations SET effect_applied=1 WHERE read_only=0 AND phase IN ('applied','compensated')"
            )
            db.execute(
                "UPDATE agent_tool_operations SET effect_compensated=1 WHERE read_only=0 AND phase='compensated'"
            )
        if "call_deadline" not in columns:
            if not (allow_legacy_migration or allow_deadline_migration):
                raise AgentError("runtime_schema", "action deadline needs a migration", 503)
            # 不为已进入尝试流程的历史动作凭空创造新的时间窗口。
            db.execute("ALTER TABLE agent_tool_operations ADD COLUMN call_deadline REAL")

    @staticmethod
    def key(context: RunContext, call: ToolCall) -> str:
        return content_digest(
            {
                "owner": context.identity.owner,
                "agent": context.agent_id,
                "run": context.run_id,
                "call": call.call_id,
            }
        ).removeprefix("sha256:")

    def begin(
        self,
        context: RunContext,
        tool: ToolDeclaration,
        call: ToolCall,
        limit: int,
        *,
        manifest_digest: str | None = None,
    ) -> dict:
        key = self.key(context, call)
        fingerprint = self.fingerprint(context, tool, call, manifest_digest=manifest_digest)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ancestors = task_tree.assert_ancestors_in(db, context, required_scope=tool.scope)
            if ancestors and limit != ancestors[-1].budget.max_tool_calls:
                raise AgentError("task_budget", "tool budget differs from task snapshot", 409)
            row = db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
            if row is not None:
                if row["binding"] != context.config_digest or row["fingerprint"] != fingerprint:
                    raise AgentError(
                        "operation_conflict", "same action has different parameters or binding", 409
                    )
                return dict(row)
            task_tree.check_quota_in(db, ancestors, "tool")
            count = db.execute(
                "SELECT COUNT(*) FROM agent_tool_operations WHERE run_id=?", (context.run_id,)
            ).fetchone()[0]
            if count >= limit:
                raise AgentError("tool_call_budget", "run logical tool-call budget exhausted", 409)
            db.execute(
                "INSERT INTO agent_tool_operations(key,owner,agent_id,run_id,call_id,binding,fingerprint,read_only,phase) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    context.identity.owner,
                    context.agent_id,
                    context.run_id,
                    call.call_id,
                    context.config_digest,
                    fingerprint,
                    int(tool.read_only),
                    "pending",
                ),
            )
            return dict(
                db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
            )

    @staticmethod
    def fingerprint(
        context: RunContext,
        tool: ToolDeclaration,
        call: ToolCall,
        *,
        manifest_digest: str | None = None,
    ) -> str:
        descriptor: dict[str, Any] = {
            "tool": tool.model_dump(mode="json"),
            "call": call.model_dump(mode="json"),
            "dependencies": context.dependency_results,
        }
        # 省略该字段可保留已配置 Tool 的原始 fingerprint。
        if manifest_digest is not None:
            descriptor["manifest"] = manifest_digest
        return content_digest(descriptor)

    def reconcile(
        self,
        owner: str,
        agent_id: str,
        run_id: str,
        inspector: ActionReceiptInspector,
    ) -> None:
        """在原子更新已有动作之前，先校验整个 Run 的证据。

        不会创建任何 action/attempt。缺失 receipt 并不能证明没有发生 effect。
        可信 inspector 绝不能提交事务、改写 SQL、await、打开另一个连接或调用 Handler。
        它必须能区分"活跃 intent"与"已确认的未知结果"，以便 Handler 保留自身的
        intent 豁免。
        """
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM agent_tool_operations WHERE owner=? AND agent_id=? AND run_id=? AND read_only=0 ORDER BY rowid",
                (owner, agent_id, run_id),
            ).fetchall()
            updates = []
            for record in rows:
                row = dict(record)
                changed = db.total_changes
                # 防止 executescript 之类的 SQLite API 在调用后的契约检查执行前
                # 隐式提交事务。这是可信扩展的守卫，而非针对任意 Python 的沙箱。
                db.set_authorizer(self._receipt_authorizer)
                try:
                    receipt = inspector(db, copy.deepcopy(row))
                except sqlite3.Error as exc:
                    raise AgentError(
                        "effect_receipt_contract", "receipt inspector must only read SQL", 503
                    ) from exc
                finally:
                    db.set_authorizer(None)
                if inspect.isawaitable(receipt):
                    if inspect.iscoroutine(receipt):
                        receipt.close()
                    raise AgentError(
                        "effect_receipt_contract", "receipt inspector must be synchronous", 503
                    )
                if not db.in_transaction or db.total_changes != changed:
                    raise AgentError(
                        "effect_receipt_contract", "receipt inspector must only read SQL", 503
                    )
                if receipt is None:
                    continue
                if not isinstance(receipt, ActionReceipt):
                    raise AgentError(
                        "effect_receipt_contract", "receipt inspector returned an invalid fact", 503
                    )
                try:
                    receipt = ActionReceipt.model_validate(receipt.model_dump(mode="json"))
                except (ValidationError, ValueError, TypeError) as exc:
                    raise AgentError(
                        "effect_receipt_contract", "receipt inspector returned an invalid fact", 503
                    ) from exc
                if row["attempts"] < 1 or not row["attempt_token"] or row["phase"] == "pending":
                    raise AgentError(
                        "effect_receipt_unclaimed", "receipt lacks an original claimed attempt", 409
                    )
                if (
                    (receipt.phase == "not_applied" and row["effect_applied"])
                    or (receipt.phase == "applied" and row["effect_compensated"])
                    or (row["phase"] == "not_applied" and receipt.phase != "not_applied")
                ):
                    raise AgentError(
                        "effect_receipt_conflict", "receipt contradicts durable effect history", 409
                    )
                value = (
                    canonical_json(receipt.value).decode()
                    if receipt.phase == "applied" and receipt.result_available
                    else None
                    if receipt.phase == "not_applied"
                    else row["result"]
                )
                if (
                    row["effect_applied"]
                    and receipt.phase == "applied"
                    and receipt.result_available
                    and row["result"] is not None
                    and row["result"] != value
                ):
                    raise AgentError(
                        "effect_receipt_conflict", "applied receipt value changed", 409
                    )
                updates.append(
                    (
                        row,
                        receipt.phase,
                        value,
                        (None if receipt.result_available else "effect_applied_result_unavailable")
                        if receipt.phase == "applied"
                        else row["error"],
                        int(
                            bool(row["effect_applied"])
                            or receipt.phase in {"applied", "compensated"}
                        ),
                        int(bool(row["effect_compensated"]) or receipt.phase == "compensated"),
                    )
                )
            # 所有事实都在首次更新之前完成校验；任何失败都会回滚整个 Run，
            # 包括后续出现的冲突 receipt。
            for row, phase, value, error, applied, compensated in updates:
                expected = tuple(row[name] for name in self._cas_fields())
                cursor = db.execute(
                    "UPDATE agent_tool_operations SET phase=?,result=?,error=?,effect_applied=?,effect_compensated=? WHERE "
                    + " AND ".join(f"{name} IS ?" for name in self._cas_fields()),
                    (phase, value, error, applied, compensated, *expected),
                )
                if cursor.rowcount != 1:
                    raise AgentError(
                        "effect_receipt_conflict", "action changed during receipt inspection", 409
                    )

    def recover_read_attempts(self, inspector: ReadAttemptRecoveryInspector) -> int:
        """仅在启动阶段执行的状态修复；保留 attempts 与原始 deadline。"""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM agent_tool_operations WHERE read_only=1 AND phase='executing' ORDER BY rowid"
                )
            ]
            accepted = []
            for row in rows:
                if (
                    row["attempts"] < 1
                    or not isinstance(row["attempt_token"], str)
                    or not row["attempt_token"]
                    or type(row["call_deadline"]) not in (int, float)
                    or not 0 < row["call_deadline"] < float("inf")
                ):
                    continue
                changed = db.total_changes
                db.set_authorizer(self._receipt_authorizer)
                try:
                    proof = inspector(db, copy.deepcopy(row))
                    if inspect.isawaitable(proof):
                        if inspect.iscoroutine(proof):
                            proof.close()
                        raise AgentError(
                            "read_recovery_contract", "startup proof must be synchronous", 503
                        )
                    if (
                        type(proof) is not bool
                        or db.total_changes != changed
                        or not db.in_transaction
                    ):
                        raise AgentError(
                            "read_recovery_contract", "startup proof must be read-only bool", 503
                        )
                except sqlite3.Error as exc:
                    raise AgentError(
                        "read_recovery_contract", "startup proof violated SQL contract", 503
                    ) from exc
                finally:
                    db.set_authorizer(None)
                if proof:
                    accepted.append(row)
            for row in accepted:
                cursor = db.execute(
                    "UPDATE agent_tool_operations SET phase='failed',error='process_lost',attempt_token=NULL WHERE "
                    + " AND ".join(f"{name} IS ?" for name in self._cas_fields()),
                    tuple(row[name] for name in self._cas_fields()),
                )
                if cursor.rowcount != 1:
                    raise AgentError(
                        "read_recovery_conflict", "action changed during startup recovery", 409
                    )
            return len(accepted)

    @staticmethod
    def _receipt_authorizer(action, *_args):
        return (
            sqlite3.SQLITE_OK
            if action
            in {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }
            else sqlite3.SQLITE_DENY
        )

    @staticmethod
    def _cas_fields():
        return (
            "key",
            "owner",
            "agent_id",
            "run_id",
            "call_id",
            "binding",
            "fingerprint",
            "read_only",
            "phase",
            "attempts",
            "result",
            "error",
            "attempt_token",
            "effect_applied",
            "effect_compensated",
            "call_deadline",
        )

    def claim_attempt(
        self,
        key: str,
        max_attempts: int,
        *,
        call_deadline: float | None = None,
        evidence: AttemptEvidence | None = None,
        commit: AttemptCommit | None = None,
    ):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
            if row is None:
                raise AgentError("operation_missing", "tool action is unavailable", 409)
            if not row["read_only"] and row["phase"] in {"intent", "unknown"}:
                raise NeedsReconciliation()
            if not row["read_only"] and row["phase"] in {"not_applied", "compensated"}:
                raise AgentError(
                    "effect_" + row["phase"], "terminal effect cannot be replayed", 409
                )
            if row["read_only"] and row["phase"] == "executing":
                raise AgentError(
                    "operation_in_progress", "same tool action is already executing", 409
                )
            if row["phase"] in {"applied", "read_success"}:
                raise AgentError("operation_complete", "tool action already completed", 409)
            if row["attempts"] >= max_attempts:
                raise AgentError("tool_attempt_budget", "tool attempt budget exhausted", 409)
            deadline = row["call_deadline"]
            if call_deadline is not None:
                if not isinstance(call_deadline, (float, int)) or isinstance(call_deadline, bool):
                    raise AgentError("tool_deadline", "invalid tool execution deadline", 409)
                if deadline is None and row["attempts"]:
                    raise AgentError(
                        "tool_deadline_missing", "original tool deadline is unavailable", 409
                    )
                deadline = call_deadline if deadline is None else deadline
                if not time.time() < deadline < float("inf"):
                    raise AgentError("tool_timeout", "tool execution deadline expired", 409)
            managed = task_tree.metadata_in(db, row["run_id"]) is not None
            if managed and evidence is None:
                raise AgentError("task_control", "managed claim requires runtime evidence", 409)
            if evidence is not None:
                self._check_evidence(dict(row), evidence)
                task_tree.assert_ancestors_in(
                    db, evidence.context, required_scope=evidence.tool.scope
                )
            if commit is not None:
                if not isinstance(commit, AttemptCommit) or not isinstance(
                    evidence, AttemptEvidence
                ):
                    raise AgentError(
                        "attempt_commit", "claim requires trusted action evidence", 503
                    )
            token = uuid.uuid4().hex
            db.execute(
                "UPDATE agent_tool_operations SET phase=?,attempts=attempts+1,attempt_token=?,call_deadline=? WHERE key=?",
                ("executing" if row["read_only"] else "intent", token, deadline, key),
            )
            if commit is not None:
                assert evidence is not None
                claimed = dict(
                    db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
                )
                private = evidence.model_copy(
                    update={
                        "context": evidence.context.model_copy(
                            update={"operation_key": key, "attempt_token": token}, deep=True
                        )
                    },
                    deep=True,
                )
                self._commit_sidecar(db, claimed, private, commit)
            if deadline is not None and time.time() >= deadline:
                raise AgentError("tool_timeout", "tool execution deadline expired", 409)
            return token

    def _check_evidence(self, row: dict, evidence: AttemptEvidence):
        context, tool, call = evidence.context, evidence.tool, evidence.call
        if (
            self.key(context, call) != row["key"]
            or context.identity.owner != row["owner"]
            or context.agent_id != row["agent_id"]
            or context.run_id != row["run_id"]
            or context.config_digest != row["binding"]
            or call.call_id != row["call_id"]
            or int(tool.read_only) != row["read_only"]
            or self.fingerprint(context, tool, call, manifest_digest=evidence.manifest_digest)
            != row["fingerprint"]
        ):
            raise AgentError("attempt_commit", "claim evidence differs from original action", 409)

    @staticmethod
    def _commit_sidecar(db, claimed, evidence, commit):
        readable = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_RECURSIVE,
        }
        functions = db.execute("PRAGMA function_list").fetchall()
        custom = {row[0].lower() for row in functions if not row[1]}
        safe_functions = (
            {row[0].lower() for row in functions if row[1]} - custom - {"load_extension"}
        )
        writable = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}

        def authorize(action, table, column, database, trigger):
            if action in readable:
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_FUNCTION
                and column is not None
                and column.lower() in safe_functions
            ):
                return sqlite3.SQLITE_OK
            if action in writable and database == "main" and table in commit.tables:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        db.set_authorizer(authorize)
        transaction = AttemptTransaction(db)
        try:
            result = commit.callback(
                transaction, copy.deepcopy(claimed), evidence.model_copy(deep=True)
            )
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise AgentError("attempt_commit", "attempt hook must be synchronous", 503)
            if result is not None:
                raise AgentError("attempt_commit", "attempt hook must return None", 503)
            transaction._active()
        finally:
            transaction._invalidate()
            db.set_authorizer(None)
        current = db.execute(
            "SELECT * FROM agent_tool_operations WHERE key=?", (claimed["key"],)
        ).fetchone()
        if not db.in_transaction or current is None or dict(current) != claimed:
            raise AgentError("attempt_commit", "attempt hook changed claim authority", 503)

    def complete(self, key: str, value: Any, attempt_token: str):
        payload = canonical_json(value).decode()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
            if (
                row is None
                or row["phase"] not in {"intent", "executing"}
                or row["attempt_token"] != attempt_token
            ):
                raise AgentError(
                    "operation_receipt", "tool receipt does not match an active attempt", 409
                )
            db.execute(
                "UPDATE agent_tool_operations SET phase=?,result=?,error=NULL,effect_applied=MAX(effect_applied,?) WHERE key=? AND attempt_token=?",
                (
                    "read_success" if row["read_only"] else "applied",
                    payload,
                    int(not row["read_only"]),
                    key,
                    attempt_token,
                ),
            )

    def failed(self, key: str, error: str, *, handler_started: bool, attempt_token: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
            if row is None or row["attempt_token"] != attempt_token:
                return
            if row["phase"] in {"applied", "read_success", "compensated", "unknown"}:
                return
            if row["phase"] == "not_applied":
                db.execute(
                    "UPDATE agent_tool_operations SET error=COALESCE(error,?) WHERE key=? AND attempt_token=?",
                    (error, key, attempt_token),
                )
                return
            phase = (
                "failed" if row["read_only"] else "unknown" if handler_started else "not_applied"
            )
            db.execute(
                "UPDATE agent_tool_operations SET phase=?,error=? WHERE key=? AND attempt_token=?",
                (phase, error, key, attempt_token),
            )

    def operation(self, key: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
        if row is None:
            raise AgentError("operation_missing", "tool action is unavailable", 409)
        return dict(row)

    def uncertain(self, run_id: str, *, owned_intent: tuple[str, str] | None = None) -> bool:
        with self.connection() as db:
            return (
                db.execute(
                    "SELECT 1 FROM agent_tool_operations WHERE run_id=? AND read_only=0 AND phase IN ('intent','unknown') AND NOT(key=? AND phase='intent' AND attempt_token IS ?) LIMIT 1",
                    (run_id, *(owned_intent or ("", ""))),
                ).fetchone()
                is not None
            )

    def operations(self, owner: str, run_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM agent_tool_operations WHERE owner=? AND run_id=? ORDER BY rowid",
                (owner, run_id),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def previous_value(row: dict):
        if (
            not row["read_only"]
            and row["phase"] == "applied"
            and (row["result"] is None or row["error"] == "effect_applied_result_unavailable")
        ):
            raise AppliedResultUnavailable()
        return json.loads(row["result"]) if row["result"] is not None else None
