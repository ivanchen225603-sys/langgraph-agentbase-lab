"""持久化且保守的 Run 预留额度，与 provider 侧的计费用量相区分。"""

import sqlite3
import time
import uuid
from typing import Protocol

from . import task_tree
from .configuration import BudgetPolicy
from .contracts import AgentError, RunContext, TokenUsage
from .schema import content_digest
from .storage import ConnectionFactory


class BudgetLedger(Protocol):
    def reserve(self, context: RunContext, policy: BudgetPolicy, input_tokens: int) -> str: ...

    def settle(self, reservation_id: str, usage: TokenUsage | None) -> bool: ...


class SqliteBudgetLedger:
    """失败时保留对应的预留额度；重启绝不会重置某个活跃 Run 的额度。

    该 ledger 仅约束单个 Run，不实现金额计费，也不实现用户级的跨 Run 配额。
    可信运行时（trusted runtime）可以在评审期间暂停其执行时钟；恢复后，稳定的
    Run/policy 绑定与预留额度保持不变。Schema 的安装属于应用层的 migration。
    """

    def __init__(self, connection: ConnectionFactory):
        self.connection = connection

    @staticmethod
    def install(db: sqlite3.Connection):
        db.execute(
            "CREATE TABLE IF NOT EXISTS agent_model_budgets(run_id TEXT PRIMARY KEY,owner TEXT NOT NULL,agent_id TEXT NOT NULL,binding TEXT NOT NULL,policy TEXT NOT NULL,deadline REAL NOT NULL,calls INTEGER NOT NULL,tokens INTEGER NOT NULL,overrun INTEGER NOT NULL DEFAULT 0)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS agent_model_reservations(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,input_estimate INTEGER NOT NULL,output_reserved INTEGER NOT NULL,usage TEXT,settled INTEGER NOT NULL DEFAULT 0)"
        )

    def reserve(self, context: RunContext, policy: BudgetPolicy, input_tokens: int) -> str:
        if type(input_tokens) is not int or input_tokens < 0:
            raise AgentError("token_count", "token estimate must be a nonnegative integer")
        if input_tokens > policy.max_input_tokens:
            raise AgentError("model_input_budget", "complete model input exceeds token budget", 409)
        if time.time() >= context.deadline:
            raise AgentError("deadline", "execution deadline expired", 409)
        signature = content_digest(policy.model_dump(mode="json"))
        reserved = input_tokens + policy.max_output_tokens
        reservation_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ancestors = task_tree.assert_ancestors_in(db, context)
            if ancestors and policy != ancestors[-1].budget:
                raise AgentError("task_budget", "model budget differs from task snapshot", 409)
            task_tree.check_quota_in(db, ancestors, "model", tokens=reserved)
            row = db.execute(
                "SELECT * FROM agent_model_budgets WHERE run_id=?", (context.run_id,)
            ).fetchone()
            if row:
                if (
                    row["owner"],
                    row["agent_id"],
                    row["binding"],
                    row["policy"],
                ) != (
                    context.identity.owner,
                    context.agent_id,
                    context.config_digest,
                    signature,
                ):
                    raise AgentError("budget_binding", "run budget binding changed", 409)
                calls, tokens, overrun = row["calls"], row["tokens"], row["overrun"]
            else:
                calls, tokens, overrun = 0, 0, 0
            if (
                overrun
                or calls >= policy.max_model_calls
                or tokens + reserved > policy.max_run_tokens
            ):
                raise AgentError("model_run_budget", "run model budget exhausted", 409)
            db.execute(
                "INSERT INTO agent_model_budgets VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET calls=excluded.calls,tokens=excluded.tokens",
                (
                    context.run_id,
                    context.identity.owner,
                    context.agent_id,
                    context.config_digest,
                    signature,
                    context.deadline,
                    calls + 1,
                    tokens + reserved,
                    overrun,
                ),
            )
            db.execute(
                "INSERT INTO agent_model_reservations(id,run_id,input_estimate,output_reserved) VALUES(?,?,?,?)",
                (reservation_id, context.run_id, input_tokens, policy.max_output_tokens),
            )
        return reservation_id

    def settle(self, reservation_id: str, usage: TokenUsage | None) -> bool:
        payload = usage.model_dump_json() if usage is not None else None
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM agent_model_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise AgentError("budget_receipt", "model reservation is not available", 409)
            if row["settled"] and row["usage"] != payload:
                raise AgentError("budget_receipt", "model usage receipt changed", 409)
            excess = usage is not None and (
                usage.input_tokens > row["input_estimate"]
                or usage.output_tokens > row["output_reserved"]
            )
            db.execute(
                "UPDATE agent_model_reservations SET usage=?,settled=1 WHERE id=?",
                (payload, reservation_id),
            )
            if excess:
                db.execute(
                    "UPDATE agent_model_budgets SET overrun=1 WHERE run_id=?", (row["run_id"],)
                )
            return not excess

    def snapshot(self, run_id: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN")
            if task_tree.ancestry_in(db, run_id):
                return task_tree.model_totals_in(db, run_id)
            row = db.execute(
                "SELECT calls,tokens,overrun FROM agent_model_budgets WHERE run_id=?", (run_id,)
            ).fetchone()
            return dict(row) if row else {"calls": 0, "tokens": 0, "overrun": 0}
