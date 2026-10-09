"""Trusted task membership and SQL-only ancestor controls.

The runtime owns admission and scheduling. This module never runs a graph, opens a
connection, resolves credentials, or invents an execution result.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Literal

from pydantic import Field, JsonValue, ValidationError, field_serializer

from .configuration import BudgetPolicy, DelegationPolicy
from .contracts import AgentError, BaseContract, Identifier, Identity, RunContext
from .schema import canonical_json, content_digest
from .storage import SqliteContextStore


class DelegationSnapshot(BaseContract):
    binding: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    scopes: frozenset[str]

    @field_serializer("scopes")
    def stable_scopes(self, value):
        return sorted(value)


class TaskMetadata(BaseContract):
    run_id: Identifier
    owner: Identifier
    agent_id: Identifier
    binding: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    root_run_id: Identifier
    parent_run_id: Identifier | None = None
    task_key: Identifier | None = None
    depth: int = Field(0, ge=0, le=8)
    hard_deadline: float
    scope_ceiling: frozenset[str]
    budget: BudgetPolicy
    delegation: DelegationPolicy | None = None
    targets: dict[Identifier, DelegationSnapshot] = Field(default_factory=dict, max_length=32)
    input_sources: list[dict[str, JsonValue]] = Field(
        default_factory=list, max_length=100, exclude_if=lambda value: not value
    )

    @field_serializer("scope_ceiling")
    def stable_scopes(self, value):
        return sorted(value)


class ChildTask(BaseContract):
    id: Identifier
    agent_id: Identifier
    input: JsonValue


class ChildReceiptReference(BaseContract):
    kind: Literal["child_receipt"] = "child_receipt"
    parent_run_id: Identifier
    child_run_id: Identifier
    agent_id: Identifier
    binding: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class TaskInputReference(BaseContract):
    kind: Literal["task_input"] = "task_input"
    run_id: Identifier
    parent_run_id: Identifier
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def _table(db: sqlite3.Connection, name: str) -> bool:
    return (
        db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        is not None
    )


def install(db: sqlite3.Connection) -> None:
    if not db.in_transaction:
        raise AgentError("task_schema", "task installation requires a migration transaction", 503)
    for sql in (
        "CREATE TABLE IF NOT EXISTS agent_task_runs(run_id TEXT PRIMARY KEY,parent_run_id TEXT,root_run_id TEXT NOT NULL,task_key TEXT,depth INTEGER NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,UNIQUE(parent_run_id,task_key))",
        "CREATE TABLE IF NOT EXISTS agent_task_closure(ancestor_run_id TEXT NOT NULL,descendant_run_id TEXT NOT NULL,distance INTEGER NOT NULL,PRIMARY KEY(ancestor_run_id,descendant_run_id))",
        "CREATE INDEX IF NOT EXISTS agent_task_descendants ON agent_task_closure(descendant_run_id,ancestor_run_id)",
        "CREATE TABLE IF NOT EXISTS agent_task_dispatch(run_id TEXT PRIMARY KEY,generation INTEGER NOT NULL,state TEXT NOT NULL,command TEXT,claim_token TEXT,expected_interrupt_id TEXT,expected_interrupt_digest TEXT)",
        "CREATE TABLE IF NOT EXISTS agent_task_waits(parent_run_id TEXT NOT NULL,wait_key TEXT NOT NULL,digest TEXT NOT NULL,children TEXT NOT NULL,phase TEXT NOT NULL,interrupt_id TEXT,checkpoint_id TEXT,PRIMARY KEY(parent_run_id,wait_key))",
        "CREATE TABLE IF NOT EXISTS agent_task_outcomes(run_id TEXT PRIMARY KEY,status TEXT NOT NULL,error TEXT,final_receipt_digest TEXT)",
        "CREATE TABLE IF NOT EXISTS agent_task_consumptions(parent_run_id TEXT NOT NULL,child_run_id TEXT NOT NULL,reference TEXT NOT NULL,PRIMARY KEY(parent_run_id,child_run_id))",
    ):
        db.execute(sql)


def metadata_in(db: sqlite3.Connection, run_id: str) -> TaskMetadata | None:
    row = (
        db.execute("SELECT * FROM agent_task_runs WHERE run_id=?", (run_id,)).fetchone()
        if _table(db, "agent_task_runs")
        else None
    )
    if row is None:
        request = (
            db.execute("SELECT payload FROM requests WHERE run_id=?", (run_id,)).fetchone()
            if _table(db, "requests")
            else None
        )
        if request is not None and json.loads(request[0]).get("_task_tree") is True:
            raise AgentError("task_membership", "original task membership is unavailable", 409)
        if (
            _table(db, "agent_task_closure")
            and db.execute(
                "SELECT 1 FROM agent_task_closure WHERE descendant_run_id=?", (run_id,)
            ).fetchone()
        ):
            raise AgentError("task_membership", "task membership is incomplete", 409)
        return None
    try:
        node = TaskMetadata.model_validate_json(row["payload"])
    except (ValidationError, ValueError) as exc:
        raise AgentError("task_membership", "task metadata is invalid", 409) from exc
    if (
        node.run_id != row["run_id"]
        or node.parent_run_id != row["parent_run_id"]
        or node.root_run_id != row["root_run_id"]
        or node.task_key != row["task_key"]
        or node.depth != row["depth"]
        or content_digest(node.model_dump(mode="json")) != row["digest"]
        or ((node.parent_run_id is None) != (node.depth == 0))
        or ((node.task_key is None) != (node.depth == 0))
    ):
        raise AgentError("task_membership", "task metadata binding differs", 409)
    return node


def ancestry_in(db: sqlite3.Connection, run_id: str) -> list[TaskMetadata]:
    node = metadata_in(db, run_id)
    if node is None:
        return []
    links = db.execute(
        "SELECT ancestor_run_id,distance FROM agent_task_closure WHERE descendant_run_id=? ORDER BY distance DESC",
        (run_id,),
    ).fetchall()
    if len(links) != node.depth + 1:
        raise AgentError("task_membership", "task ancestry is incomplete", 409)
    result: list[TaskMetadata] = []
    for depth, link in enumerate(links):
        ancestor = metadata_in(db, link[0])
        if (
            ancestor is None
            or ancestor.depth != depth
            or ancestor.root_run_id != node.root_run_id
            or ancestor.owner != node.owner
            or link[1] != node.depth - depth
            or ancestor.parent_run_id != (result[-1].run_id if result else None)
            or (not result and ancestor.run_id != node.root_run_id)
        ):
            raise AgentError("task_membership", "task ancestry binding differs", 409)
        result.append(ancestor)
    if result[-1].run_id != run_id:
        raise AgentError("task_membership", "task self membership is missing", 409)
    return result


def identity_digest(identity: Identity) -> str:
    return content_digest(
        {
            "owner": identity.owner,
            "scopes": sorted(identity.scopes),
            "version": identity.permission_version,
        }
    )


def assert_ancestors_in(
    db: sqlite3.Connection,
    context: RunContext,
    *,
    required_scope: str | None = None,
    execution: bool = True,
) -> list[TaskMetadata]:
    if not db.in_transaction:
        raise AgentError("task_transaction", "ancestor guard requires a transaction", 503)
    nodes = ancestry_in(db, context.run_id)
    for node in nodes:
        row = db.execute("SELECT * FROM requests WHERE run_id=?", (node.run_id,)).fetchone()
        control = db.execute(
            "SELECT * FROM agent_run_controls WHERE run_id=?", (node.run_id,)
        ).fetchone()
        dispatch = db.execute(
            "SELECT state FROM agent_task_dispatch WHERE run_id=?", (node.run_id,)
        ).fetchone()
        if (
            row is None
            or control is None
            or dispatch is None
            or (row["owner"], row["agent_id"], row["binding"])
            != (node.owner, node.agent_id, node.binding)
            or json.loads(row["payload"]).get("_task_tree") is not True
            or control["execution_policy"] != "fixed-deadline"
            or row["deadline"] != node.hard_deadline
            or control["original_deadline"] != node.hard_deadline
            or context.identity.owner != node.owner
            or control["identity_version"] != context.identity.permission_version
            or control["identity_digest"] != identity_digest(context.identity)
            or row["cancel"]
            or SqliteContextStore._epoch(db, node.owner, node.agent_id, control["conversation_id"])
            != row["epoch"]
        ):
            raise AgentError("task_control", "ancestor control no longer permits this task", 409)
        if execution and (node.hard_deadline <= time.time() or dispatch[0] == "terminal"):
            raise AgentError("task_inactive", "ancestor task is no longer active", 409)
        if (
            execution
            and db.execute(
                "SELECT 1 FROM agent_task_outcomes WHERE run_id=?", (node.run_id,)
            ).fetchone()
        ):
            raise AgentError("task_inactive", "ancestor task already has a committed outcome", 409)
        if (
            required_scope is not None
            and required_scope not in context.identity.scopes & node.scope_ceiling
        ):
            raise AgentError("tool_scope", "task scope ceiling denies this operation", 403)
        if node.run_id == context.run_id and (
            context.agent_id != node.agent_id
            or context.config_digest != node.binding
            or context.deadline != node.hard_deadline
            or context.epoch != row["epoch"]
            or context.conversation_id != control["conversation_id"]
            or context.scope_ceiling != node.scope_ceiling
        ):
            raise AgentError("task_control", "runtime task context differs from its snapshot", 409)
    return nodes


def model_totals_in(db: sqlite3.Connection, run_id: str) -> dict:
    row = db.execute(
        "SELECT COALESCE(SUM(b.calls),0),COALESCE(SUM(b.tokens),0),COALESCE(MAX(b.overrun),0) FROM agent_task_closure c JOIN agent_model_budgets b ON b.run_id=c.descendant_run_id WHERE c.ancestor_run_id=?",
        (run_id,),
    ).fetchone()
    return {"calls": row[0], "tokens": row[1], "overrun": row[2]}


def check_quota_in(
    db: sqlite3.Connection,
    nodes: list[TaskMetadata],
    kind: Literal["model", "tool", "step"],
    *,
    tokens: int = 0,
) -> None:
    for node in nodes:
        if kind == "model":
            used = model_totals_in(db, node.run_id)
            exhausted = (
                used["overrun"]
                or used["calls"] >= node.budget.max_model_calls
                or used["tokens"] + tokens > node.budget.max_run_tokens
            )
            code = "model_run_budget"
        else:
            table = "agent_tool_operations" if kind == "tool" else "agent_run_steps"
            used_count = db.execute(
                f"SELECT COUNT(*) FROM agent_task_closure c JOIN {table} b ON b.run_id=c.descendant_run_id WHERE c.ancestor_run_id=?",
                (node.run_id,),
            ).fetchone()[0]
            exhausted = used_count >= (
                node.budget.max_tool_calls if kind == "tool" else node.budget.max_steps
            )
            code = "tool_call_budget" if kind == "tool" else "step_budget"
        if exhausted:
            raise AgentError(code, "ancestor task budget exhausted", 409)


def add_in(db: sqlite3.Connection, node: TaskMetadata) -> None:
    """Called once inside the same request admission transaction."""
    if not db.in_transaction:
        raise AgentError("task_transaction", "task admission requires a transaction", 503)
    parent = metadata_in(db, node.parent_run_id) if node.parent_run_id is not None else None
    if parent is not None:
        ancestors = ancestry_in(db, parent.run_id)
        if (
            node.depth != parent.depth + 1
            or node.root_run_id != parent.root_run_id
            or node.owner != parent.owner
            or node.hard_deadline > parent.hard_deadline
            or not node.scope_ceiling <= parent.scope_ceiling
        ):
            raise AgentError("task_admission", "child task widens its ancestor boundary", 409)
        for ancestor in ancestors:
            policy = ancestor.delegation
            if policy is not None and (
                node.depth - ancestor.depth > policy.max_depth
                or db.execute(
                    "SELECT COUNT(*) FROM agent_task_closure WHERE ancestor_run_id=? AND distance>0",
                    (ancestor.run_id,),
                ).fetchone()[0]
                >= policy.max_descendants
            ):
                raise AgentError("task_capacity", "ancestor delegation limit exhausted", 409)
        if parent.delegation is None or node.agent_id not in parent.targets:
            raise AgentError("task_target", "target is not permitted by the parent", 403)
        target = parent.targets[node.agent_id]
        if node.binding != target.binding or not node.scope_ceiling <= target.scopes:
            raise AgentError("task_target", "child target differs from frozen delegation", 409)
        if (
            db.execute(
                "SELECT COUNT(*) FROM agent_task_runs WHERE parent_run_id=?", (parent.run_id,)
            ).fetchone()[0]
            >= parent.delegation.max_children
        ):
            raise AgentError("task_capacity", "parent child limit exhausted", 409)
        if any(
            getattr(node.budget, key) > getattr(parent.budget, key)
            for key in BudgetPolicy.model_fields
        ):
            raise AgentError("task_admission", "child budget widens its parent boundary", 409)
    elif node.parent_run_id is not None or node.root_run_id != node.run_id or node.depth != 0:
        raise AgentError("task_admission", "task root binding differs", 409)
    payload = canonical_json(node.model_dump(mode="json")).decode()
    db.execute(
        "INSERT INTO agent_task_runs VALUES(?,?,?,?,?,?,?)",
        (
            node.run_id,
            node.parent_run_id,
            node.root_run_id,
            node.task_key,
            node.depth,
            payload,
            content_digest(node.model_dump(mode="json")),
        ),
    )
    db.execute("INSERT INTO agent_task_closure VALUES(?,?,0)", (node.run_id, node.run_id))
    if parent is not None:
        db.execute(
            "INSERT INTO agent_task_closure SELECT ancestor_run_id,?,distance+1 FROM agent_task_closure WHERE descendant_run_id=?",
            (node.run_id, parent.run_id),
        )
    db.execute(
        "INSERT INTO agent_task_dispatch VALUES(?,0,'ready',NULL,NULL,NULL,NULL)", (node.run_id,)
    )


def release_waits_in(db: sqlite3.Connection, child_run_id: str) -> None:
    parent = db.execute(
        "SELECT parent_run_id FROM agent_task_runs WHERE run_id=?", (child_run_id,)
    ).fetchone()
    if parent is None or parent[0] is None:
        return
    for wait in db.execute(
        "SELECT * FROM agent_task_waits WHERE parent_run_id=? AND phase='armed'", (parent[0],)
    ).fetchall():
        children = json.loads(wait["children"])
        if not all(
            db.execute("SELECT 1 FROM agent_task_outcomes WHERE run_id=?", (run_id,)).fetchone()
            for run_id in children
        ):
            continue
        command = canonical_json(
            {"kind": "task_wait", "wait_key": wait["wait_key"], "digest": wait["digest"]}
        ).decode()
        db.execute(
            "UPDATE agent_task_waits SET phase='released' WHERE parent_run_id=? AND wait_key=? AND phase='armed'",
            (parent[0], wait["wait_key"]),
        )
        db.execute(
            "UPDATE agent_task_dispatch SET state='ready',generation=generation+1,claim_token=NULL,command=?,expected_interrupt_id=?,expected_interrupt_digest=? WHERE run_id=? AND state='waiting_children'",
            (command, wait["interrupt_id"], wait["digest"], parent[0]),
        )


def outcome_in(
    db: sqlite3.Connection,
    run_id: str,
    status: str,
    error: str | None,
    receipt_digest: str | None = None,
) -> None:
    if metadata_in(db, run_id) is None:
        return
    previous = db.execute("SELECT * FROM agent_task_outcomes WHERE run_id=?", (run_id,)).fetchone()
    if previous is not None and (previous["status"], previous["final_receipt_digest"]) != (
        status,
        receipt_digest,
    ):
        raise AgentError("task_outcome", "committed task outcome changed", 409)
    db.execute(
        "INSERT OR IGNORE INTO agent_task_outcomes VALUES(?,?,?,?)",
        (run_id, status, error, receipt_digest),
    )
    db.execute(
        "UPDATE agent_task_dispatch SET state='terminal',command=NULL,claim_token=NULL WHERE run_id=?",
        (run_id,),
    )
    release_waits_in(db, run_id)
