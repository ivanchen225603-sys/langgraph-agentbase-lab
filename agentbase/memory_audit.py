"""Content-free memory mutation evidence; never an authorization or restore token."""

import inspect
import json
import sqlite3
import time
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from .contracts import AgentError, BaseContract, Identifier, Identity, MemoryRecord
from .schema import canonical_json, content_digest

AuditDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
MemoryOperation = Literal["create", "update", "disable", "enable", "delete"]


class MemoryAuditChange(BaseContract):
    owner: Identifier
    agent_id: Identifier
    conversation_id: str = Field(max_length=64)
    memory_id: Identifier
    operation: MemoryOperation
    actor_owner: Identifier
    actor_context: Literal["host-identity", "legacy-owner"]
    actor_permission_version: int | None = Field(default=None, ge=1)
    actor_identity_digest: AuditDigest | None = None
    old_version: int | None = Field(default=None, ge=1)
    new_version: int | None = Field(default=None, ge=1)
    before_digest: AuditDigest | None = None
    after_digest: AuditDigest | None = None
    source_fingerprints: list[AuditDigest] = Field(default_factory=list, max_length=200)
    occurred_at: float


class MemoryAuditRecord(MemoryAuditChange):
    sequence: int = Field(ge=1)


def install_memory_audit(db: sqlite3.Connection):
    if not db.in_transaction:
        raise AgentError("context_schema", "memory audit installation requires a transaction")
    db.execute(
        "CREATE TABLE IF NOT EXISTS agent_memory_audit("
        "sequence INTEGER PRIMARY KEY AUTOINCREMENT,owner TEXT NOT NULL,"
        "agent_id TEXT NOT NULL,conversation_id TEXT NOT NULL,memory_id TEXT NOT NULL,"
        "payload TEXT NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS agent_memory_audit_namespace ON agent_memory_audit"
        "(owner,agent_id,conversation_id,memory_id,sequence)"
    )


def record_memory_change(
    db: sqlite3.Connection,
    before: MemoryRecord | None,
    after: MemoryRecord | None,
    actor: Identity | None,
):
    if not db.in_transaction:
        raise AgentError("memory_audit", "memory audit requires the mutation transaction", 503)
    record = after if after is not None else before
    assert record is not None
    if actor is not None and (not isinstance(actor, Identity) or actor.owner != record.owner):
        raise AgentError("memory_actor", "memory actor differs from namespace owner", 403)
    operation: MemoryOperation = "update"
    if before is None:
        operation = "create"
    elif after is None:
        operation = "delete"
    elif before.enabled != after.enabled:
        operation = "enable" if after.enabled else "disable"
    change = MemoryAuditChange(
        owner=record.owner,
        agent_id=record.agent_id,
        conversation_id=record.conversation_id,
        memory_id=record.id,
        operation=operation,
        actor_owner=record.owner,
        actor_context="host-identity" if actor is not None else "legacy-owner",
        actor_permission_version=actor.permission_version if actor is not None else None,
        actor_identity_digest=(
            content_digest(actor.model_dump(mode="json") | {"scopes": sorted(actor.scopes)})
            if actor
            else None
        ),
        old_version=before.version if before is not None else None,
        new_version=after.version if after is not None else None,
        before_digest=content_digest(before.model_dump(mode="json")) if before else None,
        after_digest=content_digest(after.model_dump(mode="json")) if after else None,
        source_fingerprints=sorted(
            {content_digest(source) for item in (before, after) if item for source in item.sources}
        ),
        occurred_at=time.time(),
    )
    db.execute(
        "INSERT INTO agent_memory_audit(owner,agent_id,conversation_id,memory_id,payload) "
        "VALUES(?,?,?,?,?)",
        (
            record.owner,
            record.agent_id,
            record.conversation_id,
            record.id,
            canonical_json(change.model_dump(mode="json")).decode(),
        ),
    )


def read_memory_audit(
    db: sqlite3.Connection,
    identity: Identity,
    agent_id: str,
    conversation_id: str,
    memory_id: str,
    *,
    guard: Callable[[sqlite3.Connection], bool],
    after: int = 0,
    limit: int = 100,
) -> list[MemoryAuditRecord]:
    """Trusted host must bind guard to current authentication and audit-read policy."""
    if not isinstance(identity, Identity):
        raise AgentError("memory_audit_forbidden", "authenticated audit identity required", 403)
    if (
        type(after) is not int
        or not 0 <= after <= (1 << 63) - 1
        or type(limit) is not int
        or not 1 <= limit <= 200
    ):
        raise AgentError("memory_audit_query", "invalid memory audit page")

    def authorize():
        result = guard(db)
        if result is not True:
            if inspect.iscoroutine(result):
                result.close()
            raise AgentError("memory_audit_forbidden", "memory audit access denied", 403)

    authorize()
    rows = db.execute(
        "SELECT sequence,payload FROM agent_memory_audit WHERE owner=? AND agent_id=? "
        "AND conversation_id=? AND memory_id=? AND sequence>? ORDER BY sequence LIMIT ?",
        (identity.owner, agent_id, conversation_id, memory_id, after, limit),
    ).fetchall()
    try:
        result = [
            MemoryAuditRecord.model_validate(
                json.loads(row["payload"]) | {"sequence": row["sequence"]}
            )
            for row in rows
        ]
    except (ValueError, TypeError, ValidationError) as exc:
        raise AgentError("memory_audit_storage", "memory audit record is invalid", 503) from exc
    if any(
        (record.owner, record.agent_id, record.conversation_id, record.memory_id)
        != (identity.owner, agent_id, conversation_id, memory_id)
        for record in result
    ):
        raise AgentError("memory_audit_storage", "memory audit namespace is invalid", 503)
    authorize()
    return result
