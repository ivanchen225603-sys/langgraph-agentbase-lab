"""SQLite 上下文存储。锁管理与显式 Schema 安装由运行时负责。"""

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from pydantic import Field

from .contracts import AgentError, BaseContract, Identity, MemoryRecord, Message
from .memory_audit import (
    MemoryAuditRecord,
    install_memory_audit,
    read_memory_audit,
    record_memory_change,
)
from .schema import canonical_json

ConnectionFactory = Callable[[], AbstractContextManager[sqlite3.Connection]]


class MemoryLimits(BaseContract):
    """Opt-in retained payload limits; disabled and expired records still consume storage."""

    max_namespace_records: int = Field(1000, ge=1)
    max_namespace_bytes: int = Field(8388608, ge=1)
    max_owner_records: int = Field(10000, ge=1)
    max_owner_bytes: int = Field(67108864, ge=1)


class SqliteDatabase:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            db = sqlite3.connect(self.path, timeout=0.15)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        except sqlite3.Error as exc:
            raise AgentError("storage_unavailable", "context storage is unavailable", 503) from exc
        finally:
            if "db" in locals():
                db.close()


class SqliteContextStore:
    """底层可信存储；所有方法都绑定完整的命名空间。

    连接的提交/回滚/关闭由工厂负责。install 是显式调用，必须在应用自己的迁移
    事务中执行，绝不能从本构造函数里触发。
    """

    def __init__(self, connection: ConnectionFactory, *, memory_limits: MemoryLimits | None = None):
        self.connection = connection
        self.memory_limits = memory_limits

    def _memory_capacity(
        self, db, record: MemoryRecord, payload: str, previous_bytes: int = 0, *, maintenance=False
    ):
        limits = self.memory_limits
        if limits is None:
            return
        if db.execute("PRAGMA encoding").fetchone()[0] != "UTF-8":
            raise AgentError("memory_encoding", "memory limits require a UTF-8 database", 503)
        incoming = len(payload.encode("utf-8"))
        for where, parameters, count_limit, byte_limit in (
            (
                "owner=? AND agent_id=? AND conversation_id=?",
                (record.owner, record.agent_id, record.conversation_id),
                limits.max_namespace_records,
                limits.max_namespace_bytes,
            ),
            ("owner=?", (record.owner,), limits.max_owner_records, limits.max_owner_bytes),
        ):
            count, size = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(length(CAST(payload AS BLOB))),0) "
                "FROM agent_memories WHERE " + where,
                parameters,
            ).fetchone()
            if (previous_bytes == 0 and count + 1 > count_limit) or (
                size - previous_bytes + incoming > byte_limit
                and incoming > previous_bytes
                and not maintenance
            ):
                raise AgentError("memory_quota", "retained memory storage limit exceeded", 409)

    @staticmethod
    def install(db: sqlite3.Connection):
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_context_meta'"
        ).fetchone()
        if exists:
            row = db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()
            if not row or row[0] not in {"1", "2"}:
                raise AgentError("context_schema", "context schema requires a supported migration")
        if not db.in_transaction:
            # Legacy installers may rely on the connection context to commit.
            # Start their migration atomically without taking over commit/rollback.
            db.execute("BEGIN IMMEDIATE")
        statements = (
            "CREATE TABLE IF NOT EXISTS agent_context_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
            "INSERT OR IGNORE INTO agent_context_meta VALUES('schema','1')",
            "CREATE TABLE IF NOT EXISTS agent_conversations(owner TEXT,agent_id TEXT,conversation_id TEXT,epoch INTEGER NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id))",
            "CREATE TABLE IF NOT EXISTS agent_history(owner TEXT,agent_id TEXT,conversation_id TEXT,epoch INTEGER,turn_id TEXT,payload TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id,epoch,turn_id))",
            "CREATE TABLE IF NOT EXISTS agent_memories(owner TEXT,agent_id TEXT,conversation_id TEXT,id TEXT,payload TEXT NOT NULL,version INTEGER NOT NULL,enabled INTEGER NOT NULL,valid_until REAL,updated_at REAL NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id,id))",
        )
        for statement in statements:
            db.execute(statement)
        install_memory_audit(db)
        db.execute("UPDATE agent_context_meta SET value='2' WHERE key='schema'")

    @staticmethod
    def _epoch(db, owner, agent_id, conversation_id) -> int:
        row = db.execute(
            "SELECT epoch FROM agent_conversations WHERE owner=? AND agent_id=? AND conversation_id=?",
            (owner, agent_id, conversation_id),
        ).fetchone()
        return row[0] if row else 0

    def epoch(self, owner: str, agent_id: str, conversation_id: str) -> int:
        with self.connection() as db:
            return self._epoch(db, owner, agent_id, conversation_id)

    def switch(self, owner: str, agent_id: str, conversation_id: str) -> int:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO agent_conversations VALUES(?,?,?,1) ON CONFLICT(owner,agent_id,conversation_id) DO UPDATE SET epoch=agent_conversations.epoch+1",
                (owner, agent_id, conversation_id),
            )
            return self._epoch(db, owner, agent_id, conversation_id)

    def turns(self, owner: str, agent_id: str, conversation_id: str, epoch: int, limit: int):
        with self.connection() as db:
            rows = db.execute(
                "SELECT payload FROM agent_history WHERE owner=? AND agent_id=? AND conversation_id=? AND epoch=? ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (owner, agent_id, conversation_id, epoch, limit),
            ).fetchall()
        return [
            [Message.model_validate(message) for message in json.loads(row[0])]
            for row in reversed(rows)
        ]

    def append_turn(
        self,
        owner: str,
        agent_id: str,
        conversation_id: str,
        epoch: int,
        turn_id: str,
        messages: list[Message],
        *,
        guard: Callable[[sqlite3.Connection], bool],
    ) -> bool:
        """由运行时在最终校验之后调用；guard 使用同一个事务。"""
        payload = canonical_json([message.model_dump(mode="json") for message in messages]).decode()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if self._epoch(db, owner, agent_id, conversation_id) != epoch or not guard(db):
                return False
            old = db.execute(
                "SELECT payload FROM agent_history WHERE owner=? AND agent_id=? AND conversation_id=? AND epoch=? AND turn_id=?",
                (owner, agent_id, conversation_id, epoch, turn_id),
            ).fetchone()
            if old:
                if old[0] != payload:
                    raise AgentError(
                        "history_conflict", "same turn ID has different final projection", 409
                    )
                return True
            db.execute(
                "INSERT INTO agent_history VALUES(?,?,?,?,?,?,?)",
                (owner, agent_id, conversation_id, epoch, turn_id, payload, time.time()),
            )
            return True

    def create_memory(
        self,
        owner: str,
        agent_id: str,
        conversation_id: str,
        content: str,
        *,
        importance: float = 0.5,
        valid_until: float | None = None,
        sources: list[dict] | None = None,
        actor: Identity | None = None,
    ) -> MemoryRecord:
        now = time.time()
        record = MemoryRecord(
            id=uuid.uuid4().hex,
            owner=owner,
            agent_id=agent_id,
            conversation_id=conversation_id,
            content=content,
            importance=importance,
            valid_until=valid_until,
            sources=sources or [],
            created_at=now,
            updated_at=now,
        )
        payload = canonical_json(record.model_dump(mode="json")).decode()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._memory_capacity(db, record, payload)
            db.execute(
                "INSERT INTO agent_memories VALUES(?,?,?,?,?,?,?,?,?)",
                (owner, agent_id, conversation_id, record.id, payload, 1, 1, valid_until, now),
            )
            record_memory_change(db, None, record, actor)
        return record

    def memory(self, owner: str, agent_id: str, conversation_id: str, memory_id: str):
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND id=?",
                (owner, agent_id, conversation_id, memory_id),
            ).fetchone()
        if not row:
            raise AgentError("memory_not_found", "memory is not available", 404)
        return MemoryRecord.model_validate_json(row[0])

    def memories(self, owner: str, agent_id: str, conversation_id: str, limit: int):
        with self.connection() as db:
            rows = db.execute(
                "SELECT payload FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND enabled=1 AND (valid_until IS NULL OR valid_until>?) ORDER BY updated_at DESC LIMIT ?",
                (owner, agent_id, conversation_id, time.time(), limit),
            ).fetchall()
        return [MemoryRecord.model_validate_json(row[0]) for row in rows]

    def update_memory(
        self,
        owner: str,
        agent_id: str,
        conversation_id: str,
        memory_id: str,
        expected_version: int,
        *,
        content: str | None = None,
        enabled: bool | None = None,
        actor: Identity | None = None,
    ) -> MemoryRecord:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND id=?",
                (owner, agent_id, conversation_id, memory_id),
            ).fetchone()
            if not row:
                raise AgentError("memory_not_found", "memory is not available", 404)
            previous = MemoryRecord.model_validate_json(row[0])
            if previous.version != expected_version:
                raise AgentError("memory_conflict", "memory version changed", 409)
            values: dict[str, Any] = previous.model_dump()
            values.update(version=previous.version + 1, updated_at=time.time())
            if content is not None:
                values["content"] = content
            if enabled is not None:
                values["enabled"] = enabled
            record = MemoryRecord.model_validate(values)
            payload = canonical_json(record.model_dump(mode="json")).decode()
            # Capacity must not prevent disabling or shrinking. Version/time and
            # enabled=false can add a few bookkeeping bytes; charge them afterwards.
            old_content_bytes = len(canonical_json(previous.content))
            new_content_bytes = len(canonical_json(record.content))
            maintenance = new_content_bytes <= old_content_bytes and (
                enabled is False or new_content_bytes < old_content_bytes
            )
            self._memory_capacity(
                db, record, payload, len(row[0].encode("utf-8")), maintenance=maintenance
            )
            db.execute(
                "UPDATE agent_memories SET payload=?,version=?,enabled=?,updated_at=? WHERE owner=? AND agent_id=? AND conversation_id=? AND id=? AND version=?",
                (
                    payload,
                    record.version,
                    int(record.enabled),
                    record.updated_at,
                    owner,
                    agent_id,
                    conversation_id,
                    memory_id,
                    expected_version,
                ),
            )
            record_memory_change(db, previous, record, actor)
            return record

    def delete_memory(
        self,
        owner: str,
        agent_id: str,
        conversation_id: str,
        memory_id: str,
        *,
        actor: Identity | None = None,
    ):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND id=?",
                (owner, agent_id, conversation_id, memory_id),
            ).fetchone()
            if row is None:
                raise AgentError("memory_not_found", "memory is not available", 404)
            previous = MemoryRecord.model_validate_json(row[0])
            db.execute(
                "DELETE FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND id=?",
                (owner, agent_id, conversation_id, memory_id),
            )
            record_memory_change(db, previous, None, actor)

    def memory_audit(
        self,
        identity: Identity,
        agent_id: str,
        conversation_id: str,
        memory_id: str,
        *,
        guard: Callable[[sqlite3.Connection], bool],
        after: int = 0,
        limit: int = 100,
    ) -> list[MemoryAuditRecord]:
        """No HTTP endpoint: the trusted host supplies fresh audit-read authorization."""
        with self.connection() as db:
            db.execute("BEGIN")
            return read_memory_audit(
                db,
                identity,
                agent_id,
                conversation_id,
                memory_id,
                guard=guard,
                after=after,
                limit=limit,
            )
