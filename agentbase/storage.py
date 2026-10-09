"""SQLite 上下文存储。锁管理与显式 Schema 安装由运行时负责。"""

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from .contracts import AgentError, MemoryRecord, Message
from .schema import canonical_json

ConnectionFactory = Callable[[], AbstractContextManager[sqlite3.Connection]]


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

    def __init__(self, connection: ConnectionFactory):
        self.connection = connection

    @staticmethod
    def install(db: sqlite3.Connection):
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_context_meta'"
        ).fetchone()
        if exists:
            row = db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()
            if not row or row[0] != "1":
                raise AgentError("context_schema", "context schema requires a supported migration")
        statements = (
            "CREATE TABLE IF NOT EXISTS agent_context_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
            "INSERT OR IGNORE INTO agent_context_meta VALUES('schema','1')",
            "CREATE TABLE IF NOT EXISTS agent_conversations(owner TEXT,agent_id TEXT,conversation_id TEXT,epoch INTEGER NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id))",
            "CREATE TABLE IF NOT EXISTS agent_history(owner TEXT,agent_id TEXT,conversation_id TEXT,epoch INTEGER,turn_id TEXT,payload TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id,epoch,turn_id))",
            "CREATE TABLE IF NOT EXISTS agent_memories(owner TEXT,agent_id TEXT,conversation_id TEXT,id TEXT,payload TEXT NOT NULL,version INTEGER NOT NULL,enabled INTEGER NOT NULL,valid_until REAL,updated_at REAL NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id,id))",
        )
        for statement in statements:
            db.execute(statement)

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
            db.execute(
                "INSERT INTO agent_memories VALUES(?,?,?,?,?,?,?,?,?)",
                (owner, agent_id, conversation_id, record.id, payload, 1, 1, valid_until, now),
            )
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
            return record

    def delete_memory(self, owner: str, agent_id: str, conversation_id: str, memory_id: str):
        with self.connection() as db:
            cursor = db.execute(
                "DELETE FROM agent_memories WHERE owner=? AND agent_id=? AND conversation_id=? AND id=?",
                (owner, agent_id, conversation_id, memory_id),
            )
            if not cursor.rowcount:
                raise AgentError("memory_not_found", "memory is not available", 404)
