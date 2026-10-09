import json
import time
import uuid

from .embedding import relevance
from .errors import LabError
from .models import MemoryInput, MemoryPatch, MemoryRecord
from .persistence import encode


class MemoryService:
    def __init__(self, store):
        self.store = store

    def list(self, owner, limit=100):
        with self.store.connection() as db:
            return [
                json.loads(x[0])
                for x in db.execute(
                    "SELECT payload FROM memories WHERE owner=? ORDER BY rowid DESC LIMIT ?",
                    (owner, limit),
                )
            ]

    def create(self, owner, data: MemoryInput):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            records = [
                json.loads(x[0])
                for x in db.execute("SELECT payload FROM memories WHERE owner=?", (owner,))
            ]
            for x in records:
                if x["content"] == data.content and x["kind"] == data.kind and x["enabled"]:
                    return x
            if len(records) >= 500:
                raise LabError("memory_limit", "owner memory limit reached", 429)
            now = time.time()
            record = MemoryRecord(
                **data.model_dump(),
                memory_id=uuid.uuid4().hex,
                owner=owner,
                created_at=now,
                updated_at=now,
            ).model_dump()
            db.execute(
                "INSERT INTO memories VALUES(?,?,?)", (record["memory_id"], owner, encode(record))
            )
            self.store.bump("memory", db)
            self.store.audit(db, owner, "memory.create", record["memory_id"])
            return record

    def patch(self, owner, memory_id, data: MemoryPatch):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM memories WHERE owner=? AND id=?", (owner, memory_id)
            ).fetchone()
            if not row:
                raise LabError("not_found", "memory not found", 404)
            record = json.loads(row[0])
            if record["version"] != data.expected_version:
                raise LabError("version_conflict", "memory changed; reload before correcting", 409)
            changes = data.model_dump(
                exclude_unset=True, exclude={"schema_version", "expected_version"}
            )
            record.update(changes)
            record.update(version=record["version"] + 1, updated_at=time.time())
            record = MemoryRecord.model_validate(record).model_dump()
            db.execute(
                "UPDATE memories SET payload=? WHERE owner=? AND id=?",
                (encode(record), owner, memory_id),
            )
            self.store.bump("memory", db)
            self.store.audit(db, owner, "memory.correct", memory_id)
            return record

    def delete(self, owner, memory_id):
        with self.store.connection() as db:
            cursor = db.execute("DELETE FROM memories WHERE owner=? AND id=?", (owner, memory_id))
            if cursor.rowcount == 0:
                raise LabError("not_found", "memory not found", 404)
            self.store.bump("memory", db)
            self.store.audit(db, owner, "memory.delete", memory_id)

    def search(self, owner, question, top_k=5):
        now = time.time()
        valid = [
            r
            for r in self.list(owner, 500)
            if r["enabled"] and (r["valid_until"] is None or r["valid_until"] > now)
        ]
        scored = []
        for record in valid:
            score = relevance(question, record["content"])
            if score < 0.12:
                continue
            recency = 1 / (1 + max(0, now - record["updated_at"]) / 86400)
            scored.append(
                {
                    **record,
                    "relevance": score,
                    "score": 0.7 * score + 0.2 * record["importance"] + 0.1 * recency,
                }
            )
        return sorted(scored, key=lambda r: (-r["score"], r["memory_id"]))[:top_k]
