"""单一 Run ledger、带命名空间的控制项、审批凭据与有界事件流。"""

import copy
import inspect
import json
import sqlite3
import time
import uuid
from collections.abc import Callable

from . import planning, task_tree
from .artifacts import SqliteArtifactStore
from .budgets import SqliteBudgetLedger
from .contracts import AgentError, Identity, RunContext, RunEvent
from .effects import SqliteActionLedger
from .runtime_contracts import EngineSettings, FinalProjection, RunOptions
from .schema import canonical_json, content_digest
from .storage import ConnectionFactory, SqliteContextStore


class SqliteRunStore:
    def __init__(self, connection: ConnectionFactory, settings: EngineSettings):
        self.connection, self.settings = connection, settings

    @staticmethod
    def install(db: sqlite3.Connection, *, legacy_agent_id: str | None = None):
        """调用方必须自行持有迁移事务与进程独占锁。"""
        if not db.in_transaction:
            raise AgentError("runtime_schema", "installation requires a migration transaction", 503)
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_runtime_meta'"
        ).fetchone()
        old_version = None
        if exists:
            row = db.execute("SELECT value FROM agent_runtime_meta WHERE key='schema'").fetchone()
            if row is None or row[0] not in {"1", "2", "3", "4", "5", "6", "7", "8"}:
                raise AgentError(
                    "runtime_schema", "runtime schema requires a supported migration", 503
                )
            old_version = row[0]
        existing = db.execute("PRAGMA table_info(requests)").fetchall()
        columns = {row[1] for row in existing}
        original = {
            "run_id",
            "owner",
            "key",
            "digest",
            "payload",
            "thread",
            "deadline",
            "epoch",
            "cancel",
        }
        if columns and columns != original | {"agent_id", "binding"}:
            if columns != original or legacy_agent_id is None:
                raise AgentError(
                    "runtime_schema", "legacy run storage needs an explicit compatible adapter", 503
                )
            db.execute("ALTER TABLE requests RENAME TO agent_requests_migration_v0")
        db.execute(
            "CREATE TABLE IF NOT EXISTS agent_runtime_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
        )
        db.execute("INSERT OR IGNORE INTO agent_runtime_meta VALUES('schema','8')")
        db.execute(
            "CREATE TABLE IF NOT EXISTS requests(run_id TEXT PRIMARY KEY,owner TEXT NOT NULL,key TEXT NOT NULL,digest TEXT NOT NULL,payload TEXT NOT NULL,thread TEXT NOT NULL,deadline REAL NOT NULL,epoch INTEGER NOT NULL,cancel INTEGER NOT NULL DEFAULT 0,agent_id TEXT NOT NULL,binding TEXT NOT NULL,UNIQUE(owner,agent_id,key))"
        )
        if columns == original:
            db.execute(
                "INSERT INTO requests SELECT run_id,owner,key,digest,payload,thread,deadline,epoch,cancel,?,'' FROM agent_requests_migration_v0",
                (legacy_agent_id,),
            )
            db.execute("DROP TABLE agent_requests_migration_v0")
        for statement in (
            "CREATE TABLE IF NOT EXISTS agent_run_controls(run_id TEXT PRIMARY KEY,identity_version INTEGER NOT NULL,conversation_id TEXT NOT NULL,execution_policy TEXT NOT NULL,original_deadline REAL NOT NULL,options TEXT NOT NULL,client_digest TEXT NOT NULL,review_id TEXT)",
            "CREATE TABLE IF NOT EXISTS agent_reviews(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,owner TEXT NOT NULL,digest TEXT NOT NULL,preview TEXT NOT NULL,binding TEXT NOT NULL,epoch INTEGER NOT NULL,paused_at REAL NOT NULL,expires_at REAL NOT NULL,remaining REAL NOT NULL,approved_at REAL,UNIQUE(run_id,digest))",
            "CREATE TABLE IF NOT EXISTS agent_review_visibility(review_id TEXT PRIMARY KEY,visible_at REAL NOT NULL,remaining REAL NOT NULL)",
            "CREATE TABLE IF NOT EXISTS agent_run_steps(run_id TEXT,step_key TEXT,binding TEXT NOT NULL,PRIMARY KEY(run_id,step_key))",
            "CREATE TABLE IF NOT EXISTS agent_final_receipts(run_id TEXT PRIMARY KEY,owner TEXT NOT NULL,binding TEXT NOT NULL,payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS agent_event_heads(run_id TEXT PRIMARY KEY,next_seq INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS agent_run_events(run_id TEXT,seq INTEGER,created_at REAL NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(run_id,seq))",
        ):
            db.execute(statement)
        SqliteContextStore.install(db)
        SqliteArtifactStore.install(db)
        task_tree.install(db)
        planning.install(db)
        SqliteBudgetLedger.install(db)
        SqliteActionLedger.install(
            db,
            allow_legacy_migration=old_version in {None, "1", "2"},
            allow_deadline_migration=old_version == "3",
        )
        if "identity_digest" not in {
            column[1] for column in db.execute("PRAGMA table_info(agent_run_controls)")
        }:
            db.execute("ALTER TABLE agent_run_controls ADD COLUMN identity_digest TEXT")
            for record in db.execute("SELECT run_id,payload FROM requests").fetchall():
                fingerprint = json.loads(record[1]).get("_identity_digest")
                db.execute(
                    "UPDATE agent_run_controls SET identity_digest=? WHERE run_id=?",
                    (fingerprint, record[0]),
                )
        db.execute("UPDATE agent_runtime_meta SET value='8' WHERE key='schema'")

    def lookup(self, owner: str, agent_id: str, key: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE owner=? AND agent_id=? AND key=?",
                (owner, agent_id, key),
            ).fetchone()
        return dict(row) if row is not None else None

    def request(self, owner: str, run_id: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE owner=? AND run_id=?", (owner, run_id)
            ).fetchone()
        if row is None:
            raise AgentError("not_found", "run not found", 404)
        return dict(row)

    def internal_request(self, run_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM requests WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise AgentError("not_found", "run not found", 404)
        return dict(row)

    def control(self, run_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def epoch(db: sqlite3.Connection, owner: str, agent_id: str, conversation_id: str) -> int:
        return SqliteContextStore._epoch(db, owner, agent_id, conversation_id)

    def admit(self, *args, **kwargs):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            return self.admit_in(db, *args, **kwargs)

    def admit_in(
        self,
        db: sqlite3.Connection,
        identity: Identity,
        agent_id: str,
        key: str,
        client_digest: str,
        payload: dict,
        binding: str,
        options: RunOptions,
        deadline: float,
        execution_policy: str,
        *,
        admission_commit: Callable[[sqlite3.Connection, dict], None] | None = None,
    ):
        if not db.in_transaction:
            raise AgentError("admission_contract", "request admission requires a transaction", 503)
        encoded = canonical_json(payload).decode()
        old = db.execute(
            "SELECT * FROM requests WHERE owner=? AND agent_id=? AND key=?",
            (identity.owner, agent_id, key),
        ).fetchone()
        if old is not None:
            control = db.execute(
                "SELECT client_digest FROM agent_run_controls WHERE run_id=?", (old["run_id"],)
            ).fetchone()
            previous_digest = (
                control[0] if control else json.loads(old["payload"]).get("client_digest")
            )
            if previous_digest != client_digest:
                raise AgentError(
                    "idempotency_conflict", "same key has different request content", 409
                )
            return dict(old), False, []
        if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= self.settings.max_runs:
            raise AgentError(
                "retention_capacity", "run ledger full; use an approved retention process", 429
            )
        epoch = self.epoch(db, identity.owner, agent_id, options.conversation_id)
        victims = []
        if options.switch_topic:
            epoch += 1
            db.execute(
                "INSERT INTO agent_conversations VALUES(?,?,?,?) ON CONFLICT(owner,agent_id,conversation_id) DO UPDATE SET epoch=excluded.epoch",
                (identity.owner, agent_id, options.conversation_id, epoch),
            )
            victims = [
                row[0]
                for row in db.execute(
                    "SELECT r.run_id FROM requests r JOIN agent_run_controls c USING(run_id) WHERE r.owner=? AND r.agent_id=? AND c.conversation_id=? AND r.epoch<? AND r.cancel=0 AND NOT EXISTS(SELECT 1 FROM agent_final_receipts f WHERE f.run_id=r.run_id)",
                    (identity.owner, agent_id, options.conversation_id, epoch),
                )
            ]
            db.executemany(
                "UPDATE requests SET cancel=1 WHERE run_id=?", [(victim,) for victim in victims]
            )
        run_id, thread = uuid.uuid4().hex, uuid.uuid4().hex
        db.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                identity.owner,
                key,
                client_digest,
                encoded,
                thread,
                deadline,
                epoch,
                0,
                agent_id,
                binding,
            ),
        )
        db.execute(
            "INSERT INTO agent_run_controls(run_id,identity_version,conversation_id,execution_policy,original_deadline,options,client_digest,review_id,identity_digest) VALUES(?,?,?,?,?,?,?,NULL,?)",
            (
                run_id,
                identity.permission_version,
                options.conversation_id,
                execution_policy,
                deadline,
                options.model_dump_json(),
                client_digest,
                payload.get("_identity_digest"),
            ),
        )
        row = dict(db.execute("SELECT * FROM requests WHERE run_id=?", (run_id,)).fetchone())
        if admission_commit is not None:
            returned = admission_commit(db, copy.deepcopy(row))
            if inspect.isawaitable(returned):
                if inspect.iscoroutine(returned):
                    returned.close()
                raise AgentError(
                    "admission_contract", "admission metadata commit must be synchronous", 503
                )
            if returned is not None:
                raise AgentError(
                    "admission_contract", "admission metadata commit must return None", 503
                )
        return row, True, victims

    def cancel(self, owner: str, run_id: str) -> bool:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM agent_final_receipts WHERE owner=? AND run_id=?", (owner, run_id)
            ).fetchone():
                return False
            cursor = db.execute(
                "UPDATE requests SET cancel=1 WHERE owner=? AND run_id=?", (owner, run_id)
            )
            if cursor.rowcount != 1:
                raise AgentError("not_found", "run not found", 404)
            return True

    def reserve_step(
        self, row: dict, step_key: str, limit: int, *, context: RunContext | None = None
    ):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            node = task_tree.metadata_in(db, row["run_id"])
            if context is not None and (
                row["run_id"],
                row["owner"],
                row["agent_id"],
                row["binding"],
                row["epoch"],
                row["deadline"],
            ) != (
                context.run_id,
                context.identity.owner,
                context.agent_id,
                context.config_digest,
                context.epoch,
                context.deadline,
            ):
                raise AgentError("task_control", "step context differs from its request", 409)
            if node is not None and context is None:
                raise AgentError("task_control", "managed steps require runtime context", 409)
            ancestors = task_tree.assert_ancestors_in(db, context) if context is not None else []
            if node is not None and limit != node.budget.max_steps:
                raise AgentError("task_budget", "step budget differs from task snapshot", 409)
            prior = db.execute(
                "SELECT binding FROM agent_run_steps WHERE run_id=? AND step_key=?",
                (row["run_id"], step_key),
            ).fetchone()
            if prior:
                if prior[0] != row["binding"]:
                    raise AgentError("binding_changed", "logical step binding changed", 409)
                return
            task_tree.check_quota_in(db, ancestors, "step")
            if (
                db.execute(
                    "SELECT COUNT(*) FROM agent_run_steps WHERE run_id=?", (row["run_id"],)
                ).fetchone()[0]
                >= limit
            ):
                raise AgentError("step_budget", "run logical step budget exhausted", 409)
            db.execute(
                "INSERT INTO agent_run_steps VALUES(?,?,?)",
                (row["run_id"], step_key, row["binding"]),
            )

    def prepare_review(self, row: dict, digest: str, preview: dict, seconds: float) -> dict:
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            if (
                current is None
                or current["cancel"]
                or current["deadline"] <= now
                or current["binding"] != row["binding"]
            ):
                raise AgentError("review_control", "run cannot prepare a new review", 409)
            old = db.execute(
                "SELECT * FROM agent_reviews WHERE run_id=? AND digest=?", (row["run_id"], digest)
            ).fetchone()
            if old:
                if old["preview"] != canonical_json(preview).decode():
                    raise AgentError("review_conflict", "review preview changed", 409)
                return dict(old)
            review_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO agent_reviews VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    review_id,
                    row["run_id"],
                    row["owner"],
                    digest,
                    canonical_json(preview).decode(),
                    row["binding"],
                    row["epoch"],
                    now,
                    now + seconds,
                    max(0.0, current["deadline"] - now),
                ),
            )
            db.execute(
                "UPDATE agent_run_controls SET review_id=? WHERE run_id=?",
                (review_id, row["run_id"]),
            )
            return dict(
                db.execute("SELECT * FROM agent_reviews WHERE id=?", (review_id,)).fetchone()
            )

    def review(self, owner: str, run_id: str, review_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM agent_reviews WHERE owner=? AND run_id=? AND id=?",
                (owner, run_id, review_id),
            ).fetchone()
        return dict(row) if row else None

    def accept_review(
        self, owner: str, run_id: str, review_id: str, digest: str, *, resume_commit=None
    ) -> dict:
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM requests WHERE owner=? AND run_id=?", (owner, run_id)
            ).fetchone()
            control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (run_id,)
            ).fetchone()
            review = db.execute(
                "SELECT * FROM agent_reviews WHERE owner=? AND run_id=? AND id=?",
                (owner, run_id, review_id),
            ).fetchone()
            visibility = db.execute(
                "SELECT * FROM agent_review_visibility WHERE review_id=?", (review_id,)
            ).fetchone()
            if (
                row is None
                or control is None
                or review is None
                or visibility is None
                or review["digest"] != digest
                or review["binding"] != row["binding"]
                or review["epoch"] != row["epoch"]
                or control["review_id"] != review_id
                or row["cancel"]
                or self.epoch(db, owner, row["agent_id"], control["conversation_id"])
                != row["epoch"]
            ):
                raise AgentError("review_mismatch", "review does not match the current run", 409)
            if review["expires_at"] <= now:
                raise AgentError("review_expired", "review validity expired", 409)
            if review["approved_at"] is None:
                if control["execution_policy"] == "fixed-deadline" and row["deadline"] <= now:
                    raise AgentError("deadline", "original execution deadline expired", 409)
                db.execute("UPDATE agent_reviews SET approved_at=? WHERE id=?", (now, review_id))
                if control["execution_policy"] == "pause-review":
                    db.execute(
                        "UPDATE requests SET deadline=? WHERE run_id=?",
                        (now + visibility["remaining"], run_id),
                    )
            elif row["deadline"] <= now:
                raise AgentError("deadline", "approved execution deadline expired", 409)
            if resume_commit is not None:
                returned = resume_commit(db)
                if returned is not None:
                    if inspect.iscoroutine(returned):
                        returned.close()
                    raise AgentError("review_contract", "resume commit must be synchronous", 503)
            return dict(
                db.execute("SELECT * FROM agent_reviews WHERE id=?", (review_id,)).fetchone()
            )

    def visibility(self, review_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM agent_review_visibility WHERE review_id=?", (review_id,)
            ).fetchone()
        return dict(row) if row else None

    def seal_review(self, row: dict, review_id: str, digest: str) -> dict:
        """受信任的 Engine 仅在已校验且可见的 checkpoint interrupt 时调用此方法。"""
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            review = db.execute(
                "SELECT * FROM agent_reviews WHERE id=? AND run_id=?", (review_id, row["run_id"])
            ).fetchone()
            if (
                current is None
                or control is None
                or review is None
                or current["cancel"]
                or (current["owner"], current["agent_id"], current["binding"], current["epoch"])
                != (row["owner"], row["agent_id"], row["binding"], row["epoch"])
                or (review["owner"], review["binding"], review["epoch"], review["digest"])
                != (row["owner"], row["binding"], row["epoch"], digest)
                or control["review_id"] != review_id
                or self.epoch(db, row["owner"], row["agent_id"], control["conversation_id"])
                != row["epoch"]
            ):
                raise AgentError(
                    "review_mismatch", "visible review does not match current control", 409
                )
            sealed = db.execute(
                "SELECT * FROM agent_review_visibility WHERE review_id=?", (review_id,)
            ).fetchone()
            if sealed:
                return dict(sealed)
            if current["deadline"] <= now or review["expires_at"] <= now:
                raise AgentError(
                    "review_visibility_unsealed_expired",
                    "unsealed review exceeded original execution deadline",
                    409,
                )
            db.execute(
                "INSERT INTO agent_review_visibility VALUES(?,?,?)",
                (review_id, now, min(review["remaining"], current["deadline"] - now)),
            )
            return dict(
                db.execute(
                    "SELECT * FROM agent_review_visibility WHERE review_id=?", (review_id,)
                ).fetchone()
            )

    def approval(self, row: dict, digest: str) -> bool:
        with self.connection() as db:
            review = db.execute(
                "SELECT 1 FROM agent_reviews WHERE run_id=? AND owner=? AND digest=? AND binding=? AND epoch=? AND approved_at IS NOT NULL AND expires_at>?",
                (row["run_id"], row["owner"], digest, row["binding"], row["epoch"], time.time()),
            ).fetchone()
        return review is not None

    def final_receipt(self, owner: str, run_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM agent_final_receipts WHERE owner=? AND run_id=?",
                (owner, run_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _event(
        self,
        db: sqlite3.Connection,
        row: dict,
        kind: str,
        data,
        provisional: bool,
        sources: list[dict],
    ) -> RunEvent:
        head = db.execute(
            "SELECT next_seq FROM agent_event_heads WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        seq = head[0] if head else 1
        event = RunEvent(
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            seq=seq,
            kind=kind,
            created_at=time.time(),
            data=json.loads(canonical_json({"value": data, "sources": sources})),
            provisional=provisional,
        )
        payload = canonical_json(event.model_dump(mode="json"), max_bytes=65536).decode()
        db.execute(
            "INSERT INTO agent_run_events VALUES(?,?,?,?)",
            (row["run_id"], seq, event.created_at, payload),
        )
        db.execute(
            "INSERT INTO agent_event_heads VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET next_seq=excluded.next_seq",
            (row["run_id"], seq + 1),
        )
        db.execute(
            "DELETE FROM agent_run_events WHERE run_id=? AND (seq<=? OR created_at<?)",
            (
                row["run_id"],
                max(0, seq - self.settings.event_count),
                time.time() - self.settings.event_seconds,
            ),
        )
        total = db.execute(
            "SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM agent_run_events WHERE run_id=?",
            (row["run_id"],),
        ).fetchone()[0]
        while total > self.settings.event_bytes:
            oldest = db.execute(
                "SELECT seq,length(CAST(payload AS BLOB)) FROM agent_run_events WHERE run_id=? ORDER BY seq LIMIT 1",
                (row["run_id"],),
            ).fetchone()
            db.execute(
                "DELETE FROM agent_run_events WHERE run_id=? AND seq=?", (row["run_id"], oldest[0])
            )
            total -= oldest[1]
        return event

    def append_event(
        self,
        row: dict,
        kind: str,
        data,
        *,
        provisional: bool = False,
        sources: list[dict] | None = None,
        guard=None,
    ) -> RunEvent:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if guard is not None:
                current = db.execute(
                    "SELECT * FROM requests WHERE run_id=?", (row["run_id"],)
                ).fetchone()
                control = db.execute(
                    "SELECT * FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
                ).fetchone()
                if (
                    current is None
                    or control is None
                    or current["cancel"]
                    or current["deadline"] <= time.time()
                    or (current["owner"], current["agent_id"], current["binding"], current["epoch"])
                    != (row["owner"], row["agent_id"], row["binding"], row["epoch"])
                    or self.epoch(db, row["owner"], row["agent_id"], control["conversation_id"])
                    != row["epoch"]
                    or not guard(db)
                ):
                    raise AgentError("event_control", "event publication control changed", 409)
            return self._event(db, row, kind, data, provisional, sources or [])

    def events(self, owner: str, run_id: str, after: int, limit: int = 128) -> list[RunEvent]:
        row = self.request(owner, run_id)
        if type(after) is not int or not 0 <= after <= 9223372036854775807:
            raise AgentError("event_cursor_invalid", "cursor must be a nonnegative integer", 400)
        if type(limit) is not int or not 1 <= limit <= 512:
            raise AgentError("event_limit", "event limit must be an integer from 1 to 512", 400)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM agent_run_events WHERE run_id=? AND created_at<?",
                (run_id, time.time() - self.settings.event_seconds),
            )
            first = db.execute(
                "SELECT MIN(seq) FROM agent_run_events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            head = db.execute(
                "SELECT next_seq FROM agent_event_heads WHERE run_id=?", (run_id,)
            ).fetchone()
            last = head[0] - 1 if head else 0
            if after > last:
                raise AgentError(
                    "event_cursor_future", "cursor exceeds the current event head", 409
                )
            if (first is not None and after < first - 1) or (
                first is None and head and after < head[0] - 1
            ):
                raise AgentError(
                    "event_cursor_expired", "event retention gap; query current run snapshot", 409
                )
            rows = db.execute(
                "SELECT seq,payload FROM agent_run_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after, limit),
            ).fetchall()
            events = [RunEvent.model_validate_json(record[1]) for record in rows]
            if any(
                event.seq != record[0]
                or event.seq != after + index + 1
                or event.run_id != run_id
                or event.agent_id != row["agent_id"]
                for index, (record, event) in enumerate(zip(rows, events, strict=True))
            ):
                raise AgentError("event_store_conflict", "event sequence or binding differs", 409)
        return events

    def commit_final(self, row: dict, projection: FinalProjection) -> bool:
        history = canonical_json(
            [message.model_dump(mode="json") for message in projection.messages]
        ).decode()
        payload = canonical_json(
            {
                "status": projection.status,
                "value": projection.value,
                "sources": projection.sources,
                "history_digest": content_digest(json.loads(history)),
            }
        ).decode()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT * FROM requests WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            control = db.execute(
                "SELECT * FROM agent_run_controls WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            previous = db.execute(
                "SELECT payload FROM agent_final_receipts WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            if current is None or (
                current["owner"],
                current["agent_id"],
                current["epoch"],
                current["binding"],
            ) != (row["owner"], row["agent_id"], row["epoch"], row["binding"]):
                raise AgentError("final_binding", "final receipt identity or binding differs", 409)
            if previous:
                if previous[0] != payload:
                    raise AgentError("final_conflict", "final receipt changed", 409)
                task_tree.outcome_in(
                    db, row["run_id"], projection.status, None, content_digest(json.loads(payload))
                )
                return True
            if (
                task_tree.metadata_in(db, row["run_id"]) is not None
                and db.execute(
                    "SELECT 1 FROM agent_task_runs t LEFT JOIN agent_task_outcomes o ON o.run_id=t.run_id WHERE t.parent_run_id=? AND o.run_id IS NULL",
                    (row["run_id"],),
                ).fetchone()
            ):
                raise AgentError(
                    "task_unfinished", "child tasks must finish before parent publication", 409
                )
            if (
                current is None
                or control is None
                or current["cancel"]
                or current["deadline"] <= time.time()
                or (current["owner"], current["agent_id"], current["epoch"])
                != (row["owner"], row["agent_id"], row["epoch"])
                or current["binding"] != row["binding"]
                or self.epoch(db, row["owner"], row["agent_id"], control["conversation_id"])
                != row["epoch"]
                or (projection.local_guard is not None and not projection.local_guard(db))
            ):
                return False
            if projection.extra_commit is not None:
                projection.extra_commit(db)
            db.execute(
                "INSERT INTO agent_final_receipts VALUES(?,?,?,?)",
                (row["run_id"], row["owner"], row["binding"], payload),
            )
            task_tree.outcome_in(
                db, row["run_id"], projection.status, None, content_digest(json.loads(payload))
            )
            if projection.messages:
                existing_history = db.execute(
                    "SELECT payload FROM agent_history WHERE owner=? AND agent_id=? AND conversation_id=? AND epoch=? AND turn_id=?",
                    (
                        row["owner"],
                        row["agent_id"],
                        control["conversation_id"],
                        row["epoch"],
                        row["run_id"],
                    ),
                ).fetchone()
                if existing_history is not None and existing_history[0] != history:
                    raise AgentError(
                        "history_conflict", "same final turn has different history", 409
                    )
                db.execute(
                    "INSERT INTO agent_history VALUES(?,?,?,?,?,?,?) ON CONFLICT(owner,agent_id,conversation_id,epoch,turn_id) DO NOTHING",
                    (
                        row["owner"],
                        row["agent_id"],
                        control["conversation_id"],
                        row["epoch"],
                        row["run_id"],
                        history,
                        time.time(),
                    ),
                )
            self._event(
                db,
                row,
                "run.final",
                {"status": projection.status, "result_available": True},
                False,
                projection.sources,
            )
            return True
