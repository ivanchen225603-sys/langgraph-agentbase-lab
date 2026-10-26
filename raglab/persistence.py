import hashlib
import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from agentbase.run_store import SqliteRunStore

from .effect_scope import owns_intent
from .errors import LabError, UnknownEffect

LEGACY_AGENT_ID = "rag"


def encode(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value) -> str:
    return hashlib.sha256((value if isinstance(value, str) else encode(value)).encode()).hexdigest()


class ProcessLock:
    """每个数据目录只允许一个应用实例；进程真正崩溃时操作系统会自动释放 lock。"""

    def __init__(self, path: Path):
        self.path = path
        self.file: BinaryIO | None = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import importlib

                fcntl = importlib.import_module("fcntl")
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise LabError(
                "already_running", "data directory already owned by another process", 503
            ) from exc

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class Store:
    """只存放业务记录，不含独立的执行状态列，也不承担 DAG 调度职责。"""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone()
            if exists and self.meta("schema", db) not in {"1", "2", "3", "4"}:
                raise LabError(
                    "schema_version", "unsupported business schema; restore matching release", 503
                )
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            old_requests = db.execute("PRAGMA table_info(requests)").fetchall()
            legacy_runs = []
            if old_requests and "agent_id" not in {column[1] for column in old_requests}:
                legacy_runs = [row[0] for row in db.execute("SELECT run_id FROM requests")]
            statements = """
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                INSERT OR IGNORE INTO meta VALUES('schema','4'),('acl','1'),('memory','1');
                CREATE TABLE IF NOT EXISTS indexes(version TEXT PRIMARY KEY,manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks(version TEXT,id TEXT,payload TEXT,vector TEXT,PRIMARY KEY(version,id));
                CREATE TABLE IF NOT EXISTS embedding_cache(key TEXT PRIMARY KEY,vector TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,owner TEXT,enabled INTEGER,valid_until REAL,revision INTEGER);
                CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY,owner TEXT,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations(owner TEXT,id TEXT,epoch INTEGER,PRIMARY KEY(owner,id));
                CREATE TABLE IF NOT EXISTS legacy_unadopted(run_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS legacy_tool_manifests(run_id TEXT PRIMARY KEY,binding TEXT NOT NULL,digest TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS legacy_execution_protocol(run_id TEXT PRIMARY KEY,binding TEXT NOT NULL,manifest TEXT NOT NULL,protocol TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS legacy_action_links(public_key TEXT NOT NULL,attempt_token TEXT NOT NULL,owner TEXT NOT NULL,agent_id TEXT NOT NULL,run_id TEXT NOT NULL,call_id TEXT NOT NULL,binding TEXT NOT NULL,manifest TEXT NOT NULL,fingerprint TEXT NOT NULL,attempt INTEGER NOT NULL,domain_key TEXT NOT NULL,evidence TEXT NOT NULL,PRIMARY KEY(public_key,attempt_token));
                CREATE TABLE IF NOT EXISTS legacy_write_actions(domain_key TEXT PRIMARY KEY,public_key TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS legacy_plan_grants(run_id TEXT PRIMARY KEY,owner TEXT NOT NULL,binding TEXT NOT NULL,epoch INTEGER NOT NULL,manifest TEXT NOT NULL,plan_digest TEXT NOT NULL,review_id TEXT NOT NULL,digest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS legacy_plan_actions(run_id TEXT NOT NULL,call_id TEXT NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(run_id,call_id));
                CREATE TABLE IF NOT EXISTS history(run_id TEXT PRIMARY KEY,owner TEXT,conversation TEXT,epoch INTEGER,payload TEXT,created REAL);
                CREATE TABLE IF NOT EXISTS operations(key TEXT PRIMARY KEY,owner TEXT,run_id TEXT,phase TEXT,payload TEXT);
                CREATE TABLE IF NOT EXISTS simulation(key TEXT PRIMARY KEY,value TEXT,compensated INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,created REAL,owner TEXT,event TEXT,object_id TEXT);
                CREATE TABLE IF NOT EXISTS model_budget(scope TEXT PRIMARY KEY,used INTEGER,deadline REAL);
            """
            for statement in statements.split(";"):
                if statement.strip():
                    db.execute(statement)
            SqliteRunStore.install(db, legacy_agent_id=LEGACY_AGENT_ID)
            db.executemany(
                "INSERT OR IGNORE INTO legacy_unadopted VALUES(?)", [(run,) for run in legacy_runs]
            )
            for conversation in db.execute("SELECT owner,id,epoch FROM conversations").fetchall():
                current = db.execute(
                    "SELECT epoch FROM agent_conversations WHERE owner=? AND agent_id=? AND conversation_id=?",
                    (conversation[0], LEGACY_AGENT_ID, conversation[1]),
                ).fetchone()
                if current is not None and current[0] != conversation[2] and legacy_runs:
                    raise LabError(
                        "schema_version",
                        "conflicting conversation epochs require explicit repair",
                        503,
                    )
                db.execute(
                    "INSERT OR IGNORE INTO agent_conversations VALUES(?,?,?,?)",
                    (conversation[0], LEGACY_AGENT_ID, conversation[1], conversation[2]),
                )
            db.execute("UPDATE meta SET value='4' WHERE key='schema'")

    @contextmanager
    def connection(self):
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=0.15)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        except sqlite3.Error as exc:
            raise LabError(
                "storage_unavailable", "business state storage unavailable", 503
            ) from exc
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def meta(key, db):
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def bump(key, db):
        db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key=?", (key,))

    @staticmethod
    def audit(db, owner, event, object_id):
        db.execute(
            "INSERT INTO audit(created,owner,event,object_id) VALUES(?,?,?,?)",
            (time.time(), owner, event, object_id),
        )

    def revisions(self) -> tuple[int, int]:
        with self.connection() as db:
            return int(self.meta("acl", db)), int(self.meta("memory", db))

    def epoch(self, owner, conversation) -> int:
        with self.connection() as db:
            row = db.execute(
                "SELECT epoch FROM agent_conversations WHERE owner=? AND agent_id=? AND conversation_id=?",
                (owner, LEGACY_AGENT_ID, conversation),
            ).fetchone()
            return row[0] if row else 0

    def request(self, owner, run_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE owner=? AND run_id=?", (owner, run_id)
            ).fetchone()
            if not row:
                raise LabError("not_found", "run not found", 404)
            return dict(row)

    def requests(self, owner=None, limit=100):
        with self.connection() as db:
            rows = (
                db.execute(
                    "SELECT * FROM requests WHERE owner=? AND agent_id=? ORDER BY rowid DESC LIMIT ?",
                    (owner, LEGACY_AGENT_ID, limit),
                )
                if owner
                else db.execute(
                    "SELECT * FROM requests WHERE agent_id=? ORDER BY rowid LIMIT ?",
                    (LEGACY_AGENT_ID, limit),
                )
            )
            return [dict(x) for x in rows]

    def request_cancel(self, owner, run_id):
        self.request(owner, run_id)
        with self.connection() as db:
            db.execute("UPDATE requests SET cancel=1 WHERE owner=? AND run_id=?", (owner, run_id))
            self.audit(db, owner, "run.cancel", run_id)

    def allowed(self, owner, document_id) -> bool:
        with self.connection() as db:
            r = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return bool(
                r
                and r["enabled"]
                and r["owner"] in ("public", owner)
                and (r["valid_until"] is None or r["valid_until"] > time.time())
            )

    def revoke(self, owner, document_id):
        with self.connection() as db:
            r = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            if not r or r["owner"] not in (owner, "public"):
                raise LabError("not_found", "document not found", 404)
            # 仅本地 demo principal 才有权管理 public 合成文档。
            if r["owner"] == "public" and owner != "local":
                raise LabError(
                    "forbidden", "public document administration requires local owner", 403
                )
            db.execute(
                "UPDATE documents SET enabled=0,revision=revision+1 WHERE id=?", (document_id,)
            )
            self.bump("acl", db)
            self.audit(db, owner, "document.revoke", document_id)

    def active_index(self, owner="local"):
        with self.connection() as db:
            version = self.meta(f"active_index:{owner}", db) or self.meta("active_index", db)
            if not version:
                return None
            return json.loads(
                db.execute("SELECT manifest FROM indexes WHERE version=?", (version,)).fetchone()[0]
            )

    def snapshot(self, version, owner):
        with self.connection() as db:
            if not db.execute("SELECT 1 FROM indexes WHERE version=?", (version,)).fetchone():
                raise LabError("index_not_ready", "index version not available", 409)
            rows = db.execute(
                """SELECT c.payload,c.vector FROM chunks c JOIN documents d
                                 ON json_extract(c.payload,'$.document_id')=d.id
                                 WHERE c.version=? AND d.enabled=1 AND (d.owner='public' OR d.owner=?)
                                 AND (d.valid_until IS NULL OR d.valid_until>?)""",
                (version, owner, time.time()),
            ).fetchall()
            return [(json.loads(r[0]), json.loads(r[1])) for r in rows]

    def cached_vector(self, key):
        with self.connection() as db:
            row = db.execute("SELECT vector FROM embedding_cache WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def publish(self, manifest, documents, chunks, vectors, cache):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO indexes VALUES(?,?)", (manifest["version"], encode(manifest)))
            db.executemany(
                "INSERT INTO chunks VALUES(?,?,?,?)",
                [
                    (manifest["version"], c["chunk_id"], encode(c), encode(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            # 重建索引绝不会隐式重新启用已撤权（revoke）的文档，也不会改变其 owner。
            db.executemany(
                "INSERT INTO documents VALUES(?,?,1,?,1) ON CONFLICT(id) DO UPDATE SET valid_until=excluded.valid_until,revision=documents.revision+1 WHERE documents.owner=excluded.owner",
                [(d["document_id"], d["owner"], d["valid_until"]) for d in documents],
            )
            db.executemany(
                "INSERT OR IGNORE INTO embedding_cache VALUES(?,?)",
                [(k, encode(v)) for k, v in cache.items()],
            )
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES(?,?)",
                (
                    "active_index"
                    if manifest["owner"] == "public"
                    else f"active_index:{manifest['owner']}",
                    manifest["version"],
                ),
            )
            self.bump("acl", db)
            self.audit(db, "system", "index.publish", manifest["version"])

    def add_history(self, owner, conversation, epoch, run_id, payload):
        if epoch != self.epoch(owner, conversation):
            return
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO history VALUES(?,?,?,?,?,?)",
                (run_id, owner, conversation, epoch, encode(payload), time.time()),
            )
            # 教学用的保留策略：短期历史有界。Graph checkpoint 另行保存执行数据。
            db.execute(
                "DELETE FROM history WHERE owner=? AND conversation=? AND run_id NOT IN (SELECT run_id FROM history WHERE owner=? AND conversation=? ORDER BY created DESC LIMIT 20)",
                (owner, conversation, owner, conversation),
            )

    def history(self, owner, conversation, epoch, turns, budget):
        with self.connection() as db:
            rows = db.execute(
                "SELECT payload FROM history WHERE owner=? AND conversation=? AND epoch=? ORDER BY created DESC LIMIT ?",
                (owner, conversation, epoch, turns),
            ).fetchall()
        return self.filter_history(
            owner, [json.loads(row[0]) for row in reversed(rows)], turns, budget
        )

    def filter_history(self, owner, items, turns, budget):
        result = []
        for item in reversed(items[-turns:]):
            if any(not self.allowed(owner, d) for d in item.get("documents", [])):
                continue
            if any(not self.memory_current(owner, m, v) for m, v in item.get("memories", [])):
                continue
            size = len(encode(item))
            if size <= budget:
                result.append(item)
                budget -= size
        return list(reversed(result))

    def memory_current(self, owner, memory_id, version):
        with self.connection() as db:
            r = db.execute(
                "SELECT payload FROM memories WHERE owner=? AND id=?", (owner, memory_id)
            ).fetchone()
        if not r:
            return False
        x = json.loads(r[0])
        return (
            x["version"] == version
            and x["enabled"]
            and (x["valid_until"] is None or x["valid_until"] > time.time())
        )

    def operation(self, key, owner, run_id):
        with self.connection() as db:
            old = db.execute("SELECT * FROM operations WHERE key=?", (key,)).fetchone()
            if old:
                if old["owner"] != owner or old["run_id"] != run_id:
                    raise LabError("forbidden", "operation owner mismatch", 403)
                if old["phase"] == "intent" and owns_intent(db, key, owner, run_id):
                    return None
                if old["phase"] in ("intent", "compensating", "unknown"):
                    raise UnknownEffect()
                if old["phase"] in ("not_applied", "compensated"):
                    raise LabError(
                        "operation_not_successful",
                        "stored operation did not apply or was compensated",
                        409,
                    )
                return json.loads(old["payload"])
            if db.execute(
                "SELECT 1 FROM legacy_execution_protocol WHERE run_id=?", (run_id,)
            ).fetchone():
                raise LabError("operation_claim", "business intent requires its public claim", 409)
            db.execute("INSERT INTO operations VALUES(?,?,?,'intent','{}')", (key, owner, run_id))
            self.audit(db, owner, "operation.intent", key)
        return None

    def receipt(self, key, value, phase="done"):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM operations WHERE key=?", (key,)).fetchone()
            if old is None:
                raise LabError("operation_receipt", "business operation is missing", 409)
            guarded = db.execute(
                "SELECT 1 FROM legacy_execution_protocol WHERE run_id=?", (old["run_id"],)
            ).fetchone()
            if guarded:
                compensation = phase == "compensated" and old["phase"] == "compensating"
                if not compensation and (
                    phase not in {"done", "not_applied"}
                    or old["phase"] not in {"intent", "unknown"}
                    or not owns_intent(db, key, old["owner"], old["run_id"], receipt=True)
                ):
                    raise LabError(
                        "operation_receipt", "receipt has no matching active attempt", 409
                    )
            cursor = db.execute(
                "UPDATE operations SET phase=?,payload=? WHERE key=? AND owner=? AND run_id=? AND phase=?",
                (phase, encode(value), key, old["owner"], old["run_id"], old["phase"]),
            )
            if cursor.rowcount != 1:
                raise LabError("operation_receipt", "business operation changed", 409)

    def uncertain(self, run_id):
        with self.connection() as db:
            return bool(
                db.execute(
                    "SELECT 1 FROM operations WHERE run_id=? AND phase IN ('intent','compensating','unknown')",
                    (run_id,),
                ).fetchone()
            )

    def operations(self, owner, run_id):
        self.request(owner, run_id)
        with self.connection() as db:
            return [
                dict(x)
                for x in db.execute(
                    "SELECT key,phase FROM operations WHERE owner=? AND run_id=?", (owner, run_id)
                )
            ]

    def simulation_effect(self, key, value):
        # 与 intent/receipt 分处不同事务，刻意暴露外部副作用（external effect）的时间窗。
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM operations WHERE key=?", (key,)).fetchone()
            if (
                old
                and db.execute(
                    "SELECT 1 FROM legacy_execution_protocol WHERE run_id=?", (old["run_id"],)
                ).fetchone()
                and (
                    old["phase"] != "intent"
                    or not owns_intent(db, key, old["owner"], old["run_id"])
                )
            ):
                raise LabError("operation_claim", "effect has no matching active intent", 409)
            db.execute("INSERT OR IGNORE INTO simulation VALUES(?,?,0)", (key, value))
