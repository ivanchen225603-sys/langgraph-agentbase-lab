"""Controlled JSON artifacts; trusted scopes, immutable revisions and mutation receipts.

This is a storage contract, not an authorization service. Callers derive scopes and operations
from trusted runtime state and supply a synchronous transaction guard. Model-visible adapters
must authorize current references and inherited sources before returning content.
"""

from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import Field, JsonValue, ValidationError

from .contracts import AgentError, BaseContract, Identifier
from .schema import canonical_json, content_digest
from .storage import ConnectionFactory

MutationGuard = Callable[[sqlite3.Connection], bool]
SQLITE_MAX_INT = (1 << 63) - 1


class ArtifactScope(BaseContract):
    owner: Identifier
    agent_id: Identifier
    namespace: Identifier

    def key(self) -> tuple[str, str, str]:
        return self.owner, self.agent_id, self.namespace


class ArtifactOperation(BaseContract):
    run_id: Identifier
    key: str = Field(min_length=1, max_length=256)
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ArtifactLimits(BaseContract):
    max_content_bytes: int = Field(262144, ge=1, le=1048576)
    max_namespace_bytes: int = Field(8388608, ge=1, le=67108864)
    max_owner_bytes: int = Field(67108864, ge=1, le=536870912)
    max_namespace_artifacts: int = Field(256, ge=1, le=4096)
    max_owner_artifacts: int = Field(4096, ge=1, le=32768)
    max_versions: int = Field(64, ge=1, le=1024)
    max_namespace_mutations: int = Field(2048, ge=1, le=32768)
    max_owner_mutations: int = Field(8192, ge=1, le=131072)


class ArtifactReference(BaseContract):
    kind: Literal["artifact"] = "artifact"
    id: Identifier
    agent_id: Identifier
    namespace: Identifier
    version: int = Field(ge=1, le=SQLITE_MAX_INT)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ArtifactInfo(ArtifactScope):
    id: Identifier
    version: int = Field(ge=1, le=SQLITE_MAX_INT)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    content_bytes: int = Field(ge=1)
    enabled: bool = True
    valid_until: float | None = None
    created_at: float
    updated_at: float

    def reference(self) -> dict[str, JsonValue]:
        return ArtifactReference(
            id=self.id,
            agent_id=self.agent_id,
            namespace=self.namespace,
            version=self.version,
            digest=self.digest,
        ).model_dump(mode="json")


class ArtifactRecord(ArtifactInfo):
    value: JsonValue
    sources: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=100)


class ArtifactStore(Protocol):
    def receipt(
        self, scope: ArtifactScope, operation: ArtifactOperation, *, guard: MutationGuard
    ) -> ArtifactRecord | None: ...

    def read(
        self, scope: ArtifactScope, artifact_id: str, *, version: int | None = None
    ) -> ArtifactRecord: ...

    def list(
        self, scope: ArtifactScope, *, limit: int = 16, offset: int = 0
    ) -> builtins.list[ArtifactInfo]: ...

    def create(
        self,
        scope: ArtifactScope,
        operation: ArtifactOperation,
        value: JsonValue,
        *,
        sources: builtins.list[dict[str, JsonValue]],
        valid_until: float | None = None,
        guard: MutationGuard,
    ) -> ArtifactRecord: ...

    def update(
        self,
        scope: ArtifactScope,
        operation: ArtifactOperation,
        artifact_id: str,
        expected_version: int,
        value: JsonValue,
        *,
        sources: builtins.list[dict[str, JsonValue]],
        valid_until: float | None = None,
        guard: MutationGuard,
    ) -> ArtifactRecord: ...

    def revoke(
        self,
        scope: ArtifactScope,
        operation: ArtifactOperation,
        artifact_id: str,
        expected_version: int,
        *,
        guard: MutationGuard,
    ) -> ArtifactRecord: ...

    def restore(
        self,
        scope: ArtifactScope,
        operation: ArtifactOperation,
        artifact_id: str,
        expected_version: int,
        source_version: int,
        *,
        guard: MutationGuard,
    ) -> ArtifactRecord: ...


class SqliteArtifactStore:
    def __init__(self, connection: ConnectionFactory, limits: ArtifactLimits | None = None):
        self.connection = connection
        self.limits = limits or ArtifactLimits()

    @staticmethod
    def install(db: sqlite3.Connection) -> None:
        if not db.in_transaction:
            raise AgentError(
                "artifact_schema", "installation requires a migration transaction", 503
            )
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_artifact_meta'"
        ).fetchone()
        if exists:
            row = db.execute("SELECT value FROM agent_artifact_meta WHERE key='schema'").fetchone()
            if row is None or row[0] not in {"1", "2"}:
                raise AgentError(
                    "artifact_schema", "artifact schema requires a supported migration", 503
                )
        for statement in (
            "CREATE TABLE IF NOT EXISTS agent_artifact_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
            "INSERT OR IGNORE INTO agent_artifact_meta VALUES('schema','1')",
            "CREATE TABLE IF NOT EXISTS agent_artifact_heads(owner TEXT,agent_id TEXT,namespace TEXT,id TEXT,version INTEGER NOT NULL,enabled INTEGER NOT NULL,PRIMARY KEY(owner,agent_id,namespace,id))",
            "CREATE TABLE IF NOT EXISTS agent_artifact_revisions(owner TEXT,agent_id TEXT,namespace TEXT,id TEXT,version INTEGER,payload TEXT NOT NULL,size_bytes INTEGER NOT NULL,PRIMARY KEY(owner,agent_id,namespace,id,version))",
            "CREATE TABLE IF NOT EXISTS agent_artifact_mutations(owner TEXT,agent_id TEXT,namespace TEXT,run_id TEXT,operation_key TEXT,fingerprint TEXT NOT NULL,artifact_id TEXT NOT NULL,version INTEGER NOT NULL,result_enabled INTEGER NOT NULL,PRIMARY KEY(owner,agent_id,namespace,run_id,operation_key))",
        ):
            db.execute(statement)
        if "config_digest" not in {
            column[1] for column in db.execute("PRAGMA table_info(agent_artifact_mutations)")
        }:
            # Original receipts keep their fingerprint. Unknown historic bindings are
            # deliberately not guessed for the new lookup API.
            db.execute("ALTER TABLE agent_artifact_mutations ADD COLUMN config_digest TEXT")
        db.execute("UPDATE agent_artifact_meta SET value='2' WHERE key='schema'")

    def receipt(self, scope, operation, *, guard) -> ArtifactRecord | None:
        """Recover a trusted operation result; this is not current source authorization."""
        scope = self._validated_scope(scope)
        operation = ArtifactOperation.model_validate(operation.model_dump())
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            guarded = guard(db)
            if guarded is not True:
                if inspect.iscoroutine(guarded):
                    guarded.close()
                raise AgentError("artifact_guard", "artifact receipt is no longer authorized", 409)
            row = db.execute(
                "SELECT * FROM agent_artifact_mutations WHERE owner=? AND agent_id=? AND namespace=? AND run_id=? AND operation_key=?",
                (*scope.key(), operation.run_id, operation.key),
            ).fetchone()
            if row is None:
                return None
            if row["config_digest"] != operation.config_digest:
                raise AgentError(
                    "artifact_operation_conflict", "artifact receipt binding differs", 409
                )
            original = self.read_in(db, scope, row["artifact_id"], version=row["version"])
            return original.model_copy(update={"enabled": bool(row["result_enabled"])})

    @staticmethod
    def _validated_scope(scope: ArtifactScope) -> ArtifactScope:
        return ArtifactScope.model_validate(scope.model_dump())

    @staticmethod
    def _revision_digest(payload: dict) -> str:
        # Includes namespace, revision, content and inherited provenance, but not mutable head state.
        raw = canonical_json(payload, max_bytes=1114112)
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    @classmethod
    def read_in(
        cls,
        db: sqlite3.Connection,
        scope: ArtifactScope,
        artifact_id: str,
        *,
        version: int | None = None,
    ) -> ArtifactRecord:
        if not db.in_transaction:
            raise AgentError("artifact_transaction", "artifact reads require a transaction", 503)
        scope = cls._validated_scope(scope)
        head = db.execute(
            "SELECT version,enabled FROM agent_artifact_heads WHERE owner=? AND agent_id=? AND namespace=? AND id=?",
            (*scope.key(), artifact_id),
        ).fetchone()
        if head is None:
            raise AgentError("artifact_not_found", "artifact is not available in this scope", 404)
        if version is not None and (type(version) is not int or not 1 <= version <= SQLITE_MAX_INT):
            raise AgentError("artifact_version", "artifact version is invalid", 422)
        if type(head[0]) is not int or not 1 <= head[0] <= SQLITE_MAX_INT:
            raise AgentError("artifact_corrupt", "artifact head version is invalid", 503)
        selected = head[0] if version is None else version
        row = db.execute(
            "SELECT payload,size_bytes FROM agent_artifact_revisions WHERE owner=? AND agent_id=? AND namespace=? AND id=? AND version=?",
            (*scope.key(), artifact_id, selected),
        ).fetchone()
        if row is None:
            raise AgentError("artifact_not_found", "artifact revision is not available", 404)
        try:
            if not isinstance(row[0], str) or len(row[0].encode("utf-8")) > 1114112:
                raise ValueError("revision payload exceeds limit")
            if len(row[0].encode("utf-8")) != row[1]:
                raise ValueError("revision byte accounting mismatch")
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                raise ValueError("revision payload must be an object")
            digest = payload.pop("digest")
            if digest != cls._revision_digest(payload):
                raise ValueError("revision digest mismatch")
            record = ArtifactRecord.model_validate(
                {**payload, "digest": digest, "enabled": bool(head[1]) and selected == head[0]}
            )
            if (record.owner, record.agent_id, record.namespace, record.id, record.version) != (
                *scope.key(),
                artifact_id,
                selected,
            ):
                raise ValueError("revision scope mismatch")
            if record.content_bytes != len(canonical_json(record.value, max_bytes=1048576)):
                raise ValueError("content byte accounting mismatch")
        except (
            ValueError,
            KeyError,
            TypeError,
            RecursionError,
            ValidationError,
            AgentError,
        ) as exc:
            raise AgentError(
                "artifact_corrupt", "artifact revision failed integrity validation", 503
            ) from exc
        return record

    def read(
        self, scope: ArtifactScope, artifact_id: str, *, version: int | None = None
    ) -> ArtifactRecord:
        with self.connection() as db:
            db.execute("BEGIN")
            return self.read_in(db, scope, artifact_id, version=version)

    def list(
        self, scope: ArtifactScope, *, limit: int = 16, offset: int = 0
    ) -> builtins.list[ArtifactInfo]:
        scope = self._validated_scope(scope)
        if (
            type(limit) is not int
            or not 1 <= limit <= 64
            or type(offset) is not int
            or not 0 <= offset <= SQLITE_MAX_INT
        ):
            raise AgentError("artifact_page", "artifact page is invalid", 422)
        with self.connection() as db:
            db.execute("BEGIN")
            rows = db.execute(
                "SELECT id FROM agent_artifact_heads WHERE owner=? AND agent_id=? AND namespace=? ORDER BY id LIMIT ? OFFSET ?",
                (*scope.key(), limit, offset),
            ).fetchall()
            return [
                ArtifactInfo.model_validate(
                    self.read_in(db, scope, row[0]).model_dump(exclude={"value", "sources"})
                )
                for row in rows
            ]

    def create(
        self, scope, operation, value, *, sources, valid_until=None, guard
    ) -> ArtifactRecord:
        return self._mutate(
            scope,
            operation,
            "create",
            value=value,
            sources=sources,
            valid_until=valid_until,
            guard=guard,
        )

    def update(
        self,
        scope,
        operation,
        artifact_id,
        expected_version,
        value,
        *,
        sources,
        valid_until=None,
        guard,
    ) -> ArtifactRecord:
        return self._mutate(
            scope,
            operation,
            "update",
            artifact_id=artifact_id,
            expected_version=expected_version,
            value=value,
            sources=sources,
            valid_until=valid_until,
            guard=guard,
        )

    def revoke(self, scope, operation, artifact_id, expected_version, *, guard) -> ArtifactRecord:
        return self._mutate(
            scope,
            operation,
            "revoke",
            artifact_id=artifact_id,
            expected_version=expected_version,
            guard=guard,
        )

    def restore(
        self, scope, operation, artifact_id, expected_version, source_version, *, guard
    ) -> ArtifactRecord:
        return self._mutate(
            scope,
            operation,
            "restore",
            artifact_id=artifact_id,
            expected_version=expected_version,
            source_version=source_version,
            guard=guard,
        )

    def _mutation_capacity(self, db, scope, *, revoking: bool) -> None:
        # Disabling an enabled head is always available, including after quota reductions.
        # At a full quota no new heads or revisions can be created; each existing head can
        # therefore add at most one reserved revocation receipt before becoming disabled.
        if revoking:
            return
        for condition, args, limit in (
            (
                "owner=? AND agent_id=? AND namespace=?",
                scope.key(),
                self.limits.max_namespace_mutations,
            ),
            (
                "owner=?",
                (scope.owner,),
                self.limits.max_owner_mutations,
            ),
        ):
            count = db.execute(
                f"SELECT COUNT(*) FROM agent_artifact_mutations WHERE {condition}", args
            ).fetchone()[0]
            if count >= limit:
                raise AgentError("artifact_quota", "artifact mutation limit exceeded", 409)

    def _revision_capacity(self, db, scope, size: int, *, creating: bool, version: int) -> None:
        if version > self.limits.max_versions:
            raise AgentError("artifact_quota", "artifact revision limit exceeded", 409)
        for condition, args, byte_limit, count_limit in (
            (
                "owner=? AND agent_id=? AND namespace=?",
                scope.key(),
                self.limits.max_namespace_bytes,
                self.limits.max_namespace_artifacts,
            ),
            (
                "owner=?",
                (scope.owner,),
                self.limits.max_owner_bytes,
                self.limits.max_owner_artifacts,
            ),
        ):
            retained = db.execute(
                f"SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM agent_artifact_revisions WHERE {condition}",
                args,
            ).fetchone()[0]
            count = db.execute(
                f"SELECT COUNT(*) FROM agent_artifact_heads WHERE {condition}", args
            ).fetchone()[0]
            if retained + size > byte_limit or (creating and count >= count_limit):
                raise AgentError("artifact_quota", "artifact retained storage limit exceeded", 409)

    def _mutate(
        self,
        scope: ArtifactScope,
        operation: ArtifactOperation,
        action: Literal["create", "update", "revoke", "restore"],
        *,
        artifact_id: str | None = None,
        expected_version: int | None = None,
        source_version: int | None = None,
        value: JsonValue = None,
        sources: builtins.list[dict[str, JsonValue]] | None = None,
        valid_until: float | None = None,
        guard: MutationGuard,
    ) -> ArtifactRecord:
        scope = self._validated_scope(scope)
        operation = ArtifactOperation.model_validate(operation.model_dump())
        if action != "create" and artifact_id is None:
            raise AgentError("artifact_not_found", "artifact ID is required", 422)
        if action != "create" and (
            type(expected_version) is not int or not 1 <= expected_version <= SQLITE_MAX_INT
        ):
            raise AgentError("artifact_version", "expected artifact version is required", 422)
        if action == "restore" and (
            type(source_version) is not int or not 1 <= source_version <= SQLITE_MAX_INT
        ):
            raise AgentError("artifact_version", "source artifact version is required", 422)
        raw = canonical_json(
            value,
            max_bytes=self.limits.max_content_bytes if action in {"create", "update"} else 4,
        )
        canonical_json(sources or [], max_bytes=32768)
        # Validate JSON sources and finite expiry before opening the mutation transaction.
        probe = ArtifactRecord(
            **scope.model_dump(),
            id=artifact_id or "probe",
            version=expected_version or 1,
            digest="sha256:" + "0" * 64,
            content_bytes=len(raw),
            created_at=0.0,
            updated_at=0.0,
            value=json.loads(raw),
            sources=sources or [],
            valid_until=valid_until,
        )
        fingerprint = content_digest(
            {
                "scope": scope.model_dump(),
                "operation": operation.model_dump(),
                "action": action,
                "id": artifact_id,
                "expected": expected_version,
                "source_version": source_version,
                "content": self._revision_digest(
                    {
                        "value": probe.value,
                        "sources": probe.sources,
                        "valid_until": probe.valid_until,
                    }
                ),
            }
        )
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            guarded = guard(db)
            if guarded is not True:
                if inspect.iscoroutine(guarded):
                    guarded.close()
                raise AgentError("artifact_guard", "artifact mutation is no longer authorized", 409)
            receipt = db.execute(
                "SELECT * FROM agent_artifact_mutations WHERE owner=? AND agent_id=? AND namespace=? AND run_id=? AND operation_key=?",
                (*scope.key(), operation.run_id, operation.key),
            ).fetchone()
            if receipt is not None:
                if receipt["fingerprint"] != fingerprint:
                    raise AgentError(
                        "artifact_operation_conflict", "artifact operation content changed", 409
                    )
                original = self.read_in(
                    db, scope, receipt["artifact_id"], version=receipt["version"]
                )
                return original.model_copy(update={"enabled": bool(receipt["result_enabled"])})
            current = self.read_in(db, scope, artifact_id) if artifact_id is not None else None
            if current is not None and current.version != expected_version:
                raise AgentError("artifact_conflict", "artifact head version changed", 409)
            self._mutation_capacity(
                db, scope, revoking=action == "revoke" and bool(current and current.enabled)
            )
            if action == "revoke":
                assert current is not None
                db.execute(
                    "UPDATE agent_artifact_heads SET enabled=0 WHERE owner=? AND agent_id=? AND namespace=? AND id=? AND version=?",
                    (*scope.key(), current.id, current.version),
                )
                result = current.model_copy(update={"enabled": False})
            else:
                if action == "update" and current is not None and not current.enabled:
                    raise AgentError(
                        "artifact_revoked", "revoked artifacts require an explicit restore", 409
                    )
                if action == "restore":
                    assert current is not None
                    restored = self.read_in(db, scope, current.id, version=source_version)
                    probe = restored
                now = time.time()
                data = {
                    **scope.model_dump(),
                    "id": current.id if current else uuid.uuid4().hex,
                    "version": current.version + 1 if current else 1,
                    "value": probe.value,
                    "sources": probe.sources,
                    "valid_until": probe.valid_until,
                    "content_bytes": len(
                        canonical_json(probe.value, max_bytes=self.limits.max_content_bytes)
                    ),
                    "created_at": current.created_at if current else now,
                    "updated_at": now,
                }
                digest = self._revision_digest(data)
                payload = canonical_json(
                    {**data, "digest": digest}, max_bytes=self.limits.max_content_bytes + 65536
                )
                self._revision_capacity(
                    db, scope, len(payload), creating=current is None, version=data["version"]
                )
                db.execute(
                    "INSERT INTO agent_artifact_revisions VALUES(?,?,?,?,?,?,?)",
                    (*scope.key(), data["id"], data["version"], payload.decode(), len(payload)),
                )
                db.execute(
                    "INSERT INTO agent_artifact_heads VALUES(?,?,?,?,?,1) ON CONFLICT(owner,agent_id,namespace,id) DO UPDATE SET version=excluded.version,enabled=1",
                    (*scope.key(), data["id"], data["version"]),
                )
                result = ArtifactRecord.model_validate({**data, "digest": digest, "enabled": True})
            db.execute(
                "INSERT INTO agent_artifact_mutations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    *scope.key(),
                    operation.run_id,
                    operation.key,
                    fingerprint,
                    result.id,
                    result.version,
                    int(result.enabled),
                    operation.config_digest,
                ),
            )
            return result
