"""Persistent intent choices. No method calls a RunEngine execution/control API."""

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, model_validator

from .contracts import AgentError, BaseContract, Identifier, Identity, Message
from .intent import (
    Digest,
    IntentAction,
    IntentContext,
    IntentReview,
    IntentSession,
    IntentTask,
    read_intent_target_in,
)
from .schema import SchemaContract, canonical_json, content_digest

if TYPE_CHECKING:
    from .runtime import RunEngine


class IntentDraft(BaseContract):
    text: str = Field(min_length=1, max_length=4000)
    action: IntentAction
    task: IntentTask | None = None
    target_run_id: Identifier | None = None

    @model_validator(mode="after")
    def coherent(self):
        if not self.text.strip():
            raise ValueError("request text must not be blank")
        if (self.task is not None) != (self.action in {"propose_task", "propose_replace"}):
            raise ValueError("task input does not match selected proposal type")
        if (self.target_run_id is not None) != (
            self.action in {"propose_replace", "propose_cancel"}
        ):
            raise ValueError("control proposals require an explicit target")
        canonical_json(self.model_dump(mode="json"), max_bytes=32768)
        return self


class IntentConfirmation(BaseContract):
    revision: int = Field(ge=1)
    request_digest: Digest
    decision_digest: Digest
    draft: IntentDraft


class IntentProposalInput(BaseContract):
    expected_revision: int = Field(ge=0, le=10000)
    draft: IntentDraft


class IntentProposalLimits(BaseContract):
    proposal_seconds: float = Field(120.0, gt=0, le=3600)
    session_seconds: float = Field(1800.0, gt=0, le=86400)
    interpreter_seconds: float = Field(10.0, gt=0, le=60)
    max_sessions: int = Field(256, ge=1, le=10000)
    sessions_per_owner: int = Field(16, ge=1, le=1000)
    max_proposals: int = Field(2048, ge=1, le=100000)
    proposals_per_owner: int = Field(128, ge=1, le=10000)
    max_inflight: int = Field(8, ge=1, le=64)


class IntentInterpreter(Protocol):
    mode: str

    async def propose(self, request: dict, draft: IntentDraft) -> object: ...


class OfflineStructuredIntent:
    """Copies explicit structured choices; makes no language-understanding claim."""

    mode = "offline-structured-fixture"

    async def propose(self, request: dict, draft: IntentDraft) -> object:
        result = {
            "request_id": request["request_id"],
            "request_digest": request["request_digest"],
            "action": draft.action,
            "interpretation": "explicit",
            "message": "离线结构化演示：按所选类型生成，仅记录提案，不执行操作。",
        }
        if draft.task is not None:
            result["task"] = draft.task.model_dump(mode="json")
        if draft.target_run_id:
            result["target_run_id"] = request["target"]["run_id"]
            result["target_input_digest"] = request["target"]["input_digest"]
        return result


def identity_fingerprint(identity: Identity) -> str:
    return content_digest(
        {
            "owner": identity.owner,
            "scopes": sorted(identity.scopes),
            "permission_version": identity.permission_version,
        }
    )


class IntentProposalService:
    def __init__(
        self,
        engine: "RunEngine",
        interpreter: IntentInterpreter,
        *,
        limits: IntentProposalLimits | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.engine, self.interpreter = engine, interpreter
        self.limits, self.clock = limits or IntentProposalLimits(), clock
        self._inflight = 0

    def install(self):
        """Host startup migration; separate from the runtime's existing schema."""
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS agent_intent_meta(version INTEGER NOT NULL)")
            version = db.execute("SELECT version FROM agent_intent_meta").fetchall()
            if version and (len(version) != 1 or version[0][0] != 1):
                raise AgentError("intent_schema", "unsupported intent store schema", 503)
            if not version:
                db.execute("INSERT INTO agent_intent_meta VALUES(1)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS agent_intent_sessions("
                "id TEXT PRIMARY KEY,owner TEXT NOT NULL,agent_id TEXT NOT NULL,"
                "conversation_id TEXT NOT NULL,identity_digest TEXT NOT NULL,"
                "credential_digest TEXT NOT NULL,binding TEXT NOT NULL,epoch INTEGER NOT NULL,"
                "revision INTEGER NOT NULL,request_id TEXT NOT NULL,expires_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS agent_intent_proposals("
                "id TEXT PRIMARY KEY,session_id TEXT NOT NULL,revision INTEGER NOT NULL,"
                "request_id TEXT NOT NULL,request_digest TEXT NOT NULL,decision_digest TEXT NOT NULL,"
                "draft_digest TEXT NOT NULL,target TEXT,payload TEXT NOT NULL,status TEXT NOT NULL,"
                "created_at REAL NOT NULL,expires_at REAL NOT NULL,confirmed_at REAL,revoked_at REAL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS agent_intent_owner ON agent_intent_sessions(owner)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS agent_intent_session ON agent_intent_proposals(session_id)"
            )

    def _session(self, db, identity, agent_id, session_id, credential):
        self._accepting()
        row = db.execute(
            "SELECT * FROM agent_intent_sessions WHERE id=? AND owner=? AND agent_id=?",
            (session_id, identity.owner, agent_id),
        ).fetchone()
        if row is None:
            raise AgentError("intent_not_found", "intent session not found", 404)
        if (
            row["identity_digest"] != identity_fingerprint(identity)
            or row["credential_digest"] != credential
        ):
            raise AgentError("intent_identity", "intent authentication context changed", 403)
        if row["expires_at"] <= self.clock():
            raise AgentError("intent_expired", "intent session expired", 410)
        if (
            row["epoch"]
            != self.engine.store.epoch(db, identity.owner, agent_id, row["conversation_id"])
            or row["binding"] != self.engine.registry.agent(agent_id).digest
        ):
            raise AgentError("intent_stale", "intent conversation or Agent version changed", 409)
        return dict(row)

    def _session_view(self, row):
        return {
            "session_id": row["id"],
            "conversation_id": row["conversation_id"],
            "epoch": row["epoch"],
            "revision": row["revision"],
            "expires_at": row["expires_at"],
            "server_time": self.clock(),
            "mode": self.interpreter.mode,
            "execution_connected": False,
        }

    def open_session(self, identity: Identity, agent_id: str, conversation: str, credential: str):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._accepting()
            counts = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(owner=?),0) FROM agent_intent_sessions",
                (identity.owner,),
            ).fetchone()
            if counts[0] >= self.limits.max_sessions or counts[1] >= self.limits.sessions_per_owner:
                raise AgentError("intent_capacity", "intent session retention limit reached", 429)
            row = {
                "id": uuid.uuid4().hex,
                "owner": identity.owner,
                "agent_id": agent_id,
                "conversation_id": conversation,
                "identity_digest": identity_fingerprint(identity),
                "credential_digest": credential,
                "binding": self.engine.registry.agent(agent_id).digest,
                "epoch": self.engine.store.epoch(db, identity.owner, agent_id, conversation),
                "revision": 0,
                "request_id": "",
                "expires_at": self.clock() + self.limits.session_seconds,
            }
            # One current intent session per authenticated owner/Agent/conversation.
            # Reopening the workflow retires previous confirmations, never their audit data.
            db.execute(
                "UPDATE agent_intent_proposals SET status='superseded' WHERE session_id IN "
                "(SELECT id FROM agent_intent_sessions WHERE owner=? AND agent_id=? AND conversation_id=?) "
                "AND status IN ('pending','confirmed')",
                (identity.owner, agent_id, conversation),
            )
            db.execute(
                "UPDATE agent_intent_sessions SET expires_at=MIN(expires_at,?) WHERE owner=? AND agent_id=? AND conversation_id=?",
                (self.clock(), identity.owner, agent_id, conversation),
            )
            db.execute(
                "INSERT INTO agent_intent_sessions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                tuple(row.values()),
            )
            return self._session_view(row)

    def _accepting(self):
        if self.engine.closing or not self.engine.available:
            raise AgentError("not_accepting", "intent runtime is unavailable", 503)

    def session_view(self, identity: Identity, agent_id: str, session_id: str, credential: str):
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            return self._session_view(self._session(db, identity, agent_id, session_id, credential))

    def invalidate(self, identity: Identity, agent_id: str, session_id: str, credential: str):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._session(db, identity, agent_id, session_id, credential)
            self._advance(db, row)
            return self._session_view(row)

    @staticmethod
    def _advance(db, row):
        if row["revision"] >= 10000:
            raise AgentError("intent_capacity", "intent session revision limit reached", 429)
        row["revision"] += 1
        row["request_id"] = uuid.uuid4().hex
        db.execute(
            "UPDATE agent_intent_sessions SET revision=?,request_id=? WHERE id=?",
            (row["revision"], row["request_id"], row["id"]),
        )
        db.execute(
            "UPDATE agent_intent_proposals SET status='superseded' WHERE session_id=? AND status IN ('pending','confirmed')",
            (row["id"],),
        )

    def _begin(self, identity, agent_id, session_id, draft, credential, expected_revision):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._session(db, identity, agent_id, session_id, credential)
            if row["revision"] != expected_revision:
                raise AgentError("intent_stale", "intent requirement revision changed", 409)
            counts = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(s.owner=?),0) FROM agent_intent_proposals p JOIN agent_intent_sessions s ON s.id=p.session_id",
                (identity.owner,),
            ).fetchone()
            if (
                counts[0] >= self.limits.max_proposals
                or counts[1] >= self.limits.proposals_per_owner
            ):
                raise AgentError("intent_capacity", "intent proposal retention limit reached", 429)
            target = (
                read_intent_target_in(db, identity, draft.target_run_id)
                if draft.target_run_id
                else None
            )
            if target and (
                not target.active
                or target.agent_id != agent_id
                or target.conversation_id != row["conversation_id"]
                or target.epoch != row["epoch"]
            ):
                raise AgentError("intent_target", "target is not current in this conversation", 409)
            self._advance(db, row)
            return row, target

    async def propose(
        self,
        identity: Identity,
        agent_id: str,
        session_id: str,
        draft: IntentDraft,
        credential: str,
        reauthorize: Callable[[], Awaitable[None]],
        expected_revision: int,
    ) -> dict:
        if self._inflight >= self.limits.max_inflight:
            raise AgentError("intent_busy", "too many intent requests", 429)
        self._inflight += 1
        try:
            row, target = self._begin(
                identity, agent_id, session_id, draft, credential, expected_revision
            )
            if draft.task:
                SchemaContract(
                    self.engine.registry.agent(agent_id).definition.input_schema
                ).validate(draft.task.input, code="intent_task_schema")
            context = IntentContext(
                request_id=row["request_id"],
                identity=identity,
                agent_id=agent_id,
                conversation_id=row["conversation_id"],
                epoch=row["epoch"],
            )
            session = IntentSession(
                context,
                Message(role="user", content=draft.text),
                origin="authenticated-user",
                target=target,
            )
            try:
                async with asyncio.timeout(self.limits.interpreter_seconds):
                    raw = await self.interpreter.propose(
                        session.model_input(), draft.model_copy(deep=True)
                    )
            except Exception as exc:
                raise AgentError(
                    "intent_interpreter", "intent interpreter unavailable", 503
                ) from exc
            await reauthorize()
            review = session.review(raw, current=context, target=target)
            if review.proposal and (
                review.proposal.action != draft.action or review.proposal.task != draft.task
            ):
                review = IntentReview(
                    action="clarify",
                    message="提案与所选需求不一致，请重新明确需求。",
                    reason="invalid_decision",
                )
            return self._save(identity, agent_id, row, draft, target, review, credential)
        finally:
            self._inflight -= 1

    def _save(self, identity, agent_id, original, draft, target, review, credential):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._session(db, identity, agent_id, original["id"], credential)
            if (row["revision"], row["request_id"]) != (
                original["revision"],
                original["request_id"],
            ):
                raise AgentError(
                    "intent_stale", "a newer intent request replaced this response", 409
                )
            if target and read_intent_target_in(db, identity, target.run_id) != target:
                raise AgentError("intent_target", "target changed while interpreting", 409)
            # Recheck capacity after the interpreter yielded.
            counts = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(s.owner=?),0) FROM agent_intent_proposals p JOIN agent_intent_sessions s ON s.id=p.session_id",
                (identity.owner,),
            ).fetchone()
            if (
                counts[0] >= self.limits.max_proposals
                or counts[1] >= self.limits.proposals_per_owner
            ):
                raise AgentError("intent_capacity", "intent proposal retention limit reached", 429)
            payload = {
                "draft": draft.model_dump(mode="json"),
                "review": review.model_dump(mode="json"),
                "mode": self.interpreter.mode,
            }
            encoded = canonical_json(payload, max_bytes=131072).decode()
            now, proposal_id = self.clock(), uuid.uuid4().hex
            if now >= row["expires_at"]:
                raise AgentError("intent_expired", "intent session expired before saving", 410)
            request_digest = (
                review.proposal.request_digest
                if review.proposal
                else content_digest({"request_id": row["request_id"], "draft": payload["draft"]})
            )
            db.execute(
                "INSERT INTO agent_intent_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
                (
                    proposal_id,
                    row["id"],
                    row["revision"],
                    row["request_id"],
                    request_digest,
                    content_digest(payload),
                    content_digest(payload["draft"]),
                    canonical_json(target.model_dump(mode="json")).decode() if target else None,
                    encoded,
                    "pending" if review.action == "proposal" else review.action,
                    now,
                    min(now + self.limits.proposal_seconds, row["expires_at"]),
                ),
            )
            return self._view(db, identity, row, proposal_id)

    def _record(self, db, session, proposal_id):
        row = db.execute(
            "SELECT * FROM agent_intent_proposals WHERE id=? AND session_id=?",
            (proposal_id, session["id"]),
        ).fetchone()
        if row is None:
            raise AgentError("intent_not_found", "intent proposal not found", 404)
        row = dict(row)
        try:
            fingerprint = content_digest(json.loads(row["payload"]))
        except (ValueError, TypeError, AgentError) as exc:
            raise AgentError("intent_storage", "intent record integrity failed", 503) from exc
        if fingerprint != row["decision_digest"]:
            raise AgentError("intent_storage", "intent record integrity failed", 503)
        return row

    def _status(self, db, identity, session, row):
        if row["revision"] != session["revision"]:
            return "superseded"
        if row["status"] in {"revoked", "superseded"}:
            return row["status"]
        if row["expires_at"] <= self.clock():
            return "expired"
        if row["target"]:
            current = read_intent_target_in(db, identity, json.loads(row["target"])["run_id"])
            if canonical_json(current.model_dump(mode="json")).decode() != row["target"]:
                return "stale"
        return row["status"]

    def _view(self, db, identity, session, proposal_id):
        row = self._record(db, session, proposal_id)
        return {
            "session_id": session["id"],
            "proposal_id": row["id"],
            "revision": row["revision"],
            "request_digest": row["request_digest"],
            "decision_digest": row["decision_digest"],
            "status": self._status(db, identity, session, row),
            "payload": json.loads(row["payload"]),
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "confirmed_at": row["confirmed_at"],
            "revoked_at": row["revoked_at"],
            "server_time": self.clock(),
            "mode": json.loads(row["payload"]).get("mode", "unknown"),
            "execution_connected": False,
            "tool_approval": False,
        }

    def view(
        self, identity: Identity, agent_id: str, session_id: str, proposal_id: str, credential: str
    ):
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            session = self._session(db, identity, agent_id, session_id, credential)
            return self._view(db, identity, session, proposal_id)

    def decide(
        self,
        identity: Identity,
        agent_id: str,
        session_id: str,
        proposal_id: str,
        body: IntentConfirmation,
        credential: str,
        *,
        revoke: bool = False,
    ):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            session = self._session(db, identity, agent_id, session_id, credential)
            row = self._record(db, session, proposal_id)
            self.assert_choice(row, body)
            status = self._status(db, identity, session, row)
            if status == "expired":
                raise AgentError("intent_expired", "intent proposal expired", 410)
            if status not in ({"pending", "confirmed"} if revoke else {"pending"}):
                raise AgentError(
                    "intent_not_pending", "intent is not available for this choice", 409
                )
            now = self.clock()
            if now >= row["expires_at"] or now >= session["expires_at"]:
                raise AgentError("intent_expired", "intent expired before recording choice", 410)
            if revoke:
                db.execute(
                    "UPDATE agent_intent_proposals SET status='revoked',revoked_at=? WHERE id=?",
                    (now, proposal_id),
                )
            else:
                db.execute(
                    "UPDATE agent_intent_proposals SET status='confirmed',confirmed_at=? WHERE id=? AND status='pending'",
                    (now, proposal_id),
                )
            return self._view(db, identity, session, proposal_id)

    @staticmethod
    def assert_choice(row, body: IntentConfirmation):
        if (
            body.revision,
            body.request_digest,
            body.decision_digest,
            content_digest(body.draft.model_dump(mode="json")),
        ) != (row["revision"], row["request_digest"], row["decision_digest"], row["draft_digest"]):
            raise AgentError("intent_binding", "confirmation parameters do not match proposal", 409)
