"""Explicit offline receipt-recovery example; never registered by production hosts.

Two SQLite files simulate local state and a remote ticket service. No network,
model, credentials, cross-run business-key deduplication or hard cancellation.
Run `python -m examples.offline_ticket_receipts submit|recover --directory NEW_DIR`.
"""

import argparse
import asyncio
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from agentbase.configuration import AgentPackage
from agentbase.contracts import AgentError, BaseContract, Identity, RunContext, ToolCall
from agentbase.effects import ActionReceipt, AttemptCommit, NeedsReconciliation, SqliteActionLedger
from agentbase.registry import ComponentRegistry
from agentbase.schema import canonical_json, content_digest
from agentbase.storage import SqliteDatabase
from agentbase.tools import ToolExecutor


class HostPrincipal(BaseContract):
    """Trusted host authentication input, never derived from model/tool arguments."""

    tenant: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=64)

    @property
    def owner(self):
        return content_digest(self.model_dump()).removeprefix("sha256:")


class OriginalAttempt(BaseContract):
    tenant: str
    subject: str
    key: str
    owner: str
    agent_id: str
    run_id: str
    call_id: str
    binding: str
    fingerprint: str
    attempt_token: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    arguments_digest: str


LINK = (
    "key",
    "owner",
    "agent_id",
    "run_id",
    "call_id",
    "binding",
    "fingerprint",
    "attempt_token",
    "attempts",
)


class RemoteReceipt(BaseContract):
    original: OriginalAttempt
    phase: Literal["applied"] = "applied"
    ticket_id: str = Field(min_length=1, max_length=80)


def conflict():
    return AgentError("ticket_receipt_binding", "receipt differs from original attempt", 409)


class FakeTicketService:
    """Durable synthetic remote. Its lookup is the trusted transport seam.

    A real adapter would authenticate the service and validate its response;
    accepting arbitrary tool output as a lookup result would violate the seam.
    """

    def __init__(self, path: Path):
        self.database = SqliteDatabase(path)
        self.lookups = 0
        with self.database.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS ticket_calls(id INTEGER PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS tickets(key TEXT PRIMARY KEY, receipt TEXT)")

    @property
    def submissions(self):
        """All received create requests, including failed duplicate requests."""
        with self.database.connection() as db:
            return db.execute("SELECT COUNT(*) FROM ticket_calls").fetchone()[0]

    @property
    def committed_tickets(self):
        with self.database.connection() as db:
            return db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

    async def create(self, original: OriginalAttempt):
        # Request telemetry must survive a failing/duplicate business transaction;
        # otherwise a duplicate request could be hidden by its SQL rollback.
        with self.database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO ticket_calls DEFAULT VALUES")
        with self.database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = RemoteReceipt(original=original, ticket_id="ticket_" + original.key[:16])
            db.execute("INSERT INTO tickets VALUES(?,?)", (original.key, receipt.model_dump_json()))
        # Remote commit precedes the lost response, even when local state is unknown.
        raise OSError("synthetic response lost after remote commit")

    async def lookup(self, original: OriginalAttempt):
        self.lookups += 1
        await asyncio.sleep(0)  # Synthetic asynchronous transport boundary, no local SQL open.
        with self.database.connection() as db:
            row = db.execute("SELECT receipt FROM tickets WHERE key=?", (original.key,)).fetchone()
        return json.loads(row[0]) if row else None


class TicketAdapter:
    """Host-owned two-step receipt import + existing synchronous reconciliation."""

    def __init__(
        self, ledger: SqliteActionLedger, remote: FakeTicketService, principal: HostPrincipal
    ):
        self.ledger, self.remote, self.principal = ledger, remote, principal
        self.attempt_commit = AttemptCommit(frozenset({"ticket_attempt_evidence"}), self.claim)

    @staticmethod
    def install(db):
        db.execute(
            "CREATE TABLE IF NOT EXISTS ticket_attempt_evidence(key TEXT PRIMARY KEY, original TEXT NOT NULL, receipt TEXT)"
        )

    def claim(self, db, operation, evidence):
        if evidence.context.identity.owner != self.principal.owner or operation["read_only"]:
            raise conflict()
        original = OriginalAttempt(
            **{name: operation[name] for name in LINK},
            **self.principal.model_dump(),
            arguments_digest=content_digest(evidence.call.arguments),
        )
        db.execute(
            "INSERT INTO ticket_attempt_evidence(key,original) VALUES(?,?)",
            (original.key, original.model_dump_json()),
        )

    def _original(self, db, key, principal):
        row = db.execute("SELECT * FROM ticket_attempt_evidence WHERE key=?", (key,)).fetchone()
        operation = db.execute("SELECT * FROM agent_tool_operations WHERE key=?", (key,)).fetchone()
        if row is None or operation is None:
            raise AgentError(
                "ticket_evidence_missing", "original attempt evidence unavailable", 409
            )
        original = OriginalAttempt.model_validate_json(row["original"])
        if (
            original.tenant != principal.tenant
            or original.subject != principal.subject
            or original.owner != principal.owner
            or any(operation[name] != getattr(original, name) for name in LINK)
            or operation["read_only"]
            or operation["phase"] == "pending"
        ):
            raise conflict()
        return original, operation, row

    async def create(self, arguments, context):
        with self.ledger.connection() as db:
            original, _, _ = self._original(db, context.operation_key, self.principal)
        if (
            context.identity.owner != original.owner
            or context.attempt_token != original.attempt_token
            or content_digest(arguments) != original.arguments_digest
        ):
            raise conflict()
        await self.remote.create(original)
        # Even a normal tool/transport return is not this adapter's trusted
        # receipt path. Only a separately authenticated lookup may establish it.
        raise AgentError("ticket_unverified_response", "ticket outcome needs trusted lookup", 503)

    def inspect(self, db: sqlite3.Connection, operation: dict):
        if (
            db.execute(
                "SELECT 1 FROM ticket_attempt_evidence WHERE key=?", (operation["key"],)
            ).fetchone()
            is None
        ):
            return None  # Pending/unowned operations have no trusted claim evidence.
        original, _, row = self._original(db, operation["key"], self.principal)
        if row["receipt"] is None:
            return None
        receipt = RemoteReceipt.model_validate_json(row["receipt"])
        if receipt.original != original:
            raise conflict()
        return ActionReceipt(phase="applied", value={"ticket_id": receipt.ticket_id})

    async def query(self, principal: HostPrincipal, key: str, *, query_seconds: float = 1.0):
        """Bounded cooperative read. No approval grant, new attempt or submission.

        Cancellation propagates. Missing/temporary/unavailable evidence stays unknown.
        Lookup authority belongs to the host principal, separately from write approval.
        """
        if (
            not isinstance(query_seconds, (int, float))
            or isinstance(query_seconds, bool)
            or not math.isfinite(query_seconds)
            or not 0 < query_seconds <= 5
        ):
            raise ValueError("lookup timeout must be within (0, 5] seconds")
        if principal != self.principal:
            raise conflict()
        with self.ledger.connection() as db:
            original, _, _ = self._original(db, key, principal)
        # The local connection is CLOSED before the first await. Never put this
        # await in ActionReceiptInspector / AttemptCommit / an SQL transaction.
        try:
            async with asyncio.timeout(query_seconds):
                payload = await self.remote.lookup(original)
        except (TimeoutError, OSError):
            return "unknown"
        if payload is None:
            return "unknown"
        if type(payload) is not dict:
            raise conflict()
        try:
            receipt = RemoteReceipt.model_validate(payload)
        except (ValidationError, ValueError, TypeError) as exc:
            raise conflict() from exc
        if receipt.original != original:
            raise conflict()
        payload_json = receipt.model_dump_json()
        with self.ledger.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current, operation, row = self._original(db, key, principal)
            if current != original or operation["phase"] not in {"intent", "unknown", "applied"}:
                raise conflict()
            if operation["effect_compensated"] or (
                operation["effect_applied"]
                and operation["result"] is not None
                and operation["result"] != canonical_json({"ticket_id": receipt.ticket_id}).decode()
            ):
                raise conflict()
            if row["receipt"] is not None:
                if RemoteReceipt.model_validate_json(row["receipt"]) != receipt:
                    raise conflict()
            else:
                db.execute(
                    "UPDATE ticket_attempt_evidence SET receipt=? WHERE key=? AND receipt IS NULL",
                    (payload_json, key),
                )
        # A crash here is safe: durable evidence can be reconciled on restart.
        # No await between receipt commit and reconciliation. A cancellation
        # cannot erase an already committed receipt, nor grant publication rights.
        self.reconcile(original)
        return "applied"

    def reconcile(self, original: OriginalAttempt):
        self.ledger.reconcile(original.owner, original.agent_id, original.run_id, self.inspect)


class OfflineTicketDemo:
    def __init__(self, directory: Path):
        self.database = SqliteDatabase(directory / "local.sqlite")
        self.ledger = SqliteActionLedger(self.database.connection)
        with self.database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self.ledger.install(db)
            TicketAdapter.install(db)
        self.principal = HostPrincipal(tenant="demo_tenant", subject="demo_subject")
        self.remote = FakeTicketService(directory / "remote.sqlite")
        self.adapter = TicketAdapter(self.ledger, self.remote, self.principal)
        self.approved = True

        async def approval(*_args):
            return self.approved

        registry = ComponentRegistry()
        registry.register_handler(
            "demo.ticket", self.adapter.create, attempt_commit=self.adapter.attempt_commit
        )
        registry.register_workflow("fixed", lambda: None)
        registry.bind(
            AgentPackage.model_validate(
                {
                    "tools": [
                        {
                            "name": "ticket.create",
                            "handler": "demo.ticket",
                            "description": "OFFLINE synthetic ticket",
                            "scope": "ticket.write",
                            "read_only": False,
                            "retry_safe": False,
                            "idempotency": "external-receipt",
                            "input_schema": {
                                "type": "object",
                                "properties": {"title": {"type": "string"}},
                                "required": ["title"],
                                "additionalProperties": False,
                            },
                            "output_schema": {
                                "type": "object",
                                "properties": {"ticket_id": {"type": "string"}},
                                "required": ["ticket_id"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                    "agents": [
                        {
                            "id": "offline-ticket",
                            "description": "TEST ONLY",
                            "workflow": "fixed",
                            "prompt": {"system_template": "Offline fixture"},
                            "input_schema": {"type": "string"},
                            "tools": ["ticket.create"],
                        }
                    ],
                }
            )
        )
        self.context = RunContext(
            identity=Identity(owner=self.principal.owner, scopes=frozenset({"ticket.write"})),
            agent_id="offline-ticket",
            run_id="demo_run",
            conversation_id="demo",
            deadline=time.time() + 60,
            config_digest=registry.agent("offline-ticket").digest,
        )
        self.call = ToolCall(
            call_id="create", name="ticket.create", arguments={"title": "Synthetic"}
        )
        self.executor = ToolExecutor(
            registry, self.ledger, approval_checker=approval, receipt_inspector=self.adapter.inspect
        )

    @property
    def key(self):
        return self.ledger.key(self.context, self.call)

    async def submit(self):
        try:
            await self.executor.execute(self.context, self.call)
        except NeedsReconciliation:
            pass
        return self.ledger.operation(self.key)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["submit", "recover"])
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "submit":
        args.directory.mkdir(exist_ok=False, parents=True)
    elif (
        not (args.directory / "local.sqlite").is_file()
        or not (args.directory / "remote.sqlite").is_file()
    ):
        parser.error("recover requires an existing explicit fixture directory")
    demo = OfflineTicketDemo(args.directory)
    if args.phase == "submit":
        await demo.submit()
    else:
        await demo.adapter.query(demo.principal, demo.key)
    row = demo.ledger.operation(demo.key)
    print(
        json.dumps(
            {
                "phase": row["phase"],
                "attempts": row["attempts"],
                "submissions": demo.remote.submissions,
                "committed_tickets": demo.remote.committed_tickets,
                "effect_applied": row["effect_applied"],
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
