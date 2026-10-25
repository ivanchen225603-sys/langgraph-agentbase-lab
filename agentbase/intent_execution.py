"""Opt-in local arithmetic fixture; not a general proposal execution adapter."""

import copy
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .configuration import AgentPackage
from .contracts import AgentError, Identity
from .fixed_workflow import fixed_factory, input_pointer
from .intent_proposals import IntentConfirmation, IntentProposalService, OfflineStructuredIntent
from .registry import ComponentRegistry
from .runtime_contracts import AdmissionFence, RunOptions
from .schema import canonical_json, content_digest

if TYPE_CHECKING:
    from .runtime import RunEngine

AGENT = "intent-synthetic"
TOOL = "intent.synthetic_double"


async def _double(arguments, context):
    return {"doubled": arguments["number"] * 2}


def _package() -> AgentPackage:
    schema = {
        "type": "object",
        "properties": {"number": {"type": "integer", "minimum": -1000, "maximum": 1000}},
        "required": ["number"],
        "additionalProperties": False,
    }
    return AgentPackage.model_validate(
        {
            "tools": [
                {
                    "name": TOOL,
                    "handler": TOOL,
                    "handler_version": "1",
                    "description": "Local synthetic integer arithmetic; no IO",
                    "read_only": True,
                    "scope": "read",
                    "input_schema": schema,
                    "output_schema": {
                        "type": "object",
                        "properties": {"doubled": {"type": "integer"}},
                        "required": ["doubled"],
                        "additionalProperties": False,
                    },
                }
            ],
            "agents": [
                {
                    "id": AGENT,
                    "description": "Explicit synthetic intent execution fixture; no model or external IO",
                    "workflow": "fixed",
                    "workflow_version": "1",
                    "prompt": {"system_template": "Run local arithmetic only."},
                    "input_schema": schema,
                    "tools": [TOOL],
                    "tasks": [
                        {"id": "double", "tool": TOOL, "input_bindings": {"number": "/number"}}
                    ],
                    "context": {"memory_scope": "none", "max_memories": 0},
                    "budget": {"timeout_seconds": 10.0, "max_tool_calls": 1},
                }
            ],
        }
    )


@dataclass(frozen=True)
class SyntheticIntentBinding:
    registry: ComponentRegistry
    digest: str

    def check(self, engine: "RunEngine", agent_id: str):
        if agent_id != AGENT or engine.registry is not self.registry:
            raise AgentError("intent_execution_disabled", "synthetic Agent is not enabled", 403)
        bound, expected = engine.registry.agent(agent_id), _package()
        if (
            bound.digest != self.digest
            or bound.definition != expected.agents[0]
            or list(bound.tools) != expected.tools
            or engine.registry.handler(bound.tools[0], bound=bound) is not _double
            or engine.registry.workflow_factory(bound.definition, bound=bound) is not fixed_factory
            or engine.tools.resolver is not None
            or engine._application_result_policy is not None
        ):
            raise AgentError(
                "intent_execution_binding", "synthetic release or handler changed", 409
            )


def register_synthetic_intent_agent(registry: ComponentRegistry) -> SyntheticIntentBinding:
    """Trusted host calls after register_workflows; no client-defined handlers/config."""
    registry.register_handler(TOOL, _double, version="1")
    registry.bind(_package())
    return SyntheticIntentBinding(registry, registry.agent(AGENT).digest)


class SyntheticIntentExecutor:
    def __init__(self, proposals: IntentProposalService, binding: SyntheticIntentBinding):
        self.proposals, self.binding, self.engine = proposals, binding, proposals.engine
        self._policy(AGENT)

    def _policy(self, agent_id):
        self.binding.check(self.engine, agent_id)
        if type(self.proposals.interpreter) is not OfflineStructuredIntent:
            raise AgentError("intent_execution_mode", "execution requires the offline fixture", 403)

    def enabled(self, agent_id):
        try:
            self._policy(agent_id)
            return True
        except AgentError:
            return False

    def install(self):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS agent_intent_execution_meta(version INTEGER NOT NULL)"
            )
            rows = db.execute("SELECT version FROM agent_intent_execution_meta").fetchall()
            if rows and (len(rows) != 1 or rows[0][0] != 1):
                raise AgentError(
                    "intent_execution_schema", "unsupported execution metadata schema", 503
                )
            if not rows:
                db.execute("INSERT INTO agent_intent_execution_meta VALUES(1)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS agent_intent_executions(proposal_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,owner TEXT NOT NULL,agent_id TEXT NOT NULL,run_id TEXT NOT NULL UNIQUE,request_key TEXT NOT NULL,proof TEXT NOT NULL,admitted_at REAL NOT NULL,UNIQUE(owner,agent_id,request_key))"
            )

    @staticmethod
    def request_key(session_id, proposal_id):
        return content_digest({"intent-synthetic": [session_id, proposal_id]})[7:]

    def _load(self, db, identity, agent_id, session_id, proposal_id, credential, body=None):
        self._policy(agent_id)
        session = self.proposals._session(db, identity, agent_id, session_id, credential)
        record = self.proposals._record(db, session, proposal_id)
        mapping = db.execute(
            "SELECT * FROM agent_intent_executions WHERE proposal_id=? AND session_id=? AND owner=? AND agent_id=?",
            (proposal_id, session_id, identity.owner, agent_id),
        ).fetchone()
        if body is not None:
            self.proposals.assert_choice(record, body)
            status = self.proposals._status(db, identity, session, record)
            if status == "expired":
                raise AgentError("intent_expired", "intent proposal expired", 410)
            if status not in {"confirmed", "executed"} or record["confirmed_at"] is None:
                raise AgentError(
                    "intent_execution_choice", "a current independent confirmation is required", 409
                )
            if (
                body.draft.action != "propose_task"
                or body.draft.task is None
                or body.draft.target_run_id is not None
            ):
                raise AgentError(
                    "intent_execution_action",
                    "only synthetic new-task proposals are executable",
                    409,
                )
            if "read" not in identity.scopes:
                raise AgentError("forbidden", "synthetic read scope is required", 403)
        return session, record, dict(mapping) if mapping else None

    def _proof(self, identity, credential, session, proposal_id, body):
        return content_digest(
            {
                "owner": identity.owner,
                "credential": credential,
                "session": session["id"],
                "epoch": session["epoch"],
                "binding": session["binding"],
                "proposal": proposal_id,
                "confirmation": body.model_dump(mode="json"),
            }
        )

    @staticmethod
    def _matches_confirmed_payload(encoded, body: IntentConfirmation) -> bool:
        task = body.draft.task
        if task is None:
            return False
        try:
            stored = json.loads(encoded)
            arguments = {}
            for declaration in _package().agents[0].tasks:
                derived = copy.deepcopy(declaration.arguments)
                derived.update(
                    {
                        name: input_pointer(task.input, pointer)
                        for name, pointer in declaration.input_bindings.items()
                    }
                )
                arguments[declaration.id] = derived
            # fixed_workflow executes task_arguments, so input alone is insufficient.
            return canonical_json(
                {"input": stored["input"], "task_arguments": stored["task_arguments"]}
            ) == canonical_json({"input": task.input, "task_arguments": arguments})
        except (AgentError, ValueError, TypeError, KeyError):
            return False

    def _admission(
        self, db, run, *, created, identity, agent_id, session_id, proposal_id, credential, body
    ):
        session, record, mapping = self._load(
            db, identity, agent_id, session_id, proposal_id, credential, body
        )
        actual = db.execute(
            "SELECT * FROM requests WHERE run_id=? AND owner=? AND agent_id=?",
            (run["run_id"], identity.owner, agent_id),
        ).fetchone()
        if actual is None:
            raise AgentError("intent_execution_binding", "admitted Run is unavailable", 409)
        run = dict(actual)
        control = db.execute(
            "SELECT * FROM agent_run_controls WHERE run_id=?", (run["run_id"],)
        ).fetchone()
        if (
            (run["owner"], run["agent_id"], run["binding"], run["epoch"])
            != (identity.owner, agent_id, self.binding.digest, session["epoch"])
            or control is None
            or control["conversation_id"] != session["conversation_id"]
            or not self._matches_confirmed_payload(run["payload"], body)
            or run["key"] != self.request_key(session_id, proposal_id)
        ):
            raise AgentError(
                "intent_execution_binding", "Run does not match the confirmed proposal", 409
            )
        proof = self._proof(identity, credential, session, proposal_id, body)
        now = self.proposals.clock()
        if now >= record["expires_at"] or now >= session["expires_at"]:
            raise AgentError("intent_expired", "intent expired before admission", 410)
        if not created:
            if (
                mapping is None
                or mapping["run_id"] != run["run_id"]
                or mapping["proof"] != proof
                or record["status"] != "executed"
            ):
                raise AgentError(
                    "intent_execution_replay", "existing Run is not this proposal's admission", 409
                )
            return
        if mapping is not None or record["status"] != "confirmed":
            raise AgentError("intent_execution_replay", "proposal has already been consumed", 409)
        db.execute(
            "INSERT INTO agent_intent_executions VALUES(?,?,?,?,?,?,?,?)",
            (
                proposal_id,
                session_id,
                identity.owner,
                agent_id,
                run["run_id"],
                run["key"],
                proof,
                now,
            ),
        )
        updated = db.execute(
            "UPDATE agent_intent_proposals SET status='executed' WHERE id=? AND status='confirmed'",
            (proposal_id,),
        )
        if updated.rowcount != 1:
            raise AgentError("intent_execution_replay", "proposal consumption changed", 409)

    async def execute(
        self,
        identity: Identity,
        agent_id: str,
        session_id: str,
        proposal_id: str,
        body: IntentConfirmation,
        credential: str,
        reauthorize: Callable[[], Awaitable[None]],
    ):
        await reauthorize()
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            session, _, _ = self._load(
                db, identity, agent_id, session_id, proposal_id, credential, body
            )
        task = body.draft.task
        assert task is not None  # _load validated action/task before any admission.

        def check(db, run, created):
            self._admission(
                db,
                run,
                created=created,
                identity=identity,
                agent_id=agent_id,
                session_id=session_id,
                proposal_id=proposal_id,
                credential=credential,
                body=body,
            )

        run = await self.engine.submit(
            identity,
            agent_id,
            self.request_key(session_id, proposal_id),
            task.input,
            RunOptions(conversation_id=session["conversation_id"]),
            admission_fence=AdmissionFence(
                reauthorize,
                lambda db, run: check(db, run, True),
                lambda db, run: check(db, run, False),
            ),
        )
        await reauthorize()
        return {
            "proposal_id": proposal_id,
            "execution_connected": True,
            "mode": "synthetic-local-read-only",
            "admitted": True,
            "run": run,
            "tool_approval": False,
        }

    async def query(self, identity, agent_id, session_id, proposal_id, credential, reauthorize):
        await reauthorize()
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            session, record, mapping = self._load(
                db, identity, agent_id, session_id, proposal_id, credential
            )
            if mapping:
                body = IntentConfirmation.model_validate(
                    {
                        "revision": record["revision"],
                        "request_digest": record["request_digest"],
                        "decision_digest": record["decision_digest"],
                        "draft": json.loads(record["payload"])["draft"],
                    }
                )
                stored = db.execute(
                    "SELECT * FROM requests WHERE owner=? AND run_id=? AND agent_id=?",
                    (identity.owner, mapping["run_id"], agent_id),
                ).fetchone()
                if (
                    stored is None
                    or stored["binding"] != self.binding.digest
                    or stored["key"] != self.request_key(session_id, proposal_id)
                    or stored["epoch"] != session["epoch"]
                    or not self._matches_confirmed_payload(stored["payload"], body)
                    or mapping["proof"]
                    != self._proof(identity, credential, session, proposal_id, body)
                ):
                    raise AgentError(
                        "intent_execution_storage", "execution mapping is inconsistent", 503
                    )
        run = await self.engine.view(identity.owner, mapping["run_id"]) if mapping else None
        await reauthorize()
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            self._load(db, identity, agent_id, session_id, proposal_id, credential)
        return {
            "proposal_id": proposal_id,
            "execution_connected": True,
            "mode": "synthetic-local-read-only",
            "admitted": mapping is not None,
            "run": run,
            "tool_approval": False,
        }
