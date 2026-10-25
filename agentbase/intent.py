"""Optional host-side intent proposals; reviewing a proposal never executes it."""

import json
import re
import sqlite3
import unicodedata
from typing import Annotated, Literal

from pydantic import Field, JsonValue, ValidationError, model_validator

from .contracts import AgentError, BaseContract, Identifier, Identity, Message
from .run_store import SqliteRunStore
from .schema import canonical_json, content_digest

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
IntentAction = Literal["respond", "clarify", "propose_task", "propose_replace", "propose_cancel"]


class IntentTask(BaseContract):
    """Business input only; execution still uses the selected Agent's schema."""

    input: JsonValue


class IntentDecision(BaseContract):
    """Untrusted model output, deliberately without identity or approval fields."""

    request_id: Identifier
    request_digest: Digest
    action: IntentAction
    interpretation: Literal["explicit", "negated", "ambiguous", "conflicting"]
    message: str = Field(min_length=1, max_length=2000)
    task: IntentTask | None = None
    target_run_id: Identifier | None = None
    target_input_digest: Digest | None = None

    @model_validator(mode="after")
    def coherent_proposal(self):
        if not self.message.strip():
            raise ValueError("intent message must not be blank")
        if (self.task is not None) != (self.action in {"propose_task", "propose_replace"}):
            raise ValueError("only task and replacement proposals require task input")
        needs_target = self.action in {"propose_replace", "propose_cancel"}
        if needs_target:
            if self.target_run_id is None or self.target_input_digest is None:
                raise ValueError("control proposals require an exact target and input digest")
        elif self.target_run_id is not None or self.target_input_digest is not None:
            raise ValueError("this action cannot name a target")
        return self


class IntentContext(BaseContract):
    """The authenticated host creates and refreshes this context, never the model."""

    request_id: Identifier
    identity: Identity
    agent_id: Identifier
    conversation_id: Identifier
    epoch: int = Field(ge=0)


class IntentTarget(BaseContract):
    """Trusted read snapshot, not an executable control token.

    active only excludes cancellation, final receipts, and old epochs; execution
    deadlines, paused review clocks, ACLs, and checkpoints remain runtime policy.
    """

    run_id: Identifier
    owner: Identifier
    agent_id: Identifier
    conversation_id: Identifier
    epoch: int = Field(ge=0)
    binding: Digest
    input_digest: Digest
    active: bool


def read_intent_target(store: SqliteRunStore, identity: Identity, run_id: str) -> IntentTarget:
    """Read a consistent snapshot without admitting, cancelling, or approving a Run."""
    with store.connection() as db:
        db.execute("BEGIN")
        return read_intent_target_in(db, identity, run_id)


def read_intent_target_in(db: sqlite3.Connection, identity: Identity, run_id: str) -> IntentTarget:
    """Reuse the caller's transaction for proposal freshness checks."""
    row = db.execute(
        "SELECT r.*, c.conversation_id, "
        "EXISTS(SELECT 1 FROM agent_final_receipts f WHERE f.run_id=r.run_id) AS final "
        "FROM requests r JOIN agent_run_controls c USING(run_id) "
        "WHERE r.owner=? AND r.run_id=?",
        (identity.owner, run_id),
    ).fetchone()
    if row is None:
        raise AgentError("not_found", "run not found", 404)
    epoch = SqliteRunStore.epoch(db, row["owner"], row["agent_id"], row["conversation_id"])
    return IntentTarget(
        run_id=row["run_id"],
        owner=row["owner"],
        agent_id=row["agent_id"],
        conversation_id=row["conversation_id"],
        epoch=row["epoch"],
        binding=row["binding"],
        input_digest=content_digest(json.loads(row["payload"])["input"]),
        active=(not row["cancel"] and not row["final"] and row["epoch"] == epoch),
    )


class IntentReview(BaseContract):
    action: Literal["respond", "clarify", "proposal"]
    message: str
    reason: Literal[
        "response", "clarification", "invalid_decision", "uncertain", "negation", "proposal_only"
    ]
    proposal: IntentDecision | None = None


# A conservative veto for common negations, not a general language classifier.
_NEGATION = re.compile(
    r"不要|别|勿|不执行|不需要|不必|先不|暂不|暂停|不想|无需|禁止|不允许|不用|"
    r"\b(?:do\s+not|don['’]t|never|not\s+now|without|hold\s+off)\b"
)


class IntentSession:
    """One model response for one authenticated request; there is no execution API.

    The host owns request IDs, current context, origin labels, and target snapshots.
    Sessions must not be reconstructed from model output or serialized as permissions.
    """

    def __init__(
        self,
        context: IntentContext,
        message: Message,
        *,
        origin: str,
        target: IntentTarget | None = None,
    ):
        if origin != "authenticated-user" or message.role != "user" or message.sources:
            raise AgentError("intent_origin", "intent requires a direct authenticated user", 403)
        if not message.content.strip():
            raise AgentError("intent_input", "intent requires a nonempty user message")
        canonical_json(message.content, max_bytes=32768)
        if target is not None and (
            not target.active
            or (target.owner, target.agent_id, target.conversation_id, target.epoch)
            != (
                context.identity.owner,
                context.agent_id,
                context.conversation_id,
                context.epoch,
            )
        ):
            raise AgentError("intent_target", "target is not current in this conversation", 409)
        self._context = context
        self._text = message.content
        self._target = target
        self._used = False
        self._digest = content_digest(
            {
                "context": context.model_dump(mode="json"),
                "text": self._text,
                "target": target.model_dump(mode="json") if target else None,
            }
        )

    def model_input(self) -> dict:
        """Supply as user data alongside IntentDecision.model_json_schema()."""
        return {
            "request_id": self._context.request_id,
            "request_digest": self._digest,
            "user_text": self._text,
            "target": (
                {"run_id": self._target.run_id, "input_digest": self._target.input_digest}
                if self._target
                else None
            ),
        }

    def review(
        self,
        value: object,
        *,
        current: IntentContext,
        target: IntentTarget | None = None,
    ) -> IntentReview:
        """Validate untrusted JSON, consuming the session even on rejected output."""
        if self._used:
            raise AgentError("intent_replay", "intent response has already been reviewed", 409)
        self._used = True
        if current != self._context:
            raise AgentError("intent_stale", "authenticated request context changed", 409)
        try:
            # Bound size/depth before validation; parsing also detaches mutable input.
            decision = IntentDecision.model_validate_json(canonical_json(value, max_bytes=32768))
        except (AgentError, ValidationError):
            return IntentReview(
                action="clarify",
                message="无法确定请求，请明确要解释、新建、替换还是取消哪个任务。",
                reason="invalid_decision",
            )
        if decision.request_id != current.request_id or decision.request_digest != self._digest:
            raise AgentError("intent_binding", "intent belongs to another request", 409)
        if decision.interpretation != "explicit":
            return IntentReview(
                action="clarify",
                message="请求含否定、歧义或冲突，请先明确需求。",
                reason="uncertain",
            )
        if decision.action in {"respond", "clarify"}:
            return IntentReview(
                action="respond" if decision.action == "respond" else "clarify",
                message=decision.message,
                reason="response" if decision.action == "respond" else "clarification",
            )
        normalized = unicodedata.normalize("NFKC", self._text).casefold()
        if _NEGATION.search(normalized):
            return IntentReview(
                action="clarify",
                message="请求含否定表达，请确认希望采取的动作。",
                reason="negation",
            )
        if decision.action in {"propose_replace", "propose_cancel"} and (
            self._target is None
            or target != self._target
            or decision.target_run_id != self._target.run_id
            or decision.target_input_digest != self._target.input_digest
        ):
            raise AgentError("intent_target", "target changed or does not match request", 409)
        return IntentReview(
            action="proposal",
            message=decision.message,
            reason="proposal_only",
            proposal=decision,
        )
