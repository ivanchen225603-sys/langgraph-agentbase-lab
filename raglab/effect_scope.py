"""在单次公开 handler 调用期间，可信地持有某个存量业务意图的所有权。"""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class EffectOwner:
    business_key: str
    public_key: str
    token: str
    fingerprint: str


effect_owner: ContextVar[EffectOwner | None] = ContextVar("rag_effect_owner", default=None)


def owns_intent(db, key, owner, run_id, *, receipt=False):
    scope = effect_owner.get()
    if scope is None or scope.business_key != key:
        return False
    return (
        db.execute(
            "SELECT 1 FROM legacy_action_links l JOIN agent_tool_operations p ON p.key=l.public_key "
            "WHERE l.domain_key=? AND l.owner=? AND l.run_id=? AND l.public_key=? AND l.attempt_token=? "
            "AND l.fingerprint=? AND p.attempt_token=l.attempt_token AND p.fingerprint=l.fingerprint "
            "AND p.owner=l.owner AND p.agent_id=l.agent_id AND p.run_id=l.run_id AND p.call_id=l.call_id "
            "AND p.binding=l.binding AND p.attempts=l.attempt AND (p.phase='intent' OR (? AND p.phase='unknown')) AND p.read_only=0",
            (key, owner, run_id, scope.public_key, scope.token, scope.fingerprint, int(receipt)),
        ).fetchone()
        is not None
    )
