"""Explicit, read-only runtime artifact tools; no paths or caller-selected namespaces."""

from typing import Any, cast

from pydantic import JsonValue

from .artifacts import ArtifactReference, ArtifactScope
from .configuration import ToolDeclaration
from .contracts import AgentError, RunContext
from .schema import canonical_json


async def artifact_read_registration(
    arguments: dict[str, JsonValue], context: RunContext
) -> JsonValue:
    raise AgentError(
        "artifact_runtime", "artifact reader requires a trusted RunEngine binding", 503
    )


def artifact_read_tool(*, name: str = "artifact.read", scope: str = "read") -> ToolDeclaration:
    return ToolDeclaration(
        name=name,
        handler="agentbase.artifact_read",
        description="Read a bounded current artifact text page. Supply the published ID, version and digest; owner, agent and namespace are derived by the runtime. Offsets count Unicode characters.",
        scope=scope,
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
                "version": {"type": "integer", "minimum": 1, "maximum": (1 << 63) - 1},
                "digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1048576},
                "limit": {"type": "integer", "minimum": 1, "maximum": 4096},
            },
            "required": ["id", "version", "digest", "offset", "limit"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "reference": {"type": "object"},
                "text": {"type": "string", "maxLength": 4096},
                "offset": {"type": "integer", "minimum": 0},
                "next_offset": {"type": "integer", "minimum": 0},
                "total_chars": {"type": "integer", "minimum": 0},
                "truncated": {"type": "boolean"},
            },
            "required": ["reference", "text", "offset", "next_offset", "total_chars", "truncated"],
            "additionalProperties": False,
        },
    )


class ArtifactReader:
    def __init__(self, engine: Any):
        self.engine = engine

    async def _record(self, context, reference):
        reference = ArtifactReference.model_validate(reference)
        if (reference.agent_id, reference.namespace) != (context.agent_id, context.conversation_id):
            raise AgentError(
                "source_revoked", "artifact reference is outside this runtime scope", 403
            )
        row = self.engine.store.internal_request(context.run_id)
        sources = [reference.model_dump(mode="json")]
        if not await self.engine.authorized_sources(row, context.identity, sources):
            raise AgentError("source_revoked", "artifact sources are unavailable", 403)
        fresh = await self.engine.context(context.run_id)
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            if not self.engine.model_request_current_in(db, fresh, sources):
                raise AgentError("source_revoked", "artifact authorization changed", 403)
            return self.engine.artifacts.read_in(
                db,
                ArtifactScope(
                    owner=fresh.identity.owner,
                    agent_id=fresh.agent_id,
                    namespace=fresh.conversation_id,
                ),
                reference.id,
            )

    @staticmethod
    def _page(record, offset, limit):
        text = canonical_json(record.value, max_bytes=1048576).decode()
        if offset > len(text):
            raise AgentError("artifact_read_range", "artifact offset exceeds content length", 422)
        selected = text[offset : offset + limit]
        return {
            "reference": record.reference(),
            "text": selected,
            "offset": offset,
            "next_offset": offset + len(selected),
            "total_chars": len(text),
            "truncated": offset + len(selected) < len(text),
        }

    async def __call__(self, arguments: dict[str, JsonValue], context: RunContext) -> JsonValue:
        reference = ArtifactReference(
            id=cast(str, arguments["id"]),
            version=cast(int, arguments["version"]),
            digest=cast(str, arguments["digest"]),
            agent_id=context.agent_id,
            namespace=context.conversation_id,
        )
        policy = self.engine.registry.resolve(
            context.agent_id, context.config_digest
        ).definition.context.management
        limit = cast(int, arguments["limit"])
        if policy is not None and limit > policy.read_chars:
            raise AgentError("artifact_read_limit", "artifact read exceeds runtime policy", 422)
        record = await self._record(context, reference.model_dump(mode="json"))
        return cast(JsonValue, self._page(record, cast(int, arguments["offset"]), limit))

    async def sources(self, context, value) -> list[dict[str, JsonValue]]:
        if not isinstance(value, dict):
            raise AgentError("artifact_read_result", "artifact reader result is invalid", 503)
        record = await self._record(context, value.get("reference"))
        offset, text = value.get("offset"), value.get("text")
        if (
            type(offset) is not int
            or not isinstance(text, str)
            or value != self._page(record, offset, len(text))
        ):
            raise AgentError("artifact_read_result", "artifact reader result changed", 503)
        return [record.reference()]
