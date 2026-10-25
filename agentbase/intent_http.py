"""Proposal routes; synthetic execution routes require an explicit trusted host opt-in."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, Request, Response

from .contracts import AgentError, BaseContract, Identifier, Identity
from .intent_proposals import IntentConfirmation, IntentProposalInput, IntentProposalService
from .schema import content_digest

if TYPE_CHECKING:
    from .http import RequestAuthenticator
    from .intent_execution import SyntheticIntentExecutor
    from .runtime import RunEngine


class IntentSessionInput(BaseContract):
    conversation_id: Identifier


def mount_intent_routes(
    app: FastAPI,
    engine_getter: Callable[[], "RunEngine"],
    service_getter: Callable[[], IntentProposalService],
    authenticate: "RequestAuthenticator",
    refresh_identity: "RequestAuthenticator",
    *,
    execution_getter: Callable[[], "SyntheticIntentExecutor"] | None = None,
):
    def describe(value, agent_id):
        if execution_getter is None:
            return value
        enabled = execution_getter().enabled(agent_id)
        return value | {
            "execution_connected": enabled,
            "execution_mode": "synthetic-local-read-only" if enabled else "disabled",
        }

    async def actor(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Authorization, Cookie"
        return await authenticate(request)

    async def current(request: Request, identity: Identity, agent_id: str):
        origin = request.headers.get("origin")
        if (
            origin and origin.rstrip("/") != str(request.base_url).rstrip("/")
        ) or request.headers.get("sec-fetch-site") == "cross-site":
            raise AgentError("origin_forbidden", "cross-origin intent access denied", 403)
        if any(
            name in request.headers for name in ("x-owner", "x-principal", "x-role", "x-scopes")
        ):
            raise AgentError("identity_spoof", "identity headers cannot grant authority", 403)
        engine = engine_getter()
        if engine.closing or not engine.available:
            raise AgentError("not_accepting", "intent host is unavailable", 503)
        if not await engine.agent_allowed(identity, agent_id):
            raise AgentError("agent_forbidden", "Agent access is not authorized", 403)
        if (
            await refresh_identity(request) != identity
            or await engine.identity(identity.owner) != identity
        ):
            raise AgentError("intent_identity", "current authentication context changed", 403)
        if not await engine.agent_allowed(identity, agent_id):
            raise AgentError("agent_forbidden", "Agent access changed during authentication", 403)
        if await engine.identity(identity.owner) != identity:
            raise AgentError("intent_identity", "identity changed during Agent authorization", 403)
        if engine_getter() is not engine or engine.closing or not engine.available:
            raise AgentError("not_accepting", "intent runtime ownership changed", 503)
        return content_digest(
            {
                "authorization": request.headers.get("authorization", ""),
                "cookie": request.headers.get("cookie", ""),
            }
        )

    @app.post("/api/v1/agents/{agent_id}/intent-sessions", status_code=201)
    async def open_session(
        request: Request,
        agent_id: Identifier,
        body: IntentSessionInput,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)
        return describe(
            service_getter().open_session(identity, agent_id, body.conversation_id, credential),
            agent_id,
        )

    @app.post("/api/v1/agents/{agent_id}/intent-sessions/{session_id}/invalidate")
    async def invalidate(
        request: Request,
        agent_id: Identifier,
        session_id: Identifier,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)
        return describe(
            service_getter().invalidate(identity, agent_id, session_id, credential), agent_id
        )

    @app.get("/api/v1/agents/{agent_id}/intent-sessions/{session_id}")
    async def session_view(
        request: Request,
        agent_id: Identifier,
        session_id: Identifier,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)
        return describe(
            service_getter().session_view(identity, agent_id, session_id, credential), agent_id
        )

    @app.post("/api/v1/agents/{agent_id}/intent-sessions/{session_id}/proposals", status_code=201)
    async def propose(
        request: Request,
        agent_id: Identifier,
        session_id: Identifier,
        body: IntentProposalInput,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)

        async def reauthorize():
            if await current(request, identity, agent_id) != credential:
                raise AgentError("intent_identity", "intent credential changed", 403)

        result = await service_getter().propose(
            identity,
            agent_id,
            session_id,
            body.draft,
            credential,
            reauthorize,
            body.expected_revision,
        )
        return describe(result, agent_id)

    @app.get("/api/v1/agents/{agent_id}/intent-sessions/{session_id}/proposals/{proposal_id}")
    async def view(
        request: Request,
        agent_id: Identifier,
        session_id: Identifier,
        proposal_id: Identifier,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)
        return describe(
            service_getter().view(identity, agent_id, session_id, proposal_id, credential), agent_id
        )

    @app.post(
        "/api/v1/agents/{agent_id}/intent-sessions/{session_id}/proposals/{proposal_id}/confirm"
    )
    async def confirm(
        request: Request,
        agent_id: Identifier,
        session_id: Identifier,
        proposal_id: Identifier,
        body: IntentConfirmation,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)
        return describe(
            service_getter().decide(identity, agent_id, session_id, proposal_id, body, credential),
            agent_id,
        )

    @app.post(
        "/api/v1/agents/{agent_id}/intent-sessions/{session_id}/proposals/{proposal_id}/revoke"
    )
    async def revoke(
        request: Request,
        agent_id: Identifier,
        session_id: Identifier,
        proposal_id: Identifier,
        body: IntentConfirmation,
        identity: Annotated[Identity, Depends(actor)],
    ):
        credential = await current(request, identity, agent_id)
        return describe(
            service_getter().decide(
                identity, agent_id, session_id, proposal_id, body, credential, revoke=True
            ),
            agent_id,
        )

    if execution_getter is not None:

        async def execution_auth(request, identity, agent_id):
            credential = await current(request, identity, agent_id)

            async def reauthorize():
                if await current(request, identity, agent_id) != credential:
                    raise AgentError("intent_identity", "intent credential changed", 403)

            return credential, reauthorize

        @app.post(
            "/api/v1/agents/{agent_id}/intent-sessions/{session_id}/proposals/{proposal_id}/execute",
            status_code=202,
        )
        async def execute(
            request: Request,
            agent_id: Identifier,
            session_id: Identifier,
            proposal_id: Identifier,
            body: IntentConfirmation,
            identity: Annotated[Identity, Depends(actor)],
        ):
            credential, reauthorize = await execution_auth(request, identity, agent_id)
            return await execution_getter().execute(
                identity, agent_id, session_id, proposal_id, body, credential, reauthorize
            )

        @app.get(
            "/api/v1/agents/{agent_id}/intent-sessions/{session_id}/proposals/{proposal_id}/execution"
        )
        async def execution(
            request: Request,
            agent_id: Identifier,
            session_id: Identifier,
            proposal_id: Identifier,
            identity: Annotated[Identity, Depends(actor)],
        ):
            credential, reauthorize = await execution_auth(request, identity, agent_id)
            return await execution_getter().query(
                identity, agent_id, session_id, proposal_id, credential, reauthorize
            )
