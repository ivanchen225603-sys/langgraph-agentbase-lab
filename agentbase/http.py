"""为既有引擎提供可信的 HTTP 组装；路由本身不拥有 worker，也不拥有存储。"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field, JsonValue, model_validator

from .contracts import AgentError, BaseContract, Identifier, Identity
from .intent_execution import SyntheticIntentExecutor
from .intent_proposals import IntentProposalService
from .runtime import RunEngine
from .runtime_contracts import RunOptions
from .sse import (
    STREAM_STATE,
    EventResponse,
    EventSendBounds,
    StreamLeases,
    StreamSession,
    event_cursor,
)

RequestAuthenticator = Callable[[Request], Awaitable[Identity]]
AgentAccess = Callable[[Identity, str], Awaitable[bool]]


class AgentHttpLimits(BaseContract):
    request_bytes: int = Field(65536, ge=1024, le=1048576)
    connections: int = Field(16, ge=1, le=1024)
    connections_per_owner: int = Field(4, ge=1, le=128)
    connections_per_run: int = Field(2, ge=1, le=32)
    duration_seconds: float = Field(120.0, ge=0.1, le=3600)
    send_seconds: float = Field(5.0, ge=0.01, le=60)
    heartbeat_seconds: float = Field(10.0, ge=0.01, le=60)
    poll_seconds: float = Field(0.1, ge=0.001, le=5)
    frame_bytes: int = Field(131072, ge=1024, le=262144)


class AgentRunInput(BaseContract):
    input: JsonValue
    options: RunOptions = Field(default_factory=RunOptions)


class AgentResumeInput(BaseContract):
    review_id: Identifier | None = None
    digest: str | None = Field(None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def paired(self):
        if (self.review_id is None) != (self.digest is None):
            raise ValueError("review_id and digest must be supplied together")
        return self


class BodyLimit:
    """在 JSON 解析之前约束实际请求字节数，涵盖分块传输的 body。"""

    def __init__(self, app, max_bytes, receive_timeout=10, prefix=None):
        self.app, self.max_bytes = app, max_bytes
        self.receive_timeout, self.prefix = receive_timeout, prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or (
            self.prefix is not None and not scope["path"].startswith(self.prefix)
        ):
            return await self.app(scope, receive, send)
        body = bytearray()
        deadline = asyncio.get_running_loop().time() + self.receive_timeout
        while True:
            try:
                message = await asyncio.wait_for(
                    receive(), max(0, deadline - asyncio.get_running_loop().time())
                )
            except TimeoutError:
                return await JSONResponse({"error": "body_timeout"}, status_code=408)(
                    scope, receive, send
                )
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                return await JSONResponse(
                    {"error": "body_limit", "message": "request body too large"}, status_code=413
                )(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        pending = True

        async def replay():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


async def agent_error_response(request: Request, exc: Exception):
    assert isinstance(exc, AgentError)
    return JSONResponse(
        {
            "error": exc.code,
            "message": exc.message,
            "request_id": getattr(request.state, "request_id", None),
        },
        status_code=exc.status,
        headers={"Retry-After": "1"} if exc.status in (429, 503) else None,
    )


def mount_agent_routes(
    app: FastAPI,
    engine_getter: Callable[[], RunEngine],
    authenticate: RequestAuthenticator,
    *,
    refresh_identity: RequestAuthenticator,
    authorize_agent: AgentAccess | None = None,
    limits: AgentHttpLimits | None = None,
    intent_service_getter: Callable[[], IntentProposalService] | None = None,
    intent_execution_getter: Callable[[], SyntheticIntentExecutor] | None = None,
):
    """只挂载一次，需在启动之前、其他用户中间件之后执行。

    这些认证函数属于可信宿主代码。authenticate 可以计入一次请求；
    refresh_identity 必须重新认证原始凭据，且不再计入新的请求。二者都不能从
    客户端输入中获取 scopes。Agent 访问权限默认对所有已配置的 Agent 生效；
    需要按 Agent 粒度的 ACL 的宿主应自行提供策略。
    """
    limits = limits or AgentHttpLimits()
    if hasattr(app.state, "agent_streams"):
        raise ValueError("Agent routes must be mounted once")
    leases = StreamLeases(limits)
    app.state.agent_streams = leases
    app.add_middleware(BodyLimit, max_bytes=limits.request_bytes, prefix="/api/v1/agents")
    if AgentError not in app.exception_handlers:
        app.add_exception_handler(AgentError, agent_error_response)

    async def actor(request: Request):
        engine_getter().add_agent_authorizer(authorize_agent)
        identity = await authenticate(request)
        if (
            not isinstance(identity, Identity)
            or await engine_getter().identity(identity.owner) != identity
        ):
            raise AgentError("identity", "current server identity differs", 403)
        return identity

    async def access(identity: Identity, agent_id: str):
        if authorize_agent is not None and await authorize_agent(identity, agent_id) is not True:
            raise AgentError("agent_forbidden", "Agent access is not authorized", 403)

    def original(identity: Identity, agent_id: str, run_id: str):
        row = engine_getter().store.request(identity.owner, run_id)
        if row["agent_id"] != agent_id:
            raise AgentError("run_not_found", "run is not available for this Agent", 404)
        return row

    @app.get("/api/v1/agents")
    async def agents(identity: Annotated[Identity, Depends(actor)]):
        result = []
        for entry in engine_getter().registry.agents():
            if (
                authorize_agent is not None
                and await authorize_agent(identity, entry["id"]) is not True
            ):
                continue
            definition = engine_getter().registry.agent(entry["id"]).definition
            result.append(
                {
                    **entry,
                    "input_schema": definition.input_schema,
                    "output_schema": definition.output_schema,
                }
            )
        if await engine_getter().identity(identity.owner) != identity:
            raise AgentError("identity", "current server identity changed", 403)
        return {"agents": result}

    @app.get("/api/v1/agents/{agent_id}/runs")
    async def run_index(
        request: Request,
        agent_id: Identifier,
        identity: Annotated[Identity, Depends(actor)],
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
        after: Identifier | None = None,
    ):
        if set(request.query_params) - {"limit", "after"} or any(
            len(request.query_params.getlist(key)) != 1 for key in request.query_params
        ):
            raise AgentError("run_list_query", "unknown or repeated Run index parameter")
        engine = engine_getter()

        async def current():
            refreshed = await refresh_identity(request)
            if (
                not isinstance(refreshed, Identity)
                or refreshed != identity
                or await engine.identity(identity.owner) != identity
            ):
                raise AgentError(
                    "permission_changed", "original credential or identity changed", 403
                )
            if not await engine.agent_allowed(identity, agent_id):
                raise AgentError("agent_forbidden", "Agent access is not authorized", 403)
            if await engine.identity(identity.owner) != identity:
                raise AgentError("permission_changed", "current server identity changed", 403)

        await current()
        engine.registry.agent(agent_id)
        identifiers, next_after = engine.store.list_run_ids(
            identity.owner, agent_id, after=after, limit=limit
        )
        await current()
        return JSONResponse(
            {
                "runs": [{"run_id": run_id, "agent_id": agent_id} for run_id in identifiers],
                "next_after": next_after,
            },
            headers={"Cache-Control": "no-store", "Vary": "Authorization, Cookie"},
        )

    @app.post("/api/v1/agents/{agent_id}/runs", status_code=202)
    async def submit(
        agent_id: Identifier,
        body: AgentRunInput,
        idempotency_key: Annotated[Identifier, Header(alias="Idempotency-Key")],
        identity: Annotated[Identity, Depends(actor)],
    ):
        await access(identity, agent_id)
        return await engine_getter().submit(
            identity, agent_id, idempotency_key, body.input, body.options
        )

    @app.get("/api/v1/agents/{agent_id}/runs/{run_id}")
    async def view(
        agent_id: Identifier, run_id: Identifier, identity: Annotated[Identity, Depends(actor)]
    ):
        original(identity, agent_id, run_id)
        await access(identity, agent_id)
        return await engine_getter().view(identity.owner, run_id)

    @app.post("/api/v1/agents/{agent_id}/runs/{run_id}/cancel")
    async def cancel(
        agent_id: Identifier, run_id: Identifier, identity: Annotated[Identity, Depends(actor)]
    ):
        original(identity, agent_id, run_id)
        await access(identity, agent_id)
        if await engine_getter().identity(identity.owner) != identity:
            raise AgentError("identity", "current server identity changed", 403)
        return await engine_getter().cancel(identity.owner, run_id)

    @app.post("/api/v1/agents/{agent_id}/runs/{run_id}/resume")
    async def resume(
        agent_id: Identifier,
        run_id: Identifier,
        body: AgentResumeInput,
        identity: Annotated[Identity, Depends(actor)],
    ):
        original(identity, agent_id, run_id)
        await access(identity, agent_id)
        return await engine_getter().resume(
            identity, run_id, review_id=body.review_id, digest=body.digest
        )

    @app.get("/health/ready/runtime")
    async def ready():
        engine = engine_getter()
        accepting = engine.available and not engine.closing and not engine.queue.full()
        return JSONResponse(
            {"ready": accepting, "accepting": accepting}, status_code=200 if accepting else 503
        )

    @app.get("/api/v1/agents/{agent_id}/runs/{run_id}/events")
    async def events(
        request: Request,
        agent_id: Identifier,
        run_id: Identifier,
        identity: Annotated[Identity, Depends(actor)],
    ):
        original(identity, agent_id, run_id)
        await access(identity, agent_id)
        cursor = event_cursor(request, run_id)
        await engine_getter().events(identity.owner, run_id, cursor, limit=1)
        refreshed = await refresh_identity(request)
        if not isinstance(refreshed, Identity) or refreshed != identity:
            raise AgentError("permission_changed", "original credential changed", 403)
        await access(identity, agent_id)
        lease = leases.acquire(identity.owner, run_id)
        session = StreamSession(
            request,
            engine_getter(),
            identity,
            agent_id,
            run_id,
            cursor,
            refresh_identity,
            authorize_agent,
            limits,
            leases,
            lease,
            asyncio.get_running_loop().time() + limits.duration_seconds,
        )
        request.scope["state"][STREAM_STATE] = session
        return EventResponse(session)

    # 最后注册者在用户中间件中位于最外层：即便内层 BaseHTTPMiddleware 预先取走了
    # body 帧，它仍能看到真实的服务端 send。
    app.add_middleware(EventSendBounds)
    if intent_service_getter is not None:
        from .intent_http import mount_intent_routes

        mount_intent_routes(
            app,
            engine_getter,
            intent_service_getter,
            actor,
            refresh_identity,
            execution_getter=intent_execution_getter,
        )
