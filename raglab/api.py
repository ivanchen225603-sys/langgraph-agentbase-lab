import asyncio
import hmac
import ipaddress
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from agentbase.context import SourceChecker
from agentbase.http import AgentAccess, BodyLimit, mount_agent_routes
from agentbase.registry import ComponentRegistry
from agentbase.tools import ResultPolicy, ToolAuthorizer

from .config import Settings
from .errors import LabError
from .models import (
    BuildInput,
    MemoryInput,
    MemoryPatch,
    Principal,
    QueryInput,
    RetrievalParams,
    RunInput,
    StrictModel,
)
from .observability import logger
from .runtime_adapter import identity as agent_identity
from .service import LabService


class CompareInput(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    left: RetrievalParams = Field(default_factory=RetrievalParams)
    right: RetrievalParams = Field(default_factory=lambda: RetrievalParams(top_k=6, mmr_lambda=0.7))


class ResumeInput(StrictModel):
    approve: bool | None = None


def create_app(
    settings: Settings | None = None,
    *,
    configure_registry: Callable[[ComponentRegistry], None] | None = None,
    source_checker: SourceChecker | None = None,
    local_source_guard=None,
    authorize_agent: AgentAccess | None = None,
    result_policy: ResultPolicy | None = None,
    tool_authorizer: ToolAuthorizer | None = None,
):
    settings = settings or Settings()
    service = LabService(
        settings,
        configure_registry=configure_registry,
        source_checker=source_checker,
        local_source_guard=local_source_guard,
        result_policy=result_policy,
        tool_authorizer=tool_authorizer,
        agent_authorizer=authorize_agent,
    )

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        app.state.service = service
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="RAG LangGraph Memory Lab", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(BodyLimit, max_bytes=settings.request_bytes)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=[settings.host, "127.0.0.1", "localhost", "[::1]"]
    )
    rate: dict[str, tuple[int, float]] = {}

    async def authenticate(request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise LabError("origin_forbidden", "cross-origin API access denied", 403)
        if any(
            name in request.headers for name in ("x-principal", "x-owner", "x-role", "x-scopes")
        ):
            raise LabError(
                "identity_spoof", "identity headers cannot grant principal or permissions", 403
            )
        if settings.auth_mode == "local":
            if not request.client or not ipaddress.ip_address(request.client.host).is_loopback:
                raise LabError("local_only", "local demo identity requires a loopback peer", 403)
            identity = Principal(owner="local")
        else:
            token = request.headers.get("authorization", "")
            if not token.startswith("Bearer "):
                raise LabError("unauthorized", "server-configured bearer token required", 401)
            owner = next(
                (
                    v
                    for k, v in settings.bearer_principals.items()
                    if hmac.compare_digest(k, token[7:])
                ),
                None,
            )
            if not owner:
                raise LabError("unauthorized", "invalid bearer token", 401)
            identity = Principal(owner=owner, scopes=frozenset({"read", "memory:write"}))
        service.runtime_adapter.remember(identity)
        request.state.principal = identity.owner
        return identity

    async def principal(request: Request):
        identity = await authenticate(request)
        now = time.monotonic()
        count, start = rate.get(identity.owner, (0, now))
        if now - start >= 60:
            count, start = 0, now
        if count >= settings.rate_per_minute:
            raise LabError("rate_limit", "principal rate quota exceeded", 429)
        rate[identity.owner] = count + 1, start
        request.state.principal = identity.owner
        return identity

    @app.middleware("http")
    async def correlate(request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        logger.info(
            "http request=%s method=%s status=%s",
            request.state.request_id,
            request.method,
            response.status_code,
        )
        return response

    @app.exception_handler(LabError)
    async def lab_error(request, exc):
        if exc.code in ("storage_unavailable", "checkpoint_unavailable"):
            service.available = False
        return JSONResponse(
            {
                "error": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "request_id": getattr(request.state, "request_id", None),
            },
            status_code=exc.status,
            headers={"Retry-After": "1"} if exc.status in (429, 503) else None,
        )

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            {
                "error": "validation_error",
                "details": [{"loc": x["loc"], "type": x["type"]} for x in exc.errors()],
            },
            status_code=422,
        )

    @app.exception_handler(TimeoutError)
    async def timeout(request, exc):
        return JSONResponse(
            {"error": "deadline_exceeded", "message": "total request deadline exceeded"},
            status_code=408,
        )

    @app.get("/health/live")
    async def live():
        return {"live": True}

    @app.get("/health/ready")
    @app.get("/health/ready/agent")
    async def ready():
        accepting = service.available and not service.closing and not service.queue.full()
        try:
            indexed = bool(service.store.active_index()) if hasattr(service, "store") else False
        except LabError:
            accepting, indexed = False, False
        return JSONResponse(
            {"ready": accepting and indexed, "accepting": accepting, "index_ready": indexed},
            status_code=200 if accepting and indexed else 503,
        )

    @app.get("/api/v1/config")
    async def config(p: Annotated[Principal, Depends(principal)]):
        return {
            **settings.public(),
            "principal": p.owner,
            "limits": {
                "queue": settings.queue_size,
                "workflows": settings.workflows,
                "tools": settings.tools_concurrency,
            },
            "tools": service.registry.describe(),
        }

    @app.get("/api/v1/index")
    async def index(p: Annotated[Principal, Depends(principal)]):
        manifest = service.store.active_index(p.owner)
        build = service.last_build if p.owner == "local" else {"status": "admin_only"}
        return {"manifest": manifest, "build": build}

    @app.post("/api/v1/index/rebuild")
    async def rebuild(body: BuildInput, p: Annotated[Principal, Depends(principal)]):
        if "index:write" not in p.scopes:
            raise LabError("forbidden", "index scope required", 403)
        return await service.build(p.owner, body)

    @app.post("/api/v1/index/documents/{document_id}/revoke")
    async def revoke(document_id: str, p: Annotated[Principal, Depends(principal)]):
        service.store.revoke(p.owner, document_id)
        return {"revoked": document_id}

    @app.post("/api/v1/answer")
    async def answer(body: QueryInput, p: Annotated[Principal, Depends(principal)]):
        return await service.answer(p, body)

    @app.post("/api/v1/search")
    async def search(body: QueryInput, p: Annotated[Principal, Depends(principal)]):
        result = await service.answer(p, body, search_only=True)
        return {
            "hits": result["evidence"],
            "trace_id": result["trace_id"],
            "stages_ms": result["stages_ms"],
            "degradation": result["degradation"],
            "index_version": result["index_version"],
        }

    @app.post("/api/v1/compare")
    async def compare(body: CompareInput, p: Annotated[Principal, Depends(principal)]):
        left, right = await asyncio.gather(
            service.answer(
                p, QueryInput(question=body.question, params=body.left), search_only=True
            ),
            service.answer(
                p, QueryInput(question=body.question, params=body.right), search_only=True
            ),
        )
        return {"left": left, "right": right}

    @app.get("/api/v1/memories")
    async def memories(p: Annotated[Principal, Depends(principal)]):
        return {"memories": service.memories.list(p.owner)}

    @app.post("/api/v1/memories", status_code=201)
    async def add_memory(body: MemoryInput, p: Annotated[Principal, Depends(principal)]):
        if "memory:write" not in p.scopes:
            raise LabError("forbidden", "memory write scope required", 403)
        return service.memories.create(p.owner, body)

    @app.post("/api/v1/memories/search")
    async def search_memory(body: QueryInput, p: Annotated[Principal, Depends(principal)]):
        return {"memories": service.memories.search(p.owner, body.question, body.params.top_k)}

    @app.patch("/api/v1/memories/{memory_id}")
    async def patch_memory(
        memory_id: str, body: MemoryPatch, p: Annotated[Principal, Depends(principal)]
    ):
        if "memory:write" not in p.scopes:
            raise LabError("forbidden", "memory write scope required", 403)
        return service.memories.patch(p.owner, memory_id, body)

    @app.delete("/api/v1/memories/{memory_id}")
    async def delete_memory(memory_id: str, p: Annotated[Principal, Depends(principal)]):
        if "memory:write" not in p.scopes:
            raise LabError("forbidden", "memory write scope required", 403)
        service.memories.delete(p.owner, memory_id)
        return {"deleted": True}

    @app.post("/api/v1/agent/runs", status_code=202)
    async def submit(
        body: RunInput,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
        p: Annotated[Principal, Depends(principal)],
    ):
        return await service.submit(p, idempotency_key, body)

    @app.get("/api/v1/agent/runs")
    async def runs(p: Annotated[Principal, Depends(principal)]):
        return {
            "runs": [
                await service.view(p.owner, r["run_id"])
                for r in service.store.requests(p.owner, 30)
            ]
        }

    @app.get("/api/v1/agent/runs/{run_id}")
    async def run(run_id: str, p: Annotated[Principal, Depends(principal)]):
        return await service.view(p.owner, run_id)

    @app.post("/api/v1/agent/runs/{run_id}/cancel")
    async def cancel(run_id: str, p: Annotated[Principal, Depends(principal)]):
        return await service.cancel(p.owner, run_id)

    @app.post("/api/v1/agent/runs/{run_id}/resume")
    async def resume(run_id: str, body: ResumeInput, p: Annotated[Principal, Depends(principal)]):
        return await service.resume(p, run_id, body.approve)

    @app.get("/api/v1/agent/runs/{run_id}/operations")
    async def operations(run_id: str, p: Annotated[Principal, Depends(principal)]):
        return {"operations": service.store.operations(p.owner, run_id)}

    @app.post("/api/v1/agent/runs/{run_id}/compensate")
    async def compensate(run_id: str, p: Annotated[Principal, Depends(principal)]):
        if not settings.enable_simulation or "memory:write" not in p.scopes:
            raise LabError("forbidden", "synthetic compensation is disabled", 403)
        return service.compensate(p.owner, run_id)

    @app.get("/api/v1/traces")
    async def traces(p: Annotated[Principal, Depends(principal)]):
        return {"traces": service.telemetry.for_owner(p.owner)}

    @app.get("/api/v1/diagnostics")
    async def diagnostics(p: Annotated[Principal, Depends(principal)]):
        return {
            "mode": settings.public(),
            "accepting": service.available,
            "queued": service.queue.qsize(),
            "active": len(service.active),
            "cache_entries": len(service.cache.values),
            "single_flights": len(service.cache.flights),
            "cache_hits": service.cache.hits,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(p: Annotated[Principal, Depends(principal)]):
        return service.telemetry.metrics()

    web = Path(__file__).parent / "web"

    @app.get("/static/sse.mjs", include_in_schema=False)
    async def sse_module():
        # Windows 的 MIME 注册表条目可能把 .mjs 映射为 text/plain，
        # 此时即使 HTTP 状态码成功，浏览器也会拒绝按模块加载。
        return FileResponse(web / "sse.mjs", media_type="application/javascript")

    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(web / "index.html")

    async def initial_actor(request: Request):
        return agent_identity(await principal(request))

    async def refresh_actor(request: Request):
        return agent_identity(await authenticate(request))

    mount_agent_routes(
        app,
        lambda: service.runtime,
        initial_actor,
        refresh_identity=refresh_actor,
        authorize_agent=authorize_agent,
        limits=settings.agent_http,
    )
    return app
