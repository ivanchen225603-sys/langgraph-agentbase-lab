"""有界的事件订阅，检查点设在最外层真实的 ASGI send 上。"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .contracts import AgentError, Identity, RunEvent
from .runtime import RunEngine
from .runtime_contracts import TERMINAL

if TYPE_CHECKING:
    from .http import AgentAccess, AgentHttpLimits, RequestAuthenticator

STREAM_STATE = "agentbase_stream_session"


def cursor_number(value: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]{0,18}", value):
        raise AgentError("event_cursor_invalid", "invalid event cursor", 400)
    number = int(value)
    if number > 9223372036854775807:
        raise AgentError("event_cursor_invalid", "cursor exceeds SQLite integer range", 400)
    return number


def event_cursor(request: Request, run_id: str) -> int:
    queries = request.query_params.getlist("after")
    headers = request.headers.getlist("last-event-id")
    if len(queries) > 1 or len(headers) > 1:
        raise AgentError("event_cursor_invalid", "duplicate event cursor", 400)
    query = cursor_number(queries[0]) if queries else None
    header = None
    if headers:
        parts = headers[0].split(":")
        if len(parts) != 2 or parts[0] != run_id:
            raise AgentError("event_cursor_invalid", "event cursor belongs to another run", 400)
        header = cursor_number(parts[1])
    if header is not None and query is not None and header != query:
        raise AgentError("event_cursor_invalid", "header and query cursors differ", 400)
    return header if header is not None else query if query is not None else 0


def encode_event(event: RunEvent) -> bytes:
    # JSON 会转义内容中的换行符，因此一条持久化事件恰好对应一个完整帧。
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.run_id}:{event.seq}\nevent: {event.kind}\ndata: {data}\n\n".encode()


def error_frame(code: str) -> bytes:
    if not re.fullmatch(r"[a-zA-Z0-9_]{1,80}", code):
        code = "stream_failed"
    data = json.dumps({"error": code, "snapshot_required": True}, separators=(",", ":"))
    return f"event: stream.error\ndata: {data}\n\n".encode()


class StreamRejected(Exception):
    def __init__(self, code):
        self.code = code


class StreamSendTimeout(Exception):
    pass


class StreamLeases:
    """单一事件循环内的配额；释放由外层中间件负责，而非生产者。"""

    def __init__(self, limits: AgentHttpLimits):
        self.limits = limits
        self.active: dict[str, tuple[str, str]] = {}

    def acquire(self, owner: str, run_id: str) -> str:
        entries = list(self.active.values())
        if (
            len(entries) >= self.limits.connections
            or sum(entry[0] == owner for entry in entries) >= self.limits.connections_per_owner
            or entries.count((owner, run_id)) >= self.limits.connections_per_run
        ):
            raise AgentError("stream_capacity", "event subscription capacity is exhausted", 429)
        key = uuid.uuid4().hex
        self.active[key] = (owner, run_id)
        return key

    def release(self, key):
        self.active.pop(key, None)


@dataclass
class StreamSession:
    request: Request
    engine: RunEngine
    identity: Identity
    agent_id: str
    run_id: str
    cursor: int
    refresh_identity: RequestAuthenticator
    authorize_agent: AgentAccess | None
    limits: AgentHttpLimits
    leases: StreamLeases
    lease: str
    deadline: float

    async def refresh(self):
        if (
            self.authorize_agent is not None
            and await self.authorize_agent(self.identity, self.agent_id) is not True
        ):
            raise AgentError("agent_forbidden", "Agent access is no longer authorized", 403)
        if await self.engine.identity(self.identity.owner) != self.identity:
            raise AgentError("permission_changed", "server identity changed", 403)
        # 最后一道异步策略检查：同一 owner 的另一个令牌不能凭更早那次 ACL await
        # 让这份原始凭证继续保持授权状态。
        actor = await self.refresh_identity(self.request)
        if not isinstance(actor, Identity) or actor != self.identity:
            raise AgentError("permission_changed", "original credential changed", 403)
        row = self.engine.store.request(actor.owner, self.run_id)
        if row["agent_id"] != self.agent_id:
            raise AgentError("run_not_found", "original Agent binding changed", 404)

    async def before_send(self, message):
        body = message.get("body", b"")
        if message["type"] not in {"http.response.start", "http.response.body"} or (
            message["type"] == "http.response.start" and body
        ):
            raise AgentError("event_store_conflict", "unexpected stream message", 409)
        if len(body) > self.limits.frame_bytes:
            raise AgentError("stream_frame_limit", "event frame is too large", 409)
        if body.startswith(b"id: "):
            first = body.split(b"\n", 1)[0][4:].decode("ascii")
            parts = first.split(":")
            if len(parts) != 2 or parts[0] != self.run_id:
                raise AgentError("event_store_conflict", "frame identity differs", 409)
            seq = cursor_number(parts[1])
            if seq != self.cursor + 1:
                raise AgentError("event_store_conflict", "frame sequence differs", 409)
            events = await self.engine.events(self.identity.owner, self.run_id, seq - 1, limit=1)
            if not events or events[0].seq != seq or encode_event(events[0]) != body:
                raise AgentError("event_store_conflict", "frame differs from retained event", 409)
            await self.refresh()
            self.engine.check_event_frame(self.identity, events[0])
        else:
            if body not in {b"", b": heartbeat\n\n"}:
                raise AgentError("event_store_conflict", "unexpected stream body", 409)
            # 即便是空闲心跳（heartbeat）也重新检查最后可见的数据源，避免被撤销的
            # 候选文本一直显示到下一个模型增量到来为止。
            events = await self.engine.events(
                self.identity.owner, self.run_id, max(0, self.cursor - 1), limit=1
            )
            await self.refresh()
            for event in events:
                self.engine.check_event_frame(self.identity, event)

    def sent(self, message):
        body = message.get("body", b"")
        if body.startswith(b"id: "):
            self.cursor = cursor_number(body.split(b"\n", 1)[0].decode().rsplit(":", 1)[1])


class EventSendBounds:
    """在用户中间件注册之后，作为最外层安装。

    对 ASGI 服务器实际提供的 send 做授权与限时，包括 response.start。带缓冲的
    中间件既无法预授权后续帧，也无法在网络发送仍被阻塞时释放自己的租约。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        scope.setdefault("state", {})
        started = False

        async def bounded(message):
            nonlocal started
            session = scope.get("state", {}).get(STREAM_STATE)
            if session is None:
                return await send(message)
            try:
                async with asyncio.timeout(
                    max(0.0, session.deadline - asyncio.get_running_loop().time())
                ):
                    await session.before_send(message)
            except TimeoutError:
                raise StreamRejected("stream_duration") from None
            except Exception as exc:
                raise StreamRejected(getattr(exc, "code", "stream_failed")) from exc
            remaining = min(
                session.limits.send_seconds, session.deadline - asyncio.get_running_loop().time()
            )
            if remaining <= 0:
                raise StreamSendTimeout()
            try:
                async with asyncio.timeout(remaining):
                    await send(message)
            except TimeoutError as exc:
                raise StreamSendTimeout() from exc
            session.sent(message)
            if message["type"] == "http.response.start":
                started = True

        try:
            await self.app(scope, receive, bounded)
        except StreamRejected as exc:
            session = scope.get("state", {}).get(STREAM_STATE)
            timeout = session.limits.send_seconds if session else 1.0
            try:
                if started:
                    await asyncio.wait_for(
                        send(
                            {
                                "type": "http.response.body",
                                "body": error_frame(exc.code),
                                "more_body": False,
                            }
                        ),
                        timeout,
                    )
                else:
                    async with asyncio.timeout(timeout):
                        await JSONResponse({"error": exc.code}, status_code=403)(
                            scope, receive, send
                        )
            except (TimeoutError, OSError):
                pass
        except (StreamSendTimeout, OSError):
            if STREAM_STATE not in scope["state"]:
                raise
            # 已超时或发生 OSError 的传输层无法再接收任何数据。
            pass
        finally:
            session = scope.get("state", {}).pop(STREAM_STATE, None)
            if session is not None:
                session.leases.release(session.lease)


class EventResponse(Response):
    def __init__(self, session: StreamSession):
        super().__init__(
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
        del self.headers["content-length"]
        self.session = session

    async def __call__(self, scope, receive, send):
        async def produce():
            cursor = self.session.cursor
            heartbeat = asyncio.get_running_loop().time()
            await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
            while True:
                if self.session.engine.closing:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    return
                events = await self.session.engine.events(
                    self.session.identity.owner, self.session.run_id, cursor, limit=1
                )
                if events:
                    event = events[0]
                    await send(
                        {
                            "type": "http.response.body",
                            "body": encode_event(event),
                            "more_body": True,
                        }
                    )
                    cursor = event.seq
                    if event.kind in {"run.final", "run.terminal", "run.awaiting_review"}:
                        await send({"type": "http.response.body", "body": b"", "more_body": False})
                        return
                else:
                    engine = self.session.engine
                    row = engine.store.request(self.session.identity.owner, self.session.run_id)
                    snapshot = await engine.checkpoint_state(row)
                    stopped = (
                        self.session.run_id not in engine.active
                        and self.session.run_id not in engine.pending
                    )
                    if engine.store.final_receipt(
                        self.session.identity.owner, self.session.run_id
                    ) is not None or (
                        stopped
                        and (
                            snapshot.values.get("status") in TERMINAL or engine.interrupts(snapshot)
                        )
                    ):
                        await send({"type": "http.response.body", "body": b"", "more_body": False})
                        return
                    now = asyncio.get_running_loop().time()
                    if now - heartbeat >= self.session.limits.heartbeat_seconds:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": b": heartbeat\n\n",
                                "more_body": True,
                            }
                        )
                        heartbeat = now
                    await asyncio.sleep(self.session.limits.poll_seconds)

        async def disconnect():
            while (await receive())["type"] != "http.disconnect":
                pass

        producer = asyncio.create_task(produce())
        listener = asyncio.create_task(disconnect())
        try:
            async with asyncio.timeout(
                max(0.0, self.session.deadline - asyncio.get_running_loop().time())
            ):
                done, _ = await asyncio.wait(
                    {producer, listener}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    await task
        except AgentError as exc:
            raise StreamRejected(exc.code) from exc
        except TimeoutError:
            raise StreamRejected("stream_duration") from None
        finally:
            for task in (producer, listener):
                if not task.done():
                    task.cancel()
            await asyncio.gather(producer, listener, return_exceptions=True)
