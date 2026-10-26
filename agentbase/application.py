"""Optional local application assembly; the existing RunEngine owns all execution."""

import asyncio
import importlib
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import BinaryIO

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .context import ContextProvider, SourceChecker
from .contracts import AgentError
from .http import AgentAccess, AgentHttpLimits, RequestAuthenticator, mount_agent_routes
from .intent_execution import SyntheticIntentBinding, SyntheticIntentExecutor
from .intent_proposals import (
    IntentInterpreter,
    IntentProposalLimits,
    IntentProposalService,
    OfflineStructuredIntent,
)
from .registry import ComponentRegistry
from .run_store import SqliteRunStore
from .runtime import RunEngine
from .runtime_contracts import AgentAuthorizer, EngineSettings, IdentityResolver
from .storage import MemoryLimits, SqliteContextStore, SqliteDatabase
from .tools import ResultPolicy, ToolAuthorizer, ToolResolver


class RuntimeProcessLock:
    """OS-held exclusive ownership, released on close or process death."""

    def __init__(self, path: Path):
        self.path = path.parent.resolve() / path.name
        self.file: BinaryIO | None = None

    def acquire(self):
        if self.file is not None:
            raise AgentError("already_running", "runtime lock is already held", 503)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise AgentError("runtime_path", "runtime lock must be a regular file", 503)
        handle = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                locks = importlib.import_module("fcntl")
                locks.flock(handle.fileno(), locks.LOCK_EX | locks.LOCK_NB)
        except BaseException:
            handle.close()
            raise
        self.file = handle

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


async def _finish_close(engine: RunEngine):
    # Cancellation of the host must not release the OS lock while Runs still execute.
    closing = asyncio.create_task(engine.close())
    cancelled = False
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            cancelled = True
    closing.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def open_runtime(
    registry: ComponentRegistry,
    *,
    data_dir: Path,
    identity_resolver: IdentityResolver,
    settings: EngineSettings | None = None,
    memory_limits: MemoryLimits | None = None,
    source_checker: SourceChecker | None = None,
    tool_authorizer: ToolAuthorizer | None = None,
    agent_authorizer: AgentAuthorizer | None = None,
    tool_resolver: ToolResolver | None = None,
    result_policy: ResultPolicy | None = None,
    local_source_guard=None,
    observer=None,
) -> AsyncIterator[RunEngine]:
    """Own a dedicated local directory; do not point this at a RAG business store.

    The trusted host binds workflows/providers/handlers before entering. Historical
    bindings must also be restored before recovery. No anonymous identity is created.
    Existing Runs are not automatically approved or resumed.
    """
    directory = await asyncio.to_thread(Path(data_dir).resolve)
    lock = RuntimeProcessLock(directory / "owner.lock")
    try:
        lock.acquire()
    except OSError as exc:
        raise AgentError("already_running", "runtime directory cannot be acquired", 503) from exc
    async with AsyncExitStack() as stack:
        stack.callback(lock.close)
        database = SqliteDatabase(directory / "runs.sqlite")
        policy = settings or EngineSettings()
        store = SqliteRunStore(database.connection, policy)
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            store.install(db)
        contexts = SqliteContextStore(database.connection, memory_limits=memory_limits)
        saver = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite"))
        )
        await saver.setup()
        engine = RunEngine(
            registry,
            store,
            saver,
            ContextProvider(contexts, contexts, source_checker),
            identity_resolver,
            settings=policy,
            tool_authorizer=tool_authorizer,
            agent_authorizer=agent_authorizer,
            tool_resolver=tool_resolver,
            result_policy=result_policy,
            local_source_guard=local_source_guard,
            observer=observer,
        )
        stack.push_async_callback(_finish_close, engine)
        engine.recover_startup_actions()
        await engine.start()
        yield engine


def create_agent_app(
    runtime_factory: Callable[[], AbstractAsyncContextManager[RunEngine]],
    *,
    authenticate: RequestAuthenticator,
    refresh_identity: RequestAuthenticator,
    authorize_agent: AgentAccess | None = None,
    limits: AgentHttpLimits | None = None,
    intent_interpreter: IntentInterpreter | None = None,
    intent_limits: IntentProposalLimits | None = None,
    synthetic_intent: SyntheticIntentBinding | None = None,
) -> FastAPI:
    """Business-independent HTTP host; authentication callbacks remain mandatory."""

    if synthetic_intent is not None and type(intent_interpreter) is not OfflineStructuredIntent:
        raise ValueError("synthetic intent execution requires OfflineStructuredIntent")

    @asynccontextmanager
    async def lifespan(app):
        async with runtime_factory() as engine:
            engine.add_agent_authorizer(authorize_agent)
            app.state.engine = engine
            try:
                if intent_interpreter is not None:
                    intents = IntentProposalService(
                        engine, intent_interpreter, limits=intent_limits
                    )
                    intents.install()
                    app.state.intents = intents
                    if synthetic_intent is not None:
                        execution = SyntheticIntentExecutor(intents, synthetic_intent)
                        execution.install()
                        app.state.intent_execution = execution
                yield
            finally:
                app.state.intents = None
                app.state.intent_execution = None
                app.state.engine = None

    app = FastAPI(title="Agent Base", lifespan=lifespan)
    app.state.engine = None
    app.state.intent_execution = None

    def current():
        if app.state.engine is None:
            raise AgentError("not_accepting", "runtime is not started", 503)
        return app.state.engine

    def current_intents():
        if app.state.intents is None:
            raise AgentError("not_accepting", "intent service is not started", 503)
        return app.state.intents

    def current_execution():
        if app.state.intent_execution is None:
            raise AgentError("not_accepting", "synthetic execution is not started", 503)
        return app.state.intent_execution

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        # Default FastAPI errors echo submitted values, including secret-bearing input.
        return JSONResponse({"error": "validation_error"}, status_code=422)

    mount_agent_routes(
        app,
        current,
        authenticate,
        refresh_identity=refresh_identity,
        authorize_agent=authorize_agent,
        limits=limits,
        intent_service_getter=current_intents if intent_interpreter is not None else None,
        intent_execution_getter=current_execution if synthetic_intent is not None else None,
    )
    return app
