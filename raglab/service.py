import asyncio
import copy
import json
import re
import time
import uuid
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langsmith import tracing_context

from agentbase.context import SourceChecker
from agentbase.contracts import AgentError
from agentbase.registry import ComponentRegistry
from agentbase.runtime_contracts import AgentAuthorizer, RunOptions
from agentbase.tools import ResultPolicy, ToolAuthorizer

from .budget import BudgetedEmbeddings, BudgetedGenerator, ProviderBudget, provider_scope
from .config import Settings
from .embedding import LexicalHashEmbeddings
from .errors import LabError, UnknownEffect
from .generation import PROMPT_VERSION, OpenAIAdapter
from .graphs import AgentWorkflow, RagWorkflow
from .ingestion import Ingestion, IngestState
from .memory import MemoryService
from .models import BuildInput, Principal, QueryInput, RunInput, RunView
from .observability import Telemetry
from .persistence import LEGACY_AGENT_ID, ProcessLock, Store, digest
from .resilience import AnswerCache, remaining
from .runtime_adapter import RagRuntimeAdapter
from .tools import ToolRegistry

TERMINAL = {"succeeded", "partial", "failed", "cancelled", "timed_out", "needs_reconciliation"}


def legacy_errors(method):
    async def wrapped(*args, **kwargs):
        try:
            return await method(*args, **kwargs)
        except AgentError as exc:
            if exc.code == "admission_timeout":
                raise TimeoutError() from exc
            message = (
                "persistent workflow admission unavailable"
                if exc.code == "not_accepting"
                else "cancel recorded but checkpoint cleanup failed"
                if exc.code == "cancel_cleanup_failed"
                else "request recorded but graph initialization failed; restart and resume same run"
                if exc.code == "checkpoint_unavailable" and method.__name__ == "submit"
                else exc.message
            )
            raise LabError(exc.code, message, exc.status) from exc

    return wrapped


def runtime_field(name):
    return property(
        lambda self: getattr(self.runtime, name),
        lambda self, value: setattr(self.runtime, name, value),
    )


class LabService:
    queue = runtime_field("queue")
    workers = runtime_field("workers")
    active = runtime_field("active")
    pending = runtime_field("pending")
    admission_lock = runtime_field("admission_lock")
    available = runtime_field("available")
    closing = runtime_field("closing")
    terminal_checkpoint = runtime_field("terminal_checkpoint")

    def __init__(
        self,
        settings: Settings,
        *,
        configure_registry: Callable[[ComponentRegistry], None] | None = None,
        source_checker: SourceChecker | None = None,
        local_source_guard=None,
        result_policy: ResultPolicy | None = None,
        tool_authorizer: ToolAuthorizer | None = None,
        agent_authorizer: AgentAuthorizer | None = None,
    ):
        if (source_checker is None) != (local_source_guard is None):
            raise ValueError("domain sources require both asynchronous and local SQL policies")
        self.configure_registry = configure_registry
        self.extension_agent_authorizer = agent_authorizer
        self.extension_result_policy, self.extension_tool_authorizer = (
            result_policy,
            tool_authorizer,
        )
        self.extension_source_checker, self.extension_source_guard = (
            source_checker,
            local_source_guard,
        )
        self.settings = settings
        self.lock = ProcessLock(settings.data_dir / "owner.lock")
        self.stack = AsyncExitStack()
        self.telemetry = Telemetry()
        self.cache = AnswerCache(settings.cache_entries, settings.workflows + settings.queue_size)
        self.query_slots = asyncio.Semaphore(settings.workflows)
        self.build_lock = asyncio.Lock()
        self.last_build: dict = {"status": "not_started"}

    async def start(self):
        self.lock.acquire()
        try:
            self.store = Store(self.settings.data_dir / "business.sqlite")
            self.memories = MemoryService(self.store)
            self.provider_budget = ProviderBudget(self.store)
            adapter = OpenAIAdapter(self.settings) if self.settings.mode == "openai" else None
            self.generator = BudgetedGenerator(adapter, self.provider_budget) if adapter else None
            self.embeddings = (
                BudgetedEmbeddings(adapter.embeddings, self.provider_budget)
                if adapter
                else LexicalHashEmbeddings(self.settings.embedding_dimensions)
            )
            self.embedding_id = (
                f"openai:{self.settings.embedding_model}:{self.settings.embedding_dimensions}"
                if self.generator
                else f"lexical-hash-v1:{self.settings.embedding_dimensions}"
            )
            self.ingestion = Ingestion(
                self.store, self.embeddings, self.settings, self.embedding_id
            )
            self.rag = RagWorkflow(self)
            saver = await self.stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(
                    str(self.settings.data_dir / "checkpoints.sqlite")
                )
            )
            await saver.setup()
            self.agent = AgentWorkflow(self, saver)
            self.registry = ToolRegistry(self)
            self.runtime_adapter = RagRuntimeAdapter(self, saver)
            self.runtime = self.runtime_adapter.engine
            await self.runtime_adapter.adopt_for_query()
            self.runtime.recover_startup_actions()
            await self.runtime.start()
        except BaseException:
            await self.stack.aclose()
            self.lock.close()
            raise
        return self

    async def close(self):
        if hasattr(self, "runtime"):
            await self.runtime.close()
        await self.stack.aclose()
        self.lock.close()

    def index_version(self, owner="local"):
        manifest = self.store.active_index(owner)
        if not manifest:
            raise LabError("index_not_ready", "build a synthetic index first", 409)
        if manifest["embedding"] != self.embedding_id:
            raise LabError(
                "index_model_mismatch", "embedding changed; rebuild index before querying", 409
            )
        return manifest["version"]

    async def build(self, owner: str, request: BuildInput):
        deadline = time.time() + request.timeout_seconds
        try:
            async with asyncio.timeout(remaining(deadline)):
                async with self.build_lock:
                    self.last_build = {"status": "building", "parameters": request.model_dump()}
                    with (
                        tracing_context(enabled=False),
                        provider_scope(
                            uuid.uuid4().hex, self.settings.max_build_model_calls, deadline
                        ),
                    ):
                        result = await cast(Any, self.ingestion.graph).ainvoke(
                            cast(
                                IngestState,
                                {
                                    "owner": owner,
                                    "request": request.model_dump(),
                                    "deadline": deadline,
                                },
                            ),
                            RunnableConfig(recursion_limit=10),
                        )
                    self.last_build = {"status": "succeeded", "manifest": result["manifest"]}
                    return self.last_build
        except BaseException as exc:
            self.last_build = {
                "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                "error": exc.code if isinstance(exc, LabError) else type(exc).__name__,
            }
            raise

    def sanitize_value(self, owner, value, *, db=None):
        result = copy.deepcopy(value)
        allowed = (
            self.store.allowed
            if db is None
            else lambda actor, document: self.runtime_adapter.local_sources(
                db, actor, [document], []
            )
        )
        memory_current = (
            self.store.memory_current
            if db is None
            else lambda actor, memory, version: self.runtime_adapter.local_sources(
                db, actor, [], [(memory, version)]
            )
        )
        if "citations" in result:
            evidence = result.get("evidence", [])
            revoked = any(not allowed(owner, c["document_id"]) for c in result["citations"])
            revoked = revoked or any(
                not allowed(owner, h["chunk"]["document_id"]) for h in evidence
            )
            revoked = revoked or any(
                not allowed(owner, d) for d in result.get("context_documents", [])
            )
            if revoked:
                result["context_documents"] = []
                result.update(
                    status="refused",
                    answer="资料已撤权或过期，请重新检索。",
                    reason="evidence_revoked",
                    citations=[],
                    evidence=[],
                    resolved_question=result.get("question", ""),
                )
        if "memories" in result:
            result["memories"] = [
                m for m in result["memories"] if memory_current(owner, m["memory_id"], m["version"])
            ]
        return result

    def cache_key(self, principal, request, version, epoch, history):
        acl, memory = self.store.revisions()
        return digest(
            [
                principal.owner,
                principal.permission_version,
                acl,
                memory,
                version,
                self.embedding_id,
                self.settings.chat_model if self.generator else "extractive-v1",
                PROMPT_VERSION,
                request.model_dump(exclude={"timeout_seconds"}),
                epoch,
                history,
            ]
        )

    async def answer(self, principal: Principal, request: QueryInput, *, search_only=False):
        deadline, version = (
            time.time() + request.timeout_seconds,
            self.index_version(principal.owner),
        )
        epoch = self.store.epoch(principal.owner, request.conversation_id)
        history = self.store.history(
            principal.owner,
            request.conversation_id,
            epoch,
            self.settings.context_turns,
            self.settings.context_chars,
        )
        key = self.cache_key(principal, request, version, epoch, history)

        async def compute():
            with (
                tracing_context(enabled=False),
                provider_scope(uuid.uuid4().hex, self.settings.max_model_calls, deadline),
            ):
                async with asyncio.timeout(remaining(deadline)), self.query_slots:
                    result = await cast(Any, self.rag.graph).ainvoke(
                        {
                            "principal": principal.owner,
                            "question": request.question,
                            "history": history,
                            "index_version": version,
                            "deadline": deadline,
                            "params": request.params.model_dump(),
                            "branches": {},
                            "degradation": [],
                        },
                        RunnableConfig(recursion_limit=12),
                    )
            return result["answer"]

        async with asyncio.timeout(remaining(deadline)):
            result = await self.cache.get_or_compute(key, compute)
        result = self.sanitize_value(principal.owner, result)
        if epoch != self.store.epoch(principal.owner, request.conversation_id):
            raise LabError("topic_changed", "late result discarded because topic changed", 409)
        if not search_only and result["status"] == "answered":
            self.store.add_history(
                principal.owner,
                request.conversation_id,
                epoch,
                result["trace_id"],
                {
                    "question": result["resolved_question"][:1000],
                    "answer": result["answer"][:1200],
                    "source_run": result["trace_id"],
                    "documents": list(
                        dict.fromkeys(
                            [c["document_id"] for c in result["citations"]]
                            + result.get("context_documents", [])
                        )
                    ),
                },
            )
        return result

    @staticmethod
    def config(row) -> RunnableConfig:
        return {
            "configurable": {"thread_id": row["thread"]},
            "recursion_limit": 64,
        }

    def initial(self, row):
        payload = json.loads(row["payload"])
        return {
            "state_version": self.settings.state_version,
            "graph_version": self.settings.graph_version,
            "principal": row["owner"],
            "permission_version": payload["permission_version"],
            "conversation_id": payload["conversation_id"],
            "topic_epoch": row["epoch"],
            "run_id": row["run_id"],
            "deadline": row["deadline"],
            "index_version": payload["index_version"],
            "message": payload["message"],
            "params": payload["params"],
            "tasks": payload["tasks"],
            "history": [],
            "results": {},
            "events": {},
            "waves": 0,
            "status": "queued",
            "approved": False,
            "approval_required": payload["pause_for_review"]
            or any(not tool["read_only"] for tool in payload["tool_contracts"].values()),
        }

    @legacy_errors
    async def submit(self, principal: Principal, key: str, request: RunInput):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key):
            raise LabError(
                "idempotency_key", "Idempotency-Key must be 1..64 ASCII identifier characters"
            )
        self.runtime_adapter.remember(principal)
        actor = await self.runtime_adapter.resolve_identity(principal.owner)
        view = await self.runtime.submit(
            actor,
            LEGACY_AGENT_ID,
            key,
            request.model_dump(),
            RunOptions(
                conversation_id=request.conversation_id,
                switch_topic=request.switch_topic,
                pause_for_review=request.pause_for_review,
                timeout_seconds=float(request.timeout_seconds),
            ),
        )
        return self.legacy_view(view)

    def enqueue(self, row, command=None):
        return self.runtime.enqueue(row, command)

    @legacy_errors
    async def checkpoint_state(self, row):
        return await self.runtime.checkpoint_state(row)

    @legacy_errors
    async def update_checkpoint(self, row, values, *, as_node):
        return await self.runtime.update_checkpoint(row, values, as_node=as_node)

    async def worker(self):
        return await self.runtime.worker()

    async def execute(self, row, command=None):
        return await self.runtime.execute(row, command)

    def legacy_view(self, view):
        row = self.runtime.store.internal_request(view["run_id"])
        payload = json.loads(row["payload"])
        fields = {name: value for name, value in view.items() if name in RunView.model_fields}
        fields.setdefault("topic_epoch", view["epoch"])
        fields.setdefault("thread_id", row["thread"])
        fields.setdefault("checkpoint_id", None)
        fields.setdefault("tasks", payload["tasks"])
        fields.setdefault("results", {})
        fields.setdefault("unfinished", [task["id"] for task in payload["tasks"]])
        fields.setdefault("decisions", [])
        return RunView.model_validate(fields).model_dump()

    def legacy_row(self, owner, run_id):
        row = self.store.request(owner, run_id)
        if row["agent_id"] != LEGACY_AGENT_ID:
            raise LabError("not_found", "legacy run not found", 404)
        return row

    @legacy_errors
    async def view(self, owner, run_id):
        self.legacy_row(owner, run_id)
        return self.legacy_view(await self.runtime.view(owner, run_id))

    @legacy_errors
    async def cancel(self, owner, run_id):
        self.legacy_row(owner, run_id)
        return self.legacy_view(await self.runtime.cancel(owner, run_id))

    @legacy_errors
    async def resume(self, principal, run_id, approve=None):
        self.legacy_row(principal.owner, run_id)
        self.runtime_adapter.remember(principal)
        actor = await self.runtime_adapter.resolve_identity(principal.owner)
        view = await self.runtime.view(principal.owner, run_id)
        review = None
        if view["status"] == "awaiting_review":
            if approve is not True or len(view["interrupts"]) != 1:
                raise LabError(
                    "approval_required",
                    "resume requires explicit approve=true for this review",
                    409,
                )
            review = view["interrupts"][0]
        return self.legacy_view(
            await self.runtime.resume(
                actor,
                run_id,
                review_id=review.get("review_id") if review else None,
                digest=review.get("digest") if review else None,
            )
        )

    def compensate(self, owner, run_id, *, inject_unknown=False):
        """显式的合成补偿：按校验过的 DAG 顺序反向执行，绝不假定回滚一定成功。"""
        row = self.legacy_row(owner, run_id)
        if run_id in self.active or run_id in self.pending or self.store.uncertain(run_id):
            raise LabError(
                "not_reconciled",
                "active or unknown operations must stop and be reconciled first",
                409,
            )
        self.store.request_cancel(owner, run_id)
        tasks = json.loads(row["payload"])["tasks"]
        # 先拓扑排序再反向执行：补偿失败或结果未知的任务会阻断其上游任务。
        ordered, remaining_tasks = [], list(tasks)
        while remaining_tasks:
            ready = [t for t in remaining_tasks if all(d in ordered for d in t["depends_on"])]
            for t in ready:
                ordered.append(t["id"])
                remaining_tasks.remove(t)
        outcomes = []
        for task_id in reversed(ordered):
            key = f"{run_id}:{task_id}"
            with self.store.connection() as db:
                receipt = db.execute(
                    "SELECT phase FROM operations WHERE key=? AND owner=?", (key, owner)
                ).fetchone()
                if not receipt or receipt[0] in ("compensated", "not_applied"):
                    continue
                db.execute("UPDATE operations SET phase='compensating' WHERE key=?", (key,))
            with self.store.connection() as db:
                db.execute("UPDATE simulation SET compensated=1 WHERE key=?", (key,))
            if inject_unknown:
                raise UnknownEffect()
            self.store.receipt(key, {"effect": "compensated", "operation_key": key}, "compensated")
            outcomes.append(key)
        return {"compensated": outcomes}
