import asyncio
import copy
import time
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from agentbase.contracts import ToolCall

from .errors import LabError, UnknownEffect
from .generation import contextualize, evidence_problem, extractive_chain, validate_claims
from .models import AgentState, AnswerResult, RagState, RetrievalParams, TaskSpec, ToolResult
from .resilience import Circuit, guarded, remaining
from .retrieval import SnapshotRetriever, fuse, lexical_rerank, mmr, validate_rerank


class RagWorkflow:
    def __init__(self, service):
        self.service = service
        self.circuits = {c: Circuit() for c in ("vector", "keyword", "generation")}
        self.reranker = lexical_rerank
        graph = StateGraph(RagState)
        for name in ("normalize", "vector", "keyword", "fuse", "generate", "validate"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "vector")
        graph.add_edge("normalize", "keyword")
        graph.add_edge(["vector", "keyword"], "fuse")
        graph.add_edge("fuse", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", END)
        self.graph = graph.compile()

    async def normalize(self, state):
        remaining(state["deadline"])
        history = self.service.store.filter_history(
            state["principal"],
            state.get("history", []),
            self.service.settings.context_turns,
            self.service.settings.context_chars,
        )
        resolved, ambiguity = contextualize(state["question"], history)
        return {
            "history": history,
            "resolved_question": resolved,
            "branches": {"normalize": {"ambiguity": ambiguity}},
        }

    async def channel(self, state, name):
        started = time.perf_counter()
        try:
            rows = self.service.store.snapshot(state["index_version"], state["principal"])
            retriever = SnapshotRetriever(
                rows=rows,
                embeddings=self.service.embeddings,
                channel=name,
                candidate_pool=state["params"]["candidate_pool"],
            )
            documents = await guarded(
                lambda: retriever.ainvoke(state["resolved_question"]),
                state["deadline"],
                self.circuits[name],
            )
            return {
                "branches": {
                    name: {
                        "documents": [d.metadata for d in documents],
                        "ms": (time.perf_counter() - started) * 1000,
                    }
                }
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, LabError) else type(exc).__name__
            return {
                "branches": {name: {"documents": [], "ms": (time.perf_counter() - started) * 1000}},
                "degradation": [f"{name}:{code}"],
            }

    async def vector(self, state):
        return await self.channel(state, "vector")

    async def keyword(self, state):
        return await self.channel(state, "keyword")

    async def fuse(self, state):
        remaining(state["deadline"])
        params = RetrievalParams.model_validate(state["params"])
        started = time.perf_counter()
        hits = fuse(
            {k: v for k, v in state["branches"].items() if k in ("vector", "keyword")},
            state["resolved_question"],
            params,
        )
        degradation = []
        try:
            result = self.reranker(copy.deepcopy(hits), state["resolved_question"])
            validate_rerank(hits, result)
            hits = result
        except Exception:
            degradation = ["rerank:failed_using_fusion"]
        hits = mmr(hits, params.mmr_lambda)
        hits = [
            x
            for x in hits
            if x["relevance"] >= params.min_relevance
            and self.service.store.allowed(state["principal"], x["chunk"]["document_id"])
        ][: params.top_k]
        return {
            "evidence": hits,
            "degradation": degradation,
            "branches": {"fusion": {"ms": (time.perf_counter() - started) * 1000}},
        }

    async def generate(self, state):
        started = time.perf_counter()
        minimum = state["params"]["min_relevance"]
        problem = state["branches"]["normalize"]["ambiguity"] or evidence_problem(
            state["resolved_question"], state["evidence"], minimum
        )
        if any(
            not self.service.store.allowed(state["principal"], h["chunk"]["document_id"])
            for h in state["evidence"]
        ):
            problem = "evidence_revoked"
        if any(
            not self.service.store.allowed(state["principal"], d)
            for item in state["history"]
            for d in item.get("documents", [])
        ):
            problem = "context_revoked"
        candidate, usage, degradation = {}, {}, []
        strategy = "exact-extractive"
        if not problem:
            if self.service.generator:
                try:
                    candidate, usage = await guarded(
                        lambda: self.service.generator.generate(
                            state["resolved_question"], copy.deepcopy(state["evidence"])
                        ),
                        state["deadline"],
                        self.circuits["generation"],
                        attempts=1,
                    )
                    strategy = "conservative-claim-validation"
                    validate_claims(
                        candidate, state["evidence"], state["index_version"], extractive=False
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    degradation.append("generation:failed_or_unsupported_claims")
                    strategy = "exact-extractive"
            if strategy == "exact-extractive":
                candidate = await extractive_chain.ainvoke(
                    {
                        "question": state["resolved_question"],
                        "hits": state["evidence"],
                        "minimum": minimum,
                    }
                )
        return {
            "answer": {
                "candidate": candidate,
                "problem": problem,
                "strategy": strategy,
                "usage": usage,
            },
            "degradation": degradation,
            "branches": {"generation": {"ms": (time.perf_counter() - started) * 1000}},
        }

    async def validate(self, state):
        remaining(state["deadline"])
        started = time.perf_counter()
        problem, answer, citations = state["answer"]["problem"], "", []
        hits = state["evidence"]
        context_documents = list(
            dict.fromkeys(d for item in state["history"] for d in item.get("documents", []))
        )
        if any(not self.service.store.allowed(state["principal"], d) for d in context_documents):
            problem, hits = "context_revoked", []
        if any(
            not self.service.store.allowed(state["principal"], h["chunk"]["document_id"])
            for h in hits
        ):
            problem, hits = "evidence_revoked", []
        if not problem:
            try:
                answer, citations = validate_claims(
                    state["answer"]["candidate"],
                    hits,
                    state["index_version"],
                    extractive=state["answer"]["strategy"] == "exact-extractive",
                )
            except (ValueError, KeyError, TypeError):
                problem = "unsupported_or_empty_claims"
        stages = {k: v["ms"] for k, v in state["branches"].items() if "ms" in v}
        stages["validation"] = (time.perf_counter() - started) * 1000
        trace_id = self.service.telemetry.trace(
            state["principal"],
            state["question"],
            state["index_version"],
            stages,
            state.get("degradation", []),
        )
        status: Literal["clarify", "refused", "answered"] = (
            "clarify"
            if problem in ("ambiguous_reference", "ambiguous_object")
            else "refused"
            if problem
            else "answered"
        )
        result = AnswerResult(
            status=status,
            answer=answer if not problem else "证据不足或存在冲突，请补充具体对象与可信资料。",
            question=state["question"],
            resolved_question=state["question"]
            if problem == "context_revoked"
            else state["resolved_question"],
            citations=citations,
            evidence=hits,
            degradation=state.get("degradation", []),
            reason=problem,
            trace_id=trace_id,
            index_version=state["index_version"],
            stages_ms=stages,
            validation=state["answer"]["strategy"],
            model_usage=state["answer"]["usage"],
            context_documents=context_documents,
        )
        return {"answer": result.model_dump()}


class AgentWorkflow:
    def __init__(self, service, saver):
        self.service = service
        graph = StateGraph(AgentState)
        for name in ("prepare", "approval", "dispatch", "execute_task", "collect", "finish"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "approval")
        graph.add_edge("approval", "dispatch")
        graph.add_conditional_edges("dispatch", self.route, ["execute_task", "finish"])
        graph.add_edge("execute_task", "collect")
        graph.add_edge("collect", "dispatch")
        graph.add_edge("finish", END)
        self.graph = graph.compile(checkpointer=saver)

    def check(self, state):
        remaining(state["deadline"])
        row = self.service.store.request(state["principal"], state["run_id"])
        if row["cancel"] or state["topic_epoch"] != self.service.store.epoch(
            state["principal"], state["conversation_id"]
        ):
            raise asyncio.CancelledError()
        if (
            state["graph_version"] != self.service.settings.graph_version
            or state["state_version"] != self.service.settings.state_version
        ):
            raise LabError(
                "checkpoint_version", "checkpoint requires matching graph/state release", 409
            )
        self.service.registry.catalog.for_run(row)

    async def prepare(self, state):
        self.check(state)
        history = self.service.store.history(
            state["principal"],
            state["conversation_id"],
            state["topic_epoch"],
            self.service.settings.context_turns,
            self.service.settings.context_chars,
        )
        return {
            "history": history,
            "status": "running",
            "events": {"prepare": "validated explicit plan; bounded authorized history assembled"},
        }

    async def approval(self, state):
        self.check(state)
        if state["approval_required"] and not state.get("approved"):
            decision = interrupt(
                {
                    "kind": "review",
                    "tasks": [{"id": t["id"], "tool": t["tool"]} for t in state["tasks"]],
                    "run_id": state["run_id"],
                }
            )
            if decision is not True:
                raise LabError("review_rejected", "review was not approved", 409)
            self.check(state)
        if state["approval_required"]:
            row = self.service.store.request(state["principal"], state["run_id"])
            if not self.service.runtime_adapter.approved(row):
                raise LabError(
                    "review_mismatch", "legacy approval lacks a durable bound review", 409
                )
        return {
            "approved": True,
            "events": {"approval": "authorized review complete; no effects before interrupt"},
        }

    async def dispatch(self, state):
        self.check(state)
        if state.get("waves", 0) > 8:
            raise LabError("step_budget", "task wave budget exceeded", 409)
        return {}

    def route(self, state):
        pending = [t for t in state["tasks"] if t["id"] not in state.get("results", {})]
        ready = [t for t in pending if all(d in state.get("results", {}) for d in t["depends_on"])]
        if not ready:
            return "finish"
        row = self.service.store.request(state["principal"], state["run_id"])
        manifest = self.service.registry.catalog.for_run(row)
        reads = [task for task in ready if manifest.tools[task["tool"]].declaration.read_only]
        ready = reads or ready[:1]
        return [
            Send(
                "execute_task",
                {
                    **state,
                    "task": task,
                    "dependency_results": {d: state["results"][d] for d in task["depends_on"]},
                },
            )
            for task in ready
        ]

    async def execute_task(self, state, config: RunnableConfig):
        self.check(state)
        self.service.runtime_adapter.validate_task(state, config)
        await self.service.runtime.context(state["run_id"])
        task = TaskSpec.model_validate(state["task"])
        row = self.service.store.request(state["principal"], state["run_id"])
        if not self.service.registry.catalog.for_run(row).tools[task.tool].declaration.read_only:
            if not self.service.runtime_adapter.approved(row):
                raise LabError("review_mismatch", "write tool requires durable plan approval", 409)
        dependencies = state["dependency_results"]
        if any(r["status"] != "succeeded" for r in dependencies.values()):
            result = ToolResult(
                task_id=task.id, tool=task.tool, status="skipped", error="dependency_failed"
            )
        else:
            selected = self.service.registry.catalog.for_run(row).tools[task.tool]
            if not selected.declaration.read_only and task.use_dependencies:
                context = await self.service.registry.bridge.action_context(row, task, dependencies)
                call = ToolCall(call_id=task.id, name=task.tool, arguments=task.arguments)
                manifest = self.service.registry.catalog.for_run(row)
                challenge = self.service.runtime.tools.challenge(context, call)
                await self.service.runtime.require_review(
                    row["run_id"],
                    challenge,
                    {
                        "kind": "tool_action",
                        "call": call.model_dump(mode="json"),
                        "dependencies": context.dependency_results,
                        "manifest": manifest.digest,
                    },
                )
            result = await self.service.registry.execute(task, state, dependencies, config)
        self.check(state)
        return {
            "results": {task.id: result.model_dump()},
            "events": {f"task:{task.id}": f"{task.tool}: {result.status}"},
        }

    async def collect(self, state):
        self.check(state)
        if any(r["status"] == "needs_reconciliation" for r in state["results"].values()):
            raise UnknownEffect()
        return {"waves": state.get("waves", 0) + 1}

    async def finish(self, state):
        self.check(state)
        results = state.get("results", {})
        count = sum(r["status"] == "succeeded" for r in results.values())
        status = "succeeded" if count == len(state["tasks"]) else "partial" if count else "failed"
        return {
            "status": status,
            "events": {"finish": f"{count}/{len(state['tasks'])} tasks completed"},
        }
