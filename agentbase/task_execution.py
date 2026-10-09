"""Durable task coordination around the existing RunEngine workers."""

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command, interrupt
from pydantic import TypeAdapter

from . import task_tree
from .configuration import BudgetPolicy
from .contracts import AgentError, Identifier, RunContext
from .runtime_contracts import PreparedAdmission, RunOptions
from .schema import SchemaContract, canonical_json, content_digest
from .task_tree import (
    ChildReceiptReference,
    ChildTask,
    DelegationSnapshot,
    TaskInputReference,
    TaskMetadata,
)


@dataclass(frozen=True)
class TaskTicket:
    generation: int


class TaskCoordinator:
    def __init__(self, engine: Any):
        self.engine = engine
        self.wake = asyncio.Event()
        self.dispatcher: asyncio.Task | None = None

    def metadata(self, run_id: str) -> TaskMetadata | None:
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            return task_tree.metadata_in(db, run_id)

    def state(self, run_id: str) -> dict | None:
        with self.engine.store.connection() as db:
            row = db.execute(
                "SELECT * FROM agent_task_dispatch WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def outcome(self, run_id: str) -> dict | None:
        with self.engine.store.connection() as db:
            row = db.execute(
                "SELECT * FROM agent_task_outcomes WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def consumed(self, run_id: str) -> list[dict]:
        with self.engine.store.connection() as db:
            return self.consumed_in(db, run_id)

    @staticmethod
    def consumed_in(db, run_id: str) -> list[dict]:
        return [
            json.loads(row[0])
            for row in db.execute(
                "SELECT reference FROM agent_task_consumptions WHERE parent_run_id=? ORDER BY child_run_id",
                (run_id,),
            )
        ]

    def runtime_sources(self, run_id: str) -> list[dict]:
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            return self.runtime_sources_in(db, run_id)

    @classmethod
    def runtime_sources_in(cls, db, run_id: str) -> list[dict]:
        sources = cls.consumed_in(db, run_id)
        node = task_tree.metadata_in(db, run_id)
        if node is not None and node.parent_run_id is not None and node.input_sources:
            sources.append(
                TaskInputReference(
                    run_id=run_id,
                    parent_run_id=node.parent_run_id,
                    digest=content_digest(node.input_sources),
                ).model_dump(mode="json")
            )
        return sources

    async def _consume(self, context: RunContext, children: list[str], result: list[dict]):
        references = [source for item in result for source in item["sources"]]
        row = self.engine.store.internal_request(context.run_id)
        if not await self.engine.authorized_sources(row, context.identity, references):
            raise AgentError("source_revoked", "child result sources are no longer authorized", 403)
        fresh = await self.engine.context(context.run_id)
        if not self.engine._same_control(fresh, context):
            raise AgentError("task_control", "parent control changed before child consumption", 409)
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                not self.engine.model_request_current_in(db, fresh, references)
                or self._outcomes_in(db, fresh, children) != result
            ):
                raise AgentError("source_revoked", "child receipt changed before consumption", 403)
            for reference in references:
                encoded = canonical_json(reference).decode()
                previous = db.execute(
                    "SELECT reference FROM agent_task_consumptions WHERE parent_run_id=? AND child_run_id=?",
                    (context.run_id, reference["child_run_id"]),
                ).fetchone()
                if previous is not None and previous[0] != encoded:
                    raise AgentError("task_outcome", "consumed child receipt changed", 409)
                db.execute(
                    "INSERT OR IGNORE INTO agent_task_consumptions VALUES(?,?,?)",
                    (context.run_id, reference["child_run_id"], encoded),
                )
        return result

    def children(self, run_id: str) -> list[dict]:
        with self.engine.store.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT t.run_id,t.task_key AS task_id,r.agent_id,d.state,o.status,o.error FROM agent_task_runs t JOIN requests r USING(run_id) JOIN agent_task_dispatch d USING(run_id) LEFT JOIN agent_task_outcomes o USING(run_id) WHERE t.parent_run_id=? ORDER BY t.rowid",
                    (run_id,),
                )
            ]

    def awaiting_review(self, run_id: str):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE agent_task_dispatch SET state='awaiting_review',command=NULL,claim_token=NULL WHERE run_id=? AND state='running'",
                (run_id,),
            )

    def resume_commit(self, context: RunContext, snapshot, value, digest):
        expected_id = self.interrupt_id(snapshot)
        if expected_id is None:
            raise AgentError("task_command", "resume requires one real checkpoint interrupt", 409)
        encoded = canonical_json(value).decode()

        def commit(db):
            task_tree.assert_ancestors_in(db, context)
            state = db.execute(
                "SELECT state FROM agent_task_dispatch WHERE run_id=?", (context.run_id,)
            ).fetchone()
            if state is None or state[0] != "awaiting_review":
                raise AgentError("task_command", "task is not waiting for this approval", 409)
            db.execute(
                "UPDATE agent_task_dispatch SET state='ready',generation=generation+1,command=?,claim_token=NULL,expected_interrupt_id=?,expected_interrupt_digest=? WHERE run_id=?",
                (encoded, expected_id, digest, context.run_id),
            )

        return commit

    def targets(self, definition, scopes) -> dict[str, DelegationSnapshot]:
        return (
            {
                target.agent_id: DelegationSnapshot(
                    binding=self.engine.registry.agent(target.agent_id).digest,
                    scopes=scopes if target.scopes is None else scopes & target.scopes,
                )
                for target in definition.delegation.targets
            }
            if definition.delegation is not None
            else {}
        )

    def root_commit(self, bound, identity, deadline, original_commit):
        targets = self.targets(bound.definition, identity.scopes)

        def commit(db, row):
            task_tree.add_in(
                db,
                TaskMetadata(
                    run_id=row["run_id"],
                    owner=identity.owner,
                    agent_id=bound.definition.id,
                    binding=bound.digest,
                    root_run_id=row["run_id"],
                    hard_deadline=deadline,
                    scope_ceiling=identity.scopes,
                    budget=bound.definition.budget,
                    delegation=bound.definition.delegation,
                    targets=targets,
                ),
            )
            if original_commit is not None:
                returned = original_commit(db, row)
                if returned is not None:
                    if asyncio.iscoroutine(returned):
                        returned.close()
                    raise AgentError("admission_contract", "admission commit must return None", 503)

        return commit

    async def dispatch(
        self, parent_run_id: str, declarations: list[ChildTask], *, sources=None
    ) -> list[str]:
        if (
            not declarations
            or len(declarations) > 64
            or len({task.id for task in declarations}) != len(declarations)
        ):
            raise AgentError(
                "task_declaration", "child declarations must be bounded and unique", 409
            )
        context = await self.engine.context(parent_run_id)
        parent = self.metadata(parent_run_id)
        if parent is None or parent.delegation is None:
            raise AgentError("task_delegation", "run has no trusted delegation policy", 403)
        sources = list(
            {
                canonical_json(source): source
                for source in [*(sources or []), *self.runtime_sources(parent_run_id)]
            }.values()
        )
        if len(sources) > 100 or not await self.engine.authorized_sources(
            self.engine.store.internal_request(parent_run_id), context.identity, sources
        ):
            raise AgentError("source_revoked", "child input sources are unavailable", 403)
        prepared = []
        for task in declarations:
            target = parent.targets.get(task.agent_id)
            if target is None:
                raise AgentError("task_target", "child target is not allowed", 403)
            bound = self.engine.registry.resolve(task.agent_id, target.binding)
            SchemaContract(bound.definition.input_schema).validate(task.input, code="agent_input")
            scopes = context.effective_scopes & target.scopes
            identity = context.identity.model_copy(update={"scopes": scopes})
            if not await self.engine.agent_allowed(identity, task.agent_id):
                raise AgentError("agent_forbidden", "child Agent access is not authorized", 403)
            graph = self.engine.binding({"agent_id": task.agent_id, "binding": bound.digest})
            payload, callback = {"input": copy.deepcopy(task.input)}, None
            if graph.prepare is not None:
                value = await graph.prepare(identity, copy.deepcopy(task.input))
                if isinstance(value, PreparedAdmission):
                    payload, callback = value.payload, value.commit
                else:
                    payload = value
            if not isinstance(payload, dict):
                raise AgentError("admission_contract", "child payload must be a JSON object", 503)
            payload = copy.deepcopy(payload)
            payload.update(
                _task_tree=True, _identity_digest=self.engine.identity_digest(context.identity)
            )
            budget = BudgetPolicy(
                **{
                    key: min(getattr(bound.definition.budget, key), getattr(parent.budget, key))
                    for key in BudgetPolicy.model_fields
                }
            )
            key = content_digest({"parent": parent_run_id, "task": task.id})[7:]
            digest = content_digest(
                {"task": task.model_dump(mode="json"), "target": target.model_dump(mode="json")}
            )
            options = RunOptions(conversation_id="child_" + key[:32])
            targets = self.targets(bound.definition, scopes)
            prepared.append(
                (task, bound, scopes, budget, key, digest, options, payload, callback, targets)
            )
        fresh = await self.engine.context(parent_run_id)
        for task, _, scopes, *_ in prepared:
            if not await self.engine.agent_allowed(
                context.identity.model_copy(update={"scopes": scopes}), task.agent_id
            ):
                raise AgentError(
                    "agent_forbidden", "child Agent access changed before admission", 403
                )
        fresh = await self.engine.context(parent_run_id)
        if not self.engine._same_control(fresh, context):
            raise AgentError("task_control", "parent changed during child preparation", 409)
        result = []
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self.engine.model_request_current_in(db, fresh, sources):
                raise AgentError(
                    "source_revoked", "child input sources changed before admission", 403
                )
            ancestors = task_tree.assert_ancestors_in(db, context)
            step_key = (
                "dispatch:"
                + content_digest([task.model_dump(mode="json") for task in declarations])[7:]
            )
            previous = db.execute(
                "SELECT binding FROM agent_run_steps WHERE run_id=? AND step_key=?",
                (parent_run_id, step_key),
            ).fetchone()
            if previous is None:
                task_tree.check_quota_in(db, ancestors, "step")
                db.execute(
                    "INSERT INTO agent_run_steps VALUES(?,?,?)",
                    (parent_run_id, step_key, context.config_digest),
                )
            for (
                task,
                bound,
                scopes,
                budget,
                key,
                digest,
                options,
                payload,
                callback,
                targets,
            ) in prepared:
                deadline = min(parent.hard_deadline, time.time() + budget.timeout_seconds)

                def commit(
                    connection,
                    row,
                    *,
                    task=task,
                    bound=bound,
                    scopes=scopes,
                    budget=budget,
                    callback=callback,
                    targets=targets,
                    deadline=deadline,
                ):
                    task_tree.add_in(
                        connection,
                        TaskMetadata(
                            run_id=row["run_id"],
                            owner=context.identity.owner,
                            agent_id=task.agent_id,
                            binding=bound.digest,
                            root_run_id=parent.root_run_id,
                            parent_run_id=parent_run_id,
                            task_key=task.id,
                            depth=parent.depth + 1,
                            hard_deadline=deadline,
                            scope_ceiling=scopes,
                            budget=budget,
                            delegation=bound.definition.delegation,
                            targets=targets,
                            input_sources=copy.deepcopy(sources),
                        ),
                    )
                    if callback is not None:
                        returned = callback(connection, row)
                        if returned is not None:
                            if asyncio.iscoroutine(returned):
                                returned.close()
                            raise AgentError(
                                "admission_contract", "child admission commit must return None", 503
                            )

                row, created, _ = self.engine.store.admit_in(
                    db,
                    context.identity,
                    task.agent_id,
                    key,
                    digest,
                    payload,
                    bound.digest,
                    options,
                    deadline,
                    "fixed-deadline",
                    admission_commit=commit,
                )
                membership = task_tree.metadata_in(db, row["run_id"])
                if (
                    membership is None
                    or membership.parent_run_id != parent_run_id
                    or membership.task_key != task.id
                ):
                    raise AgentError("task_membership", "child admission relationship differs", 409)
                if created:
                    self.engine.store._event(
                        db, row, "run.accepted", {"parent_run_id": parent_run_id}, False, []
                    )
                result.append(row["run_id"])
        self.wake.set()
        return result

    def _outcomes_in(self, db, context, children):
        result = []
        pending = False
        for run_id in children:
            node = task_tree.metadata_in(db, run_id)
            if (
                node is None
                or node.parent_run_id != context.run_id
                or node.owner != context.identity.owner
            ):
                raise AgentError("task_membership", "wait requires direct owned child tasks", 409)
            outcome = db.execute(
                "SELECT * FROM agent_task_outcomes WHERE run_id=?", (run_id,)
            ).fetchone()
            if outcome is None:
                pending = True
                continue
            item = {
                "run_id": run_id,
                "task_id": node.task_key,
                "agent_id": node.agent_id,
                "status": outcome["status"],
                "error": outcome["error"],
                "value": None,
                "sources": [],
            }
            if outcome["final_receipt_digest"] is not None:
                receipt = db.execute(
                    "SELECT * FROM agent_final_receipts WHERE run_id=?", (run_id,)
                ).fetchone()
                if (
                    receipt is None
                    or receipt["owner"] != node.owner
                    or receipt["binding"] != node.binding
                ):
                    raise AgentError("task_outcome", "child receipt binding differs", 409)
                value = json.loads(receipt["payload"])
                if (
                    content_digest(value) != outcome["final_receipt_digest"]
                    or value["status"] != outcome["status"]
                ):
                    raise AgentError("task_outcome", "child receipt digest differs", 409)
                item.update(
                    value=value["value"],
                    sources=[
                        ChildReceiptReference(
                            parent_run_id=context.run_id,
                            child_run_id=run_id,
                            agent_id=node.agent_id,
                            binding=node.binding,
                            digest=outcome["final_receipt_digest"],
                        ).model_dump(mode="json")
                    ],
                )
            result.append(item)
        return None if pending else result

    async def wait(self, parent_run_id: str, wait_key: str, children: list[str]) -> list[dict]:
        TypeAdapter(Identifier).validate_python(wait_key)
        if not children or len(children) > 64 or len(set(children)) != len(children):
            raise AgentError("task_wait", "wait children must be bounded and unique", 409)
        children = sorted(children)
        context = await self.engine.context(parent_run_id)
        digest = content_digest({"parent": parent_run_id, "wait": wait_key, "children": children})
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            task_tree.assert_ancestors_in(db, context)
            result = self._outcomes_in(db, context, children)
            previous = db.execute(
                "SELECT digest FROM agent_task_waits WHERE parent_run_id=? AND wait_key=?",
                (parent_run_id, wait_key),
            ).fetchone()
            if previous is not None and previous[0] != digest:
                raise AgentError("task_wait", "wait declaration changed", 409)
            if (
                result is None
                and db.execute(
                    "SELECT 1 FROM agent_task_waits WHERE parent_run_id=? AND wait_key!=? AND phase!='released'",
                    (parent_run_id, wait_key),
                ).fetchone()
            ):
                raise AgentError("task_wait", "only one outstanding child wait is allowed", 409)
            if result is None:
                db.execute(
                    "INSERT OR IGNORE INTO agent_task_waits(parent_run_id,wait_key,digest,children,phase) VALUES(?,?,?,?,'registered')",
                    (parent_run_id, wait_key, digest, canonical_json(children).decode()),
                )
        if result is not None:
            return await self._consume(context, children, result)
        resumed = interrupt({"kind": "task_wait", "wait_key": wait_key, "digest": digest})
        if resumed != {"kind": "task_wait", "wait_key": wait_key, "digest": digest}:
            raise AgentError("task_wait", "wait resume token differs", 409)
        context = await self.engine.context(parent_run_id)
        with self.engine.store.connection() as db:
            db.execute("BEGIN")
            task_tree.assert_ancestors_in(db, context)
            result = self._outcomes_in(db, context, children)
            if result is None:
                raise AgentError("task_wait", "child completion is not committed", 409)
        return await self._consume(context, children, result)

    @staticmethod
    def interrupt_id(snapshot) -> str | None:
        values = [value for task in snapshot.tasks for value in task.interrupts]
        return values[0].id if len(values) == 1 else None

    def park(self, row, snapshot) -> bool:
        raw = self.engine.interrupts(snapshot)
        if len(raw) != 1 or raw[0].get("kind") != "task_wait":
            return False
        value = raw[0]
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            wait = db.execute(
                "SELECT * FROM agent_task_waits WHERE parent_run_id=? AND wait_key=?",
                (row["run_id"], value["wait_key"]),
            ).fetchone()
            if (
                wait is None
                or wait["digest"] != value["digest"]
                or self.interrupt_id(snapshot) is None
            ):
                raise AgentError(
                    "task_wait", "checkpoint wait differs from durable registration", 409
                )
            db.execute(
                "UPDATE agent_task_dispatch SET state='waiting_children',command=NULL,claim_token=NULL WHERE run_id=? AND state='running'",
                (row["run_id"],),
            )
            db.execute(
                "UPDATE agent_task_waits SET phase='armed',interrupt_id=?,checkpoint_id=? WHERE parent_run_id=? AND wait_key=?",
                (
                    self.interrupt_id(snapshot),
                    snapshot.config["configurable"]["checkpoint_id"],
                    row["run_id"],
                    value["wait_key"],
                ),
            )
            for child in json.loads(wait["children"]):
                task_tree.release_waits_in(db, child)
        self.wake.set()
        return True

    def claim(self, row, ticket: TaskTicket) -> bool:
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            return (
                db.execute(
                    "UPDATE agent_task_dispatch SET state='running',claim_token=? WHERE run_id=? AND generation=? AND state='ready'",
                    (uuid.uuid4().hex, row["run_id"], ticket.generation),
                ).rowcount
                == 1
            )

    def command(self, row, snapshot):
        state = self.state(row["run_id"])
        if state is None or state["command"] is None:
            return None
        value = json.loads(state["command"])
        raw = self.engine.interrupts(snapshot)
        if self.interrupt_id(snapshot) != state["expected_interrupt_id"]:
            return None
        visible = (
            raw
            if raw and raw[0].get("kind") == "task_wait"
            else self.engine.visible_reviews(row, snapshot)
        )
        if len(visible) != 1 or visible[0].get("digest") != state["expected_interrupt_digest"]:
            return None
        if visible[0].get("kind") != "task_wait":
            expected = {
                "review_id": visible[0].get("review_id"),
                "digest": visible[0].get("digest"),
            }
            if value != expected or not self.engine.store.approval(row, visible[0]["digest"]):
                raise AgentError("task_command", "durable approval command differs", 409)
            policy = self.engine.binding(row).resume_value
            if policy is not None:
                value = policy(row, visible[0])
        return Command(resume=value)

    async def start(self):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE agent_task_dispatch SET state='ready',claim_token=NULL WHERE state='running'"
            )
            for row in db.execute("SELECT run_id FROM agent_task_outcomes").fetchall():
                task_tree.release_waits_in(db, row[0])
            terminal = [
                dict(row)
                for row in db.execute(
                    "SELECT r.*,o.status AS outcome_status FROM requests r JOIN agent_task_outcomes o USING(run_id) JOIN agent_task_dispatch d USING(run_id)"
                )
            ]
        for record in terminal:
            snapshot = await self.engine.checkpoint_state(record)
            controls = self.engine.binding(record).control_fields
            if snapshot.values and any(name not in snapshot.values for name in controls):
                if any(
                    name in snapshot.values and snapshot.values[name] != record[column]
                    for name, column in controls.items()
                ):
                    raise AgentError(
                        "checkpoint_control", "terminal checkpoint control differs", 409
                    )
                # Complete this proven terminal repair before startup admits queries.
                # No graph execution or external action is needed.
                await self.engine.terminal_checkpoint(
                    record, record["outcome_status"], "task_outcome_recovered"
                )
                self.engine._validate_snapshot(record, await self.engine.checkpoint_state(record))
                continue
            self.engine._validate_snapshot(record, snapshot)
            if snapshot.next or snapshot.values.get("status") != record["outcome_status"]:
                self.repair(record["run_id"])
        self.dispatcher = asyncio.create_task(self.loop())

    def repair(self, run_id: str):
        with self.engine.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE agent_task_dispatch SET state='ready',generation=generation+1,command=NULL,claim_token=NULL WHERE run_id=? AND state='terminal'",
                (run_id,),
            )
        self.wake.set()

    async def close(self):
        if self.dispatcher is not None:
            self.dispatcher.cancel()
            await asyncio.gather(self.dispatcher, return_exceptions=True)
            self.dispatcher = None

    async def loop(self):
        try:
            await self._loop()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.engine.available = False
            self.failure_code = (
                error.code if isinstance(error, AgentError) else "dispatcher_failure"
            )

    async def _loop(self):
        while not self.engine.closing:
            self.wake.clear()
            with self.engine.store.connection() as db:
                expired = [
                    dict(row)
                    for row in db.execute(
                        "SELECT r.* FROM agent_task_dispatch d JOIN requests r USING(run_id) WHERE d.state IN ('waiting_children','awaiting_review') AND (r.deadline<=? OR r.cancel=1) ORDER BY r.rowid",
                        (time.time(),),
                    )
                ]
            for row in expired:
                if self.outcome(row["run_id"]) is not None:
                    continue
                if row["run_id"] not in self.engine.active:
                    await self.engine.cancel(
                        row["owner"], row["run_id"], _timeout=row["deadline"] <= time.time()
                    )
            with self.engine.store.connection() as db:
                candidates = [
                    dict(row)
                    for row in db.execute(
                        "SELECT r.*,d.generation FROM agent_task_dispatch d JOIN requests r USING(run_id) WHERE d.state='ready' ORDER BY r.rowid LIMIT ?",
                        (self.engine.settings.max_runs,),
                    )
                ]
            for row in candidates:
                run_id = row["run_id"]
                if run_id in self.engine.pending or run_id in self.engine.active:
                    continue
                try:
                    self.engine.queue.put_nowait((row, TaskTicket(row.pop("generation"))))
                    self.engine.pending.add(run_id)
                except asyncio.QueueFull:
                    break
            try:
                await asyncio.wait_for(self.wake.wait(), 0.05)
            except TimeoutError:
                pass
