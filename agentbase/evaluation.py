"""Small bounded business case runner; never approves a write operation."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .contracts import AgentError, BaseContract, Identifier, Identity, RunEvent
from .fixed_workflow import input_pointer
from .runtime import RunEngine
from .runtime_contracts import TERMINAL, RunOptions
from .schema import SchemaContract, canonical_json


class ResultCheck(BaseContract):
    pointer: str = Field("", max_length=1024)
    operation: Literal["equals", "contains", "not_contains", "schema"] = "equals"
    expected: JsonValue

    @model_validator(mode="after")
    def valid_check(self):
        # Validate pointer syntax without assuming a particular case's output shape.
        import re

        if self.pointer and (
            not self.pointer.startswith("/") or re.search(r"~(?![01])", self.pointer)
        ):
            raise ValueError("check pointer must be RFC 6901")
        if self.operation == "schema":
            if not isinstance(self.expected, dict):
                raise ValueError("schema check requires a schema object")
            SchemaContract(self.expected)
        return self


class EvalCase(BaseContract):
    id: Identifier
    input: JsonValue
    expected_status: Literal[
        "succeeded",
        "partial",
        "failed",
        "cancelled",
        "timed_out",
        "awaiting_review",
        "needs_reconciliation",
        "effect_applied_result_unavailable",
        "rejected",
    ] = "succeeded"
    checks: list[ResultCheck] = Field(default_factory=list, max_length=64)
    expected_error: str | None = Field(None, max_length=128)
    expected_tool_calls: list[str] | None = Field(None, max_length=128)
    expected_applied_operations: int | None = Field(None, ge=0, le=128)
    timeout_seconds: float = Field(10, gt=0, le=120)

    @model_validator(mode="after")
    def quality_contract(self):
        canonical_json(self.input)
        if self.expected_status in {"succeeded", "partial"} and not self.checks:
            raise ValueError("successful cases require business result checks")
        if self.expected_status == "rejected" and self.expected_error is None:
            raise ValueError("rejected cases require an exact error code")
        return self


def _json_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def load_cases(path: Path, *, max_bytes: int = 1048576) -> list[EvalCase]:
    with path.open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise AgentError("evaluation_limit", "case file byte limit exceeded")
    try:
        cases = [EvalCase.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise AgentError("evaluation_config", "invalid evaluation cases") from exc
    if not cases or len(cases) > 200 or len({case.id for case in cases}) != len(cases):
        raise AgentError("evaluation_config", "require 1..200 uniquely identified cases")
    return cases


async def wait_for_outcome(engine: RunEngine, owner: str, run_id: str, *, timeout_seconds=10):
    """Return only terminal or actionable review; intermediate projections cannot pass."""
    async with asyncio.timeout(timeout_seconds):
        while True:
            view = await engine.view(owner, run_id)
            if view["status"] in TERMINAL or view["status"] == "awaiting_review":
                return view
            await asyncio.sleep(0.02)


async def diagnose_run(engine: RunEngine, owner: str, run_id: str) -> dict:
    """Local metadata only: no prompts, arguments, outputs, receipt payloads or secrets."""
    view = await engine.view(owner, run_id)
    diagnosis = {
        "run_id": run_id,
        "agent_id": view["agent_id"],
        "status": view["status"],
        "error": view.get("error"),
        "resume_reason": view.get("resume_reason"),
        "result_visibility": view.get("result_visibility", "visible"),
        "review_count": len(view.get("interrupts", [])),
        "events": [],
        "warnings": [],
    }
    identity = await engine.identity(owner)
    if view.get("result_visibility") in {"revoked", "unverified"} or not await engine.agent_allowed(
        identity, view["agent_id"]
    ):
        return diagnosis
    events: list[RunEvent] = []
    try:
        # Events are bounded by EngineSettings; replay uses the runtime authorization guards.
        after = 0
        while len(events) < engine.settings.event_count:
            batch = await engine.events(owner, run_id, after, limit=512)
            if not batch:
                break
            events.extend(batch)
            after = batch[-1].seq
        diagnosis["events"] = [
            {"seq": event.seq, "kind": event.kind, "provisional": event.provisional}
            for event in events
        ]
    except AgentError as exc:
        diagnosis["warnings"].append(exc.code)
    tool_calls = []
    for event in events:
        data = event.data.get("value") if isinstance(event.data, dict) else None
        if (
            event.kind == "tool.started"
            and isinstance(data, dict)
            and isinstance(data.get("name"), str)
        ):
            tool_calls.append(data["name"])
    diagnosis["tool_calls"] = tool_calls
    operations = engine.actions.operations(owner, run_id)
    diagnosis["operation_phases"] = [operation["phase"] for operation in operations]
    diagnosis["applied_operations"] = sum(
        operation["phase"] == "applied" for operation in operations
    )
    with engine.store.connection() as db:
        budget = db.execute(
            "SELECT calls,tokens,overrun FROM agent_model_budgets WHERE run_id=?", (run_id,)
        ).fetchone()
        reservations = db.execute(
            "SELECT usage,settled FROM agent_model_reservations WHERE run_id=?", (run_id,)
        ).fetchall()
        diagnosis["has_final_receipt"] = engine.store.final_receipt(owner, run_id) is not None
    usage = [
        json.loads(row["usage"])
        for row in reservations
        if row["settled"] and row["usage"] is not None
    ]
    diagnosis["budget"] = dict(budget) if budget else {"calls": 0, "tokens": 0, "overrun": 0}
    known = {
        key: sum(item.get(key, 0) for item in usage) for key in ("input_tokens", "output_tokens")
    }
    diagnosis["usage_coverage"] = {
        "known_calls": len(usage),
        "unknown_calls": len(reservations) - len(usage),
        "unsettled_calls": sum(not row["settled"] for row in reservations),
    }
    diagnosis["known_usage_subtotal"] = known
    diagnosis["reported_usage"] = known if len(usage) == len(reservations) else None
    return diagnosis


def result_failures(case: EvalCase, view: dict) -> list[str]:
    """Pure business assertions, also useful against recorded authorized snapshots."""
    failures = []
    if view.get("status") != case.expected_status:
        failures.append("unexpected_status")
    if view.get("result_visibility") in {"revoked", "unverified"}:
        failures.append("result_unavailable")
    if case.expected_error is not None and view.get("error") != case.expected_error:
        failures.append("unexpected_error")
    if case.expected_status == "awaiting_review" and (
        not view.get("interrupts") or view.get("recoverable") is not True
    ):
        failures.append("review_unavailable")
    for index, check in enumerate(case.checks):
        try:
            value = input_pointer(view.get("result"), check.pointer)
            if check.operation == "schema":
                if not isinstance(check.expected, dict):
                    raise AgentError("evaluation_config", "schema check requires an object")
                SchemaContract(check.expected).validate(value)
                passed = True
            elif check.operation == "equals":
                canonical_json(value)
                canonical_json(check.expected)
                passed = _json_equal(value, check.expected)
            else:
                # Restrict membership to comparable JSON collections, never arbitrary objects.
                found = (
                    isinstance(value, str)
                    and isinstance(check.expected, str)
                    and check.expected in value
                ) or (
                    isinstance(value, list)
                    and any(_json_equal(item, check.expected) for item in value)
                )
                passed = (
                    found
                    if check.operation == "contains"
                    else (
                        (
                            isinstance(value, list)
                            or (isinstance(value, str) and isinstance(check.expected, str))
                        )
                        and not found
                    )
                )
            if not passed:
                failures.append(f"result_check_{index}")
        except (AgentError, TypeError, ValueError):
            failures.append(f"result_check_{index}")
    return failures


async def evaluate_cases(
    engine: RunEngine,
    identity: Identity,
    agent_id: str,
    cases: list[EvalCase],
    *,
    mode: Literal["offline", "model"] = "offline",
) -> dict:
    """Each case gets a fresh conversation and key. Report never certifies semantic quality."""
    if not cases or len(cases) > 200 or len({case.id for case in cases}) != len(cases):
        raise AgentError("evaluation_config", "require 1..200 unique cases")
    outcomes = []
    for case in cases:
        started, run_id = time.monotonic(), None
        key = uuid.uuid4().hex
        view = None
        failures, diagnosis = [], None
        try:
            async with asyncio.timeout(case.timeout_seconds):
                try:
                    created = await engine.submit(
                        identity, agent_id, key, case.input, options=RunOptions(conversation_id=key)
                    )
                except AgentError as exc:
                    if case.expected_status == "rejected" and exc.code == case.expected_error:
                        view = {"status": "rejected", "error": exc.code}
                    else:
                        raise
                else:
                    run_id = created["run_id"]
                    view = await wait_for_outcome(
                        engine, identity.owner, run_id, timeout_seconds=case.timeout_seconds
                    )
                    diagnosis = await diagnose_run(engine, identity.owner, run_id)
                failures = result_failures(case, view)
                if case.expected_status in {"succeeded", "partial"} and (
                    diagnosis is None or not diagnosis.get("has_final_receipt")
                ):
                    failures.append("final_receipt_missing")
                if case.expected_tool_calls is not None and (
                    diagnosis is None
                    or diagnosis.get("warnings")
                    or diagnosis.get("tool_calls") != case.expected_tool_calls
                ):
                    failures.append("unexpected_tool_calls")
                if case.expected_applied_operations is not None and (
                    diagnosis is None
                    or diagnosis.get("applied_operations") != case.expected_applied_operations
                ):
                    failures.append("unexpected_applied_operations")
                status = view["status"]
        except TimeoutError:
            status, failures = "incomplete", ["evaluation_timeout"]
        except AgentError as exc:
            status, failures = "evaluation_error", [exc.code]
        except Exception:
            status, failures = "evaluation_error", ["evaluation_error"]
        finally:
            if run_id is None:
                try:
                    # submit can commit admission and then time out while projecting its view.
                    admitted = engine.store.lookup(identity.owner, agent_id, key)
                    if admitted is not None:
                        run_id = admitted["run_id"]
                except Exception:
                    failures.append("cleanup_lookup_failed")
            if run_id is not None and (view is None or view.get("status") not in TERMINAL):
                try:
                    # Retain durable metadata, but never leave an evaluation's write awaiting approval.
                    async with asyncio.timeout(min(3, case.timeout_seconds)):
                        await engine.cancel(identity.owner, run_id)
                except Exception:
                    failures.append("cleanup_failed")
        outcomes.append(
            {
                "id": case.id,
                "run_id": run_id,
                "status": status,
                "passed": not failures,
                "failures": failures,
                "seconds": round(time.monotonic() - started, 4),
                "diagnosis": diagnosis,
            }
        )
    passed = sum(1 for item in outcomes if item["passed"] is True)
    return {
        "mode": mode,
        "agent_id": agent_id,
        "total": len(outcomes),
        "passed": passed,
        "failed": len(outcomes) - passed,
        "completed_successfully": sum(
            1 for item in outcomes if item["status"] == "succeeded" and item["passed"] is True
        ),
        "review_checks_passed": sum(
            1 for item in outcomes if item["status"] == "awaiting_review" and item["passed"] is True
        ),
        "cases": outcomes,
        "real_model_quality_verified": False,
        "quality_note": "Case checks cover declared assertions only; domain review and held-out model evaluation remain required.",
    }
