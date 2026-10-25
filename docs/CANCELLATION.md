# Cancellation and admission

`RunEngine.cancel` records the cancellation fence synchronously in SQLite before
waiting for cleanup. It does not acquire the global admission lock. Topic changes
use the same cancellation owner for superseded runs. Repeated requests share that
owner and do not inject additional cancellation into a handler's cleanup.

`EngineSettings.cancel_wait_seconds` bounds how long the cancel caller waits for
execution/finalization to drain (default 1 second; allowed range 0–5 seconds).
Checkpoint reads, identity resolution and application projection hooks still have
their own I/O characteristics; this is not a total HTTP response deadline.

While cleanup remains active, run queries report `status="cancelling"`,
`cancel_requested=true`, `cleanup_pending=true`, and `recoverable=false`.
There is no terminal-success or rollback claim. Previously applied effects remain
visible through the existing receipt projection. Unknown write outcomes remain
subject to reconciliation, never automatic replay.

New work can be admitted during cleanup. The old execution continues to own its
worker and tool-concurrency slot until it exits. Queued cancelled runs have one
finalization owner and cannot start another execution. Admission therefore does
not bypass existing resource serialization, scopes, approvals, attempt tokens or
idempotency checks. Cancellation that loses to a committed final receipt preserves
that receipt. Late cancelled results cannot enter final history.

Caller disconnection does not cancel the owned finalizer. Shutdown waits for both
active execution and cancellation finalizers before the host may close storage or
release its process lock. A persistence/finalization failure stops new admission.
Once shutdown begins, new cancellation owners are rejected with `not_accepting`;
requests for an already-owned cleanup still reuse that owner. Admission rechecks
availability after asynchronous preparation/authorization, before persisting any
request, topic epoch change, event or task-tree metadata.
Concurrent close callers share an owned shutdown task. Cancelling a close caller
does not send another cancellation into handler cleanup; the caller waits for
shutdown to drain and then receives `CancelledError`. `start()` is rejected while
that shutdown task is still running. Shutdown remains a host lifecycle operation.

Python cancellation is cooperative. A synchronous blocking handler can prevent the
event loop from processing even a cancellation request. A handler suppressing
cancellation indefinitely can retain execution resources and delay shutdown.
The response wait budget does not force-kill either handler or release its slots.
Such handlers need a host-controlled process boundary for enforceable termination;
that isolation and any external-service rollback are outside this change.

The browser's cancellation panel observes at most four runs in its current Agent /
identity scope. Starting or querying another run of the same Agent keeps the old
summary, while results update only when both Agent and run ID match the selection.
Repeated Stop clicks reuse the observation and never repeat POST /cancel while it
is active. No tool invocation is retried by this observer.

Observation uses one shared timer / GET flight: two-second spacing, at most 30 GETs
per run, a 120-second observation age and a five-second GET timeout. These bounds
stop observation, not backend execution. Eviction, access loss and exhausted
budgets explicitly leave an unconfirmed status. Terminal wording requires an
explicit cleanup_pending=false plus a recognized terminal status; a failed cleanup
with a nonterminal status remains unknown. Responses must bind to the original
Agent and run, and a delayed cancellation POST cannot downgrade a confirmed final
GET. Changing Agent/identity or pagehide clears timers, aborts the current GET and
invalidates late responses. Observation is page-local, not a persistent alert.

The test-only `python -m tests.ui_cancel_e2e_server --evidence-dir <NEW_DIR>` entry
point serves the product UI with a gated read-only synthetic tool on loopback.
`--fail-cleanup` injects a terminal-storage failure. Its /__test controls and data
exist only in that explicit test host; they are never mounted by production entry
points. It does not configure a model or call an external service.
