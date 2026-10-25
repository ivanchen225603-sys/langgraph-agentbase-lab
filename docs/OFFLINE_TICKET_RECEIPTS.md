# Offline ticket receipt recovery

This is an explicit checkout-only synthetic example, not a production integration.
It is not registered by application entry points, included in their default
configuration, or packaged by the existing wheel/sdist rules. It uses the existing
locked Pydantic/SQLite/runtime interfaces, with no new dependencies or credentials.

```powershell
python -m examples.offline_ticket_receipts submit --directory <NEW_TEST_DIRECTORY>
python -m examples.offline_ticket_receipts recover --directory <SAME_TEST_DIRECTORY>
python -m examples.offline_ticket_receipts recover --directory <SAME_TEST_DIRECTORY>
```

The first process reports `unknown`, one original attempt, one remote create
request, and one committed ticket. The recovery processes report `applied` with
the same counts. The two SQLite files simulate separate local and remote stores;
there is no network service or model. A test deliberately bypasses the adapter and
issues a duplicate remote create: it reports two requests but still one ticket.
Request counting commits separately so a failed duplicate cannot hide its request
by rolling back the business transaction.

## Existing interfaces and trust boundaries

`SqliteActionLedger.key` is owner/Agent/run/call scoped. The fingerprint binds the
tool declaration, arguments/call, dependencies and optional manifest. Neither is
a cross-run business idempotency key. This example supplies no such business-key
deduplication and does not pretend that a new run cannot create another ticket.

The host supplies `HostPrincipal(tenant, subject)` after authentication; its owner
is a digest of that pair. This is a synthetic mapping, not an authentication
implementation. Tenant/subject never come from tool arguments. `AttemptCommit`
persists the pair, owner, operation key, Agent/run/call, configuration binding,
fingerprint, argument digest, attempt token and attempt count atomically with the
original claim in `ticket_attempt_evidence`.

The creation handler deliberately treats any normal returned value as unverified.
Only the separate host-owned lookup can supply a receipt. A real implementation
must authenticate its remote service/response and map its evidence into this seam;
arbitrary tool/model output is not a trusted lookup result. The local synthetic
remote stores the full original binding and supports a read-only lookup by key.

## Recovery sequence

1. Read and validate the original host principal and claim evidence, then CLOSE
   the local SQL connection.
2. Await the synthetic remote lookup outside all local SQL transactions. A
   cooperative timeout is bounded to at most five seconds. Cancellation propagates;
   temporary failure, missing or unknown evidence does not authorize submission.
3. Validate receipt schema and every original binding field. Under a new short
   `BEGIN IMMEDIATE`, re-read both operation and sidecar, including attempt token
   and count. Reject stale identities, compensated history and contradictory
   applied values before writing any receipt. Identical receipts are idempotent;
   conflicting receipts never replace existing evidence.
4. Commit the trusted local receipt, then call existing `ledger.reconcile`.
   Its synchronous, SQL-read-only inspector independently validates the stored
   binding and returns `ActionReceipt`. No query is awaited inside reconcile.

A crash between steps 3 and 4 leaves durable evidence for later reconciliation.
Cancellation after receipt commit never deletes it. Current write approval or
run timeout may block publication while host recovery still records the submitted
fact. An expired historical attempt deadline does not invalidate that fact.
Unknown, absent, timed-out, cancelled or inconsistent query outcomes never create
another attempt or call create again. Cancellation is not compensation or rollback.

## Verification

`tests/test_offline_ticket_receipts.py` covers response loss, absent/unknown and
temporary outcomes, timeout, cancelled query, full binding/schema tampering,
query-time state changes, no local connection across await, independent SQL writer
progress, concurrent duplicate and conflicting queries, sticky effect history,
approval/current-run/historical deadline boundaries, durable-receipt recovery,
untrusted self-reported success and actual separate-process recovery.

The cancellation-after-commit and crash-between-commit/reconcile cases inject
exceptions at the exact boundary. They are not UI cancellation or OS crash tests.
The CLI subprocess case does execute separate Python processes. This example does
not claim real provider authentication, exactly-once external effects, cross-run
business idempotency, enforcement against malicious Python host extensions, or
hard interruption of non-cooperative/blocking code.

No changes to core runtime/ledger, production configuration, original checkout or
the pre-existing `tests/test_agent_ui.mjs` are required for this example.
