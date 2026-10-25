import assert from 'node:assert/strict';
import test from 'node:test';
import {CancellationMonitor, cancellationState} from '../raglab/web/cancel-status.mjs';

const flush = async () => {for (let i = 0; i < 12; i++) await Promise.resolve();};
function deferred() {let resolve; const promise = new Promise(done => {resolve = done;}); return {promise, resolve};}
function harness(fetchStatus, options = {}) {
  let time = 0, sequence = 0;
  const timers = new Map(), changes = [], views = [], calls = [];
  const monitor = new CancellationMonitor({
    fetchStatus: (...args) => {calls.push(args); return fetchStatus(...args);},
    onChange: value => changes.push(value), onView: (...args) => views.push(args),
    now: () => time, schedule: (fn, delay) => {timers.set(++sequence, {fn, at: time + delay}); return sequence;},
    cancelSchedule: id => timers.delete(id), ...options,
  });
  const next = async () => {
    const entry = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
    assert.ok(entry, 'expected an owned timer'); timers.delete(entry[0]); time = entry[1].at;
    entry[1].fn(); await flush();
  };
  return {monitor, timers, changes, views, calls, next};
}
const view = (id, status = 'cancelling', extra = {}) => ({agent_id: 'A', run_id: id, status,
  cancel_requested: true, cleanup_pending: status === 'cancelling', result: null, ...extra});

test('pending cancellation reaches confirmed terminal automatically, with no remaining timers', async () => {
  let count = 0;
  const h = harness(async () => view('old', ++count === 1 ? 'cancelling' : 'cancelled'));
  h.monitor.track('A', 'old', view('old'));
  assert.match(h.changes.at(-1).records[0].label, /尚未结束/);
  await h.next(); await h.next();
  assert.equal(h.calls.length, 2); assert.equal(h.timers.size, 0);
  assert.match(h.changes.at(-1).records[0].label, /清理已完成/);
  assert.equal(h.changes.at(-1).records[0].active, false);
});

test('scope reset aborts an in-flight GET, ignores its late response, and leaves no timers', async () => {
  const gate = deferred();
  const h = harness(() => gate.promise);
  h.monitor.track('A', 'old'); await h.next();
  const signal = h.calls[0][2]; h.monitor.clear(); await flush();
  assert.equal(signal.aborted, true); assert.equal(h.timers.size, 0);
  gate.resolve(view('old', 'cancelled', {result: {private: 'OLD'}})); await flush();
  assert.equal(h.views.length, 0); assert.deepEqual(h.changes.at(-1).records, []);
});

test('repeated tracking neither resets budgets nor overlaps requests; exhaustion remains unknown', async () => {
  const h = harness(async () => view('old'), {maxAttempts: 2});
  h.monitor.track('A', 'old'); const record = h.monitor.records.values().next().value;
  const deadline = record.deadline;
  await h.next(); h.monitor.track('A', 'old'); await h.next();
  h.monitor.track('A', 'old'); await h.next();
  assert.equal(h.calls.length, 2); assert.equal(record.deadline, deadline);
  assert.equal(h.timers.size, 0); assert.match(record.label, /上限.*未确认结束/);
});

test('request timeout is bounded even when fetch ignores abort, and other records can progress', async () => {
  const gate = deferred();
  const h = harness((agent, id) => id === 'old' ? gate.promise : Promise.resolve(view(id, 'cancelled')), {maxAttempts: 1});
  h.monitor.track('A', 'old'); h.monitor.track('A', 'new');
  await h.next(); await h.next(); // poll old, then timeout it
  assert.equal(h.calls[0][2].aborted, true);
  await h.next(); // exhaust old, query new
  assert.equal(h.calls.length, 2); assert.equal(h.timers.size, 0);
  gate.resolve(view('old', 'cancelled')); await flush();
  assert.ok(!h.views.some(([, id]) => id === 'old'));
});

test('capacity eviction aborts the removed owner without stalling remaining observations', async () => {
  const gate = deferred();
  const h = harness((agent, id) => id === 'first' ? gate.promise : Promise.resolve(view(id, 'cancelled')), {maxRuns: 2});
  h.monitor.track('A', 'first'); await h.next();
  h.monitor.track('A', 'second'); h.monitor.track('A', 'third'); await flush();
  assert.equal(h.calls[0][2].aborted, true); assert.equal(h.monitor.records.size, 2);
  assert.match(h.changes.at(-1).notice, /first.*状态仍未知/);
  await h.next(); await h.next(); assert.equal(h.timers.size, 0);
  gate.resolve(view('first', 'cancelled')); await flush();
  assert.ok(!h.views.some(([, id]) => id === 'first'));
});

test('age limit ends observation without claiming termination', async () => {
  const h = harness(async () => {throw new Error('should not request');}, {maxAgeMs: 1});
  h.monitor.track('A', 'old'); await h.next();
  assert.equal(h.calls.length, 0); assert.equal(h.timers.size, 0);
  assert.match(h.changes.at(-1).records[0].label, /未确认结束/);
});

test('failed cleanup and inaccessible status stop safely; no stopped claim or result in summary', async () => {
  const h = harness(async () => {throw Object.assign(new Error('denied'), {status: 403});});
  h.monitor.track('A', 'old'); await h.next();
  assert.equal(h.timers.size, 0); assert.match(h.changes.at(-1).records[0].label, /状态未知/);
  h.monitor.track('A', 'failure'); h.monitor.fail('A', 'failure');
  assert.equal(h.timers.size, 0); assert.match(h.changes.at(-1).records.at(-1).label, /收尾失败/);
  assert.equal(JSON.stringify(h.changes).includes('result'), false);
  assert.equal(cancellationState(view('broken', 'interrupted')).active, false);
  assert.match(cancellationState(view('broken', 'interrupted')).label, /状态未知/);
});

test('binding mismatches and missing cleanup proof never become confirmed stopped', async () => {
  const h = harness(async () => view('wrong', 'cancelled', {result: {secret: 'not allowed'}}));
  h.monitor.track('A', 'old'); await h.next();
  assert.equal(h.views.length, 0); assert.match(h.changes.at(-1).records[0].label, /响应与原任务不符/);
  const missing = {status: 'cancelled'};
  assert.equal(cancellationState(missing).active, true);
  assert.match(cancellationState(view('old', 'needs_reconciliation')).label, /仍待核对/);
});

test('a delayed cancel response cannot replace a terminal observation with pending state', async () => {
  const h = harness(async () => view('old', 'cancelled'));
  h.monitor.track('A', 'old'); await h.next();
  const before = h.views.length;
  h.monitor.track('A', 'old', view('old'));
  assert.equal(h.views.length, before); assert.equal(h.timers.size, 0);
  assert.match(h.changes.at(-1).records[0].label, /清理已完成/);
});
