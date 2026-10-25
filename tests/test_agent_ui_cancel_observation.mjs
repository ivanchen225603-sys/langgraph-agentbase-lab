// Independent DOM and clock fixture for cancellation observation.
// Execute the shipped UI module against controllable HTTP promises.
import assert from 'node:assert/strict';

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, value: '', textContent: '', disabled: false, hidden: true, dataset: {}, listeners: {}, options: [],
    addEventListener(name, callback) {this.listeners[name] = callback;},
    replaceChildren() {this.options = [];},
    append(option) {this.options.push(option); if (this.options.length === 1) this.value = option.value;},
    get selectedOptions() {return this.options.filter(option => option.value === this.value);},
  });
  return elements.get(id);
}
globalThis.document = {getElementById: element, createElement: () => ({dataset: {}})};
const pageListeners = {}, timers = new Map(); let timerSequence = 0;
globalThis.window = {addEventListener(name, callback) {pageListeners[name] = callback;}};
// Browser timer functions reject a monitor object as their receiver.
globalThis.setTimeout = function(fn, delay) {assert.equal(this, undefined); timers.set(++timerSequence, {fn, delay}); return timerSequence;};
globalThis.clearTimeout = function(id) {assert.equal(this, undefined); return timers.delete(id);};
const calls = [], pending = [];
const schema = {type: 'string'};
const listing = {agents: [{id: 'A', description: 'first', input_schema: schema}, {id: 'B', description: 'second', input_schema: schema}]};
const view = (run, status = 'succeeded', result = {run}, review = null) => ({agent_id: run === 'runB' ? 'B' : 'A', run_id: run, status, cleanup_pending: status === 'cancelling', cancel_requested: ['cancelling','cancelled'].includes(status), result, recoverable: !!review, interrupts: review ? [review] : []});
const response = (data, ok = true) => ({ok, status: ok ? 200 : 503, json: async () => data});
globalThis.fetch = async (url, options) => {
  calls.push({url, options});
  if (url === '/api/v1/agents') return response(listing);
  const next = pending.shift();
  assert.ok(next, `unexpected request: ${url}`);
  return typeof next === 'function' ? next(url, options) : response(next);
};
await import('../raglab/web/agent-ui.js');
await new Promise(resolve => setImmediate(resolve));
function switchAgent(id) {element('generic-agent').value = id; element('generic-agent').listeners.change();}
async function get(run, result = view(run)) {element('generic-run-id').value = run; pending.push(result); await element('generic-get').listeners.click();}
function deferred() {let resolve; const promise = new Promise(done => {resolve = done;}); return {promise, resolve};}
async function poll() {
  const timer = [...timers].find(([, value]) => value.delay === 2000);
  assert.ok(timer, 'expected bounded cleanup observation'); timers.delete(timer[0]); timer[1].fn();
  for (let i = 0; i < 15; i++) await Promise.resolve();
}
switchAgent('A'); await get('old', view('old', 'running', null));
pending.push(view('old', 'cancelling', null)); await element('generic-cancel').listeners.click();
assert.match(element('generic-status').textContent, /正在收尾/);
assert.match(element('generic-cleanup-status').textContent, /old.*尚未结束/);
const cancelCount = calls.filter(call => call.url.endsWith('/old/cancel')).length;
await element('generic-cancel').listeners.click();
assert.equal(calls.filter(call => call.url.endsWith('/old/cancel')).length, cancelCount);
await get('new', view('new', 'succeeded', {marker: 'NEW'}));
pending.push(view('old', 'cancelled', {marker: 'OLD_MUST_NOT_OVERWRITE'})); await poll();
assert.equal(element('generic-result').textContent, JSON.stringify({marker: 'NEW'}, null, 2));
assert.match(element('generic-cleanup-status').textContent, /old.*清理已完成/);
assert.equal(timers.size, 0);

await get('failure', view('failure', 'running', null));
pending.push(() => response({error: 'cancel_cleanup_failed'}, false));
await element('generic-cancel').listeners.click();
assert.match(element('generic-cleanup-status').textContent, /收尾失败.*状态未知/);
assert.equal(timers.size, 0);

await get('leaving', view('leaving', 'running', null));
pending.push(view('leaving', 'cancelling', null)); await element('generic-cancel').listeners.click();
const leaving = deferred(); pending.push(() => leaving.promise); await poll();
const request = calls.findLast(call => call.url.endsWith('/leaving'));
pageListeners.pagehide();
for (let i = 0; i < 15; i++) await Promise.resolve();
assert.equal(request.options.signal.aborted, true); assert.equal(timers.size, 0);
leaving.resolve(response(view('leaving', 'cancelled', {marker: 'LATE_AFTER_PAGEHIDE'})));
for (let i = 0; i < 15; i++) await Promise.resolve();
assert.equal(element('generic-cleanup-panel').hidden, true);
assert.ok(!element('generic-result').textContent.includes('LATE_AFTER_PAGEHIDE'));
console.log('Agent UI: automatic cleanup summary survives same-Agent run changes, never replaces new output, and cancels on pagehide');
