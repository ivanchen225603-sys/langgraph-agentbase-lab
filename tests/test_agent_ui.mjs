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
globalThis.window = {addEventListener() {}};
const calls = [], pending = [];
const schema = {type: 'string'};
const listing = {agents: [{id: 'A', description: 'first', input_schema: schema}, {id: 'B', description: 'second', input_schema: schema}]};
const view = (run, status = 'succeeded', result = {run}, review = null) => ({run_id: run, status, result, recoverable: !!review, interrupts: review ? [review] : []});
const response = (data, ok = true) => ({ok, json: async () => data});
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
for (const action of ['cancel', 'approve']) {
  switchAgent('A');
  const review = {review_id: 'original', digest: 'a'.repeat(64)};
  await get('runA', view('runA', 'awaiting_review', null, review));
  const gate = deferred(); pending.push(() => gate.promise);
  const old = element(`generic-${action}`).listeners.click();
  switchAgent('B'); await get('runB');
  gate.resolve(response(view('runA', 'cancelled', {private: 'A old result'})));
  await old;
  assert.equal(element('generic-result').textContent, JSON.stringify({run: 'runB'}, null, 2));
  assert.equal(element('generic-status').textContent, 'succeeded · ');
  assert.equal(element('generic-approve').disabled, true);
  assert.equal(element('error').hidden, true);
  assert.ok(!calls.some(call => call.url.endsWith('/runB/events')));
  if (action === 'approve') assert.deepEqual(JSON.parse(calls.findLast(call => call.url.endsWith('/resume')).options.body), review);
}
switchAgent('A'); await get('runA');
const gate = deferred(); pending.push(() => gate.promise);
const old = element('generic-cancel').listeners.click();
element('bearer-token').value = 'new credential'; element('bearer-token').listeners.input();
gate.resolve(response({error: 'old_identity_error'}, false)); await old;
assert.equal(element('generic-result').textContent, '尚无运行');
assert.equal(element('error').hidden, true);
assert.equal(element('generic-cancel').disabled, false);
switchAgent('A'); await get('runA', view('runA', 'running', null));
const delta = {run_id: 'runA', seq: 1, kind: 'model.delta', provisional: true, data: {value: {cycle: 1, text: 'unfinished candidate'}}};
pending.push(() => ({ok: true, body: new ReadableStream({start(controller) {
  controller.enqueue(new TextEncoder().encode(`id: runA:1\nevent: model.delta\ndata: ${JSON.stringify(delta)}\n\n`)); controller.close();
}})}), view('runA'));
await element('generic-stream').listeners.click();
assert.equal(element('generic-candidate-panel').hidden, true);
assert.equal(element('generic-candidate').textContent, '');
assert.equal(element('generic-result').textContent, JSON.stringify({run: 'runA'}, null, 2));
console.log('Agent UI: stale cancel/approve/error responses after Agent or credential changes cannot replace the current view');
