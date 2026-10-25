// Isolated DOM/HTTP fixtures exercise the shipped module; no real browser is used.
import assert from 'node:assert/strict';
import {test} from 'node:test';

let sequence = 0;
const turn = () => new Promise(resolve => setImmediate(resolve));
function deferred() {let resolve; const promise = new Promise(done => {resolve = done;}); return {promise, resolve};}
const reply = (data, status = 200) => ({ok: status >= 200 && status < 300, status, json: async () => data});
const view = (id = 'r1', options = {}) => ({agent_id: 'A', run_id: id, status: 'interrupted', result: null, recoverable: true, interrupts: [], ...options});
const page = (ids, next = null, agent = 'A') => ({runs: ids.map(run_id => ({run_id, agent_id: agent})), next_after: next});
async function fixture() {
  const elements = new Map(), calls = [], pending = [], events = {};
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      id, value: '', textContent: '', disabled: false, hidden: true, dataset: {}, listeners: {}, options: [],
      addEventListener(event, fn) {this.listeners[event] = fn;},
      replaceChildren() {this.options = []; this.value = '';},
      append(option) {this.options.push(option); if (this.options.length === 1) this.value = option.value;},
      get selectedOptions() {return this.options.filter(option => option.value === this.value);},
    });
    return elements.get(id);
  }
  globalThis.document = {getElementById: element, createElement: () => ({dataset: {}})};
  globalThis.window = {addEventListener(event, fn) {events[event] = fn;}};
  globalThis.fetch = async (url, options) => {
    calls.push({url, options});
    if (url === '/api/v1/agents') return reply({agents: ['A', 'B'].map(id => ({id, description: id, input_schema: {type: 'string'}}))});
    const next = pending.shift(); assert.ok(next, `unexpected HTTP request ${url}`);
    return typeof next === 'function' ? next(url, options) : reply(next);
  };
  await import(`../raglab/web/agent-ui.js?closure=${++sequence}`); await turn();
  const click = id => element(id).listeners.click();
  async function get(run = view()) {element('generic-run-id').value = run.run_id; pending.push(run); await click('generic-get');}
  async function listing(data = page(['r1'])) {pending.push(data); await click('generic-run-refresh');}
  function choose(id) {element('generic-run-list').value = id; element('generic-run-list').listeners.change();}
  function agent(id) {element('generic-agent').value = id; element('generic-agent').listeners.change();}
  const writes = () => calls.filter(call => call.options.method === 'POST');
  return {element, calls, pending, events, click, get, listing, choose, agent, writes};
}

test('index refresh, bounded next page and explicit open use only GET; selection grants no recovery', async () => {
  const f = await fixture();
  assert.equal(f.calls.length, 1);
  await f.listing(page(['r1', 'r2'], 'r2'));
  assert.match(f.calls.at(-1).url, /runs\?limit=20$/);
  assert.equal(f.element('generic-run-next').disabled, false);
  f.choose('r1');
  assert.equal(f.element('generic-run-id').value, 'r1');
  assert.equal(f.writes().length, 0); assert.equal(f.element('generic-resume').disabled, true);
  f.pending.push(view('r1', {status: 'succeeded', result: {text: 'AUTHORIZED'}, recoverable: false}));
  await f.click('generic-run-open');
  assert.equal(f.calls.at(-1).url, '/api/v1/agents/A/runs/r1');
  assert.match(f.element('generic-result').textContent, /AUTHORIZED/);
  f.pending.push(page(['r3'])); await f.click('generic-run-next');
  assert.match(f.calls.at(-1).url, /limit=20&after=r2$/);
  assert.equal(f.element('generic-run-next').disabled, true);
  const count = f.calls.length; await f.click('generic-run-next'); assert.equal(f.calls.length, count);
  await f.listing(page([])); assert.match(f.element('generic-run-notice').textContent, /没有/);
  assert.equal(f.element('generic-run-open').disabled, true);
  assert.equal(f.writes().length, 0);
});

test('pending pagination is single flight; Agent switch aborts and discards late private IDs', async () => {
  const f = await fixture(), gate = deferred(); f.pending.push(() => gate.promise);
  const first = f.click('generic-run-refresh'); const selected = f.calls.at(-1);
  await f.click('generic-run-refresh'); assert.equal(f.calls.filter(call => call.url.includes('?limit')).length, 1);
  f.agent('B'); assert.equal(selected.options.signal.aborted, true);
  gate.resolve(reply(page(['PRIVATE_OLD']))); await first;
  assert.equal(f.element('generic-run-list').options.length, 0);
  assert.equal(f.element('generic-run-next').disabled, true);
  await f.listing(page(['b1'], null, 'B')); assert.equal(f.element('generic-run-list').options.at(-1).value, 'b1');
  assert.equal(f.writes().length, 0);
});

for (const event of ['credential', 'page']) test(`${event} clears IDs/cursors, releases busy actions and rejects old list/detail responses`, async () => {
  const f = await fixture(); await f.listing(page(['PRIVATE_OLD'], 'PRIVATE_OLD')); f.choose('PRIVATE_OLD');
  const gate = deferred(); f.pending.push(() => gate.promise); const old = f.click('generic-run-open');
  if (event === 'credential') {f.element('bearer-token').value = 'new'; f.element('bearer-token').listeners.input();}
  else {f.events.pagehide(); f.events.pageshow();}
  assert.equal(f.element('generic-run-id').value, ''); assert.equal(f.element('generic-run-list').options.length, 0);
  assert.equal(f.element('generic-run-next').disabled, true); assert.equal(f.element('generic-resume').disabled, true);
  gate.resolve(reply(view('PRIVATE_OLD', {result: {private: 'OLD_RESULT'}}))); await old;
  assert.doesNotMatch(f.element('generic-result').textContent, /OLD_RESULT/);
  const count = f.calls.length; await turn(); assert.equal(f.calls.length, count);
  await f.listing(page(['new1'])); assert.equal(f.element('generic-run-list').options.at(-1).value, 'new1');
  assert.equal(f.writes().length, 0);
});

test('pagehide aborts an in-flight list and restored page does not replay any request', async () => {
  const f = await fixture(), gate = deferred(); f.pending.push(() => gate.promise);
  const old = f.click('generic-run-refresh'), call = f.calls.at(-1);
  f.events.pagehide(); f.events.pageshow(); const count = f.calls.length;
  assert.equal(call.options.signal.aborted, true);
  gate.resolve(reply(page(['OLD_ID']))); await old;
  assert.equal(f.element('generic-run-list').options.length, 0); assert.equal(f.calls.length, count);
  await f.listing(page(['fresh'])); assert.equal(f.element('generic-run-list').options.at(-1).value, 'fresh');
});

test('bad index shape, duplicate/nonadvancing IDs and failed query clear page and allow manual retry', async () => {
  const f = await fixture();
  for (const invalid of [page(['x', 'x']), page(['<script>']), page(['b', 'a']), page(['x'], 'other'), page(['x'], null, 'B'), page(Array.from({length: 21}, (_, i) => `x${i}`))]) {
    await f.listing(invalid); assert.equal(f.element('generic-run-list').options.length, 0);
    assert.equal(f.element('generic-run-refresh').disabled, false); assert.equal(f.element('generic-run-next').disabled, true);
  }
  await f.listing(page(['r2'], 'r2')); f.pending.push(page(['r1'])); await f.click('generic-run-next');
  assert.equal(f.element('generic-run-list').options.length, 0);
  f.pending.push(() => reply({error: 'unavailable'}, 503)); await f.click('generic-run-refresh');
  assert.match(f.element('generic-run-notice').textContent, /尚未核实/);
  await f.listing(page(['r3'])); assert.equal(f.element('generic-run-list').options.at(-1).value, 'r3');
  assert.equal(f.writes().length, 0);
});

test('current credential rejection clears private IDs and old result without probing another identity', async () => {
  const f = await fixture(); await f.listing(page(['private'])); await f.get(view('private', {result: {private: 'SECRET'}}));
  f.pending.push(() => reply({error: 'forbidden'}, 403)); await f.click('generic-run-refresh');
  assert.equal(f.element('generic-run-list').options.length, 0); assert.equal(f.element('generic-run-id').value, '');
  assert.doesNotMatch(f.element('generic-result').textContent, /SECRET/); assert.equal(f.writes().length, 0);
});

test('a timed-out list is not accepted even if its late transport response succeeds', async () => {
  const f = await fixture(), gate = deferred();
  const originalSet = globalThis.setTimeout, originalClear = globalThis.clearTimeout;
  let timeout;
  try {
    globalThis.setTimeout = (fn, delay) => {assert.equal(delay, 10000); timeout = fn; return 123;};
    globalThis.clearTimeout = () => {};
    f.pending.push(() => gate.promise); const old = f.click('generic-run-refresh');
    timeout(); assert.equal(f.calls.at(-1).options.signal.aborted, true);
    gate.resolve(reply(page(['TOO_LATE']))); await old;
    assert.equal(f.element('generic-run-list').options.length, 0);
    assert.equal(f.element('generic-run-refresh').disabled, false);
  } finally {globalThis.setTimeout = originalSet; globalThis.clearTimeout = originalClear;}
  await f.listing(page(['fresh'])); assert.equal(f.element('generic-run-list').options.at(-1).value, 'fresh');
});

test('late denial of an old Run does not clear a newer selection', async () => {
  const f = await fixture(), gate = deferred();
  f.element('generic-run-id').value = 'old'; f.pending.push(() => gate.promise);
  const old = f.click('generic-get');
  f.element('generic-run-id').listeners.input();
  await f.get(view('new', {status: 'succeeded', result: {current: 'NEW'}, recoverable: false}));
  gate.resolve(reply({error: 'forbidden'}, 403)); await old;
  assert.match(f.element('generic-result').textContent, /NEW/); assert.equal(f.element('error').hidden, true);
});

test('ordinary recovery rechecks detail, sends one explicit empty resume, and then reads the existing result', async () => {
  const f = await fixture(); await f.get(); assert.equal(f.element('generic-resume').disabled, false);
  const gate = deferred(); f.pending.push(() => gate.promise);
  const resume = f.click('generic-resume'); await f.click('generic-resume');
  assert.equal(f.writes().length, 0);
  const completed = view('r1', {status: 'succeeded', recoverable: false, result: {ok: true}});
  f.pending.push(completed, () => ({ok: true, body: new ReadableStream({start(controller) {controller.close();}})}), completed);
  gate.resolve(reply(view())); await resume;
  const posts = f.writes(); assert.equal(posts.length, 1);
  assert.equal(posts[0].url, '/api/v1/agents/A/runs/r1/resume'); assert.deepEqual(JSON.parse(posts[0].options.body), {});
  assert.equal(f.element('generic-resume').disabled, true); assert.match(f.element('generic-result').textContent, /true/);
});

test('review, changed status, revoked result and wrong Run response cannot use ordinary recovery', async () => {
  const f = await fixture();
  for (const fresh of [view('r1', {status: 'awaiting_review', interrupts: [{review_id: 'review1', digest: 'sha256:' + 'a'.repeat(64)}]}), view('r1', {status: 'cancelled', recoverable: false}), view('r1', {result_visibility: 'revoked'}), view('OTHER'), view('r1', {agent_id: 'B'})]) {
    await f.get(); f.pending.push(fresh); await f.click('generic-resume');
    assert.equal(f.writes().length, 0); assert.equal(f.element('generic-resume').disabled, true);
  }
  await f.get(view('r1', {status: 'awaiting_review', interrupts: [{review_id: 'review1', digest: 'sha256:' + 'a'.repeat(64)}]}));
  assert.equal(f.element('generic-resume').disabled, true); assert.equal(f.element('generic-approve').disabled, false);
  const count = f.calls.length; await f.click('generic-resume'); assert.equal(f.calls.length, count);
});

for (const event of ['Agent', 'credential', 'page', 'input', 'selection']) test(`${event} change invalidates pending recovery preflight and prior approval`, async () => {
  const f = await fixture(); await f.get(); const gate = deferred(); f.pending.push(() => gate.promise);
  const old = f.click('generic-resume');
  if (event === 'Agent') f.agent('B');
  else if (event === 'credential') {f.element('bearer-token').value = 'changed'; f.element('bearer-token').listeners.input();}
  else if (event === 'page') {f.events.pagehide(); f.events.pageshow();}
  else if (event === 'input') {f.element('generic-input').value = 'changed'; f.element('generic-input').listeners.input();}
  else {await f.listing(page(['r2'])); f.choose('r2');}
  gate.resolve(reply(view())); await old;
  assert.equal(f.writes().length, 0); assert.equal(f.element('generic-resume').disabled, true); assert.equal(f.element('generic-approve').disabled, true);
});

test('lost resume response stays unknown; no repeat resume until a separate manual detail query', async () => {
  const f = await fixture(); await f.get();
  f.pending.push(view(), () => reply({error: 'unavailable'}, 503)); await f.click('generic-resume');
  assert.equal(f.writes().length, 1); assert.match(f.element('generic-status').textContent, /尚未核实/);
  assert.equal(f.element('generic-resume').disabled, true); const count = f.calls.length;
  await f.click('generic-resume'); assert.equal(f.calls.length, count);
  await f.get(view('r1', {status: 'succeeded', result: {recovered: true}, recoverable: false}));
  assert.match(f.element('generic-result').textContent, /recovered/); assert.equal(f.writes().length, 1);
});

test('a late accepted resume cannot replace the view after credentials change', async () => {
  const f = await fixture(); await f.get(); const gate = deferred(); f.pending.push(view(), () => gate.promise);
  const old = f.click('generic-resume'); await turn(); assert.equal(f.writes().length, 1);
  f.element('bearer-token').value = 'new'; f.element('bearer-token').listeners.input();
  gate.resolve(reply(view('r1', {result: {private: 'OLD_RESUME'}}))); await old;
  assert.doesNotMatch(f.element('generic-result').textContent, /OLD_RESUME/);
  assert.equal(f.element('generic-resume').disabled, true);
  assert.equal(f.calls.filter(call => call.url.endsWith('/events')).length, 0);
});

test('editing input, conversation, topic or manual ID removes a previously visible exact approval', async () => {
  const f = await fixture();
  for (const id of ['generic-input', 'generic-conversation', 'generic-topic', 'generic-run-id']) {
    await f.get(view('r1', {status: 'awaiting_review', interrupts: [{review_id: 'review', digest: 'sha256:' + 'b'.repeat(64)}]}));
    assert.equal(f.element('generic-approve').disabled, false);
    f.element(id).listeners.input(); assert.equal(f.element('generic-approve').disabled, true);
    await f.click('generic-approve');
  }
  assert.equal(f.writes().length, 0);
});
