import assert from 'node:assert/strict';
import test from 'node:test';

let sequence = 0;
async function setup() {
  const elements = new Map(), timers = new Map(), calls = [];
  let timerId = 0;
  function element(id) {
    if (!elements.has(id)) elements.set(id, {id, value: '', checked: false, disabled: false,
      textContent: '', listeners: {}, options: [],
      addEventListener(name, fn) {this.listeners[name] = fn;},
      replaceChildren() {this.options = []; this.value = '';},
      append(option) {this.options.push(option); if (this.options.length === 1) this.value = option.value;},
    });
    return elements.get(id);
  }
  globalThis.document = {getElementById: element, createElement: () => ({})};
  const events = {};
  globalThis.window = {addEventListener(name, fn) {events[name] = fn;}};
  globalThis.setTimeout = (fn, delay) => {timers.set(++timerId, {fn, delay}); return timerId;};
  globalThis.clearTimeout = id => timers.delete(id);
  element('intent-action').value = 'propose_task';
  element('intent-conversation').value = 'same';
  element('intent-text').value = 'create proposal';
  element('intent-input').value = '{"text":"synthetic"}';
  const response = (data, status = 200) => {
    const snapshot = structuredClone(data);
    return {ok: status < 400, status, json: async () => structuredClone(snapshot)};
  };
  const server = {revision: 0, record: null, confirmGate: null, proposalGate: null,
    queryGate: null, queryFailures: 0, lostConfirm: false};
  globalThis.fetch = async (url, options) => {
    calls.push({url, options});
    assert.ok(url === '/api/v1/agents' || url.includes('/intent-sessions'), 'no business execution route');
    if (url === '/api/v1/agents') return response({agents: [{id: 'worker', description: 'fixture'}]});
    if (url.endsWith('/intent-sessions')) return response({session_id: 's1', revision: server.revision});
    if (url.endsWith('/invalidate')) {server.revision++; if (server.record) server.record.status = 'superseded'; return response({revision: server.revision});}
    if (url.endsWith('/proposals')) {
      const body = JSON.parse(options.body);
      assert.equal(body.expected_revision, server.revision);
      server.revision++;
      server.record = {session_id: 's1', proposal_id: 'p' + server.revision, revision: server.revision,
        request_digest: 'sha256:' + 'a'.repeat(64), decision_digest: 'sha256:' + 'b'.repeat(64),
        status: 'pending', payload: {draft: body.draft}, expires_at: 120, server_time: 0,
        execution_connected: false, tool_approval: false};
      const result = response(server.record);
      if (server.proposalGate) await server.proposalGate.promise;
      return result;
    }
    if (url.endsWith('/confirm')) {
      server.record.status = 'confirmed';
      const result = response(server.record);
      if (server.confirmGate) await server.confirmGate.promise;
      if (server.lostConfirm) throw new Error('response lost after record commit');
      return result;
    }
    if (url.endsWith('/revoke')) {server.record.status = 'revoked'; return response(server.record);}
    if (server.queryFailures > 0) {server.queryFailures--; throw new Error('temporary query failure');}
    const result = response(server.record);
    if (server.queryGate) await server.queryGate.promise;
    return result;
  };
  const flush = async () => {for (let i = 0; i < 30; i++) await Promise.resolve();};
  await import('../raglab/web/intent-ui.mjs?fixture=' + ++sequence);
  await flush();
  const fire = (id, event = 'click') => element(id).listeners[event]();
  return {element, calls, timers, server, flush, fire, events};
}
function deferred() {let resolve; const promise = new Promise(done => {resolve = done;}); return {promise, resolve};}

test('independent checkbox confirmation is one request; withdrawal records only metadata', async () => {
  const ui = await setup();
  await ui.fire('intent-generate');
  assert.equal(ui.element('intent-confirm').disabled, true);
  await ui.fire('intent-confirm');
  assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, 0);
  ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
  assert.equal(ui.element('intent-confirm').disabled, false);
  await ui.fire('intent-confirm'); await ui.fire('intent-confirm');
  assert.match(ui.element('intent-status').textContent, /已确认.*执行端未接入/);
  assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, 1);
  assert.equal(ui.element('intent-confirm').disabled, true);
  await ui.fire('intent-revoke');
  assert.match(ui.element('intent-status').textContent, /已撤销/);
  assert.equal(ui.element('intent-revoke').disabled, true);
});

test('draft edits invalidate old versions before a new proposal is requested', async () => {
  const ui = await setup(); await ui.fire('intent-generate');
  ui.element('intent-text').value = 'new requirement'; ui.fire('intent-text', 'input');
  assert.equal(ui.element('intent-confirm').disabled, true);
  await ui.fire('intent-confirm'); await ui.fire('intent-generate');
  assert.equal(ui.server.record.payload.draft.text, 'new requirement');
  const latest = ui.calls.findLast(c => c.url.endsWith('/proposals'));
  assert.equal(JSON.parse(latest.options.body).expected_revision, 2);
  assert.ok(ui.calls.some(c => c.url.endsWith('/invalidate')));
});

test('late proposal after an edit cannot reenable confirmation or replace the draft', async () => {
  const ui = await setup(); ui.server.proposalGate = deferred();
  const old = ui.fire('intent-generate'); await ui.flush();
  ui.element('intent-text').value = 'new draft'; ui.fire('intent-text', 'input');
  await ui.flush(); ui.server.proposalGate.resolve(); await old;
  assert.equal(ui.element('intent-confirm').disabled, true);
  assert.ok(!ui.element('intent-preview').textContent.includes('synthetic'));
});

test('local expiry disables confirmation even if checkbox is clicked again', async () => {
  const ui = await setup(); await ui.fire('intent-generate');
  const timer = [...ui.timers.values()].find(t => t.delay === 120000);
  assert.ok(timer); timer.fn();
  ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
  await ui.fire('intent-confirm');
  assert.match(ui.element('intent-status').textContent, /已过期/);
  assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, 0);
});

test('unknown confirmation response can be queried without retrying confirmation', async () => {
  const ui = await setup(); await ui.fire('intent-generate');
  ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
  ui.server.lostConfirm = true; await ui.fire('intent-confirm');
  assert.match(ui.element('intent-status').textContent, /尚未核实/);
  assert.equal(ui.element('intent-get').disabled, false);
  await ui.fire('intent-get');
  assert.match(ui.element('intent-status').textContent, /已确认/);
  assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, 1);
});

test('late confirmation from a different context does not update the current UI', async () => {
  const ui = await setup(); await ui.fire('intent-generate');
  ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
  ui.server.confirmGate = deferred(); const old = ui.fire('intent-confirm');
  ui.element('intent-conversation').value = 'other'; ui.fire('intent-conversation', 'input');
  ui.server.confirmGate.resolve(); await old;
  assert.equal(ui.element('intent-confirm').disabled, true);
  assert.ok(!ui.element('intent-status').textContent.includes('已确认'));
  ui.events.pagehide(); assert.equal(ui.timers.size, 0);
});

test('a failed query after a lost confirmation keeps the choice and can be retried without invalidation', async () => {
  const ui = await setup(); await ui.fire('intent-generate');
  ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
  ui.server.lostConfirm = true; await ui.fire('intent-confirm');
  ui.server.queryFailures = 1; await ui.fire('intent-get'); await ui.flush();
  assert.equal(ui.server.record.status, 'confirmed');
  assert.equal(ui.calls.filter(c => c.url.endsWith('/invalidate')).length, 0);
  assert.match(ui.element('intent-status').textContent, /尚未核实/);
  assert.equal(ui.element('intent-get').disabled, false);
  assert.equal(ui.element('intent-confirm').disabled, true);
  await ui.fire('intent-get');
  assert.match(ui.element('intent-status').textContent, /已确认/);
  assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, 1);
  assert.equal(ui.calls.filter(c => c.url.endsWith('/invalidate')).length, 0);
});

for (const operation of ['generate', 'confirm', 'query']) {
  test(`page restoration releases ${operation} ownership and discards its late response`, async () => {
    const ui = await setup();
    if (operation !== 'generate') await ui.fire('intent-generate');
    const gate = deferred();
    ui.server[{generate: 'proposalGate', confirm: 'confirmGate', query: 'queryGate'}[operation]] = gate;
    if (operation === 'confirm') {
      ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
    }
    const old = ui.fire({generate: 'intent-generate', confirm: 'intent-confirm', query: 'intent-get'}[operation]);
    await ui.flush();
    assert.equal(ui.element('intent-generate').disabled, true);
    ui.events.pagehide();
    if (operation === 'query') {ui.server.record.status = 'expired'; ui.server.queryGate = null;}
    ui.events.pageshow?.({persisted: true}); await ui.flush();
    assert.equal(ui.element('intent-generate').disabled, false);
    assert.equal(ui.element('intent-confirm').disabled, true);
    if (operation === 'confirm') assert.match(ui.element('intent-status').textContent, /已确认/);
    if (operation === 'query') assert.match(ui.element('intent-status').textContent, /已过期/);
    const preview = ui.element('intent-preview').textContent;
    const status = ui.element('intent-status').textContent;
    gate.resolve(); await old;
    assert.equal(ui.element('intent-preview').textContent, preview);
    assert.equal(ui.element('intent-status').textContent, status);
    assert.equal(ui.element('intent-generate').disabled, false);
    assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, operation === 'confirm' ? 1 : 0);
    assert.equal(ui.calls.filter(c => c.url.endsWith('/invalidate')).length, 0);
    if (operation === 'generate') {
      ui.server.proposalGate = null; await ui.fire('intent-generate');
      assert.match(ui.element('intent-status').textContent, /待独立确认/);
    }
  });
}

test('credential changes clear the old choice and prevent a late confirmation from restoring it', async () => {
  const ui = await setup(); await ui.fire('intent-generate');
  ui.element('intent-checked').checked = true; ui.fire('intent-checked', 'change');
  ui.server.confirmGate = deferred(); const old = ui.fire('intent-confirm');
  await ui.flush();
  ui.element('bearer-token').value = 'synthetic-new-identity'; ui.fire('bearer-token', 'input');
  await ui.flush(); ui.server.confirmGate.resolve(); await old;
  assert.equal(ui.element('intent-confirm').disabled, true);
  assert.equal(ui.element('intent-get').disabled, true);
  assert.ok(!ui.element('intent-status').textContent.includes('已确认'));
  assert.equal(ui.calls.filter(c => c.url.endsWith('/confirm')).length, 1);
});
