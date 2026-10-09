import {SseDecoder, validateEvent} from './sse.mjs';

const el = id => document.getElementById(id);
const state = {run: null, agent: null, seq: 0, generation: 0, selection: 0, controller: null, review: null, candidate: '', cycle: null};
const actions = new Map();
const text = (id, value) => {el(id).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);};
const authorization = () => el('bearer-token').value ? {Authorization: 'Bearer ' + el('bearer-token').value} : {};

async function request(path, body, method = body ? 'POST' : 'GET', headers = {}) {
  const response = await fetch('/api/v1/agents' + path, {method, headers: {'Content-Type': 'application/json', ...authorization(), ...headers}, body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'request_failed');
  return data;
}
function clearCandidate() {state.candidate = ''; state.cycle = null; text('generic-candidate', ''); el('generic-candidate-panel').hidden = true;}
function stop() {state.generation++; state.controller?.abort(); state.controller = null;}
function clearRun() {stop(); state.selection++; actions.clear(); for (const id of ['generic-refresh', 'generic-submit', 'generic-get', 'generic-stream', 'generic-stop', 'generic-cancel']) el(id).disabled = false; state.run = null; state.agent = null; state.seq = 0; state.review = null; clearCandidate(); text('generic-status', ''); text('generic-result', '尚无运行'); text('generic-review', ''); el('generic-approve').disabled = true; el('error').hidden = true;}
function showRun(run) {
  if (!['queued', 'running'].includes(run.status) || run.resume_reason || run.result !== null) clearCandidate();
  text('generic-status', `${run.status} · ${run.resume_reason || ''}`);
  text('generic-result', run.result === null ? '当前没有可展示的最终结果。' : run.result);
  state.review = run.recoverable && run.interrupts?.length === 1 ? run.interrupts[0] : null;
  text('generic-review', state.review || '无待确认内容');
  el('generic-approve').disabled = !state.review;
}
async function snapshot(own = state.generation) {
  const result = await request(`/${state.agent}/runs/${state.run}`);
  if (own === state.generation) showRun(result);
  return result;
}
async function subscribe() {
  if (!state.run) throw new Error('请先创建或查询一个运行。');
  stop();
  const own = state.generation;
  const controller = new AbortController(); state.controller = controller;
  const agent = state.agent, run = state.run;
  try {
    const response = await fetch(`/api/v1/agents/${agent}/runs/${run}/events`, {headers: {...authorization(), ...(state.seq ? {'Last-Event-ID': `${run}:${state.seq}`} : {})}, signal: controller.signal});
    if (own !== state.generation) return;
    if (!response.ok) throw new Error((await response.json()).error || 'stream_failed');
    if (!response.body) throw new Error('stream_unavailable');
    const reader = response.body.getReader(), decoder = new SseDecoder();
    try {
      while (own === state.generation) {
        const chunk = await reader.read();
        if (own !== state.generation) return;
        for (const frame of decoder.push(chunk.value || new Uint8Array(), chunk.done)) {
          if (own !== state.generation) return;
          const event = validateEvent(frame, run, state.seq);
          if (!event) continue;
          state.seq = event.seq;
          if (event.kind === 'model.delta' && event.provisional) {
            if (state.cycle !== event.data?.value?.cycle) clearCandidate();
            state.cycle = event.data?.value?.cycle;
            state.candidate += event.data?.value?.text || '';
            if (state.candidate.length > 262144) throw new Error('stream_candidate_limit');
            el('generic-candidate-panel').hidden = false; text('generic-candidate', state.candidate);
          }
          if (['run.final', 'run.terminal', 'run.awaiting_review'].includes(event.kind)) {
            clearCandidate(); await snapshot(own);
          }
        }
        if (chunk.done) {await snapshot(own); break;}
      }
    } finally {try {await reader.cancel();} catch {} reader.releaseLock();}
  } catch (error) {
    if (own !== state.generation || error.name === 'AbortError') return;
    clearCandidate();
    await snapshot(own).catch(() => {});
    throw error;
  } finally {if (own === state.generation) state.controller = null;}
}
function bind(id, fn) {
  el(id).addEventListener('click', async () => {
    el('error').hidden = true;
    const ticket = Symbol(id);
    let selection = state.selection;
    try {
      const pending = fn();
      selection = state.selection; actions.set(id, ticket); el(id).disabled = true;
      await pending;
    } catch (error) {if (selection === state.selection && actions.get(id) === ticket) {text('error', error.message); el('error').hidden = false;}}
    finally {if (actions.get(id) === ticket) {actions.delete(id); el(id).disabled = id === 'generic-approve' ? !state.review : false;}}
  });
}
function example(schema, depth = 0) {
  if (depth > 5) return null;
  if ('default' in schema) return schema.default;
  if (schema.enum?.length) return schema.enum[0];
  if (schema.type === 'object') return Object.fromEntries((schema.required || []).map(key => [key, example(schema.properties?.[key] || {}, depth + 1)]));
  if (schema.type === 'array') return [];
  if (schema.type === 'string') return '请输入内容';
  if (schema.type === 'boolean') return false;
  if (['number', 'integer'].includes(schema.type)) return schema.minimum || 0;
  return null;
}
async function agents() {
  const own = state.generation;
  const entries = (await request('')).agents;
  if (own !== state.generation) return;
  el('generic-agent').replaceChildren();
  for (const entry of entries) {
    const option = document.createElement('option'); option.value = entry.id; option.textContent = `${entry.id} · ${entry.description}`;
    option.dataset.schema = JSON.stringify(entry.input_schema); el('generic-agent').append(option);
  }
  selectAgent();
}
function selectAgent() {clearRun(); const schema = JSON.parse(el('generic-agent').selectedOptions[0]?.dataset.schema || '{}'); el('generic-input').value = JSON.stringify(example(schema), null, 2); text('generic-schema', schema);}
bind('generic-refresh', agents);
el('generic-agent').addEventListener('change', selectAgent);
bind('generic-submit', async () => {
  clearRun();
  const agent = el('generic-agent').value, own = state.generation;
  const run = await request(`/${agent}/runs`, {input: JSON.parse(el('generic-input').value), options: {conversation_id: el('generic-conversation').value, switch_topic: el('generic-topic').checked}}, 'POST', {'Idempotency-Key': crypto.randomUUID().replaceAll('-', '')});
  if (own !== state.generation) return;
  state.agent = agent; state.run = run.run_id; el('generic-run-id').value = run.run_id; showRun(run); await subscribe();
});
bind('generic-get', async () => {clearRun(); state.agent = el('generic-agent').value; state.run = el('generic-run-id').value; await snapshot();});
bind('generic-stream', subscribe);
bind('generic-stop', async () => {stop(); clearCandidate(); text('generic-status', '订阅已断开，后台运行继续。');});
bind('generic-cancel', async () => {
  if (!state.run) throw new Error('请先选择运行。');
  stop(); clearCandidate();
  const own = state.generation, agent = state.agent, runId = state.run;
  const run = await request(`/${agent}/runs/${runId}/cancel`, {});
  if (own === state.generation && agent === state.agent && runId === state.run) showRun(run);
});
bind('generic-approve', async () => {
  if (!state.review) throw new Error('请先查询并复核当前确认内容。');
  const {review_id, digest} = state.review;
  stop(); clearCandidate();
  const own = state.generation, agent = state.agent, runId = state.run;
  const run = await request(`/${agent}/runs/${runId}/resume`, {review_id, digest});
  if (own !== state.generation || agent !== state.agent || runId !== state.run) return;
  showRun(run); await subscribe();
});
el('bearer-token').addEventListener('input', () => {clearRun();});
el('bearer-token').addEventListener('change', () => {agents().catch(() => {});});
window.addEventListener('pagehide', stop);
agents().catch(error => text('generic-status', error.message));
