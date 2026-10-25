import {SseDecoder, validateEvent} from './sse.mjs';
import {CancellationMonitor} from './cancel-status.mjs?v=1';

const el = id => document.getElementById(id);
const state = {run: null, agent: null, seq: 0, generation: 0, selection: 0, scope: 0, active: true, controller: null, review: null, resumable: false, candidate: '', cycle: null};
const index = {epoch: 0, agent: null, ids: [], next: null, busy: false, controller: null, timer: null};
const actions = new Map();
const text = (id, value) => {el(id).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);};
const authorization = () => el('bearer-token').value ? {Authorization: 'Bearer ' + el('bearer-token').value} : {};

async function request(path, body, method = body ? 'POST' : 'GET', headers = {}, signal) {
  const scope = state.scope, selection = state.selection;
  const response = await fetch('/api/v1/agents' + path, {method, signal, headers: {'Content-Type': 'application/json', ...authorization(), ...headers}, body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) {
    const target = /^\/([^/]+)\/runs\/([^/?]+)/.exec(path);
    if ([401, 403].includes(response.status) && scope === state.scope && selection === state.selection && state.active && (!target || target[1] === state.agent && target[2] === state.run)) {
      clearRun(true); text('generic-status', '当前认证或权限已变化，已清除原运行信息，请重新查询。');
    }
    throw Object.assign(new Error(data.error || 'request_failed'), {status: response.status});
  }
  return data;
}
function clearCandidate() {state.candidate = ''; state.cycle = null; text('generic-candidate', ''); el('generic-candidate-panel').hidden = true;}
function stop() {state.generation++; state.controller?.abort(); state.controller = null;}
function controls() {
  const blocked = {
    'generic-approve': !state.review,
    'generic-resume': !state.resumable,
    'generic-run-refresh': index.busy || !el('generic-agent').value,
    'generic-run-next': index.busy || !index.next,
    'generic-run-open': index.busy || index.agent !== el('generic-agent').value || !index.ids.includes(el('generic-run-list').value),
  };
  for (const id of ['generic-refresh', 'generic-submit', 'generic-get', 'generic-stream', 'generic-stop', 'generic-cancel', ...Object.keys(blocked)]) {
    el(id).disabled = !state.active || actions.has(id) || !!blocked[id];
  }
  el('generic-run-list').disabled = !state.active || index.busy || !index.ids.length;
}
function resetIndex(notice = '请刷新运行列表。') {
  index.epoch++; index.controller?.abort(); clearTimeout(index.timer);
  Object.assign(index, {agent: null, ids: [], next: null, busy: false, controller: null, timer: null});
  el('generic-run-list').replaceChildren(); text('generic-run-notice', notice);
}
function clearRun(resetCleanup = false) {
  stop();
  if (resetCleanup) {state.scope++; cancellations.clear(); resetIndex(); el('generic-run-id').value = '';}
  state.selection++; actions.clear(); state.run = null; state.agent = null; state.seq = 0; state.review = null; state.resumable = false;
  clearCandidate(); text('generic-status', ''); text('generic-result', '尚无运行'); text('generic-review', ''); el('error').hidden = true; controls();
}
function currentRun(own, agent, run) {
  return state.active && own === state.generation && agent === state.agent && run === state.run && agent === el('generic-agent').value;
}
async function loadIndex(next = false) {
  if (index.busy || !state.active) return;
  const agent = el('generic-agent').value;
  if (!agent) throw new Error('请先选择 Agent。');
  const after = next ? index.next : null;
  if (next && (!after || index.agent !== agent)) return;
  resetIndex('正在查询运行 ID…');
  const epoch = index.epoch, scope = state.scope, controller = new AbortController();
  index.busy = true; index.controller = controller;
  index.timer = setTimeout(() => controller.abort(), 10000); controls();
  try {
    const page = await request(`/${agent}/runs?limit=20${after ? '&after=' + encodeURIComponent(after) : ''}`, undefined, 'GET', {}, controller.signal);
    if (!state.active || scope !== state.scope || epoch !== index.epoch || agent !== el('generic-agent').value) return;
    if (controller.signal.aborted) throw new Error('运行列表查询超时，请手动刷新。');
    const ids = page.runs?.map(row => row.agent_id === agent ? row.run_id : null);
    if (!Array.isArray(ids) || ids.length > 20 || ids.some((id, position) => typeof id !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(id) || id <= (position ? ids[position - 1] : after || '')) || !(page.next_after === null || ids.length && page.next_after === ids.at(-1))) {
      throw new Error('运行列表响应无效，请重新查询。');
    }
    index.agent = agent; index.ids = ids; index.next = page.next_after;
    const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = '请选择运行 ID'; el('generic-run-list').append(placeholder);
    for (const id of ids) {const option = document.createElement('option'); option.value = id; option.textContent = id; el('generic-run-list').append(option);}
    text('generic-run-notice', ids.length ? `本页 ${ids.length} 条 · 按 ID 排序${index.next ? '，还有下一页' : '，已到页尾'}` : '当前没有运行记录。');
  } catch (error) {
    if (scope === state.scope && epoch === index.epoch && state.active) {
      resetIndex('运行列表尚未核实，请手动刷新。');
      throw error;
    }
  } finally {
    if (epoch === index.epoch) {clearTimeout(index.timer); index.busy = false; index.controller = null; index.timer = null; controls();}
  }
}
const cancellations = new CancellationMonitor({
  fetchStatus: (agent, runId, signal) => request(`/${agent}/runs/${runId}`, undefined, 'GET', {}, signal),
  onChange: ({records, notice}) => {
    text('generic-cleanup-status', [...records.map(record => `运行 ${record.runId}：${record.label}`), notice].filter(Boolean).join('\n'));
    el('generic-cleanup-panel').hidden = records.length === 0 && !notice;
  },
  onView: (agent, runId, run) => {if (agent === state.agent && runId === state.run) showRun(run);},
});
function showRun(run) {
  if (!['queued', 'running'].includes(run.status) || run.resume_reason || run.result !== null) clearCandidate();
  text('generic-status', `${run.status} · ${run.cleanup_pending === true ? '正在收尾，任务尚未结束' : run.resume_reason || ''}`);
  text('generic-result', run.result === null ? '当前没有可展示的最终结果。' : run.result);
  state.review = run.recoverable && run.interrupts?.length === 1 ? run.interrupts[0] : null;
  state.resumable = run.run_id === state.run && run.agent_id === state.agent && run.recoverable === true && ['interrupted', 'initialization_incomplete', 'finalization_incomplete'].includes(run.status) && Array.isArray(run.interrupts) && run.interrupts.length === 0 && !run.resume_reason && !run.cancel_requested && !run.cleanup_pending && !['revoked', 'unverified'].includes(run.result_visibility);
  text('generic-review', state.review || '无待确认内容');
  controls();
}
async function snapshot(own = state.generation) {
  const result = await request(`/${state.agent}/runs/${state.run}`);
  if (state.active && own === state.generation) showRun(result);
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
    if (!state.active || el(id).disabled || actions.has(id)) return;
    el('error').hidden = true;
    const ticket = Symbol(id);
    let selection = state.selection;
    try {
      const pending = fn();
      selection = state.selection; actions.set(id, ticket); el(id).disabled = true;
      await pending;
    } catch (error) {if (selection === state.selection && actions.get(id) === ticket) {text('error', error.message); el('error').hidden = false;}}
    finally {if (actions.get(id) === ticket) {actions.delete(id); controls();}}
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
  clearRun(true);
  const own = state.generation;
  let entries;
  try {entries = (await request('')).agents;}
  catch (error) {if (state.active && own === state.generation) throw error; return;}
  if (!state.active || own !== state.generation) return;
  el('generic-agent').replaceChildren();
  for (const entry of entries) {
    const option = document.createElement('option'); option.value = entry.id; option.textContent = `${entry.id} · ${entry.description}`;
    option.dataset.schema = JSON.stringify(entry.input_schema); el('generic-agent').append(option);
  }
  selectAgent();
}
function selectAgent() {clearRun(true); const schema = JSON.parse(el('generic-agent').selectedOptions[0]?.dataset.schema || '{}'); el('generic-input').value = JSON.stringify(example(schema), null, 2); text('generic-schema', schema);}
bind('generic-refresh', agents);
bind('generic-run-refresh', () => loadIndex());
bind('generic-run-next', () => loadIndex(true));
el('generic-run-list').addEventListener('change', () => {clearRun(); el('generic-run-id').value = index.ids.includes(el('generic-run-list').value) ? el('generic-run-list').value : ''; controls();});
bind('generic-run-open', async () => {
  const run = el('generic-run-list').value, agent = el('generic-agent').value;
  if (index.agent !== agent || !index.ids.includes(run)) return;
  clearRun(); state.agent = agent; state.run = run; el('generic-run-id').value = run; await snapshot();
});
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
  const agent = state.agent, runId = state.run, scope = cancellations.epoch;
  if (cancellations.isWatching(agent, runId)) return;
  cancellations.track(agent, runId);
  text('generic-status', '已请求取消，尚未确认任务结束');
  try {
    const run = await request(`/${agent}/runs/${runId}/cancel`, {});
    if (scope === cancellations.epoch) cancellations.track(agent, runId, run);
  } catch (error) {
    if (scope === cancellations.epoch) cancellations.fail(agent, runId);
    throw error;
  }
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
bind('generic-resume', async () => {
  if (!state.resumable || !state.run) return;
  stop(); clearCandidate();
  const own = state.generation, agent = state.agent, runId = state.run;
  state.resumable = false; controls();
  let submitted = false;
  try {
    const fresh = await request(`/${agent}/runs/${runId}`);
    if (!currentRun(own, agent, runId)) return;
    if (fresh.run_id !== runId || fresh.agent_id !== agent) throw new Error('运行查询响应与当前选择不一致。');
    showRun(fresh);
    if (!state.resumable) throw new Error('当前状态不允许普通恢复，请查询并核对最新状态或审批。');
    state.resumable = false; controls();
    submitted = true;
    const run = await request(`/${agent}/runs/${runId}/resume`, {});
    if (!currentRun(own, agent, runId)) return;
    if (run.run_id !== runId || run.agent_id !== agent) throw new Error('恢复响应与当前选择不一致，请重新查询。');
    showRun(run); await subscribe();
  } catch (error) {
    if (currentRun(own, agent, runId)) {
      state.resumable = false; state.review = null; controls();
      text('generic-status', submitted ? '恢复结果尚未核实，请手动查询；不会自动重试或批准。' : '普通恢复未发起，请查询当前状态；待审批须独立核对。');
    }
    throw error;
  }
});
for (const id of ['generic-run-id', 'generic-input', 'generic-conversation', 'generic-topic']) {
  el(id).addEventListener('input', () => clearRun());
  el(id).addEventListener('change', () => clearRun());
}
el('bearer-token').addEventListener('input', () => {clearRun(true);});
el('bearer-token').addEventListener('change', () => {agents().catch(() => {});});
window.addEventListener('pagehide', () => {state.active = false; clearRun(true);});
window.addEventListener('pageshow', (event = {}) => {if (state.active && !event.persisted) return; state.active = true; clearRun(true); text('generic-status', '请刷新列表或手动查询运行；不会自动恢复或批准。');});
agents().catch(error => text('generic-status', error.message));
