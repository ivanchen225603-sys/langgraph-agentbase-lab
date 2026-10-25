// Proposal metadata only. There are no calls to RunEngine execution/control routes.
const el = id => document.getElementById(id);
const state = {generation: 0, session: null, record: null, busy: false, timer: null,
  controller: null, barrier: Promise.resolve(), dirty: false, reopen: false, suspended: false};
const authorization = () => el('bearer-token').value ? {Authorization: 'Bearer ' + el('bearer-token').value} : {};
const text = value => {el('intent-status').textContent = value;};
const context = () => ({agent: el('intent-agent').value, conversation: el('intent-conversation').value});
const base = agent => '/api/v1/agents/' + encodeURIComponent(agent) + '/intent-sessions';
const labels = {
  pending: '待独立确认：请先核对提案与版本。',
  confirmed: '已确认，仅记录选择。执行端未接入；未创建或取消任务，也未授予工具审批。',
  revoked: '已撤销本次选择。执行端未接入。',
  expired: '提案或确认记录已过期，请重新生成。执行端未接入。',
  superseded: '需求版本已变化，旧提案已失效。',
  stale: '目标任务已变化，旧提案不可确认。',
  clarify: '需要澄清，请先修改需求。',
  respond: '仅解释，无可确认的动作。',
};

async function request(path, body, method = 'POST', signal) {
  const response = await fetch(path, {method, signal,
    headers: {'Content-Type': 'application/json', ...authorization()},
    body: body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'intent_request_failed');
  return data;
}
function controls() {
  const record = state.record;
  const blocked = state.busy || state.suspended;
  el('intent-generate').disabled = blocked;
  el('intent-confirm').disabled = blocked || record?.status !== 'pending' || !el('intent-checked').checked;
  el('intent-revoke').disabled = blocked || !['pending', 'confirmed'].includes(record?.status);
  el('intent-get').disabled = blocked || !record;
}
function clearTimer() {
  if (state.timer !== null) clearTimeout(state.timer);
  state.timer = null;
}
function render(record) {
  clearTimer(); state.record = record;
  el('intent-checked').checked = false;
  el('intent-preview').textContent = JSON.stringify(record, null, 2);
  text(labels[record.status] || '提案状态未知，请重新查询。');
  if (['pending', 'confirmed'].includes(record.status)) {
    const generation = state.generation;
    state.timer = setTimeout(() => {
      if (generation !== state.generation || state.record !== record) return;
      record.status = 'expired'; state.timer = null;
      el('intent-checked').checked = false;
      text(labels.expired); controls();
    }, Math.max(0, (record.expires_at - record.server_time) * 1000));
  }
  controls();
}
function queueInvalidation(session) {
  state.barrier = state.barrier.catch(() => {}).then(async () => {
    try {
      const info = await request(base(session.agent) + '/' + session.session_id + '/invalidate');
      if (state.session === session) session.revision = Math.max(session.revision, info.revision);
    } catch {
      if (state.session === session) state.reopen = true;
    }
  });
}
function changed(resetSession = false) {
  state.generation++; state.controller?.abort(); state.controller = null;
  state.busy = false; state.record = null; clearTimer();
  el('intent-checked').checked = false;
  el('intent-preview').textContent = '需求已变化，旧提案不再用于确认。';
  text('需求或身份上下文已变化，请重新生成提案。');
  if (state.session && !state.dirty) queueInvalidation(state.session);
  state.dirty = true;
  if (resetSession) {state.session = null; state.reopen = false;}
  controls();
}
function draft() {
  const action = el('intent-action').value;
  return {
    text: el('intent-text').value,
    action,
    task: ['propose_task', 'propose_replace'].includes(action) ? {input: JSON.parse(el('intent-input').value)} : null,
    target_run_id: ['propose_replace', 'propose_cancel'].includes(action) ? el('intent-target').value : null,
  };
}
async function generate() {
  if (state.busy || state.suspended) return;
  const generation = state.generation;
  let value;
  try {value = draft();} catch {text('任务输入不是有效 JSON。'); return;}
  const selected = context();
  if (!selected.agent) {text('请先选择 Agent。'); return;}
  state.busy = true; state.record = null; clearTimer(); el('intent-checked').checked = false; controls();
  try {
    await state.barrier;
    if (generation !== state.generation) return;
    if (!state.session || state.reopen) {
      const info = await request(base(selected.agent), {conversation_id: selected.conversation});
      const opened = {...info, agent: selected.agent};
      if (generation !== state.generation) {queueInvalidation(opened); return;}
      state.session = opened; state.reopen = false;
    }
    const session = state.session;
    const controller = new AbortController(); state.controller = controller; state.dirty = false;
    text('正在生成提案；不会执行任务。');
    const result = await request(base(session.agent) + '/' + session.session_id + '/proposals',
      {expected_revision: session.revision, draft: value}, 'POST', controller.signal);
    if (generation !== state.generation || state.session !== session) return;
    session.revision = result.revision;
    render(result);
  } catch (error) {
    if (generation === state.generation) {
      state.reopen = true; text('提案未生成：' + error.message + '。请重新生成。');
    }
  } finally {
    if (generation === state.generation) {state.busy = false; state.controller = null; controls();}
  }
}
async function choice(action) {
  if (state.busy || state.suspended || !state.record) return;
  if (action === 'confirm' && (state.record.status !== 'pending' || !el('intent-checked').checked)) return;
  if (action === 'revoke' && !['pending', 'confirmed'].includes(state.record.status)) return;
  const generation = state.generation, record = state.record, session = state.session;
  let body;
  try {
    body = {revision: record.revision, request_digest: record.request_digest,
      decision_digest: record.decision_digest, draft: draft()};
  } catch {changed(); text('需求已变化，请重新生成。'); return;}
  state.busy = true; controls();
  try {
    const result = await request(base(session.agent) + '/' + session.session_id + '/proposals/' + record.proposal_id + '/' + action, body);
    if (generation === state.generation && state.record === record) render(result);
  } catch (error) {
    if (generation === state.generation) {
      render({...record, status: 'unknown'});
      text('选择结果尚未核实：' + error.message + '。请查询提案状态。执行端未接入。');
    }
  } finally {
    if (generation === state.generation) {state.busy = false; controls();}
  }
}
async function refreshRecord() {
  if (state.busy || state.suspended || !state.record) return;
  const generation = state.generation, record = state.record, session = state.session;
  state.busy = true; controls();
  try {
    const result = await request(base(session.agent) + '/' + session.session_id + '/proposals/' + record.proposal_id, undefined, 'GET');
    if (generation === state.generation && state.record === record) render(result);
  } catch (error) {
    if (generation === state.generation) {
      render({...record, status: 'unknown'});
      text('状态尚未核实：' + error.message + '。请再次查询提案状态。执行端未接入。');
    }
  } finally {
    if (generation === state.generation) {state.busy = false; controls();}
  }
}
async function refreshAgents() {
  const credential = el('bearer-token').value;
  try {
    const data = await request('/api/v1/agents', undefined, 'GET');
    if (credential !== el('bearer-token').value) return;
    const selected = el('intent-agent').value;
    el('intent-agent').replaceChildren();
    for (const agent of data.agents) {
      const option = document.createElement('option'); option.value = agent.id; option.textContent = agent.id + ' · ' + agent.description;
      el('intent-agent').append(option);
    }
    if (data.agents.some(agent => agent.id === selected)) el('intent-agent').value = selected;
    changed(true);
  } catch (error) {text('无法读取 Agent：' + error.message);}
}
function suspendPage() {
  state.generation++; state.suspended = true;
  state.controller?.abort(); state.controller = null; state.busy = false;
  clearTimer(); el('intent-checked').checked = false;
  if (state.record) render({...state.record, status: 'unknown'});
  else state.reopen = true;
  controls();
}
function resumePage() {
  if (!state.suspended) return;
  state.suspended = false; controls();
  if (state.record) void refreshRecord();
  else text('页面已恢复，请重新生成提案。执行端未接入。');
}

el('intent-generate').addEventListener('click', generate);
el('intent-confirm').addEventListener('click', () => choice('confirm'));
el('intent-revoke').addEventListener('click', () => choice('revoke'));
el('intent-get').addEventListener('click', refreshRecord);
el('intent-refresh').addEventListener('click', refreshAgents);
el('intent-checked').addEventListener('change', controls);
el('intent-agent').addEventListener('change', () => changed(true));
el('intent-conversation').addEventListener('input', () => changed(true));
for (const id of ['intent-text', 'intent-input', 'intent-target']) el(id).addEventListener('input', () => changed());
el('intent-action').addEventListener('change', () => changed());
el('bearer-token').addEventListener('input', () => {changed(true); void refreshAgents();});
window.addEventListener('pagehide', suspendPage);
window.addEventListener('pageshow', resumePage);
controls(); void refreshAgents();
