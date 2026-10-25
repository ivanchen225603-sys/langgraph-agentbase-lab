// Bounded observation only: never retries a cancellation or a tool operation.
const terminal = new Set(['succeeded', 'partial', 'failed', 'cancelled', 'timed_out', 'needs_reconciliation', 'effect_applied_result_unavailable']);

export function cancellationState(view) {
  if (view.cleanup_pending === true || view.status === 'cancelling') {
    return {active: true, label: '正在收尾，任务尚未结束'};
  }
  if (view.cleanup_pending === false && terminal.has(view.status)) {
    const labels = {
      cancelled: '取消流程已结束，清理已完成',
      succeeded: '任务已完成，完成结果先于取消生效',
      partial: '任务部分完成，收尾已结束',
      failed: '任务失败，收尾已结束',
      timed_out: '任务超时，收尾已结束',
      needs_reconciliation: '执行已结束，外部操作结果仍待核对',
      effect_applied_result_unavailable: '执行已结束，已有操作生效，结果仍需核对',
    };
    return {active: false, label: labels[view.status]};
  }
  if (view.cancel_requested === true && view.cleanup_pending === false) {
    return {active: false, label: '收尾状态未知，尚未确认任务已结束'};
  }
  return {active: true, label: '仍未确认任务结束，正在有限核对'};
}

export class CancellationMonitor {
  constructor({fetchStatus, onChange, onView = () => {}, now = Date.now,
    schedule = (callback, delay) => setTimeout(callback, delay),
    cancelSchedule = handle => clearTimeout(handle),
    maxRuns = 4, maxAttempts = 30, maxAgeMs = 120000,
    intervalMs = 2000, requestTimeoutMs = 5000}) {
    Object.assign(this, {fetchStatus, onChange, onView, now, schedule, cancelSchedule,
      maxRuns, maxAttempts, maxAgeMs, intervalMs, requestTimeoutMs});
    this.records = new Map(); this.epoch = 0; this.timer = null; this.active = null;
    this.turn = 0; this.notice = '';
  }
  key(agent, runId) {return JSON.stringify([agent, runId]);}
  isWatching(agent, runId) {return this.records.get(this.key(agent, runId))?.active === true;}
  emit() {
    this.onChange({records: [...this.records.values()].map(({agent, runId, label, active}) => ({agent, runId, label, active})), notice: this.notice});
  }
  track(agent, runId, view = null) {
    const key = this.key(agent, runId);
    let record = this.records.get(key);
    if (!record) {
      if (this.records.size >= this.maxRuns) {
        const victim = [...this.records.values()].find(item => !item.active) || this.records.values().next().value;
        if (victim.active) this.notice = `运行 ${victim.runId} 已停止自动核对，状态仍未知，请按运行 ID 查询。`;
        this.records.delete(victim.key);
        if (this.active?.record === victim) this.active.controller.abort();
      }
      record = {key, agent, runId, epoch: this.epoch, attempts: 0, turn: 0,
        deadline: this.now() + this.maxAgeMs, active: true, proved: false,
        label: '已请求取消，尚未确认任务结束'};
      this.records.set(key, record);
    }
    if (view) this.apply(record, view);
    this.emit(); this.arm();
  }
  fail(agent, runId) {
    const record = this.records.get(this.key(agent, runId));
    if (!record) return;
    record.active = false;
    record.label = '取消或收尾失败，状态未知，尚未确认任务已结束';
    if (this.active?.record === record) this.active.controller.abort();
    this.emit(); this.arm();
  }
  current(record) {return record.epoch === this.epoch && this.records.get(record.key) === record;}
  apply(record, view) {
    if (!this.current(record)) return;
    if (view.run_id !== record.runId || view.agent_id !== record.agent) {
      record.active = false; record.label = '响应与原任务不符，状态未知'; return;
    }
    const next = cancellationState(view);
    // A delayed POST response must not downgrade a confirmed terminal GET.
    if (record.proved && !(view.cleanup_pending === false && terminal.has(view.status))) return;
    Object.assign(record, next);
    if (!next.active && view.cleanup_pending === false && terminal.has(view.status)) record.proved = true;
    this.onView(record.agent, record.runId, view);
  }
  arm() {
    if (![...this.records.values()].some(record => record.active)) {
      if (this.timer !== null) this.cancelSchedule(this.timer);
      this.timer = null; return;
    }
    if (this.timer === null && this.active === null) {
      this.timer = this.schedule(() => {this.timer = null; void this.poll();}, this.intervalMs);
    }
  }
  async poll() {
    for (const record of this.records.values()) {
      if (record.active && (record.attempts >= this.maxAttempts || this.now() >= record.deadline)) {
        record.active = false; record.label = '自动核对已到上限，任务仍未确认结束，请查询核对';
      }
    }
    const record = [...this.records.values()].filter(item => item.active).sort((a, b) => a.turn - b.turn)[0];
    if (!record) {this.emit(); this.arm(); return;}
    const controller = new AbortController(), flight = {record, controller};
    this.active = flight; record.attempts++; record.turn = ++this.turn;
    let rejectAbort;
    const aborted = new Promise((_, reject) => {rejectAbort = reject;});
    const onAbort = () => rejectAbort(Object.assign(new Error('observation_aborted'), {name: 'AbortError'}));
    controller.signal.addEventListener('abort', onAbort, {once: true});
    const timeout = this.schedule(() => controller.abort(), this.requestTimeoutMs);
    try {
      const view = await Promise.race([this.fetchStatus(record.agent, record.runId, controller.signal), aborted]);
      if (this.current(record)) this.apply(record, view);
    } catch (error) {
      if (this.current(record) && record.active) {
        record.label = '状态核对暂不可用，尚未确认结束，将有限重试';
        if ([401, 403, 404].includes(error.status)) {
          record.active = false; record.label = '无法访问原任务，状态未知，已停止自动核对';
        }
      }
    } finally {
      this.cancelSchedule(timeout);
      controller.signal.removeEventListener('abort', onAbort);
      const owns = this.active === flight;
      if (owns) this.active = null;
      if (this.current(record)) this.emit();
      if (owns) this.arm();
    }
  }
  clear() {
    this.epoch++;
    if (this.timer !== null) this.cancelSchedule(this.timer);
    this.timer = null;
    this.active?.controller.abort(); this.active = null;
    this.records.clear(); this.notice = ''; this.emit();
  }
}
