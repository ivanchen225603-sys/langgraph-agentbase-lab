import assert from 'node:assert/strict';
import {SseDecoder, validateEvent} from '../raglab/web/sse.mjs';

const event = {run_id: 'runA', seq: 1, kind: 'model.delta', provisional: true, data: {text: '中文\n第二行'}};
const source = `id: runA:1\r\nevent: model.delta\r\ndata: ${JSON.stringify(event)}\r\n\r\n: heartbeat\r\n\r\n`;
const bytes = new TextEncoder().encode(source);
for (let cut = 0; cut <= bytes.length; cut++) {
  const decoder = new SseDecoder();
  const frames = [...decoder.push(bytes.slice(0, cut)), ...decoder.push(bytes.slice(cut), true)];
  assert.equal(frames.length, 1);
  assert.deepEqual(validateEvent(frames[0], 'runA', 0), event);
  assert.equal(validateEvent(frames[0], 'runA', 1), null);
}
const bytewise = new SseDecoder();
const frames = Array.from(bytes).flatMap(byte => bytewise.push(Uint8Array.of(byte)));
assert.deepEqual(validateEvent(frames[0], 'runA', 0), event);
bytewise.push(new Uint8Array(), true);
const frame = {id: 'runA:1', event: event.kind, data: event};
assert.throws(() => validateEvent(frame, 'other', 0), /stream_event_binding/);
assert.throws(() => validateEvent({...frame, id: 'runA:3', data: {...event, seq: 3}}, 'runA', 1), /event_cursor_expired/);
assert.throws(() => validateEvent({id: null, event: 'stream.error', data: {error: 'evidence_revoked'}}, 'runA', 0), /evidence_revoked/);
assert.throws(() => validateEvent({id: 'runA:1', event: 'stream.error', data: {}}, 'runA', 0), /stream_control_id/);
assert.throws(() => new SseDecoder().push(new TextEncoder().encode('data: {}'), true), /stream_incomplete_frame/);
assert.throws(() => new SseDecoder(4).push(new TextEncoder().encode('data: {}\n\n')), /stream_frame_limit/);
assert.throws(() => new SseDecoder().push(Uint8Array.of(255), true));
console.log(`SSE decoder: ${bytes.length + 1} UTF-8/CRLF boundaries, bytewise, cursor, control and limit cases passed`);
