// Decode complete SSE frames and retain split UTF-8 characters between chunks.
export class SseDecoder {
  constructor(maxBytes = 262144) {
    this.decoder = new TextDecoder('utf-8', {fatal: true});
    this.buffer = '';
    this.maxBytes = maxBytes;
  }
  push(bytes, final = false) {
    this.buffer += this.decoder.decode(bytes, {stream: !final});
    const frames = [];
    while (true) {
      const boundary = /\r?\n\r?\n/.exec(this.buffer);
      if (!boundary) break;
      const text = this.buffer.slice(0, boundary.index);
      if (new TextEncoder().encode(text).length > this.maxBytes) throw new Error('stream_frame_limit');
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      const frame = {event: 'message', id: null, data: []};
      for (const line of text.split(/\r?\n/)) {
        if (!line || line.startsWith(':')) continue;
        const index = line.indexOf(':');
        const field = index < 0 ? line : line.slice(0, index);
        const value = index < 0 ? '' : line.slice(index + 1).replace(/^ /, '');
        if (field === 'event') frame.event = value;
        if (field === 'id') frame.id = value;
        if (field === 'data') frame.data.push(value);
      }
      if (frame.data.length) frames.push({...frame, data: JSON.parse(frame.data.join('\n'))});
    }
    if (new TextEncoder().encode(this.buffer).length > this.maxBytes) throw new Error('stream_frame_limit');
    if (final && this.buffer.trim()) throw new Error('stream_incomplete_frame');
    return frames;
  }
}

export function validateEvent(frame, runId, lastSeq) {
  const event = frame.data;
  if (frame.event === 'stream.error') {
    if (frame.id !== null) throw new Error('stream_control_id');
    throw new Error(event.error || 'stream_failed');
  }
  if (!event || event.run_id !== runId || !Number.isSafeInteger(event.seq) || event.seq < 1 || frame.id !== `${runId}:${event.seq}` || frame.event !== event.kind) throw new Error('stream_event_binding');
  if (event.seq <= lastSeq) return null;
  if (event.seq !== lastSeq + 1) throw new Error('event_cursor_expired');
  return event;
}
