import type { SSEEvent } from "@/types";

// POST /research/stream returns text/event-stream.
// The server's _summarize() (app/api/server.py) strips the full plan/findings/
// critique payloads down to counts and IDs — see SSEUpdate in @/types. If the
// UI needs rationale/excerpt/score breakdown for live runs, expand
// _summarize() on the backend; the demo mode uses local mock data.

export interface StreamHandle {
  events: AsyncIterable<SSEEvent>;
  abort: () => void;
}

const API_BASE = "/api";

export function streamResearch(query: string, sessionId?: string): StreamHandle {
  const controller = new AbortController();

  async function* parse(): AsyncIterable<SSEEvent> {
    const res = await fetch(`${API_BASE}/research/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ query, session_id: sessionId }),
      signal: controller.signal,
    });

    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}: ${await res.text().catch(() => "")}`);
    }

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      let nl: number;
      while ((nl = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 2);
        const ev = parseSSEChunk(chunk);
        if (ev) yield ev;
      }
    }
  }

  return { events: parse(), abort: () => controller.abort() };
}

function parseSSEChunk(chunk: string): SSEEvent | null {
  let event: string | null = null;
  let dataLines: string[] = [];
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!event || dataLines.length === 0) return null;
  try {
    const data = JSON.parse(dataLines.join("\n"));
    return { event, data } as SSEEvent;
  } catch {
    return null;
  }
}

export async function checkHealth(): Promise<{ ok: boolean; mode?: string }> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return { ok: false };
    const j = await res.json();
    return { ok: j.status === "ok", mode: j.mode };
  } catch {
    return { ok: false };
  }
}
