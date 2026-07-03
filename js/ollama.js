// ═══════════════════════════════════════════════
// Ollama HTTP API 클라이언트
//  - /api/tags   : 모델 목록
//  - /api/chat   : 스트리밍 대화 (NDJSON)
//  - /api/embed  : 임베딩 (배치)
// ═══════════════════════════════════════════════

export class OllamaClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/+$/, '');
  }

  /** 서버 연결 + 모델 목록. 실패 시 throw */
  async listModels(timeoutMs = 5000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(`${this.baseUrl}/api/tags`, { signal: ctrl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return (data.models || []).map((m) => ({
        name: m.name,
        size: m.size,
        family: m.details?.family || '',
      }));
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * 스트리밍 채팅.
   * @returns {Promise<{text, ttftMs, totalMs, promptTokens, evalTokens, evalDurationMs, aborted}>}
   */
  async chatStream({ model, messages, options = {}, format, onToken, signal }) {
    const t0 = performance.now();
    let ttftMs = null;
    let text = '';
    let stats = {};
    let aborted = false;

    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: true, options, ...(format ? { format } : {}) }),
      signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`Ollama 응답 오류 (HTTP ${res.status}) ${body.slice(0, 200)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let obj;
          try {
            obj = JSON.parse(line);
          } catch {
            continue; // 불완전 라인 무시
          }
          if (obj.error) throw new Error(obj.error);
          const piece = obj.message?.content || '';
          if (piece) {
            if (ttftMs === null) ttftMs = performance.now() - t0;
            text += piece;
            onToken?.(piece, text);
          }
          if (obj.done) {
            stats = {
              promptTokens: obj.prompt_eval_count ?? null,
              evalTokens: obj.eval_count ?? null,
              evalDurationMs: obj.eval_duration ? obj.eval_duration / 1e6 : null,
              loadDurationMs: obj.load_duration ? obj.load_duration / 1e6 : null,
            };
          }
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') aborted = true;
      else throw e;
    }

    return {
      text,
      ttftMs,
      totalMs: performance.now() - t0,
      aborted,
      ...stats,
    };
  }

  /** 비스트리밍 JSON 응답 (그래프 추출용) */
  async chatJSON({ model, messages, options = {}, signal }) {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: false, format: 'json', options }),
      signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`Ollama 응답 오류 (HTTP ${res.status}) ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    return data.message?.content || '';
  }

  /** 임베딩 (input: string 배열) → Float32Array 배열 */
  async embed({ model, input, signal }) {
    const res = await fetch(`${this.baseUrl}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, input }),
      signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`임베딩 오류 (HTTP ${res.status}) ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    const arr = data.embeddings || [];
    return arr.map((v) => {
      const f = new Float32Array(v);
      normalize(f);
      return f;
    });
  }
}

/** L2 정규화 (in-place) — 이후 코사인 = 내적 */
export function normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= s;
  return v;
}

export function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}
