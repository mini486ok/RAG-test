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
   * idleTimeoutMs: 토큰 간 무응답이 이 시간을 넘으면 자동 중단.
   * @returns {Promise<{text, ttftMs, totalMs, promptTokens, evalTokens, evalDurationMs, loadDurationMs, aborted, incomplete}>}
   */
  async chatStream({ model, messages, options = {}, format, onToken, signal, idleTimeoutMs = 120000 }) {
    const t0 = performance.now();
    let ttftMs = null;
    let text = '';
    let stats = {};
    let aborted = false;
    let doneReceived = false;
    let timedOut = false;

    // 사용자 signal + 유휴 타임아웃을 결합
    const ctrl = new AbortController();
    const onUserAbort = () => ctrl.abort();
    signal?.addEventListener('abort', onUserAbort, { once: true });
    let idleTimer = idleTimeoutMs
      ? setTimeout(() => { timedOut = true; ctrl.abort(); }, idleTimeoutMs)
      : null;
    const resetIdle = () => {
      if (!idleTimer) return;
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { timedOut = true; ctrl.abort(); }, idleTimeoutMs);
    };

    const handleLine = (line) => {
      if (!line) return;
      let obj;
      try {
        obj = JSON.parse(line);
      } catch {
        return; // 불완전 라인 무시
      }
      if (obj.error) throw new Error(obj.error);
      const piece = obj.message?.content || '';
      if (piece) {
        if (ttftMs === null) ttftMs = performance.now() - t0;
        text += piece;
        onToken?.(piece, text);
      }
      if (obj.done) {
        doneReceived = true;
        stats = {
          promptTokens: obj.prompt_eval_count ?? null,
          evalTokens: obj.eval_count ?? null,
          evalDurationMs: obj.eval_duration ? obj.eval_duration / 1e6 : null,
          loadDurationMs: obj.load_duration ? obj.load_duration / 1e6 : null,
        };
      }
    };

    try {
      const res = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages, stream: true, options, ...(format ? { format } : {}) }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`Ollama 응답 오류 (HTTP ${res.status}) ${body.slice(0, 200)}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        resetIdle();
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          handleLine(line);
        }
      }
      // 디코더/버퍼 잔여분 flush (개행 없이 끝나는 마지막 라인 처리)
      buf += decoder.decode();
      handleLine(buf.trim());
    } catch (e) {
      if (e.name === 'AbortError') aborted = true;
      else throw e;
    } finally {
      if (idleTimer) clearTimeout(idleTimer);
      signal?.removeEventListener('abort', onUserAbort);
    }

    if (timedOut) throw new Error(`응답이 ${Math.round(idleTimeoutMs / 1000)}초간 없어 중단했습니다. Ollama 상태를 확인하세요.`);

    return {
      text,
      ttftMs,
      totalMs: performance.now() - t0,
      aborted,
      incomplete: !aborted && !doneReceived,
      ...stats,
    };
  }

  /** 비스트리밍 JSON 응답 (그래프 추출용) — timeoutMs 내 미완료 시 실패 */
  async chatJSON({ model, messages, options = {}, signal, timeoutMs = 180000 }) {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: false, format: 'json', options }),
      signal: composeSignal(signal, timeoutMs),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`Ollama 응답 오류 (HTTP ${res.status}) ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    return data.message?.content || '';
  }

  /** 임베딩 (input: string 배열) → Float32Array 배열 */
  async embed({ model, input, signal, timeoutMs = 120000 }) {
    const res = await fetch(`${this.baseUrl}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, input }),
      signal: composeSignal(signal, timeoutMs),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`임베딩 오류 (HTTP ${res.status}) ${body.slice(0, 200)}`);
    }
    const data = await res.json();
    const arr = data.embeddings || [];
    if (arr.length !== input.length) {
      throw new Error(`임베딩 개수 불일치 (요청 ${input.length}, 응답 ${arr.length}). 임베딩 모델을 확인하세요.`);
    }
    return arr.map((v) => {
      const f = new Float32Array(v);
      normalize(f);
      return f;
    });
  }
}

/** 사용자 signal과 타임아웃을 결합한 AbortSignal */
function composeSignal(signal, timeoutMs) {
  if (!timeoutMs) return signal;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException('시간 초과', 'TimeoutError')), timeoutMs);
  const clean = () => clearTimeout(timer);
  ctrl.signal.addEventListener('abort', clean, { once: true });
  if (signal) {
    if (signal.aborted) ctrl.abort(signal.reason);
    else signal.addEventListener('abort', () => ctrl.abort(signal.reason), { once: true });
  }
  return ctrl.signal;
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
