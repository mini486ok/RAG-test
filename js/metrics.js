// ═══════════════════════════════════════════════
// 성능 지표 수집 · 비교 차트(DOM 바) · 실험 기록
// 차트 규칙: 지표당 하나의 축(미니차트 분리), 시리즈 고정 색,
//            직접 라벨 + 수치 표기 (dataviz 방법론 준수)
// ═══════════════════════════════════════════════

import { MODES, LS_KEYS } from './config.js';
import { $, fmtMs, fmtNum, escapeHtml } from './ui.js';

const MODE_KEYS = ['basic', 'vector', 'graph'];

const CHART_DEFS = [
  { key: 'totalMs', title: '총 응답시간', sub: '낮을수록 빠름', fmt: fmtMs },
  { key: 'ttftMs', title: '첫 토큰 지연 (TTFT)', sub: '체감 반응 속도', fmt: fmtMs },
  { key: 'retrievalMs', title: '검색(Retrieval) 시간', sub: 'RAG 전처리 비용', fmt: fmtMs },
  { key: 'promptTokens', title: '입력(프롬프트) 토큰', sub: '컨텍스트 주입량', fmt: fmtNum },
  { key: 'evalTokens', title: '생성 토큰', sub: '응답 길이', fmt: fmtNum },
  { key: 'tps', title: '생성 속도', sub: 'tokens/sec · 높을수록 좋음', fmt: (v) => (v == null ? '–' : v.toFixed(1)) },
];

export function emptyRunMetrics() {
  const m = {};
  for (const k of MODE_KEYS) {
    m[k] = { retrievalMs: null, ttftMs: null, totalMs: null, promptTokens: null, evalTokens: null, tps: null, status: 'idle' };
  }
  return m;
}

export function computeTps(evalTokens, evalDurationMs) {
  if (!evalTokens || !evalDurationMs) return null;
  return evalTokens / (evalDurationMs / 1000);
}

/** 비교 차트 렌더 */
export function renderCharts(run) {
  const grid = $('#chartGrid');
  const section = $('#compareSection');
  const hasData = MODE_KEYS.some((k) => run[k].totalMs != null);
  section.hidden = !hasData;
  if (!hasData) return;

  grid.innerHTML = '';
  for (const def of CHART_DEFS) {
    const values = MODE_KEYS.map((k) => run[k][def.key]);
    if (values.every((v) => v == null)) continue;
    const max = Math.max(...values.filter((v) => v != null), 1e-9);

    const chart = document.createElement('div');
    chart.className = 'mini-chart';
    chart.innerHTML = `<h4>${def.title}<small>${def.sub}</small></h4>`;

    MODE_KEYS.forEach((k) => {
      const v = run[k][def.key];
      const row = document.createElement('div');
      row.className = 'bar-row' + (v == null ? ' na' : '');
      const pct = v == null ? 0 : Math.max(2, (v / max) * 100);
      row.innerHTML =
        `<span class="bar-label"><span class="bar-swatch" style="background:var(--c-${k})"></span>${MODES[k].name}</span>` +
        `<div class="bar-track"><div class="bar-fill" style="width:0%;background:var(--c-${k})"></div></div>` +
        `<span class="bar-value">${v == null ? 'N/A' : def.fmt(v)}</span>`;
      row.title = `${MODES[k].name} — ${def.title}: ${v == null ? '해당 없음' : def.fmt(v)}`;
      chart.appendChild(row);
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          row.querySelector('.bar-fill').style.width = pct + '%';
        })
      );
    });
    grid.appendChild(chart);
  }
}

// ── 실험 기록 ──────────────────────────────

export function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEYS.history) || '[]');
  } catch {
    return [];
  }
}

export function saveHistoryEntry(query, model, run) {
  const list = loadHistory();
  list.unshift({
    at: Date.now(),
    query,
    model,
    m: Object.fromEntries(
      MODE_KEYS.map((k) => [k, { totalMs: run[k].totalMs, evalTokens: run[k].evalTokens }])
    ),
  });
  const trimmed = list.slice(0, 30);
  localStorage.setItem(LS_KEYS.history, JSON.stringify(trimmed));
  return trimmed;
}

export function clearHistory() {
  localStorage.removeItem(LS_KEYS.history);
}

export function renderHistory() {
  const list = loadHistory();
  const section = $('#historySection');
  section.hidden = !list.length;
  $('#historyCount').textContent = list.length ? `${list.length}건` : '';
  const body = $('#historyBody');
  body.innerHTML = list
    .map((h) => {
      const t = new Date(h.at);
      const time = `${String(t.getMonth() + 1).padStart(2, '0')}/${String(t.getDate()).padStart(2, '0')} ${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;
      const cells = MODE_KEYS.map((k) => {
        const m = h.m?.[k] || {};
        return `<td class="num">${fmtMs(m.totalMs)}</td><td class="num">${fmtNum(m.evalTokens)}</td>`;
      }).join('');
      return `<tr><td class="num">${time}</td><td class="q-cell" title="${escapeHtml(h.query)}">${escapeHtml(h.query)}</td><td class="num">${escapeHtml(h.model || '')}</td>${cells}</tr>`;
    })
    .join('');
}
