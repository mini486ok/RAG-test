// ═══════════════════════════════════════════════
// RAIL·RAG LAB — 메인 애플리케이션 컨트롤러
// ═══════════════════════════════════════════════

import { DEFAULTS, MODES, SYSTEM_PROMPT_BASE, LS_KEYS, buildBasicMessages, buildVectorMessages, buildGraphMessages } from './config.js';
import { OllamaClient } from './ollama.js';
import { idb } from './db.js';
import { parseFile, cleanText, detectType } from './parser.js';
import { VectorStore } from './vectorstore.js';
import { GraphStore } from './graphstore.js';
import { VectorViz } from './viz-vector.js';
import { GraphViz } from './viz-graph.js';
import { emptyRunMetrics, computeTps, renderCharts, renderHistory, saveHistoryEntry, clearHistory } from './metrics.js';
import { $, $$, toast, makeStreamRenderer, fmtMs, fmtNum, fmtBytes, escapeHtml, uid } from './ui.js';

// ── 전역 상태 ──────────────────────────────
const settings = loadSettings();
applyServerParam(); // ?server= 공유 링크 지원 (settings 로드 직후)
const client = new OllamaClient(settings.ollamaUrl);
let authUser = null; // 원격 공유 모드 로그인 사용자

/** 공유 링크의 ?server=https://... 파라미터를 서버 주소로 적용 */
function applyServerParam() {
  try {
    const raw = new URLSearchParams(location.search).get('server');
    if (!raw) return;
    const u = new URL(raw);
    if (u.protocol === 'http:' || u.protocol === 'https:') {
      settings.ollamaUrl = u.origin + (u.pathname === '/' ? '' : u.pathname.replace(/\/+$/, ''));
      saveSettings();
    }
  } catch { /* 잘못된 파라미터는 무시 */ }
}

/** 로컬(자기 PC) Ollama가 아닌 원격 공유 서버인지 */
function isRemoteServer() {
  try {
    const host = new URL(settings.ollamaUrl).hostname;
    return host !== 'localhost' && host !== '127.0.0.1';
  } catch {
    return false;
  }
}
const vectorStore = new VectorStore();
const graphStore = new GraphStore();
let connected = false;
let models = [];
let docs = []; // {id, name, type, size, textLen, text, addedAt}
let running = false;
let building = false;
let runAbort = null;
let buildAbort = null;
let vizVectorDirty = true;
let vizGraphDirty = true;
let vectorViz = null;
let graphViz = null;
const panels = {};

function loadSettings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(LS_KEYS.settings) || '{}');
  } catch { /* 무시 */ }
  const base = typeof structuredClone === 'function'
    ? structuredClone(DEFAULTS)
    : JSON.parse(JSON.stringify(DEFAULTS));
  // 저장값 손상·변조 대비: 수치 필드는 로드 시에도 클램프
  const numOr = (v, min, max, def) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : def;
  };
  const s = {
    ...base,
    ...saved,
    vector: { ...base.vector, ...(saved.vector || {}) },
    graph: {
      ...base.graph,
      ...(saved.graph || {}),
      // DEFAULTS 원본 오염 방지: 배열은 반드시 새로 복사
      entityTypes: Array.isArray(saved.graph?.entityTypes)
        ? saved.graph.entityTypes.filter((t) => typeof t === 'string' && t.trim()).slice(0, 30)
        : [...base.graph.entityTypes],
    },
    seqMode: saved.seqMode !== false, // 기본: 순차 실행(공정 측정)
    sysPrompt: typeof saved.sysPrompt === 'string' && saved.sysPrompt.trim() ? saved.sysPrompt : SYSTEM_PROMPT_BASE,
  };
  s.temperature = numOr(s.temperature, 0, 2, base.temperature);
  s.numCtx = numOr(s.numCtx, 512, 131072, base.numCtx);
  s.numPredict = numOr(s.numPredict, 64, 8192, base.numPredict);
  s.vector.chunkSize = numOr(s.vector.chunkSize, 100, 4000, base.vector.chunkSize);
  s.vector.overlap = numOr(s.vector.overlap, 0, 1000, base.vector.overlap);
  s.vector.topK = numOr(s.vector.topK, 1, 20, base.vector.topK);
  s.vector.minScore = numOr(s.vector.minScore, 0, 1, base.vector.minScore);
  s.vector.mmrLambda = numOr(s.vector.mmrLambda, 0, 1, base.vector.mmrLambda);
  s.graph.maxTriples = numOr(s.graph.maxTriples, 3, 30, base.graph.maxTriples);
  s.graph.hops = numOr(s.graph.hops, 1, 3, base.graph.hops);
  s.graph.topKEntities = numOr(s.graph.topKEntities, 1, 20, base.graph.topKEntities);
  s.graph.maxCtxTriples = numOr(s.graph.maxCtxTriples, 5, 80, base.graph.maxCtxTriples);
  s.graph.chunkTopK = numOr(s.graph.chunkTopK, 0, 8, base.graph.chunkTopK);
  s.graph.gleaning = numOr(s.graph.gleaning, 0, 2, base.graph.gleaning);
  s.graph.extractChunkSize = numOr(s.graph.extractChunkSize, 500, 4000, base.graph.extractChunkSize);
  s.graphModel = typeof s.graphModel === 'string' ? s.graphModel : '';
  // v2 마이그레이션: 시연 최적 기본값 적용 (사용자가 바꾼 적 없는 옛 기본값만 교체)
  if ((saved.paramVersion || 1) < 2) {
    if (s.graph.maxTriples === 12) s.graph.maxTriples = 8;
    if (s.graph.extractChunkSize === 1200) s.graph.extractChunkSize = 1800;
    s.paramVersion = 2;
  }
  return s;
}

function saveSettings() {
  localStorage.setItem(LS_KEYS.settings, JSON.stringify(settings));
}

// ── 엔진 패널 ──────────────────────────────
class EnginePanel {
  constructor(mode) {
    const tpl = $('#enginePanelTpl');
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.mode = mode;
    node.querySelector('.engine-name').textContent = MODES[mode].name;
    node.querySelector('.engine-desc').textContent = MODES[mode].desc;
    $('#engineGrid').appendChild(node);
    this.mode = mode;
    this.root = node;
    this.signal = node.querySelector('.signal-box');
    this.stage = node.querySelector('.engine-stage');
    this.budget = node.querySelector('.ctx-budget');
    this.answer = node.querySelector('.answer');
    this.empty = node.querySelector('.engine-empty');
    this.ctxDetails = node.querySelector('.ctx-details');
    this.ctxCount = node.querySelector('.ctx-count');
    this.ctxList = node.querySelector('.ctx-list');
    this.reset();
  }

  reset() {
    this.setSignal('idle');
    this.setStage('STANDBY · 대기');
    this.budget.hidden = true;
    this.budget.innerHTML = '';
    this.answer.innerHTML = '';
    this.answer.classList.remove('streaming');
    this.empty.innerHTML = '';
    this.ctxDetails.hidden = true;
    this.ctxDetails.open = false;
    this.ctxList.innerHTML = '';
    this.ctxCount.textContent = '';
    for (const el of this.root.querySelectorAll('.m-value')) el.textContent = '–';
  }

  setSignal(state) {
    this.signal.dataset.state = state;
  }

  setStage(text) {
    this.stage.textContent = text;
  }

  setEmpty(html) {
    this.empty.innerHTML = html || '';
  }

  setMetric(key, text) {
    const el = this.root.querySelector(`.m-value[data-m="${key}"]`);
    if (!el) return;
    el.textContent = text;
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }

  /** RAG 컨텍스트 사용률 게이지: 사전 추정치 표시 */
  setBudget(estTokens, budgetTokens, trimmed) {
    const pct = Math.min(100, Math.round((estTokens / budgetTokens) * 100));
    const warn = trimmed > 0
      ? `<span class="b-warn">⚠ 예산 초과로 ${trimmed}건 절삭됨</span>`
      : '';
    this.budget.innerHTML =
      `<span class="b-label">주입 컨텍스트</span>` +
      `<span class="b-bar"><span class="b-fill${pct >= 90 ? ' hot' : ''}" style="width:${pct}%"></span></span>` +
      `<span class="b-text">≈${fmtNum(estTokens)} / ${fmtNum(budgetTokens)} tok (${pct}%)</span>${warn}`;
    this.budget.hidden = false;
    this._budgetShown = true;
  }

  /** 생성 완료 후 Ollama 실측값(prompt_eval_count)으로 게이지 갱신 */
  updateBudgetActual(promptTokens, numCtx) {
    if (!this._budgetShown || promptTokens == null) return;
    const pct = Math.min(100, Math.round((promptTokens / numCtx) * 100));
    const text = this.budget.querySelector('.b-text');
    if (text) text.textContent = `실측 ${fmtNum(promptTokens)} tok · num_ctx ${fmtNum(numCtx)}의 ${pct}%`;
    const fill = this.budget.querySelector('.b-fill');
    if (fill) {
      fill.style.width = pct + '%';
      fill.classList.toggle('hot', pct >= 90);
    }
  }

  applyMetrics(m) {
    this.setMetric('retrieval', fmtMs(m.retrievalMs));
    this.setMetric('ttft', fmtMs(m.ttftMs));
    this.setMetric('total', fmtMs(m.totalMs));
    this.setMetric('ptok', fmtNum(m.promptTokens));
    this.setMetric('etok', fmtNum(m.evalTokens));
    this.setMetric('tps', m.tps == null ? '–' : m.tps.toFixed(1));
  }

  showChunks(items) {
    if (!items.length) return;
    this.ctxDetails.hidden = false;
    this.ctxCount.textContent = `청크 ${items.length}건`;
    this.ctxList.innerHTML = items
      .map(
        (c, i) =>
          `<div class="ctx-item" title="클릭하여 전체 보기">
            <div class="ctx-item-head"><b>근거 ${i + 1}</b><span>${escapeHtml(c.docName)} · 유사도 ${c.score.toFixed(3)}</span></div>
            <div class="ctx-item-text">${escapeHtml(c.text)}</div>
          </div>`
      )
      .join('');
    bindCtxItemToggles(this.ctxList);
  }

  showGraphCtx(result) {
    const total = result.triples.length + result.ctxChunks.length;
    if (!total) return;
    this.ctxDetails.hidden = false;
    this.ctxCount.textContent = `트리플 ${result.triples.length} · 청크 ${result.ctxChunks.length}`;
    const seedHtml = result.seeds.length
      ? `<div class="ctx-item"><div class="ctx-item-head"><b>시드 개체</b></div><div class="ctx-item-text">${result.seeds
          .map((s) => `${escapeHtml(s.name)} (${s.score.toFixed(2)})`)
          .join(' · ')}</div></div>`
      : '';
    const tripleHtml = result.triples.length
      ? `<div class="ctx-item"><div class="ctx-item-head"><b>지식그래프 트리플</b></div><div class="ctx-triples">${result.triples
          .slice(0, 20)
          .map(
            (t) =>
              `<div class="triple"><span class="t-ent">${escapeHtml(t.source)}</span><span class="t-rel">${escapeHtml(t.relation)}</span><span class="t-ent">${escapeHtml(t.target)}</span></div>`
          )
          .join('')}${result.triples.length > 20 ? `<div class="ctx-item-text">외 ${result.triples.length - 20}건…</div>` : ''}</div></div>`
      : '';
    const chunkHtml = result.ctxChunks
      .map(
        (c, i) =>
          `<div class="ctx-item"><div class="ctx-item-head"><b>원문 ${i + 1}</b><span>${escapeHtml(c.docName)}</span></div><div class="ctx-item-text">${escapeHtml(c.text)}</div></div>`
      )
      .join('');
    this.ctxList.innerHTML = seedHtml + tripleHtml + chunkHtml;
    bindCtxItemToggles(this.ctxList);
  }
}

/** 참조 컨텍스트 아이템: 클릭/Enter/Space로 펼침 (키보드 접근성 + 상태 알림) */
function bindCtxItemToggles(listEl) {
  const items = listEl.querySelectorAll('.ctx-item');
  items.forEach((el, i) => {
    el.tabIndex = 0;
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', `근거 ${i + 1} 펼치기`);
    el.setAttribute('aria-expanded', 'false');
    const toggle = () => {
      const expanded = el.classList.toggle('expanded');
      el.setAttribute('aria-expanded', String(expanded));
      el.setAttribute('aria-label', `근거 ${i + 1} ${expanded ? '접기' : '펼치기'}`);
    };
    el.addEventListener('click', toggle);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggle();
      }
    });
  });
}

// ── 초기화 ──────────────────────────────
async function init() {
  initTheme();
  initTabs();
  initPanels();
  initQueryConsole();
  initDrawer();
  initAuth();
  initBuildTab();
  initVizTab();

  await Promise.all([loadDocs(), vectorStore.load(), graphStore.load()]);
  refreshDocList();
  refreshDbStats();
  renderHistory();

  await connect();
}

function initTheme() {
  const saved = localStorage.getItem(LS_KEYS.theme);
  if (saved) document.body.dataset.theme = saved;
  $('#themeToggle').addEventListener('click', () => {
    const next = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    document.body.dataset.theme = next;
    localStorage.setItem(LS_KEYS.theme, next);
    vectorViz?.scheduleDraw?.();
    graphViz?.draw?.();
  });
}

function initTabs() {
  const tabs = $$('.tab-btn');
  for (const btn of tabs) {
    btn.addEventListener('click', () => {
      for (const b of tabs) {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', String(b === btn));
      }
      for (const p of $$('.tab-panel')) p.classList.toggle('active', p.id === `tab-${btn.dataset.tab}`);
      if (btn.dataset.tab === 'viz') refreshViz();
    });
    // 좌우 화살표로 탭 이동 (WAI-ARIA 탭 패턴)
    btn.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const idx = tabs.indexOf(btn);
      const next = tabs[(idx + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      next.click();
    });
  }

  // 파라미터 도움말(?)을 키보드/스크린리더에서도 읽을 수 있게
  for (const h of $$('.hint')) {
    h.tabIndex = 0;
    h.setAttribute('role', 'button');
    h.setAttribute('aria-label', h.dataset.tip || '도움말');
  }
}

function initPanels() {
  for (const mode of Object.keys(MODES)) panels[mode] = new EnginePanel(mode);
  setIdleEmptyStates();
}

function setIdleEmptyStates() {
  panels.basic.setEmpty('LLM 자체 지식만으로 답변합니다.<br>질문을 입력하고 실행을 눌러주세요.');
  panels.vector.setEmpty(
    vectorStore.size
      ? `Vector DB 준비 완료 (${vectorStore.size} 청크)<br>질문을 입력하고 실행을 눌러주세요.`
      : 'Vector DB가 비어 있습니다.<br><button class="btn-ghost btn-sm goto-build">문서 · DB 구축으로 이동</button>'
  );
  panels.graph.setEmpty(
    graphStore.nodeCount
      ? `Graph DB 준비 완료 (${graphStore.nodeCount} 노드 · ${graphStore.edgeCount} 관계)<br>질문을 입력하고 실행을 눌러주세요.`
      : 'Graph DB가 비어 있습니다.<br><button class="btn-ghost btn-sm goto-build">문서 · DB 구축으로 이동</button>'
  );
  for (const btn of $$('.goto-build')) {
    btn.addEventListener('click', () => $('.tab-btn[data-tab="build"]').click());
  }
}

// ── 연결 ──────────────────────────────
/** 연결 시도 본체 — 성공 시 UI 갱신, 실패 시 throw (호출자가 UI 처리) */
async function connectRaw() {
  const box = $('#connBox');
  box.className = 'conn-box';
  $('#connText').textContent = '연결 확인 중…';
  client.setBaseUrl(settings.ollamaUrl);
  models = await client.listModels();
  if (!models.length) {
    const err = new Error('설치된 모델이 없습니다.');
    err.noModels = true;
    throw err;
  }
  connected = true;
  box.classList.add('ok');
  const who = authUser ? ` · ${authUser}` : '';
  $('#connText').textContent = `${isRemoteServer() ? '원격 서버' : 'Ollama'} 연결됨 · 모델 ${models.length}개${who}`;
  populateModelSelects();
  $('#drawerInfo').textContent = `서버: ${settings.ollamaUrl}\n모델 ${models.length}개 감지${authUser ? `\n로그인: ${authUser}` : ''}`;
}

async function connect() {
  try {
    await connectRaw();
  } catch (e) {
    connected = false;
    const box = $('#connBox');
    box.className = 'conn-box fail';
    if (e.status === 401) {
      $('#connText').textContent = '로그인 필요 — 클릭하여 로그인';
      showLoginModal(authUser ? '저장된 로그인 정보가 만료되었거나 잘못되었습니다. 다시 로그인하세요.' : undefined);
    } else if (e.noModels) {
      $('#connText').textContent = 'Ollama에 설치된 모델이 없습니다';
      toast('설치된 모델이 없습니다. 터미널에서 `ollama pull exaone3.5:7.8b`와 `ollama pull bge-m3`를 실행하세요.', 'err', 9000);
    } else if (isRemoteServer()) {
      $('#connText').textContent = '원격 서버 연결 실패 — 클릭하여 재시도';
      toast(`원격 서버(${settings.ollamaUrl})에 연결할 수 없습니다. 서버 운영자에게 문의하세요.`, 'err', 7000);
    } else {
      $('#connText').textContent = 'Ollama 연결 실패 — 클릭하여 안내 보기';
      showCorsModal();
    }
    console.warn('연결 실패:', e);
  }
}

function populateModelSelects() {
  const chatSel = $('#chatModel');
  const embedSel = $('#embedModel');
  const graphSel = $('#graphModel');
  const opts = models.map((m) => `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)}</option>`).join('');
  chatSel.innerHTML = opts;
  embedSel.innerHTML = opts;
  graphSel.innerHTML = `<option value="">(응답 모델과 동일)</option>` + opts;
  graphSel.value = models.some((m) => m.name === settings.graphModel) ? settings.graphModel : '';

  // 기본값 매칭 (정확 일치 → 접두 일치)
  const pick = (want, fallbackHint) => {
    const names = models.map((m) => m.name);
    if (names.includes(want)) return want;
    const pre = names.find((n) => n.startsWith(want.split(':')[0]));
    if (pre) return pre;
    const hint = fallbackHint && names.find((n) => n.includes(fallbackHint));
    return hint || names[0] || '';
  };
  chatSel.value = pick(settings.chatModel, 'exaone');
  embedSel.value = pick(settings.embedModel, 'bge');
  settings.chatModel = chatSel.value;
  settings.embedModel = embedSel.value;
  saveSettings();

  // 리스너는 재연결 시 중복 등록되지 않도록 1회만 바인딩
  if (!chatSel.dataset.bound) {
    chatSel.dataset.bound = '1';
    chatSel.addEventListener('change', () => {
      settings.chatModel = chatSel.value;
      saveSettings();
      toast(`응답 모델: ${chatSel.value}`, 'info', 2000);
    });
    embedSel.addEventListener('change', () => {
      settings.embedModel = embedSel.value;
      saveSettings();
      if (vectorStore.size && vectorStore.embedModel && vectorStore.embedModel !== embedSel.value) {
        toast('임베딩 모델이 변경되었습니다. Vector/Graph DB를 [전체 재구축]해야 검색이 동작합니다.', 'warn', 6000);
      }
    });
    graphSel.addEventListener('change', () => {
      settings.graphModel = graphSel.value;
      saveSettings();
      toast(`Graph 추출 모델: ${graphSel.value || '(응답 모델과 동일)'}`, 'info', 2500);
    });
  }
}

// ── 질의 콘솔 ──────────────────────────────
function initQueryConsole() {
  $('#queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      runQuery();
    }
  });
  $('#seqMode').checked = settings.seqMode;
  $('#seqMode').addEventListener('change', (e) => {
    settings.seqMode = e.target.checked;
    saveSettings();
    toast(e.target.checked
      ? '순차 실행: 세 방식을 하나씩 실행해 공정하게 측정합니다.'
      : '동시 실행: 빠르지만 GPU 경합으로 시간 지표가 서로 간섭될 수 있습니다.', 'info', 4000);
  });
  $('#btnRun').addEventListener('click', runQuery);
  $('#btnStop').addEventListener('click', () => runAbort?.abort());
  $('#btnClearHistory').addEventListener('click', () => {
    clearHistory();
    renderHistory();
  });

  $('#connBox').addEventListener('click', () => {
    if (!connected) showCorsModal();
  });
  $('#connBox').addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !connected) {
      e.preventDefault();
      showCorsModal();
    }
  });
  $('#btnRetryConn').addEventListener('click', async () => {
    hideCorsModal();
    await connect();
    if (connected) toast('Ollama에 연결되었습니다.', 'ok');
  });
  $('#btnCloseCors').addEventListener('click', hideCorsModal);
}

async function runQuery() {
  const query = $('#queryInput').value.trim();
  if (!query) {
    toast('질문을 입력해 주세요.', 'warn');
    return;
  }
  if (running) return;
  if (building) {
    toast('DB 구축이 진행 중입니다. 완료 후 질의를 실행하세요.', 'warn');
    return;
  }
  if (!connected) {
    showCorsModal();
    return;
  }

  running = true;
  $('#btnRun').disabled = true;
  $('#btnStop').disabled = false;
  runAbort = new AbortController();
  const signal = runAbort.signal;
  const run = emptyRunMetrics();

  try {
    for (const p of Object.values(panels)) p.reset();
    $('#compareSection').hidden = true;

    const genOptions = {
      temperature: settings.temperature,
      num_ctx: settings.numCtx,
      num_predict: settings.numPredict,
    };

    // 질의 임베딩은 1회만 수행해 Vector/Graph가 공유 —
    // 중복 비용 제거 + 임베딩 모델 콜드로드가 한 모드에만 전가되는 비대칭 방지
    const shared = { qVec: null, embedMs: 0, embedError: null };
    if (vectorStore.size || graphStore.nodeCount) {
      try {
        const tE = performance.now();
        [shared.qVec] = await client.embed({ model: settings.embedModel, input: [query], signal });
        shared.embedMs = performance.now() - tE;
      } catch (e) {
        shared.embedError = e;
      }
    }

    const runners = [
      () => runBasic(query, run, genOptions, signal),
      () => runVector(query, run, genOptions, signal, shared),
      () => runGraph(query, run, genOptions, signal, shared),
    ];

    if (settings.seqMode) {
      // 순차 실행: GPU 경합·모델 스왑 없이 공정한 지표 측정
      for (const r of runners) {
        if (signal.aborted) break;
        await r();
      }
    } else {
      await Promise.allSettled(runners.map((r) => r()));
    }

    renderCharts(run);
  } finally {
    running = false;
    $('#btnRun').disabled = false;
    $('#btnStop').disabled = true;
    runAbort = null;
  }

  try {
    saveHistoryEntry(query, settings.chatModel, run);
    renderHistory();
  } catch (e) {
    console.warn('기록 저장 실패:', e);
  }
}

async function streamAnswer(panel, messages, run, genOptions, signal) {
  const m = run[panel.mode];
  panel.setStage('GENERATING · 응답 생성 중');
  panel.answer.classList.add('streaming');
  const renderer = makeStreamRenderer(panel.answer);

  const res = await client.chatStream({
    model: settings.chatModel,
    messages,
    options: genOptions,
    signal,
    onToken: (_piece, full) => renderer.update(full),
  });

  renderer.finish(res.text);
  panel.answer.classList.remove('streaming');

  // TTFT·총시간 모두 모델 로드 시간을 제외해 웜/콜드 조건 차이를 보정
  // (순차 실행 시 첫 모드만 콜드로드를 뒤집어쓰는 편향 방지)
  m.ttftMs = res.ttftMs != null ? Math.max(0, res.ttftMs - (res.loadDurationMs || 0)) : null;
  m.totalMs = (m.retrievalMs || 0) + Math.max(0, res.totalMs - (res.loadDurationMs || 0));
  m.promptTokens = res.promptTokens;
  m.evalTokens = res.evalTokens;
  m.tps = computeTps(res.evalTokens, res.evalDurationMs);
  panel.applyMetrics(m);

  // 컨텍스트 게이지를 실측값(prompt_eval_count)으로 갱신
  panel.updateBudgetActual(res.promptTokens, settings.numCtx);

  if (res.aborted) {
    panel.setStage('ABORTED · 사용자 중단');
    panel.setSignal('idle');
  } else if (res.incomplete) {
    panel.setStage('INCOMPLETE · 응답이 완결되지 않음');
    panel.setSignal('error');
    toast(`${MODES[panel.mode].name}: 스트림이 완결 신호 없이 종료되어 응답이 잘렸을 수 있습니다.`, 'warn');
  } else {
    panel.setStage('COMPLETE · 완료');
    panel.setSignal('done');
  }
}

/**
 * 컨텍스트가 num_ctx 예산을 넘지 않도록 항목을 뒤에서부터 줄임.
 * 한국어 기준 보수적으로 1토큰 ≈ 1.6자 가정.
 * 첫 항목조차 예산을 넘으면 truncate 콜백으로 잘라서라도 포함.
 */
function fitContextBudget(items, itemChars, baseChars, truncate) {
  const budgetTokens = settings.numCtx - settings.numPredict - 256;
  const budgetChars = Math.max(800, budgetTokens * 1.6);
  let total = baseChars;
  const kept = [];
  for (const it of items) {
    const len = itemChars(it);
    if (total + len > budgetChars) break;
    total += len;
    kept.push(it);
  }
  if (!kept.length && items.length && truncate) {
    const maxChars = Math.max(200, Math.floor(budgetChars - baseChars - 80));
    kept.push(truncate(items[0], maxChars));
    return { kept, trimmed: items.length - 1, usedChars: baseChars + maxChars, budgetTokens };
  }
  return { kept, trimmed: items.length - kept.length, usedChars: total, budgetTokens };
}

function handleRunError(panel, e) {
  if (e.name === 'AbortError') {
    panel.setStage('ABORTED · 사용자 중단');
    panel.setSignal('idle');
    return;
  }
  panel.setSignal('error');
  panel.setStage('ERROR');
  panel.setEmpty(`⚠ 오류: ${escapeHtml(e.message || String(e))}`);
  if (e.status === 401) showLoginModal('세션이 만료되었습니다. 다시 로그인하세요.');
  console.error(`[${panel.mode}]`, e);
}

async function runBasic(query, run, genOptions, signal) {
  const panel = panels.basic;
  try {
    panel.setSignal('work');
    panel.setEmpty('');
    scrollPanelIntoView(panel);
    await streamAnswer(panel, buildBasicMessages(settings.sysPrompt, query), run, genOptions, signal);
  } catch (e) {
    handleRunError(panel, e);
  }
}

async function runVector(query, run, genOptions, signal, shared) {
  const panel = panels.vector;
  const m = run.vector;
  if (!vectorStore.size) {
    panel.setStage('NO DATABASE · DB 없음');
    panel.setEmpty('Vector DB가 비어 있어 실행하지 않았습니다.<br><button class="btn-ghost btn-sm goto-build">문서 · DB 구축으로 이동</button>');
    panel.empty.querySelector('.goto-build')?.addEventListener('click', () => $('.tab-btn[data-tab="build"]').click());
    return;
  }
  try {
    panel.setSignal('work');
    panel.setEmpty('');
    scrollPanelIntoView(panel);
    panel.setStage('RETRIEVING · 유사 청크 검색 중');
    if (shared.embedError) throw shared.embedError;
    const t0 = performance.now();
    const results = vectorStore.search(shared.qVec, settings.vector);
    m.retrievalMs = shared.embedMs + (performance.now() - t0);
    panel.setMetric('retrieval', fmtMs(m.retrievalMs));

    if (!results.length) {
      panel.setStage('NO MATCH · 관련 청크 없음');
      toast('Vector: 유사도 임계값을 넘는 청크가 없습니다. 임계값을 낮추거나 문서를 추가하세요.', 'warn');
    }
    let ctxChunks = results.map((r) => ({ ...r.chunk, docName: r.chunk.docName, text: r.chunk.text, score: r.score }));
    const fit = fitContextBudget(
      ctxChunks,
      (c) => c.text.length + 60,
      query.length + settings.sysPrompt.length + 400,
      (c, n) => ({ ...c, text: c.text.slice(0, n) })
    );
    if (fit.trimmed > 0) {
      ctxChunks = fit.kept;
      toast(`Vector: 컨텍스트 예산 초과로 청크 ${fit.trimmed}건을 제외했습니다. (num_ctx 상향 가능)`, 'info');
    }
    // 컨텍스트 수용성 게이지 (사전 추정 → 생성 후 실측으로 갱신)
    panel.setBudget(Math.round(fit.usedChars / 1.6), fit.budgetTokens, fit.trimmed);
    // 근거 패널에는 실제로 LLM에 주입되는 청크만 표시 (표시-주입 일치)
    panel.showChunks(ctxChunks);
    await streamAnswer(panel, buildVectorMessages(settings.sysPrompt, query, ctxChunks), run, genOptions, signal);
  } catch (e) {
    handleRunError(panel, e);
  }
}

async function runGraph(query, run, genOptions, signal, shared) {
  const panel = panels.graph;
  const m = run.graph;
  if (!graphStore.nodeCount) {
    panel.setStage('NO DATABASE · DB 없음');
    panel.setEmpty('Graph DB가 비어 있어 실행하지 않았습니다.<br><button class="btn-ghost btn-sm goto-build">문서 · DB 구축으로 이동</button>');
    panel.empty.querySelector('.goto-build')?.addEventListener('click', () => $('.tab-btn[data-tab="build"]').click());
    return;
  }
  try {
    panel.setSignal('work');
    panel.setEmpty('');
    scrollPanelIntoView(panel);
    panel.setStage('TRAVERSING · 지식그래프 탐색 중');
    if (shared.embedError) throw shared.embedError;
    const t0 = performance.now();
    const result = graphStore.search(query, shared.qVec, settings.graph);
    m.retrievalMs = shared.embedMs + (performance.now() - t0);
    panel.setMetric('retrieval', fmtMs(m.retrievalMs));

    if (!result.triples.length && !result.ctxChunks.length) {
      panel.setStage('NO MATCH · 관련 개체 없음');
      toast('Graph: 질의와 연결되는 개체를 찾지 못했습니다. 시드 개체 수를 늘리거나 문서를 추가하세요.', 'warn');
    }
    // 컨텍스트 예산: 트리플 → 원문 청크 순으로 예산을 배분해 초과분 절삭
    const baseChars = query.length + settings.sysPrompt.length + 400;
    const tripleLen = (t) => t.source.length + t.target.length + t.relation.length + (t.desc?.length || 0) + 20;
    const fitT = fitContextBudget(result.triples, tripleLen, baseChars);
    const triples = fitT.kept;
    const fitC = fitContextBudget(
      result.ctxChunks,
      (c) => c.text.length + 60,
      fitT.usedChars,
      (c, n) => ({ ...c, text: c.text.slice(0, n) })
    );
    const ctxChunks = fitC.kept;
    if (fitT.trimmed > 0 || fitC.trimmed > 0) {
      toast(`Graph: 컨텍스트 예산 초과로 트리플 ${fitT.trimmed}건·원문 청크 ${fitC.trimmed}건을 제외했습니다.`, 'info');
    }
    // 컨텍스트 수용성 게이지 (사전 추정 → 생성 후 실측으로 갱신)
    panel.setBudget(Math.round(fitC.usedChars / 1.6), fitC.budgetTokens, fitT.trimmed + fitC.trimmed);
    // 근거 패널에는 실제 주입분만 표시 (표시-주입 일치)
    panel.showGraphCtx({ seeds: result.seeds, triples, ctxChunks });
    await streamAnswer(panel, buildGraphMessages(settings.sysPrompt, query, triples, ctxChunks), run, genOptions, signal);
  } catch (e) {
    handleRunError(panel, e);
  }
}

/** 순차 실행 시 좁은 화면에서 현재 실행 중 패널이 보이도록 스크롤 */
function scrollPanelIntoView(panel) {
  if (settings.seqMode && window.innerWidth <= 1180) {
    panel.root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ── 오버레이 접근성 헬퍼 (포커스 이동/복원 + Tab 트랩 + ESC) ──────
// 드로어·모달이 중첩될 수 있으므로 복원 지점을 오버레이별로 분리 저장
let drawerLastFocus = null;
let modalLastFocus = null;

function trapFocus(container, e) {
  const focusables = [...container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )].filter((el) => !el.disabled && el.offsetParent !== null);
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

function showCorsModal() {
  // 원격 공유 서버 모드에서는 CORS 안내 대신 로그인 모달이 관문
  if (isRemoteServer()) {
    showLoginModal();
    return;
  }
  const modal = $('#corsModal');
  if (!modal.hidden) return;
  modalLastFocus = document.activeElement;
  modal.hidden = false;
  $('#btnRetryConn').focus();
}

function hideCorsModal() {
  $('#corsModal').hidden = true;
  modalLastFocus?.focus?.();
}

// ── 로그인 (원격 공유 서버) ──────────────────────────────
function loadStoredAuth() {
  try {
    const raw = JSON.parse(localStorage.getItem(LS_KEYS.auth) || 'null');
    if (raw?.u && raw?.p) {
      client.setAuth(raw.u, atob(raw.p));
      authUser = raw.u;
    }
  } catch { /* 무시 */ }
}

function storeAuth(user, pw) {
  localStorage.setItem(LS_KEYS.auth, JSON.stringify({ u: user, p: btoa(pw) }));
}

function clearAuth() {
  localStorage.removeItem(LS_KEYS.auth);
  client.clearAuth();
  authUser = null;
  refreshAuthUI();
}

function refreshAuthUI() {
  const box = $('#authStatus');
  box.hidden = !authUser;
  if (authUser) $('#authUserText').innerHTML = `<b>●</b> <span>${escapeHtml(authUser)}</span> 로그인됨`;
}

function showLoginModal(message) {
  const modal = $('#loginModal');
  $('#loginServerInfo').textContent = `서버: ${settings.ollamaUrl}`;
  const err = $('#loginError');
  err.hidden = !message;
  if (message) err.textContent = message;
  if (!modal.hidden) return;
  modalLastFocus = document.activeElement;
  modal.hidden = false;
  $('#loginId').focus();
}

function hideLoginModal() {
  $('#loginModal').hidden = true;
  modalLastFocus?.focus?.();
}

function initAuth() {
  loadStoredAuth();
  refreshAuthUI();

  $('#loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = $('#loginId').value.trim();
    const pw = $('#loginPw').value;
    if (!id || !pw) return;
    $('#btnLogin').disabled = true;
    client.setAuth(id, pw);
    try {
      await connectRaw();
      authUser = id;
      storeAuth(id, pw);
      refreshAuthUI();
      hideLoginModal();
      $('#loginPw').value = '';
      toast(`${id}님, 로그인되었습니다.`, 'ok');
    } catch (err2) {
      client.clearAuth();
      showLoginModal(err2.status === 401 ? '아이디 또는 비밀번호가 올바르지 않습니다.' : `연결 실패: ${err2.message}`);
    } finally {
      $('#btnLogin').disabled = false;
    }
  });
  $('#btnCloseLogin').addEventListener('click', hideLoginModal);
  $('#btnLogout').addEventListener('click', async () => {
    clearAuth();
    connected = false;
    toast('로그아웃되었습니다.', 'info');
    // 서버가 인증을 요구하면 connect()가 401을 받아 로그인 모달을 자동 표시
    await connect();
  });
}

// ── 설정 드로어 ──────────────────────────────
function initDrawer() {
  const drawer = $('#drawer');
  const backdrop = $('#drawerBackdrop');
  const openDrawer = () => {
    drawerLastFocus = document.activeElement;
    drawer.hidden = false;
    backdrop.hidden = false;
    requestAnimationFrame(() => {
      drawer.classList.add('open');
      $('#ollamaUrl').focus();
    });
  };
  const closeDrawer = () => {
    drawer.classList.remove('open');
    backdrop.hidden = true;
    setTimeout(() => (drawer.hidden = true), 300);
    drawerLastFocus?.focus?.();
  };
  $('#btnSettings').addEventListener('click', openDrawer);
  $('#btnCloseDrawer').addEventListener('click', closeDrawer);
  backdrop.addEventListener('click', closeDrawer);

  // ESC로 오버레이 닫기 + Tab 포커스 트랩
  document.addEventListener('keydown', (e) => {
    const modal = $('#corsModal');
    const login = $('#loginModal');
    if (e.key === 'Escape') {
      if (!login.hidden) hideLoginModal();
      else if (!modal.hidden) hideCorsModal();
      else if (drawer.classList.contains('open')) closeDrawer();
      return;
    }
    if (e.key === 'Tab') {
      if (!login.hidden) trapFocus(login, e);
      else if (!modal.hidden) trapFocus(modal, e);
      else if (drawer.classList.contains('open')) trapFocus(drawer, e);
    }
  });

  $('#ollamaUrl').value = settings.ollamaUrl;
  $('#temperature').value = settings.temperature;
  $('#numCtx').value = settings.numCtx;
  $('#numPredict').value = settings.numPredict;
  $('#sysPrompt').value = settings.sysPrompt;

  $('#btnReconnect').addEventListener('click', async () => {
    settings.ollamaUrl = $('#ollamaUrl').value.trim() || DEFAULTS.ollamaUrl;
    saveSettings();
    await connect();
    if (connected) toast('연결되었습니다.', 'ok');
  });
  $('#temperature').addEventListener('change', (e) => {
    settings.temperature = clampNum(e.target, 0, 2, DEFAULTS.temperature);
    saveSettings();
  });
  $('#numCtx').addEventListener('change', (e) => {
    settings.numCtx = clampNum(e.target, 512, 131072, DEFAULTS.numCtx);
    saveSettings();
  });
  $('#numPredict').addEventListener('change', (e) => {
    settings.numPredict = clampNum(e.target, 64, 8192, DEFAULTS.numPredict);
    saveSettings();
  });
  $('#sysPrompt').addEventListener('change', (e) => {
    settings.sysPrompt = e.target.value.trim() || SYSTEM_PROMPT_BASE;
    saveSettings();
  });
}

function clampNum(input, min, max, fallback) {
  let v = parseFloat(input.value);
  if (Number.isNaN(v)) v = fallback;
  v = Math.min(max, Math.max(min, v));
  input.value = v;
  return v;
}

// ── 문서 · DB 구축 탭 ──────────────────────────────
async function loadDocs() {
  docs = await idb.getAll('docs');
  docs.sort((a, b) => a.addedAt - b.addedAt);
}

function initBuildTab() {
  const dz = $('#dropzone');
  const fi = $('#fileInput');
  dz.addEventListener('click', () => fi.click());
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fi.click();
  });
  dz.addEventListener('dragover', (e) => {
    e.preventDefault();
    dz.classList.add('dragover');
  });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dragover');
    handleFiles([...e.dataTransfer.files]);
  });
  fi.addEventListener('change', () => {
    handleFiles([...fi.files]);
    fi.value = '';
  });

  $('#btnLoadSample').addEventListener('click', loadSampleDocs);
  $('#btnClearAll').addEventListener('click', clearAllData);

  // 파라미터 입력 초기화 + 바인딩
  bindVectorParams();
  bindGraphParams();

  $('#btnBuildVector').addEventListener('click', () => buildVector(false));
  $('#btnBuildGraph').addEventListener('click', () => buildGraph(false));
  $('#btnBuildAll').addEventListener('click', async () => {
    if (await buildVector(false)) await buildGraph(false);
  });
  $('#btnRebuildAll').addEventListener('click', async () => {
    if (!confirm('기존 Vector/Graph DB를 비우고 모든 문서를 처음부터 다시 구축합니다. 계속할까요?')) return;
    if (await buildVector(true)) await buildGraph(true);
  });
  $('#btnCancelBuild').addEventListener('click', () => buildAbort?.abort());
}

function bindVectorParams() {
  const v = settings.vector;
  $('#vChunkSize').value = v.chunkSize;
  $('#vOverlap').value = v.overlap;
  $('#vTopK').value = v.topK;
  $('#vMinScore').value = v.minScore;
  $('#vMmr').checked = v.mmr;
  $('#vMmrLambda').value = v.mmrLambda;

  $('#vChunkSize').addEventListener('change', (e) => { v.chunkSize = clampNum(e.target, 100, 4000, DEFAULTS.vector.chunkSize); saveSettings(); });
  $('#vOverlap').addEventListener('change', (e) => { v.overlap = clampNum(e.target, 0, 1000, DEFAULTS.vector.overlap); saveSettings(); });
  $('#vTopK').addEventListener('change', (e) => { v.topK = clampNum(e.target, 1, 20, DEFAULTS.vector.topK); saveSettings(); });
  $('#vMinScore').addEventListener('change', (e) => { v.minScore = clampNum(e.target, 0, 1, DEFAULTS.vector.minScore); saveSettings(); });
  $('#vMmr').addEventListener('change', (e) => { v.mmr = e.target.checked; saveSettings(); });
  $('#vMmrLambda').addEventListener('change', (e) => { v.mmrLambda = clampNum(e.target, 0, 1, DEFAULTS.vector.mmrLambda); saveSettings(); });
}

function bindGraphParams() {
  const g = settings.graph;
  $('#gMaxTriples').value = g.maxTriples;
  $('#gHops').value = g.hops;
  // (추출 청크 크기는 아래에서 별도 바인딩)
  $('#gTopKEntities').value = g.topKEntities;
  $('#gMaxCtxTriples').value = g.maxCtxTriples;
  $('#gChunkTopK').value = g.chunkTopK;
  $('#gGleaning').value = g.gleaning;

  $('#gMaxTriples').addEventListener('change', (e) => { g.maxTriples = clampNum(e.target, 3, 30, DEFAULTS.graph.maxTriples); saveSettings(); });
  $('#gHops').addEventListener('change', (e) => { g.hops = clampNum(e.target, 1, 3, DEFAULTS.graph.hops); saveSettings(); });
  $('#gTopKEntities').addEventListener('change', (e) => { g.topKEntities = clampNum(e.target, 1, 20, DEFAULTS.graph.topKEntities); saveSettings(); });
  $('#gMaxCtxTriples').addEventListener('change', (e) => { g.maxCtxTriples = clampNum(e.target, 5, 80, DEFAULTS.graph.maxCtxTriples); saveSettings(); });
  $('#gChunkTopK').addEventListener('change', (e) => { g.chunkTopK = clampNum(e.target, 0, 8, DEFAULTS.graph.chunkTopK); saveSettings(); });
  $('#gGleaning').addEventListener('change', (e) => { g.gleaning = clampNum(e.target, 0, 2, DEFAULTS.graph.gleaning); saveSettings(); });
  $('#gExtractChunk').value = g.extractChunkSize;
  $('#gExtractChunk').addEventListener('change', (e) => { g.extractChunkSize = clampNum(e.target, 500, 4000, DEFAULTS.graph.extractChunkSize); saveSettings(); });

  renderEntityTags();
}

function renderEntityTags() {
  const box = $('#gEntityTypes');
  box.innerHTML = '';
  settings.graph.entityTypes.forEach((t, i) => {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.innerHTML = `${escapeHtml(t)}<button aria-label="${escapeHtml(t)} 제거">✕</button>`;
    tag.querySelector('button').addEventListener('click', () => {
      settings.graph.entityTypes.splice(i, 1);
      saveSettings();
      renderEntityTags();
    });
    box.appendChild(tag);
  });
  const input = document.createElement('input');
  input.className = 'tag-input';
  input.placeholder = '+ 유형 추가';
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && input.value.trim()) {
      e.preventDefault();
      const val = input.value.trim();
      if (!settings.graph.entityTypes.includes(val)) {
        settings.graph.entityTypes.push(val);
        saveSettings();
        renderEntityTags();
      }
      input.value = '';
    }
  });
  box.appendChild(input);
}

async function handleFiles(files) {
  if (building || running) {
    toast('DB 구축·질의 실행 중에는 문서를 추가할 수 없습니다.', 'warn');
    return;
  }
  for (const file of files) {
    // 파싱 await 중에 구축이 시작되었으면 남은 파일 처리 중단 (docs 변형 레이스 방지)
    if (building || running) {
      toast('DB 구축·질의가 시작되어 남은 파일 추가를 중단했습니다.', 'warn');
      break;
    }
    if (!detectType(file.name)) {
      toast(`지원하지 않는 형식: ${file.name}`, 'warn');
      continue;
    }
    try {
      toast(`${file.name} 분석 중…`, 'info', 1800);
      const raw = await parseFile(file);
      const text = cleanText(raw);
      if (text.length < 50) {
        toast(`${file.name}: 추출된 텍스트가 너무 적습니다.`, 'warn');
        continue;
      }
      // 동일 이름 문서는 교체
      const existing = docs.find((d) => d.name === file.name);
      if (existing) {
        await removeDocEverywhere(existing.id, { silent: true });
      }
      const doc = {
        id: uid(),
        name: file.name,
        type: detectType(file.name),
        size: file.size,
        textLen: text.length,
        text,
        addedAt: Date.now(),
      };
      await idb.put('docs', doc);
      docs.push(doc);
      toast(`${file.name} 추가됨 (${fmtNum(text.length)}자)`, 'ok');
    } catch (e) {
      toast(`${file.name} 처리 실패: ${e.message}`, 'err', 6000);
      console.error(e);
    }
  }
  refreshDocList();
}

async function loadSampleDocs() {
  if (building || running) {
    toast('DB 구축·질의 실행 중에는 샘플을 추가할 수 없습니다.', 'warn');
    return;
  }
  const { SAMPLE_DOCS } = await import('./sample-data.js');
  let added = 0;
  for (const s of SAMPLE_DOCS) {
    if (docs.some((d) => d.name === s.name)) continue;
    const text = cleanText(s.text);
    const doc = {
      id: uid(),
      name: s.name,
      type: 'md',
      size: text.length,
      textLen: text.length,
      text,
      addedAt: Date.now(),
    };
    await idb.put('docs', doc);
    docs.push(doc);
    added++;
  }
  refreshDocList();
  if (added) toast(`철도 샘플 문서 ${added}건을 불러왔습니다. 이제 DB를 구축해 보세요.`, 'ok', 5000);
  else toast('샘플 문서가 이미 모두 추가되어 있습니다.', 'info');
}

function refreshDocList() {
  $('#docCount').textContent = docs.length;
  const list = $('#docList');
  list.innerHTML = '';
  for (const doc of docs) {
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.innerHTML =
      `<span class="doc-icon">${doc.type.toUpperCase()}</span>` +
      `<span class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>` +
      `<span class="doc-meta">${fmtNum(doc.textLen)}자${doc.size ? ' · ' + fmtBytes(doc.size) : ''}</span>` +
      `<button class="doc-del" title="문서 삭제" aria-label="${escapeHtml(doc.name)} 삭제">✕</button>`;
    li.querySelector('.doc-del').addEventListener('click', () => removeDocEverywhere(doc.id));
    list.appendChild(li);
  }
}

async function removeDocEverywhere(docId, { silent } = {}) {
  if (building) {
    toast('DB 구축 중에는 문서를 삭제할 수 없습니다.', 'warn');
    return;
  }
  if (running) {
    toast('질의 실행 중에는 문서를 삭제할 수 없습니다.', 'warn');
    return;
  }
  const doc = docs.find((d) => d.id === docId);
  await idb.delete('docs', docId);
  await vectorStore.removeDoc(docId);
  await graphStore.removeDoc(docId);
  docs = docs.filter((d) => d.id !== docId);
  refreshDocList();
  refreshDbStats();
  vizVectorDirty = true;
  vizGraphDirty = true;
  if (!silent && doc) toast(`${doc.name} 및 관련 DB 데이터가 삭제되었습니다.`, 'ok');
}

async function clearAllData() {
  if (building) {
    toast('DB 구축 중에는 초기화할 수 없습니다. 먼저 구축을 중단하세요.', 'warn');
    return;
  }
  if (running) {
    toast('질의 실행 중에는 초기화할 수 없습니다. 먼저 실행을 중단하세요.', 'warn');
    return;
  }
  if (!confirm('업로드 문서와 Vector/Graph DB, 실험 기록을 모두 삭제합니다. 계속할까요?')) return;
  await idb.clearAll();
  await vectorStore.clear();
  await graphStore.clear();
  docs = [];
  clearHistory();
  refreshDocList();
  refreshDbStats();
  renderHistory();
  setIdleEmptyStates();
  vizVectorDirty = true;
  vizGraphDirty = true;
  toast('모든 데이터가 초기화되었습니다.', 'ok');
}

function setBuildingUI(kind, active) {
  building = active;
  $('#btnBuildVector').disabled = active;
  $('#btnBuildGraph').disabled = active;
  $('#btnBuildAll').disabled = active;
  $('#btnRebuildAll').disabled = active;
  $('#btnCancelBuild').hidden = !active;
  $(`#${kind}ProgressBox`).hidden = !active;
  // 구축 중 데이터 변경 차단 (정합성 보호)
  $('#btnClearAll').disabled = active;
  $('#btnLoadSample').disabled = active;
  $('#fileInput').disabled = active;
  $('#dropzone').classList.toggle('disabled', active);
  for (const b of $$('.doc-del')) b.disabled = active;
  // 다른 탭에서도 보이는 전역 진행 칩
  $('#globalBuildChip').hidden = !active;
  if (active) $('#globalBuildText').textContent = kind === 'v' ? 'Vector DB 구축 중' : 'Graph DB 구축 중';
}

function updateProgress(kind, done, total, stage) {
  const pct = total ? Math.round((done / total) * 100) : 0;
  $(`#${kind}ProgressFill`).style.width = pct + '%';
  $(`#${kind}ProgressText`).textContent = `${stage} ${done}/${total} (${pct}%)`;
  $('#globalBuildText').textContent = `${kind === 'v' ? 'Vector' : 'Graph'} 구축 ${pct}%`;
}

async function buildVector(full = false) {
  if (building) return false;
  if (running) {
    toast('질의 실행 중에는 DB를 구축할 수 없습니다.', 'warn');
    return false;
  }
  if (!docs.length) {
    toast('먼저 문서를 업로드하거나 샘플 문서를 로드하세요.', 'warn');
    return false;
  }
  if (!connected) {
    showCorsModal();
    return false;
  }
  if (settings.vector.overlap >= settings.vector.chunkSize) {
    toast('청크 중첩은 청크 크기보다 작아야 합니다.', 'err');
    return false;
  }

  // 증분 구축: 아직 DB에 반영되지 않은 신규 문서만 처리
  let targets = docs;
  if (!full) {
    const built = new Set(vectorStore.docIds());
    targets = docs.filter((d) => !built.has(d.id));
    if (!targets.length) {
      toast('Vector: 새로 추가된 문서가 없습니다. 파라미터 변경을 반영하려면 [전체 재구축]을 사용하세요.', 'info', 5000);
      return true;
    }
    if (vectorStore.size && vectorStore.embedModel && vectorStore.embedModel !== settings.embedModel) {
      toast('임베딩 모델이 기존 DB와 달라 증분 구축이 불가합니다. [전체 재구축]을 사용하세요.', 'err', 7000);
      return false;
    }
  }

  setBuildingUI('v', true);
  buildAbort = new AbortController();
  const t0 = performance.now();
  try {
    const count = await vectorStore.build(
      targets,
      settings.vector,
      client,
      settings.embedModel,
      (done, total, stage) => updateProgress('v', done, total, stage),
      buildAbort.signal,
      full // 전체 재구축도 성공 시에만 원자적으로 교체 (중단·실패 시 기존 DB 무손상)
    );
    toast(`Vector DB ${full ? '전체 재구축' : '증분 구축'} 완료: 문서 ${targets.length}건 → +${count}개 청크 (총 ${fmtNum(vectorStore.size)}개, ${fmtMs(performance.now() - t0)})`, 'ok', 5000);
    return true;
  } catch (e) {
    if (e.name === 'AbortError') toast('Vector DB 구축이 중단되었습니다. (기존 DB는 유지됩니다)', 'warn');
    else {
      toast(`Vector DB 구축 실패: ${e.message} (기존 DB는 유지됩니다)`, 'err', 7000);
      console.error(e);
    }
    return false;
  } finally {
    setBuildingUI('v', false);
    buildAbort = null;
    // 성공/실패와 무관하게 화면 상태를 실제 DB와 동기화
    vizVectorDirty = true;
    refreshDbStats();
    setIdleEmptyStates();
  }
}

async function buildGraph(full = false) {
  if (building) return false;
  if (running) {
    toast('질의 실행 중에는 DB를 구축할 수 없습니다.', 'warn');
    return false;
  }
  if (!docs.length) {
    toast('먼저 문서를 업로드하거나 샘플 문서를 로드하세요.', 'warn');
    return false;
  }
  if (!connected) {
    showCorsModal();
    return false;
  }

  // 증분 구축: 아직 그래프에 반영되지 않은 신규 문서만 추출
  let targets = docs;
  if (!full) {
    const built = new Set(graphStore.builtDocIds());
    targets = docs.filter((d) => !built.has(d.id));
    if (!targets.length) {
      toast('Graph: 새로 추가된 문서가 없습니다. 파라미터 변경을 반영하려면 [전체 재구축]을 사용하세요.', 'info', 5000);
      return true;
    }
    if (graphStore.nodeCount && graphStore.embedModel && graphStore.embedModel !== settings.embedModel) {
      toast('임베딩 모델이 기존 Graph DB와 달라 증분 구축이 불가합니다. [전체 재구축]을 사용하세요.', 'err', 7000);
      return false;
    }
  }

  setBuildingUI('g', true);
  buildAbort = new AbortController();
  const t0 = performance.now();
  try {
    const { nodes, edges, failed } = await graphStore.build(
      targets,
      settings.graph,
      client,
      { chatModel: settings.graphModel || settings.chatModel, embedModel: settings.embedModel, numCtx: settings.numCtx },
      (done, total, stage) => updateProgress('g', done, total, stage),
      buildAbort.signal,
      full // 전체 재구축도 성공 시에만 원자적으로 교체 (중단·실패 시 기존 DB 무손상)
    );
    toast(`Graph DB ${full ? '전체 재구축' : '증분 구축'} 완료: 문서 ${targets.length}건 처리 → 노드 ${fmtNum(nodes)} · 관계 ${fmtNum(edges)} (${fmtMs(performance.now() - t0)})`, 'ok', 6000);
    if (failed > 0) {
      const model = settings.graphModel || settings.chatModel;
      toast(`추출 실패 청크 ${failed}건은 건너뛰었습니다. 실패가 많으면 추출 모델(${model})을 더 큰 모델로 바꿔보세요.`, 'warn', 7000);
    }
    return true;
  } catch (e) {
    if (e.name === 'AbortError') toast('Graph DB 구축이 중단되었습니다. (기존 DB는 유지됩니다)', 'warn');
    else {
      toast(`Graph DB 구축 실패: ${e.message} (기존 DB는 유지됩니다)`, 'err', 7000);
      console.error(e);
    }
    return false;
  } finally {
    setBuildingUI('g', false);
    buildAbort = null;
    // 성공/실패와 무관하게 화면 상태를 실제 DB와 동기화
    vizGraphDirty = true;
    refreshDbStats();
    setIdleEmptyStates();
  }
}

function refreshDbStats() {
  $('#vStatChunks').textContent = fmtNum(vectorStore.size);
  $('#vStatDim').textContent = vectorStore.dim || '–';
  $('#vStatDocs').textContent = vectorStore.docIds().length;
  $('#gStatNodes').textContent = fmtNum(graphStore.nodeCount);
  $('#gStatEdges').textContent = fmtNum(graphStore.edgeCount);
  $('#gStatTypes').textContent = graphStore.typeSet().length || 0;
  $('#badgeVectorCount').textContent = fmtNum(vectorStore.size);
  $('#badgeGraphCount').textContent = fmtNum(graphStore.nodeCount);

  idb.get('meta', 'vectorBuiltAt').then((m) => {
    $('#vBuiltAt').textContent = m?.value ? `구축: ${new Date(m.value).toLocaleString('ko-KR')}` : '';
  });
  idb.get('meta', 'graphBuiltAt').then((m) => {
    $('#gBuiltAt').textContent = m?.value ? `구축: ${new Date(m.value).toLocaleString('ko-KR')}` : '';
  });
}

// ── 시각화 탭 ──────────────────────────────
function initVizTab() {
  vectorViz = new VectorViz($('#vectorCanvas'), $('#vectorTooltip'));
  graphViz = new GraphViz($('#graphCanvas'), $('#graphTooltip'), $('#nodeDetail'));
  graphViz.onSelect = (key) => graphStore.neighbors(key);

  $('#vizVectorBtn').addEventListener('click', () => switchViz('vector'));
  $('#vizGraphBtn').addEventListener('click', () => switchViz('graph'));
  $('#graphSearch').addEventListener('input', (e) => graphViz.setSearch(e.target.value));
  const syncPhysicsBtn = (on) => {
    const btn = $('#btnPhysics');
    btn.textContent = on ? '물리 시뮬레이션 ⏸' : '물리 시뮬레이션 ▶';
    btn.setAttribute('aria-pressed', String(on));
    btn.setAttribute('aria-label', on ? '물리 시뮬레이션 일시정지' : '물리 시뮬레이션 재생');
  };
  graphViz.onStateChange = syncPhysicsBtn; // 자동 안정화 정지 시에도 라벨 동기화
  $('#btnPhysics').addEventListener('click', () => syncPhysicsBtn(graphViz.toggle()));
  syncPhysicsBtn(false);
}

function switchViz(kind) {
  $('#vizVectorBtn').classList.toggle('active', kind === 'vector');
  $('#vizGraphBtn').classList.toggle('active', kind === 'graph');
  $('#vizVectorBtn').setAttribute('aria-selected', String(kind === 'vector'));
  $('#vizGraphBtn').setAttribute('aria-selected', String(kind === 'graph'));
  $('#vizVectorStage').hidden = kind !== 'vector';
  $('#vizGraphStage').hidden = kind !== 'graph';
  refreshViz();
}

function refreshViz() {
  const vectorActive = !$('#vizVectorStage').hidden;
  if (vectorActive && vizVectorDirty) {
    vizVectorDirty = false;
    if (vectorStore.size >= 2) {
      $('#vectorVizEmpty').hidden = true;
      const projected = vectorStore.project2D();
      vectorViz.setData(projected);
      $('#vectorVizInfo').textContent =
        `청크 ${fmtNum(vectorStore.size)}개 · ${vectorStore.dim}차원 → PCA 2D` +
        (projected.length < vectorStore.size ? ` (${projected.length}개 샘플 표시)` : '');
    } else {
      $('#vectorVizEmpty').hidden = false;
      $('#vectorVizInfo').textContent = 'Vector DB가 비어 있습니다.';
    }
  }
  const graphActive = !$('#vizGraphStage').hidden;
  if (graphActive && vizGraphDirty) {
    vizGraphDirty = false;
    if (graphStore.nodeCount) {
      $('#graphVizEmpty').hidden = true;
      const data = graphStore.vizData();
      graphViz.setData(data);
      $('#graphVizInfo').textContent =
        `노드 ${fmtNum(graphStore.nodeCount)} · 관계 ${fmtNum(graphStore.edgeCount)}` +
        (data.truncated ? ` (상위 ${data.nodes.length}개 노드 표시)` : '');
    } else {
      $('#graphVizEmpty').hidden = false;
      $('#graphVizInfo').textContent = 'Graph DB가 비어 있습니다.';
    }
  }
}

// ── 시작 ──────────────────────────────
init().catch((e) => {
  console.error('초기화 실패:', e);
  toast(`초기화 오류: ${e.message}`, 'err', 8000);
});
