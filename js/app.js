// ═══════════════════════════════════════════════
// RAIL·RAG LAB — 메인 애플리케이션 컨트롤러
// ═══════════════════════════════════════════════

import { DEFAULTS, MODES, EXAMPLE_QUESTIONS, SYSTEM_PROMPT_BASE, LS_KEYS, buildBasicMessages, buildVectorMessages, buildGraphMessages } from './config.js';
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
const client = new OllamaClient(settings.ollamaUrl);
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
  return {
    ...structuredClone(DEFAULTS),
    ...saved,
    vector: { ...DEFAULTS.vector, ...(saved.vector || {}) },
    graph: { ...DEFAULTS.graph, ...(saved.graph || {}) },
    sysPrompt: saved.sysPrompt || SYSTEM_PROMPT_BASE,
  };
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
    this.answer = node.querySelector('.answer');
    this.empty = node.querySelector('.engine-empty');
    this.ctxDetails = node.querySelector('.ctx-details');
    this.ctxCount = node.querySelector('.ctx-count');
    this.ctxList = node.querySelector('.ctx-list');
    this.reset();
  }

  reset() {
    this.setSignal('idle');
    this.setStage('STANDBY');
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
    for (const el of this.ctxList.querySelectorAll('.ctx-item')) {
      el.addEventListener('click', () => el.classList.toggle('expanded'));
    }
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
    for (const el of this.ctxList.querySelectorAll('.ctx-item')) {
      el.addEventListener('click', () => el.classList.toggle('expanded'));
    }
  }
}

// ── 초기화 ──────────────────────────────
async function init() {
  initTheme();
  initTabs();
  initPanels();
  initQueryConsole();
  initDrawer();
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
  for (const btn of $$('.tab-btn')) {
    btn.addEventListener('click', () => {
      for (const b of $$('.tab-btn')) {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', String(b === btn));
      }
      for (const p of $$('.tab-panel')) p.classList.toggle('active', p.id === `tab-${btn.dataset.tab}`);
      if (btn.dataset.tab === 'viz') refreshViz();
    });
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
async function connect() {
  const box = $('#connBox');
  box.className = 'conn-box';
  $('#connText').textContent = '연결 확인 중…';
  client.setBaseUrl(settings.ollamaUrl);
  try {
    models = await client.listModels();
    connected = true;
    box.classList.add('ok');
    $('#connText').textContent = `Ollama 연결됨 · 모델 ${models.length}개`;
    populateModelSelects();
    $('#drawerInfo').textContent = `서버: ${settings.ollamaUrl}\n모델 ${models.length}개 감지`;
  } catch (e) {
    connected = false;
    box.classList.add('fail');
    $('#connText').textContent = 'Ollama 연결 실패 — 클릭하여 안내 보기';
    $('#corsModal').hidden = false;
    console.warn('Ollama 연결 실패:', e);
  }
}

function populateModelSelects() {
  const chatSel = $('#chatModel');
  const embedSel = $('#embedModel');
  const opts = models.map((m) => `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)}</option>`).join('');
  chatSel.innerHTML = opts;
  embedSel.innerHTML = opts;

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

  chatSel.addEventListener('change', () => {
    settings.chatModel = chatSel.value;
    saveSettings();
    toast(`응답 모델: ${chatSel.value}`, 'info', 2000);
  });
  embedSel.addEventListener('change', () => {
    settings.embedModel = embedSel.value;
    saveSettings();
    if (vectorStore.size) toast('임베딩 모델이 변경되었습니다. Vector DB를 다시 구축해야 검색 정합성이 유지됩니다.', 'warn', 5000);
  });
}

// ── 질의 콘솔 ──────────────────────────────
function initQueryConsole() {
  const chipBox = $('#exampleChips');
  for (const q of EXAMPLE_QUESTIONS) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = q;
    chip.addEventListener('click', () => {
      $('#queryInput').value = q;
      $('#queryInput').focus();
    });
    chipBox.appendChild(chip);
  }

  $('#queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      runQuery();
    }
  });
  $('#btnRun').addEventListener('click', runQuery);
  $('#btnStop').addEventListener('click', () => runAbort?.abort());
  $('#btnClearHistory').addEventListener('click', () => {
    clearHistory();
    renderHistory();
  });

  $('#connBox').addEventListener('click', () => {
    if (!connected) $('#corsModal').hidden = false;
  });
  $('#btnRetryConn').addEventListener('click', async () => {
    $('#corsModal').hidden = true;
    await connect();
    if (connected) toast('Ollama에 연결되었습니다.', 'ok');
  });
  $('#btnCloseCors').addEventListener('click', () => ($('#corsModal').hidden = true));
}

async function runQuery() {
  const query = $('#queryInput').value.trim();
  if (!query) {
    toast('질문을 입력해 주세요.', 'warn');
    return;
  }
  if (running) return;
  if (!connected) {
    $('#corsModal').hidden = false;
    return;
  }

  running = true;
  $('#btnRun').disabled = true;
  $('#btnStop').disabled = false;
  runAbort = new AbortController();
  const signal = runAbort.signal;
  const run = emptyRunMetrics();

  for (const p of Object.values(panels)) p.reset();
  $('#compareSection').hidden = true;

  const genOptions = {
    temperature: settings.temperature,
    num_ctx: settings.numCtx,
    num_predict: settings.numPredict,
  };

  const tasks = [
    runBasic(query, run, genOptions, signal),
    runVector(query, run, genOptions, signal),
    runGraph(query, run, genOptions, signal),
  ];
  await Promise.allSettled(tasks);

  renderCharts(run);
  saveHistoryEntry(query, settings.chatModel, run);
  renderHistory();

  running = false;
  $('#btnRun').disabled = false;
  $('#btnStop').disabled = true;
  runAbort = null;
}

async function streamAnswer(panel, messages, run, genOptions, signal) {
  const m = run[panel.mode];
  panel.setStage('GENERATING — 응답 생성 중');
  panel.answer.classList.add('streaming');
  const renderer = makeStreamRenderer(panel.answer);

  const res = await client.chatStream({
    model: settings.chatModel,
    messages,
    options: genOptions,
    signal,
    onToken: (_piece, full) => {
      if (m.ttftMs == null) {
        // TTFT는 chatStream이 정확히 계산하지만 UI는 즉시 반영
      }
      renderer.update(full);
    },
  });

  renderer.finish(res.text);
  panel.answer.classList.remove('streaming');

  m.ttftMs = res.ttftMs;
  m.totalMs = (m.retrievalMs || 0) + res.totalMs;
  m.promptTokens = res.promptTokens;
  m.evalTokens = res.evalTokens;
  m.tps = computeTps(res.evalTokens, res.evalDurationMs);
  panel.applyMetrics(m);

  if (res.aborted) {
    panel.setStage('ABORTED — 사용자 중단');
    panel.setSignal('idle');
  } else {
    panel.setStage('COMPLETE');
    panel.setSignal('done');
  }
}

function handleRunError(panel, e) {
  if (e.name === 'AbortError') {
    panel.setStage('ABORTED — 사용자 중단');
    panel.setSignal('idle');
    return;
  }
  panel.setSignal('error');
  panel.setStage('ERROR');
  panel.setEmpty(`⚠ 오류: ${escapeHtml(e.message || String(e))}`);
  console.error(`[${panel.mode}]`, e);
}

async function runBasic(query, run, genOptions, signal) {
  const panel = panels.basic;
  try {
    panel.setSignal('work');
    panel.setEmpty('');
    await streamAnswer(panel, buildBasicMessages(settings.sysPrompt, query), run, genOptions, signal);
  } catch (e) {
    handleRunError(panel, e);
  }
}

async function runVector(query, run, genOptions, signal) {
  const panel = panels.vector;
  const m = run.vector;
  if (!vectorStore.size) {
    panel.setStage('NO DATABASE');
    panel.setEmpty('Vector DB가 비어 있어 실행하지 않았습니다.<br><button class="btn-ghost btn-sm goto-build">문서 · DB 구축으로 이동</button>');
    panel.empty.querySelector('.goto-build')?.addEventListener('click', () => $('.tab-btn[data-tab="build"]').click());
    return;
  }
  try {
    panel.setSignal('work');
    panel.setEmpty('');
    panel.setStage('RETRIEVING — 유사 청크 검색 중');
    const t0 = performance.now();
    const [qVec] = await client.embed({ model: settings.embedModel, input: [query], signal });
    const results = vectorStore.search(qVec, settings.vector);
    m.retrievalMs = performance.now() - t0;
    panel.setMetric('retrieval', fmtMs(m.retrievalMs));
    panel.showChunks(results.map((r) => ({ ...r.chunk, score: r.score })));

    if (!results.length) {
      panel.setStage('NO MATCH — 관련 청크 없음');
      toast('Vector: 유사도 임계값을 넘는 청크가 없습니다. 임계값을 낮추거나 문서를 추가하세요.', 'warn');
    }
    const ctxChunks = results.map((r) => ({ docName: r.chunk.docName, text: r.chunk.text, score: r.score }));
    await streamAnswer(panel, buildVectorMessages(settings.sysPrompt, query, ctxChunks), run, genOptions, signal);
  } catch (e) {
    handleRunError(panel, e);
  }
}

async function runGraph(query, run, genOptions, signal) {
  const panel = panels.graph;
  const m = run.graph;
  if (!graphStore.nodeCount) {
    panel.setStage('NO DATABASE');
    panel.setEmpty('Graph DB가 비어 있어 실행하지 않았습니다.<br><button class="btn-ghost btn-sm goto-build">문서 · DB 구축으로 이동</button>');
    panel.empty.querySelector('.goto-build')?.addEventListener('click', () => $('.tab-btn[data-tab="build"]').click());
    return;
  }
  try {
    panel.setSignal('work');
    panel.setEmpty('');
    panel.setStage('TRAVERSING — 지식그래프 탐색 중');
    const t0 = performance.now();
    const [qVec] = await client.embed({ model: settings.embedModel, input: [query], signal });
    const result = graphStore.search(query, qVec, settings.graph);
    m.retrievalMs = performance.now() - t0;
    panel.setMetric('retrieval', fmtMs(m.retrievalMs));
    panel.showGraphCtx(result);

    if (!result.triples.length && !result.ctxChunks.length) {
      panel.setStage('NO MATCH — 관련 개체 없음');
      toast('Graph: 질의와 연결되는 개체를 찾지 못했습니다. 시드 개체 수를 늘리거나 문서를 추가하세요.', 'warn');
    }
    await streamAnswer(panel, buildGraphMessages(settings.sysPrompt, query, result.triples, result.ctxChunks), run, genOptions, signal);
  } catch (e) {
    handleRunError(panel, e);
  }
}

// ── 설정 드로어 ──────────────────────────────
function initDrawer() {
  const drawer = $('#drawer');
  const backdrop = $('#drawerBackdrop');
  const openDrawer = () => {
    drawer.hidden = false;
    backdrop.hidden = false;
    requestAnimationFrame(() => drawer.classList.add('open'));
  };
  const closeDrawer = () => {
    drawer.classList.remove('open');
    backdrop.hidden = true;
    setTimeout(() => (drawer.hidden = true), 300);
  };
  $('#btnSettings').addEventListener('click', openDrawer);
  $('#btnCloseDrawer').addEventListener('click', closeDrawer);
  backdrop.addEventListener('click', closeDrawer);

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

  $('#btnBuildVector').addEventListener('click', () => buildVector());
  $('#btnBuildGraph').addEventListener('click', () => buildGraph());
  $('#btnBuildAll').addEventListener('click', async () => {
    if (await buildVector()) await buildGraph();
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
  for (const file of files) {
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
  $('#btnCancelBuild').hidden = !active;
  $(`#${kind}ProgressBox`).hidden = !active;
}

function updateProgress(kind, done, total, stage) {
  const pct = total ? Math.round((done / total) * 100) : 0;
  $(`#${kind}ProgressFill`).style.width = pct + '%';
  $(`#${kind}ProgressText`).textContent = `${stage} ${done}/${total} (${pct}%)`;
}

async function buildVector() {
  if (building) return false;
  if (!docs.length) {
    toast('먼저 문서를 업로드하거나 샘플 문서를 로드하세요.', 'warn');
    return false;
  }
  if (!connected) {
    $('#corsModal').hidden = false;
    return false;
  }
  if (settings.vector.overlap >= settings.vector.chunkSize) {
    toast('청크 중첩은 청크 크기보다 작아야 합니다.', 'err');
    return false;
  }

  setBuildingUI('v', true);
  buildAbort = new AbortController();
  const t0 = performance.now();
  try {
    const count = await vectorStore.build(
      docs,
      settings.vector,
      client,
      settings.embedModel,
      (done, total, stage) => updateProgress('v', done, total, stage),
      buildAbort.signal
    );
    toast(`Vector DB 구축 완료: ${count}개 청크 (${fmtMs(performance.now() - t0)})`, 'ok', 5000);
    vizVectorDirty = true;
    refreshDbStats();
    setIdleEmptyStates();
    return true;
  } catch (e) {
    if (e.name === 'AbortError') toast('Vector DB 구축이 중단되었습니다.', 'warn');
    else {
      toast(`Vector DB 구축 실패: ${e.message}`, 'err', 6000);
      console.error(e);
    }
    return false;
  } finally {
    setBuildingUI('v', false);
    buildAbort = null;
  }
}

async function buildGraph() {
  if (building) return false;
  if (!docs.length) {
    toast('먼저 문서를 업로드하거나 샘플 문서를 로드하세요.', 'warn');
    return false;
  }
  if (!connected) {
    $('#corsModal').hidden = false;
    return false;
  }

  setBuildingUI('g', true);
  buildAbort = new AbortController();
  const t0 = performance.now();
  try {
    const { nodes, edges } = await graphStore.build(
      docs,
      settings.graph,
      client,
      { chatModel: settings.chatModel, embedModel: settings.embedModel, numCtx: settings.numCtx },
      (done, total, stage) => updateProgress('g', done, total, stage),
      buildAbort.signal
    );
    toast(`Graph DB 구축 완료: ${nodes}개 노드 · ${edges}개 관계 (${fmtMs(performance.now() - t0)})`, 'ok', 6000);
    vizGraphDirty = true;
    refreshDbStats();
    setIdleEmptyStates();
    return true;
  } catch (e) {
    if (e.name === 'AbortError') toast('Graph DB 구축이 중단되었습니다.', 'warn');
    else {
      toast(`Graph DB 구축 실패: ${e.message}`, 'err', 6000);
      console.error(e);
    }
    return false;
  } finally {
    setBuildingUI('g', false);
    buildAbort = null;
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
  $('#btnPhysics').addEventListener('click', () => {
    const on = graphViz.toggle();
    $('#btnPhysics').textContent = on ? '물리 시뮬레이션 ⏸' : '물리 시뮬레이션 ▶';
  });
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
