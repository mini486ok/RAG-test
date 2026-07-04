// ═══════════════════════════════════════════════
// Graph Store (지식그래프)
//  - 구축: 청크별 LLM 개체·관계 추출(JSON) → 병합 → 개체 임베딩
//    ※ 임시 상태(work)에 구축 후 성공 시에만 교체 — 중단/실패 시 기존 상태 무손상
//  - 검색: 질의 임베딩 ↔ 개체 임베딩 + 어휘/별칭 매칭 → n-hop BFS 확장
//  - 저장: IndexedDB 단일 트랜잭션 (nodes / edges / meta 원자적 교체)
// ═══════════════════════════════════════════════

import { idb } from './db.js';
import { chunkText } from './chunker.js';
import { dot } from './ollama.js';
import { buildExtractionPrompt, buildGleaningPrompt, RAIL_ALIASES } from './config.js';
import { uid } from './ui.js';

function normName(name) {
  return String(name).trim().replace(/\s+/g, ' ').toLowerCase();
}

// 흔한 한국어 조사(말미) — 기존 노드가 있을 때만 병합에 사용하는 보수적 휴리스틱.
// 1자 조사는 오병합 위험이 낮은 것만 유지 ('과/로/이/가' 등은 복합명사·부서명의
// 마지막 음절과 충돌: 신호과, 전차선로, 전기 등 → 제외)
const PARTICLES_2 = ['으로', '에서', '부터', '까지', '에게'];
const PARTICLES_1 = ['은', '는', '을', '를'];

/** 개체명 키 결정: 정확 일치 → 조사 제거 형태가 이미 존재하면 그 키로 병합 */
function resolveKey(state, rawName) {
  const k = normName(rawName);
  if (state.nodes.has(k)) return k;
  for (const p of PARTICLES_2) {
    if (k.length > p.length + 1 && k.endsWith(p)) {
      const stripped = k.slice(0, -p.length);
      if (state.nodes.has(stripped)) return stripped;
    }
  }
  for (const p of PARTICLES_1) {
    if (k.length > 2 && k.endsWith(p)) {
      const stripped = k.slice(0, -1);
      if (state.nodes.has(stripped)) return stripped;
    }
  }
  return k;
}

/** 추출 결과를 상태(state)에 병합 */
function mergeInto(state, ext, unit, entityTypes) {
  for (const e of ext.entities) {
    if (!e.name || typeof e.name !== 'string') continue;
    const key = resolveKey(state, e.name);
    if (!key || key.length > 60) continue;
    let node = state.nodes.get(key);
    if (!node) {
      node = {
        id: uid(),
        name: e.name.trim(),
        // 하이브리드 유형: 기본 목록은 프롬프트 가이드로 쓰되,
        // LLM이 제시한 새 유형도 수용 (범례는 자동 확장)
        type: (typeof e.type === 'string' && e.type.trim()) ? e.type.trim().slice(0, 20) : '개념/용어',
        desc: (e.description || '').slice(0, 300),
        degree: 0,
        weight: 0,
        chunkIds: [],
        docIds: [],
        vec: null,
      };
      state.nodes.set(key, node);
    } else if (e.description && node.desc.length < 240 && !node.desc.includes(e.description.slice(0, 40))) {
      node.desc = (node.desc + ' ' + e.description).slice(0, 300);
    }
    node.weight++;
    if (!node.chunkIds.includes(unit.chunkId)) node.chunkIds.push(unit.chunkId);
    if (!node.docIds.includes(unit.docId)) node.docIds.push(unit.docId);
  }

  for (const r of ext.relations) {
    if (!r.source || !r.target || !r.relation) continue;
    const sKey = resolveKey(state, String(r.source));
    const tKey = resolveKey(state, String(r.target));
    if (!sKey || !tKey || sKey === tKey) continue;
    // 관계에만 등장한 개체도 노드로 승격
    for (const [k, orig] of [[sKey, r.source], [tKey, r.target]]) {
      if (!state.nodes.has(k)) {
        state.nodes.set(k, {
          id: uid(), name: String(orig).trim(), type: '개념/용어', desc: '',
          degree: 0, weight: 1, chunkIds: [unit.chunkId], docIds: [unit.docId], vec: null,
        });
      }
    }
    const eKey = `${sKey}→${String(r.relation).trim()}→${tKey}`;
    let edge = state.edges.get(eKey);
    if (!edge) {
      edge = {
        id: eKey,
        source: state.nodes.get(sKey).name,
        target: state.nodes.get(tKey).name,
        sourceKey: sKey,
        targetKey: tKey,
        relation: String(r.relation).trim().slice(0, 40),
        desc: (r.description || '').slice(0, 200),
        weight: 0,
        chunkIds: [],
      };
      state.edges.set(eKey, edge);
      state.nodes.get(sKey).degree++;
      state.nodes.get(tKey).degree++;
    }
    edge.weight++;
    if (!edge.chunkIds.includes(unit.chunkId)) edge.chunkIds.push(unit.chunkId);
  }
}

/** 상태에서 특정 문서 흔적 제거 + degree 재계산 */
function removeDocFromState(state, docId) {
  for (const [key, node] of state.nodes) {
    node.docIds = node.docIds.filter((d) => d !== docId);
    node.chunkIds = node.chunkIds.filter((c) => !c.startsWith(docId + '-g'));
    if (!node.docIds.length) state.nodes.delete(key);
  }
  for (const [key, edge] of state.edges) {
    edge.chunkIds = edge.chunkIds.filter((c) => !c.startsWith(docId + '-g'));
    if (!edge.chunkIds.length || !state.nodes.has(edge.sourceKey) || !state.nodes.has(edge.targetKey)) {
      state.edges.delete(key);
    }
  }
  for (const cid of [...state.chunkTexts.keys()]) {
    if (cid.startsWith(docId + '-g')) state.chunkTexts.delete(cid);
  }
  for (const n of state.nodes.values()) n.degree = 0;
  for (const e of state.edges.values()) {
    const s = state.nodes.get(e.sourceKey);
    const t = state.nodes.get(e.targetKey);
    if (s) s.degree++;
    if (t) t.degree++;
  }
}

/** 상태 딥카피 (vec은 불변이므로 참조 공유) */
function cloneState(store) {
  return {
    nodes: new Map([...store.nodes].map(([k, n]) => [k, { ...n, chunkIds: [...n.chunkIds], docIds: [...n.docIds] }])),
    edges: new Map([...store.edges].map(([k, e]) => [k, { ...e, chunkIds: [...e.chunkIds] }])),
    chunkTexts: new Map(store.chunkTexts),
  };
}

/** 질의에 철도 약어·동의어 별칭 확장 적용 */
function expandQuery(query) {
  const q = query.toLowerCase();
  let extra = '';
  for (const group of RAIL_ALIASES) {
    if (group.some((term) => q.includes(term.toLowerCase()))) {
      extra += ' ' + group.join(' ');
    }
  }
  return q + extra.toLowerCase();
}

export class GraphStore {
  constructor() {
    this.nodes = new Map(); // normName → node
    this.edges = new Map(); // edgeKey → edge
    this.chunkTexts = new Map(); // chunkId → {text, docName}
    this.embedModel = null;
    this.dim = 0;
  }

  async load() {
    const [nodes, edges, meta, builtMeta] = await Promise.all([
      idb.getAll('nodes'),
      idb.getAll('edges'),
      idb.get('meta', 'graphChunks'),
      idb.get('meta', 'graphBuiltAt'),
    ]);
    this.nodes = new Map(
      nodes.map((n) => [normName(n.name), { ...n, vec: n.vec ? (n.vec instanceof Float32Array ? n.vec : new Float32Array(n.vec)) : null }])
    );
    this.edges = new Map(edges.map((e) => [e.id, e]));
    this.chunkTexts = new Map(Object.entries(meta?.value || {}));
    this.embedModel = builtMeta?.model || null;
    this.dim = builtMeta?.dim || [...this.nodes.values()].find((n) => n.vec)?.vec.length || 0;
  }

  get nodeCount() { return this.nodes.size; }
  get edgeCount() { return this.edges.size; }

  typeSet() {
    return [...new Set([...this.nodes.values()].map((n) => n.type || '기타'))];
  }

  /** 그래프에 반영된(추출 처리된) 문서 ID 목록 — 증분 구축 판단용 */
  builtDocIds() {
    const ids = new Set();
    for (const cid of this.chunkTexts.keys()) {
      const m = cid.match(/^(.+)-g\d+$/);
      if (m) ids.add(m[1]);
    }
    return [...ids];
  }

  /**
   * 그래프 구축 — 임시 상태에 수행하고 성공 시에만 커밋.
   * @returns {{nodes:number, edges:number, failed:number}}
   */
  async build(docs, params, client, models, onProgress, signal) {
    const work = cloneState(this);
    for (const doc of docs) removeDocFromState(work, doc.id);

    // 1) 추출용 청킹 (벡터 청크보다 큰 단위)
    const units = [];
    for (const doc of docs) {
      const pieces = chunkText(doc.text, { chunkSize: params.extractChunkSize, overlap: 0 });
      pieces.forEach((text, i) => units.push({ chunkId: `${doc.id}-g${i}`, docId: doc.id, docName: doc.name, text }));
    }
    if (!units.length) throw new Error('추출할 텍스트가 없습니다.');

    // 2) 청크별 LLM 추출
    let done = 0;
    let failed = 0;
    for (const unit of units) {
      if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
      work.chunkTexts.set(unit.chunkId, { text: unit.text, docName: unit.docName });

      const messages = [
        { role: 'user', content: buildExtractionPrompt(unit.text, params.entityTypes, params.maxTriples) },
      ];
      try {
        const raw = await client.chatJSON({
          model: models.chatModel,
          messages,
          options: { temperature: 0.1, num_ctx: models.numCtx, num_predict: 2000 },
          signal,
        });
        const ext = parseExtraction(raw);
        let gotAny = ext.entities.length > 0 || ext.relations.length > 0;
        mergeInto(work, ext, unit, params.entityTypes);

        // gleaning: 추가 추출 반복
        for (let g = 0; g < (params.gleaning || 0); g++) {
          if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
          const gRaw = await client.chatJSON({
            model: models.chatModel,
            messages: [...messages, { role: 'assistant', content: raw }, { role: 'user', content: buildGleaningPrompt() }],
            options: { temperature: 0.1, num_ctx: models.numCtx, num_predict: 1200 },
            signal,
          });
          const gExt = parseExtraction(gRaw);
          if (gExt.entities.length || gExt.relations.length) gotAny = true;
          mergeInto(work, gExt, unit, params.entityTypes);
        }
        if (!gotAny) failed++;
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        failed++;
        console.warn('추출 실패(스킵):', unit.chunkId, e.message);
      }
      done++;
      onProgress?.(done, units.length, failed ? `개체·관계 추출 (실패 ${failed})` : '개체·관계 추출');
    }

    if (!work.nodes.size) throw new Error('추출된 개체가 없습니다. 모델 또는 문서를 확인하세요.');

    // 3) 개체 임베딩 (검색용)
    const nodeArr = [...work.nodes.values()].filter((n) => !n.vec);
    const batch = 12;
    for (let i = 0; i < nodeArr.length; i += batch) {
      if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
      const slice = nodeArr.slice(i, i + batch);
      const inputs = slice.map((n) => `${n.name} (${n.type}): ${n.desc || ''}`.slice(0, 500));
      const vecs = await client.embed({ model: models.embedModel, input: inputs, signal });
      slice.forEach((n, j) => (n.vec = vecs[j]));
      onProgress?.(Math.min(i + batch, nodeArr.length), nodeArr.length, '개체 임베딩');
    }

    // 4) 원자적 저장 성공 후에만 메모리 상태 교체
    const dim = [...work.nodes.values()].find((n) => n.vec)?.vec.length || 0;
    await persistState(work, { model: models.embedModel, dim });
    this.nodes = work.nodes;
    this.edges = work.edges;
    this.chunkTexts = work.chunkTexts;
    this.embedModel = models.embedModel;
    this.dim = dim;
    return { nodes: this.nodes.size, edges: this.edges.size, failed };
  }

  async removeDoc(docId) {
    const work = cloneState(this);
    removeDocFromState(work, docId);
    await persistState(work, { model: this.embedModel, dim: this.dim });
    this.nodes = work.nodes;
    this.edges = work.edges;
    this.chunkTexts = work.chunkTexts;
  }

  async clear() {
    await idb.atomicWrite([
      { store: 'nodes', clear: true },
      { store: 'edges', clear: true },
      { store: 'meta', deleteKeys: ['graphChunks', 'graphBuiltAt'] },
    ]);
    this.nodes.clear();
    this.edges.clear();
    this.chunkTexts.clear();
    this.embedModel = null;
    this.dim = 0;
  }

  /**
   * 그래프 검색: 시드 개체(임베딩+어휘·별칭) → n-hop BFS(프론티어 단위) → 트리플 랭킹
   * @returns {{triples:Array, seeds:Array, ctxChunks:Array}}
   */
  search(query, queryVec, { topKEntities = 6, hops = 2, maxCtxTriples = 30, chunkTopK = 2 } = {}) {
    if (!this.nodes.size) return { triples: [], seeds: [], ctxChunks: [] };
    if (queryVec && this.dim && queryVec.length !== this.dim) {
      throw new Error(`임베딩 차원 불일치 (Graph DB ${this.dim} vs 질의 ${queryVec.length}). 임베딩 모델이 바뀌었으면 Graph DB를 다시 구축하세요.`);
    }
    const qExpanded = expandQuery(query);

    // 1) 시드 개체 점수: 임베딩 유사도 + 어휘/별칭 포함 부스트
    const scored = [];
    for (const node of this.nodes.values()) {
      let score = 0;
      if (node.vec && queryVec && node.vec.length === queryVec.length) {
        score = dot(queryVec, node.vec);
      }
      const nLower = node.name.toLowerCase();
      if (nLower.length >= 2 && qExpanded.includes(nLower)) score += 0.35;
      scored.push({ node, score });
    }
    scored.sort((a, b) => b.score - a.score);
    const seeds = scored.slice(0, topKEntities).filter((s) => s.score > 0.15);
    if (!seeds.length) return { triples: [], seeds: [], ctxChunks: [] };

    const seedScore = new Map(seeds.map((s) => [normName(s.node.name), s.score]));

    // 2) n-hop BFS 확장 — 각 hop은 직전 프론티어만 확장 (거리 정확성 보장)
    const nodeDist = new Map([...seedScore.keys()].map((k) => [k, 0]));
    let frontier = new Set(seedScore.keys());
    for (let h = 1; h <= hops && frontier.size; h++) {
      const next = new Set();
      for (const e of this.edges.values()) {
        if (frontier.has(e.sourceKey) && !nodeDist.has(e.targetKey)) {
          nodeDist.set(e.targetKey, h);
          next.add(e.targetKey);
        }
        if (frontier.has(e.targetKey) && !nodeDist.has(e.sourceKey)) {
          nodeDist.set(e.sourceKey, h);
          next.add(e.sourceKey);
        }
      }
      frontier = next;
    }

    // 3) 서브그래프 엣지 랭킹
    const rankedEdges = [];
    for (const e of this.edges.values()) {
      if (!nodeDist.has(e.sourceKey) || !nodeDist.has(e.targetKey)) continue;
      const sScore = seedScore.get(e.sourceKey) || 0;
      const tScore = seedScore.get(e.targetKey) || 0;
      const dist = Math.min(nodeDist.get(e.sourceKey), nodeDist.get(e.targetKey));
      const rank = (sScore + tScore + 0.05) * Math.pow(0.55, dist) * Math.log2(1 + e.weight);
      rankedEdges.push({ edge: e, rank });
    }
    rankedEdges.sort((a, b) => b.rank - a.rank);

    const triples = rankedEdges.slice(0, maxCtxTriples).map(({ edge, rank }) => {
      // 출처 문서명: 원본 문서 기준 인용을 위해 트리플마다 부착 (최대 2개)
      const docNames = [...new Set(
        edge.chunkIds.map((cid) => this.chunkTexts.get(cid)?.docName).filter(Boolean)
      )].slice(0, 2);
      return {
        source: edge.source,
        target: edge.target,
        relation: edge.relation,
        desc: edge.desc,
        rank,
        chunkIds: edge.chunkIds,
        docNames,
      };
    });

    // 4) 시드 개체 관련 원문 청크 — 시드 간 라운드로빈으로 다양성 확보
    const ctxChunks = [];
    if (chunkTopK > 0) {
      const seen = new Set();
      outer: for (let round = 0; round < 8; round++) {
        let advanced = false;
        for (const s of seeds) {
          const cid = s.node.chunkIds[round];
          if (!cid || seen.has(cid)) continue;
          seen.add(cid);
          advanced = true;
          const rec = this.chunkTexts.get(cid);
          if (rec) ctxChunks.push({ chunkId: cid, text: rec.text, docName: rec.docName, score: s.score });
          if (ctxChunks.length >= chunkTopK) break outer;
        }
        if (!advanced) break;
      }
    }

    return {
      triples,
      seeds: seeds.map((s) => ({ name: s.node.name, type: s.node.type, score: s.score })),
      ctxChunks,
    };
  }

  /** 시각화용 데이터 (노드 400 · 엣지 800 상한) */
  vizData(maxNodes = 400, maxEdges = 800) {
    let nodes = [...this.nodes.values()];
    nodes.sort((a, b) => (b.degree + b.weight) - (a.degree + a.weight));
    const keep = new Set(nodes.slice(0, maxNodes).map((n) => normName(n.name)));
    const vNodes = nodes.slice(0, maxNodes).map((n) => ({
      key: normName(n.name),
      name: n.name,
      type: n.type || '기타',
      desc: n.desc,
      degree: n.degree,
      weight: n.weight,
    }));
    const vEdges = [...this.edges.values()]
      .filter((e) => keep.has(e.sourceKey) && keep.has(e.targetKey))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, maxEdges)
      .map((e) => ({ source: e.sourceKey, target: e.targetKey, relation: e.relation, weight: e.weight }));
    return {
      nodes: vNodes,
      edges: vEdges,
      truncated: this.nodes.size > maxNodes || this.edges.size > vEdges.length,
    };
  }

  /** 특정 노드의 인접 관계 목록 */
  neighbors(nodeKey) {
    const out = [];
    for (const e of this.edges.values()) {
      if (e.sourceKey === nodeKey) out.push({ dir: 'out', other: e.target, relation: e.relation });
      else if (e.targetKey === nodeKey) out.push({ dir: 'in', other: e.source, relation: e.relation });
    }
    return out;
  }
}

/** 상태를 IndexedDB에 원자적으로 기록 */
async function persistState(state, embedMeta = {}) {
  await idb.atomicWrite([
    { store: 'nodes', clear: true, put: [...state.nodes.values()] },
    { store: 'edges', clear: true, put: [...state.edges.values()] },
    {
      store: 'meta',
      put: [
        { key: 'graphChunks', value: Object.fromEntries(state.chunkTexts) },
        { key: 'graphBuiltAt', value: Date.now(), model: embedMeta.model || null, dim: embedMeta.dim || 0 },
      ],
    },
  ]);
}

/** LLM JSON 응답 → {entities, relations} (관대한 파싱) */
export function parseExtraction(raw) {
  const empty = { entities: [], relations: [] };
  if (!raw) return empty;
  let text = raw.trim();
  // 코드펜스 제거
  text = text.replace(/^```(?:json)?/m, '').replace(/```\s*$/m, '').trim();
  // 첫 { ~ 마지막 } 만 취함
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start < 0 || end <= start) return empty;
  try {
    const obj = JSON.parse(text.slice(start, end + 1));
    return {
      entities: Array.isArray(obj.entities) ? obj.entities : [],
      relations: Array.isArray(obj.relations) ? obj.relations : [],
    };
  } catch {
    return empty;
  }
}
