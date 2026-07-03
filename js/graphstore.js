// ═══════════════════════════════════════════════
// Graph Store (지식그래프)
//  - 구축: 청크별 LLM 개체·관계 추출(JSON) → 병합 → 개체 임베딩
//  - 검색: 질의 임베딩 ↔ 개체 임베딩 + 어휘 매칭 → n-hop 확장
//  - 저장: IndexedDB (nodes / edges)
// ═══════════════════════════════════════════════

import { idb } from './db.js';
import { chunkText } from './chunker.js';
import { dot } from './ollama.js';
import { buildExtractionPrompt, buildGleaningPrompt } from './config.js';
import { uid } from './ui.js';

function normName(name) {
  return name.trim().replace(/\s+/g, ' ').toLowerCase();
}

export class GraphStore {
  constructor() {
    this.nodes = new Map(); // normName → {id, name, type, desc, degree, weight, chunkIds:[], docIds:[], vec}
    this.edges = new Map(); // key → {id, source, target, relation, desc, weight, chunkIds:[]}
    this.chunkTexts = new Map(); // chunkId → {text, docName}
  }

  async load() {
    const [nodes, edges, meta] = await Promise.all([
      idb.getAll('nodes'),
      idb.getAll('edges'),
      idb.get('meta', 'graphChunks'),
    ]);
    this.nodes = new Map(
      nodes.map((n) => [normName(n.name), { ...n, vec: n.vec ? (n.vec instanceof Float32Array ? n.vec : new Float32Array(n.vec)) : null }])
    );
    this.edges = new Map(edges.map((e) => [e.id, e]));
    this.chunkTexts = new Map(Object.entries(meta?.value || {}));
  }

  get nodeCount() { return this.nodes.size; }
  get edgeCount() { return this.edges.size; }

  typeSet() {
    return [...new Set([...this.nodes.values()].map((n) => n.type || '기타'))];
  }

  /**
   * 그래프 구축.
   * @param {Array<{id,name,text}>} docs
   * @param {{maxTriples,gleaning,extractChunkSize,entityTypes}} params
   * @param {OllamaClient} client
   * @param {{chatModel, embedModel, numCtx}} models
   * @param {(done,total,stage)=>void} onProgress
   * @param {AbortSignal} signal
   */
  async build(docs, params, client, models, onProgress, signal) {
    // 재구축 대상 문서의 기존 흔적 제거
    for (const doc of docs) this.removeDocLocal(doc.id);

    // 1) 추출용 청킹 (벡터 청크보다 큰 단위)
    const units = [];
    for (const doc of docs) {
      const pieces = chunkText(doc.text, { chunkSize: params.extractChunkSize, overlap: 0 });
      pieces.forEach((text, i) => units.push({ chunkId: `${doc.id}-g${i}`, docId: doc.id, docName: doc.name, text }));
    }
    if (!units.length) throw new Error('추출할 텍스트가 없습니다.');

    // 2) 청크별 LLM 추출
    let done = 0;
    for (const unit of units) {
      if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
      this.chunkTexts.set(unit.chunkId, { text: unit.text, docName: unit.docName });

      const messages = [
        { role: 'user', content: buildExtractionPrompt(unit.text, params.entityTypes, params.maxTriples) },
      ];
      try {
        const raw = await client.chatJSON({
          model: models.chatModel,
          messages,
          options: { temperature: 0.1, num_ctx: models.numCtx, num_predict: 1200 },
          signal,
        });
        this.mergeExtraction(parseExtraction(raw), unit, params.entityTypes);

        // gleaning: 추가 추출 반복
        for (let g = 0; g < (params.gleaning || 0); g++) {
          if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
          const gRaw = await client.chatJSON({
            model: models.chatModel,
            messages: [...messages, { role: 'assistant', content: raw }, { role: 'user', content: buildGleaningPrompt() }],
            options: { temperature: 0.1, num_ctx: models.numCtx, num_predict: 800 },
            signal,
          });
          this.mergeExtraction(parseExtraction(gRaw), unit, params.entityTypes);
        }
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        console.warn('추출 실패(스킵):', unit.chunkId, e.message);
      }
      done++;
      onProgress?.(done, units.length, '개체·관계 추출');
    }

    if (!this.nodes.size) throw new Error('추출된 개체가 없습니다. 모델 또는 문서를 확인하세요.');

    // 3) 개체 임베딩 (검색용)
    const nodeArr = [...this.nodes.values()].filter((n) => !n.vec);
    const batch = 12;
    for (let i = 0; i < nodeArr.length; i += batch) {
      if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
      const slice = nodeArr.slice(i, i + batch);
      const inputs = slice.map((n) => `${n.name} (${n.type}): ${n.desc || ''}`.slice(0, 500));
      const vecs = await client.embed({ model: models.embedModel, input: inputs, signal });
      slice.forEach((n, j) => (n.vec = vecs[j]));
      onProgress?.(Math.min(i + batch, nodeArr.length), nodeArr.length, '개체 임베딩');
    }

    await this.persist();
    return { nodes: this.nodes.size, edges: this.edges.size };
  }

  mergeExtraction(ext, unit, entityTypes) {
    const typeSet = new Set(entityTypes);
    for (const e of ext.entities) {
      if (!e.name || typeof e.name !== 'string') continue;
      const key = normName(e.name);
      if (!key || key.length > 60) continue;
      let node = this.nodes.get(key);
      if (!node) {
        node = {
          id: uid(),
          name: e.name.trim(),
          type: typeSet.has(e.type) ? e.type : (e.type || '개념/용어'),
          desc: (e.description || '').slice(0, 300),
          degree: 0,
          weight: 0,
          chunkIds: [],
          docIds: [],
          vec: null,
        };
        this.nodes.set(key, node);
      } else if (e.description && node.desc.length < 240 && !node.desc.includes(e.description.slice(0, 40))) {
        node.desc = (node.desc + ' ' + e.description).slice(0, 300);
      }
      node.weight++;
      if (!node.chunkIds.includes(unit.chunkId)) node.chunkIds.push(unit.chunkId);
      if (!node.docIds.includes(unit.docId)) node.docIds.push(unit.docId);
    }

    for (const r of ext.relations) {
      if (!r.source || !r.target || !r.relation) continue;
      const sKey = normName(String(r.source));
      const tKey = normName(String(r.target));
      if (!sKey || !tKey || sKey === tKey) continue;
      // 관계에만 등장한 개체도 노드로 승격
      for (const [k, orig] of [[sKey, r.source], [tKey, r.target]]) {
        if (!this.nodes.has(k)) {
          this.nodes.set(k, {
            id: uid(), name: String(orig).trim(), type: '개념/용어', desc: '',
            degree: 0, weight: 1, chunkIds: [unit.chunkId], docIds: [unit.docId], vec: null,
          });
        }
      }
      const eKey = `${sKey}→${String(r.relation).trim()}→${tKey}`;
      let edge = this.edges.get(eKey);
      if (!edge) {
        edge = {
          id: eKey,
          source: this.nodes.get(sKey).name,
          target: this.nodes.get(tKey).name,
          sourceKey: sKey,
          targetKey: tKey,
          relation: String(r.relation).trim().slice(0, 40),
          desc: (r.description || '').slice(0, 200),
          weight: 0,
          chunkIds: [],
        };
        this.edges.set(eKey, edge);
        this.nodes.get(sKey).degree++;
        this.nodes.get(tKey).degree++;
      }
      edge.weight++;
      if (!edge.chunkIds.includes(unit.chunkId)) edge.chunkIds.push(unit.chunkId);
    }
  }

  async persist() {
    await idb.clear('nodes');
    await idb.clear('edges');
    await idb.bulkPut('nodes', [...this.nodes.values()]);
    await idb.bulkPut('edges', [...this.edges.values()]);
    await idb.put('meta', { key: 'graphChunks', value: Object.fromEntries(this.chunkTexts) });
    await idb.put('meta', { key: 'graphBuiltAt', value: Date.now() });
  }

  removeDocLocal(docId) {
    for (const [key, node] of this.nodes) {
      node.docIds = node.docIds.filter((d) => d !== docId);
      node.chunkIds = node.chunkIds.filter((c) => !c.startsWith(docId + '-g'));
      if (!node.docIds.length) this.nodes.delete(key);
    }
    for (const [key, edge] of this.edges) {
      edge.chunkIds = edge.chunkIds.filter((c) => !c.startsWith(docId + '-g'));
      if (!edge.chunkIds.length || !this.nodes.has(edge.sourceKey) || !this.nodes.has(edge.targetKey)) {
        this.edges.delete(key);
      }
    }
    for (const cid of [...this.chunkTexts.keys()]) {
      if (cid.startsWith(docId + '-g')) this.chunkTexts.delete(cid);
    }
    // degree 재계산
    for (const n of this.nodes.values()) n.degree = 0;
    for (const e of this.edges.values()) {
      const s = this.nodes.get(e.sourceKey);
      const t = this.nodes.get(e.targetKey);
      if (s) s.degree++;
      if (t) t.degree++;
    }
  }

  async removeDoc(docId) {
    this.removeDocLocal(docId);
    await this.persist();
  }

  async clear() {
    await idb.clear('nodes');
    await idb.clear('edges');
    await idb.delete('meta', 'graphChunks');
    this.nodes.clear();
    this.edges.clear();
    this.chunkTexts.clear();
  }

  /**
   * 그래프 검색: 시드 개체(임베딩+어휘) → n-hop 확장 → 트리플 랭킹
   * @returns {{triples:Array, seeds:Array, ctxChunks:Array}}
   */
  search(query, queryVec, { topKEntities = 6, hops = 2, maxCtxTriples = 30, chunkTopK = 2 } = {}) {
    if (!this.nodes.size) return { triples: [], seeds: [], ctxChunks: [] };
    const qLower = query.toLowerCase();

    // 1) 시드 개체 점수: 임베딩 유사도 + 어휘 포함 부스트
    const scored = [];
    for (const node of this.nodes.values()) {
      let score = node.vec && queryVec ? dot(queryVec, node.vec) : 0;
      const nLower = node.name.toLowerCase();
      if (qLower.includes(nLower) && nLower.length >= 2) score += 0.35;
      scored.push({ node, score });
    }
    scored.sort((a, b) => b.score - a.score);
    const seeds = scored.slice(0, topKEntities).filter((s) => s.score > 0.15);
    if (!seeds.length) return { triples: [], seeds: [], ctxChunks: [] };

    const seedScore = new Map(seeds.map((s) => [normName(s.node.name), s.score]));

    // 2) n-hop 확장 (BFS, 거리 감쇠)
    const nodeDist = new Map([...seedScore.keys()].map((k) => [k, 0]));
    let frontier = [...seedScore.keys()];
    for (let h = 1; h <= hops; h++) {
      const next = [];
      for (const e of this.edges.values()) {
        const sIn = nodeDist.has(e.sourceKey);
        const tIn = nodeDist.has(e.targetKey);
        if (sIn && !nodeDist.has(e.targetKey)) { nodeDist.set(e.targetKey, h); next.push(e.targetKey); }
        else if (tIn && !nodeDist.has(e.sourceKey)) { nodeDist.set(e.sourceKey, h); next.push(e.sourceKey); }
      }
      frontier = next;
      if (!frontier.length) break;
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

    const triples = rankedEdges.slice(0, maxCtxTriples).map(({ edge, rank }) => ({
      source: edge.source,
      target: edge.target,
      relation: edge.relation,
      desc: edge.desc,
      rank,
      chunkIds: edge.chunkIds,
    }));

    // 4) 시드 개체 관련 원문 청크
    const ctxChunks = [];
    const seen = new Set();
    for (const s of seeds) {
      for (const cid of s.node.chunkIds) {
        if (seen.has(cid)) continue;
        seen.add(cid);
        const rec = this.chunkTexts.get(cid);
        if (rec) ctxChunks.push({ chunkId: cid, text: rec.text, docName: rec.docName, score: s.score });
        if (ctxChunks.length >= chunkTopK) break;
      }
      if (ctxChunks.length >= chunkTopK) break;
    }

    return {
      triples,
      seeds: seeds.map((s) => ({ name: s.node.name, type: s.node.type, score: s.score })),
      ctxChunks,
    };
  }

  /** 시각화용 데이터 */
  vizData(maxNodes = 400) {
    let nodes = [...this.nodes.values()];
    // 연결도 높은 순으로 제한
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
      .map((e) => ({ source: e.sourceKey, target: e.targetKey, relation: e.relation, weight: e.weight }));
    return { nodes: vNodes, edges: vEdges, truncated: this.nodes.size > maxNodes };
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
