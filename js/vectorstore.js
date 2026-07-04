// ═══════════════════════════════════════════════
// Vector Store
//  - 구축: 청킹 → bge-m3 임베딩(배치) → IndexedDB
//  - 검색: 코사인 Top-K (+ MMR 재순위화)
//  - 시각화: 파워이터레이션 PCA 2D 투영
// ═══════════════════════════════════════════════

import { idb } from './db.js';
import { chunkText } from './chunker.js';
import { dot } from './ollama.js';
import { uid } from './ui.js';

export class VectorStore {
  constructor() {
    this.chunks = []; // {id, docId, docName, seq, text, vec:Float32Array}
    this.dim = 0;
  }

  async load() {
    const rows = await idb.getAll('chunks');
    this.chunks = rows
      .map((r) => ({ ...r, vec: r.vec instanceof Float32Array ? r.vec : new Float32Array(r.vec) }))
      .sort((a, b) => a.docId.localeCompare(b.docId) || a.seq - b.seq);
    this.dim = this.chunks[0]?.vec.length || 0;
    const meta = await idb.get('meta', 'vectorBuiltAt');
    this.embedModel = meta?.model || null;
  }

  get size() {
    return this.chunks.length;
  }

  docIds() {
    return [...new Set(this.chunks.map((c) => c.docId))];
  }

  /**
   * 문서들로부터 벡터 DB 구축 (기존 청크 중 같은 문서는 교체)
   * @param {Array<{id,name,text}>} docs
   * @param {{chunkSize,overlap,embedBatch}} params
   * @param {OllamaClient} client
   * @param {string} embedModel
   * @param {(done:number,total:number,stage:string)=>void} onProgress
   * @param {AbortSignal} signal
   */
  async build(docs, params, client, embedModel, onProgress, signal, replaceAll = false) {
    // 1) 청킹
    const newChunks = [];
    for (const doc of docs) {
      const pieces = chunkText(doc.text, params);
      pieces.forEach((text, i) => {
        newChunks.push({ id: uid(), docId: doc.id, docName: doc.name, seq: i, text });
      });
    }
    if (!newChunks.length) throw new Error('청킹 결과가 비어 있습니다. 문서 내용을 확인하세요.');

    // 2) 임베딩 (배치)
    const batch = Math.max(1, params.embedBatch || 12);
    for (let i = 0; i < newChunks.length; i += batch) {
      if (signal?.aborted) throw new DOMException('중단됨', 'AbortError');
      const slice = newChunks.slice(i, i + batch);
      const vecs = await client.embed({ model: embedModel, input: slice.map((c) => c.text), signal });
      if (vecs.length !== slice.length) throw new Error('임베딩 개수가 일치하지 않습니다.');
      slice.forEach((c, j) => (c.vec = vecs[j]));
      onProgress?.(Math.min(i + batch, newChunks.length), newChunks.length, '임베딩');
    }

    // 3) 저장: 단일 트랜잭션 (원자성) — 전체 재구축이어도 성공 시에만 기존 데이터 교체
    const rebuiltDocIds = new Set(docs.map((d) => d.id));
    const stale = this.chunks.filter((c) => rebuiltDocIds.has(c.docId)).map((c) => c.id);
    await idb.atomicWrite([
      replaceAll
        ? { store: 'chunks', clear: true, put: newChunks }
        : { store: 'chunks', deleteKeys: stale, put: newChunks },
      { store: 'meta', put: [{ key: 'vectorBuiltAt', value: Date.now(), model: embedModel, dim: newChunks[0].vec.length, params: { chunkSize: params.chunkSize, overlap: params.overlap } }] },
    ]);

    this.chunks = replaceAll
      ? newChunks
      : [...this.chunks.filter((c) => !rebuiltDocIds.has(c.docId)), ...newChunks];
    this.dim = this.chunks[0]?.vec.length || 0;
    this.embedModel = embedModel;
    return newChunks.length;
  }

  async removeDoc(docId) {
    const ids = this.chunks.filter((c) => c.docId === docId).map((c) => c.id);
    await idb.bulkDelete('chunks', ids);
    this.chunks = this.chunks.filter((c) => c.docId !== docId);
  }

  async clear() {
    await idb.clear('chunks');
    this.chunks = [];
    this.dim = 0;
  }

  /**
   * 검색: 코사인 Top-K + (옵션) MMR
   * @returns {Array<{chunk, score}>}
   */
  search(queryVec, { topK = 4, minScore = 0, mmr = true, mmrLambda = 0.6 } = {}) {
    if (!this.chunks.length) return [];
    if (!queryVec || !queryVec.length) throw new Error('질의 임베딩이 비어 있습니다. 임베딩 모델을 확인하세요.');
    if (this.dim && queryVec.length !== this.dim) {
      throw new Error(`임베딩 차원 불일치 (DB ${this.dim} vs 질의 ${queryVec.length}). 임베딩 모델이 바뀌었으면 Vector DB를 다시 구축하세요.`);
    }
    const scored = this.chunks
      .map((chunk) => ({ chunk, score: dot(queryVec, chunk.vec) }))
      .filter((r) => r.score >= minScore)
      .sort((a, b) => b.score - a.score);

    if (!mmr || scored.length <= topK) return scored.slice(0, topK);

    // MMR: 후보 풀(topK*4)에서 관련성-다양성 균형 선택
    const pool = scored.slice(0, Math.min(scored.length, topK * 4));
    const selected = [];
    while (selected.length < topK && pool.length) {
      let bestIdx = 0;
      let bestVal = -Infinity;
      for (let i = 0; i < pool.length; i++) {
        const rel = pool[i].score;
        let maxSim = 0;
        for (const s of selected) {
          const sim = dot(pool[i].chunk.vec, s.chunk.vec);
          if (sim > maxSim) maxSim = sim;
        }
        const val = mmrLambda * rel - (1 - mmrLambda) * maxSim;
        if (val > bestVal) {
          bestVal = val;
          bestIdx = i;
        }
      }
      selected.push(pool.splice(bestIdx, 1)[0]);
    }
    return selected;
  }

  /**
   * PCA 2D 투영 (파워 이터레이션, 공분산 행렬 비생성 — O(n·d·iter))
   * @returns {Array<{x,y,chunk}>}
   */
  project2D(maxPoints = 1500) {
    const n = this.chunks.length;
    if (n < 2) return [];
    const sample = n > maxPoints
      ? this.chunks.filter((_, i) => i % Math.ceil(n / maxPoints) === 0)
      : this.chunks;
    const m = sample.length;
    const d = this.dim;

    // 평균 벡터
    const mean = new Float64Array(d);
    for (const c of sample) for (let j = 0; j < d; j++) mean[j] += c.vec[j];
    for (let j = 0; j < d; j++) mean[j] /= m;

    // X·v (중심화 적용) 및 Xᵀ·(Xv) 계산 헬퍼
    const centered = (c, j) => c.vec[j] - mean[j];
    const covMul = (v, deflateWith) => {
      const out = new Float64Array(d);
      for (const c of sample) {
        let s = 0;
        for (let j = 0; j < d; j++) s += centered(c, j) * v[j];
        for (let j = 0; j < d; j++) out[j] += centered(c, j) * s;
      }
      for (let j = 0; j < d; j++) out[j] /= m;
      if (deflateWith) {
        // 1차 성분 제거(직교화)
        let proj = 0;
        for (let j = 0; j < d; j++) proj += out[j] * deflateWith[j];
        for (let j = 0; j < d; j++) out[j] -= proj * deflateWith[j];
      }
      return out;
    };

    const powerIter = (deflateWith) => {
      let v = new Float64Array(d);
      for (let j = 0; j < d; j++) v[j] = Math.sin(j * 12.9898 + 78.233) * 0.5; // 결정적 초기값
      if (deflateWith) {
        let proj = 0;
        for (let j = 0; j < d; j++) proj += v[j] * deflateWith[j];
        for (let j = 0; j < d; j++) v[j] -= proj * deflateWith[j];
      }
      for (let it = 0; it < 16; it++) {
        const nv = covMul(v, deflateWith);
        let norm = 0;
        for (let j = 0; j < d; j++) norm += nv[j] * nv[j];
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < d; j++) nv[j] /= norm;
        v = nv;
      }
      return v;
    };

    const pc1 = powerIter(null);
    const pc2 = powerIter(pc1);

    return sample.map((chunk) => {
      let x = 0, y = 0;
      for (let j = 0; j < d; j++) {
        const cj = centered(chunk, j);
        x += cj * pc1[j];
        y += cj * pc2[j];
      }
      return { x, y, chunk };
    });
  }
}
