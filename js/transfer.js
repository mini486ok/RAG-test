// ═══════════════════════════════════════════════
// DB 스냅샷 내보내기/가져오기
//  - 문서·Vector 청크·Graph 노드/엣지·메타를 하나의 파일로 직렬화
//  - 임베딩(Float32Array)은 base64로 압축 없이도 JSON 대비 ~8배 절약
//  - CompressionStream 지원 시 gzip 압축 (.json.gz)
// ═══════════════════════════════════════════════

import { idb } from './db.js';

const FORMAT = 'railbrain-db';
const VERSION = 1;

// ── base64 ↔ Float32Array ──────────────────────
function vecToB64(vec) {
  const bytes = new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength);
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

function b64ToVec(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// ── 스냅샷 생성 ──────────────────────
/** @returns {Promise<{blob: Blob, stats: object}>} */
export async function exportSnapshot() {
  const [docs, chunks, nodes, edges, graphChunks, vMeta, gMeta] = await Promise.all([
    idb.getAll('docs'),
    idb.getAll('chunks'),
    idb.getAll('nodes'),
    idb.getAll('edges'),
    idb.get('meta', 'graphChunks'),
    idb.get('meta', 'vectorBuiltAt'),
    idb.get('meta', 'graphBuiltAt'),
  ]);

  if (!docs.length && !chunks.length && !nodes.length) {
    throw new Error('내보낼 데이터가 없습니다. 먼저 문서를 업로드하고 DB를 구축하세요.');
  }

  const payload = {
    format: FORMAT,
    version: VERSION,
    exportedAt: new Date().toISOString(),
    data: {
      docs,
      chunks: chunks.map((c) => ({ ...c, vec: c.vec ? vecToB64(c.vec instanceof Float32Array ? c.vec : new Float32Array(c.vec)) : null })),
      nodes: nodes.map((n) => ({ ...n, vec: n.vec ? vecToB64(n.vec instanceof Float32Array ? n.vec : new Float32Array(n.vec)) : null })),
      edges,
      meta: {
        graphChunks: graphChunks || null,
        vectorBuiltAt: vMeta || null,
        graphBuiltAt: gMeta || null,
      },
    },
  };

  const json = JSON.stringify(payload);
  let blob;
  let compressed = false;
  if (typeof CompressionStream === 'function') {
    const stream = new Blob([json]).stream().pipeThrough(new CompressionStream('gzip'));
    blob = await new Response(stream).blob();
    compressed = true;
  } else {
    blob = new Blob([json], { type: 'application/json' });
  }

  return {
    blob,
    stats: {
      docs: docs.length,
      chunks: chunks.length,
      nodes: nodes.length,
      edges: edges.length,
      bytes: blob.size,
      compressed,
    },
  };
}

// ── 스냅샷 복원 ──────────────────────
/**
 * @param {ArrayBuffer} buf 파일 내용 (gzip 또는 순수 JSON)
 * @returns {Promise<{docs,chunks,nodes,edges, embedModel}>}
 */
export async function importSnapshot(buf) {
  const bytes = new Uint8Array(buf);
  let json;
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    // gzip 매직 넘버
    if (typeof DecompressionStream !== 'function') {
      throw new Error('이 브라우저는 압축 해제를 지원하지 않습니다. 최신 Chrome/Edge를 사용하세요.');
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    json = await new Response(stream).text();
  } else {
    json = new TextDecoder().decode(buf);
  }

  let payload;
  try {
    payload = JSON.parse(json);
  } catch {
    throw new Error('파일 형식을 해석할 수 없습니다. Rail-Brain에서 내보낸 DB 파일인지 확인하세요.');
  }
  if (payload.format !== FORMAT) {
    throw new Error('Rail-Brain DB 파일이 아닙니다.');
  }
  if (payload.version > VERSION) {
    throw new Error(`이 파일은 더 새로운 버전(v${payload.version})에서 만들어졌습니다. 페이지를 새로고침해 최신 버전으로 사용하세요.`);
  }

  const d = payload.data || {};
  const docs = Array.isArray(d.docs) ? d.docs : [];
  const chunks = (Array.isArray(d.chunks) ? d.chunks : []).map((c) => ({
    ...c,
    vec: c.vec ? b64ToVec(c.vec) : null,
  }));
  const nodes = (Array.isArray(d.nodes) ? d.nodes : []).map((n) => ({
    ...n,
    vec: n.vec ? b64ToVec(n.vec) : null,
  }));
  const edges = Array.isArray(d.edges) ? d.edges : [];
  const meta = d.meta || {};

  // 기존 데이터를 전부 교체 (단일 트랜잭션 — 원자성)
  const metaPut = [];
  if (meta.graphChunks) metaPut.push(meta.graphChunks);
  if (meta.vectorBuiltAt) metaPut.push(meta.vectorBuiltAt);
  if (meta.graphBuiltAt) metaPut.push(meta.graphBuiltAt);
  await idb.atomicWrite([
    { store: 'docs', clear: true, put: docs },
    { store: 'chunks', clear: true, put: chunks },
    { store: 'nodes', clear: true, put: nodes },
    { store: 'edges', clear: true, put: edges },
    { store: 'meta', clear: true, put: metaPut },
  ]);

  return {
    docs: docs.length,
    chunks: chunks.length,
    nodes: nodes.length,
    edges: edges.length,
    embedModel: meta.vectorBuiltAt?.model || meta.graphBuiltAt?.model || null,
  };
}
