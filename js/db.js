// ═══════════════════════════════════════════════
// IndexedDB 영속화 계층
//  - docs   : 업로드 문서 메타
//  - chunks : 벡터 청크 (text + Float32Array 임베딩)
//  - nodes  : 그래프 개체 노드
//  - edges  : 그래프 관계 엣지
//  - meta   : 구축 시각 등 부가 정보
// ═══════════════════════════════════════════════

const DB_NAME = 'raglab';
const DB_VER = 1;
let dbPromise = null;

function open() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VER);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('docs')) db.createObjectStore('docs', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('chunks')) {
        const s = db.createObjectStore('chunks', { keyPath: 'id' });
        s.createIndex('docId', 'docId');
      }
      if (!db.objectStoreNames.contains('nodes')) db.createObjectStore('nodes', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('edges')) db.createObjectStore('edges', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta', { keyPath: 'key' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx(db, store, mode) {
  return db.transaction(store, mode).objectStore(store);
}

function reqAsPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export const idb = {
  async getAll(store) {
    const db = await open();
    return reqAsPromise(tx(db, store, 'readonly').getAll());
  },

  async get(store, key) {
    const db = await open();
    return reqAsPromise(tx(db, store, 'readonly').get(key));
  },

  async put(store, value) {
    const db = await open();
    return reqAsPromise(tx(db, store, 'readwrite').put(value));
  },

  async bulkPut(store, values) {
    if (!values.length) return;
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, 'readwrite');
      const s = t.objectStore(store);
      for (const v of values) s.put(v);
      t.oncomplete = () => resolve();
      t.onerror = () => reject(t.error);
    });
  },

  async delete(store, key) {
    const db = await open();
    return reqAsPromise(tx(db, store, 'readwrite').delete(key));
  },

  async bulkDelete(store, keys) {
    if (!keys.length) return;
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, 'readwrite');
      const s = t.objectStore(store);
      for (const k of keys) s.delete(k);
      t.oncomplete = () => resolve();
      t.onerror = () => reject(t.error);
    });
  },

  async clear(store) {
    const db = await open();
    return reqAsPromise(tx(db, store, 'readwrite').clear());
  },

  /**
   * 여러 스토어에 걸친 clear/delete/put을 하나의 트랜잭션으로 수행 (원자성 보장).
   * @param {Array<{store:string, clear?:boolean, deleteKeys?:any[], put?:any[]}>} ops
   */
  async atomicWrite(ops) {
    const db = await open();
    const names = [...new Set(ops.map((o) => o.store))];
    return new Promise((resolve, reject) => {
      const t = db.transaction(names, 'readwrite');
      for (const op of ops) {
        const s = t.objectStore(op.store);
        if (op.clear) s.clear();
        for (const k of op.deleteKeys || []) s.delete(k);
        for (const v of op.put || []) s.put(v);
      }
      t.oncomplete = () => resolve();
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error || new Error('트랜잭션 중단'));
    });
  },

  async clearAll() {
    for (const s of ['docs', 'chunks', 'nodes', 'edges', 'meta']) await this.clear(s);
  },
};
