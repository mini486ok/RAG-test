// ═══════════════════════════════════════════════
// 문장 경계 인식 청킹 (한국어 대응)
//  - 문장 단위로 누적하여 chunkSize를 넘지 않게 분할
//  - overlap 만큼 직전 청크 꼬리를 가져와 문맥 연결
// ═══════════════════════════════════════════════

/** 텍스트 → 문장 배열 (한국어 종결어미/구두점 기준) */
export function splitSentences(text) {
  const out = [];
  // 문단 우선 분리 후 문장 분리
  for (const para of text.split(/\n{2,}/)) {
    const p = para.trim();
    if (!p) continue;
    // 마침표/물음표/느낌표+공백, "다." 종결, 줄바꿈 기준 (lookbehind 미사용 — Safari 구버전 호환)
    const marked = p
      .replace(/([.!?。])\s+/g, '$1\u0000')
      .replace(/(다\.)(?![\d)])\s*/g, '$1\u0000');
    const pieces = marked.split(/\u0000|\n/);
    for (const s of pieces) {
      const t = s.trim();
      if (t) out.push(t);
    }
  }
  return out;
}

/**
 * @param {string} text
 * @param {{chunkSize:number, overlap:number}} opt
 * @returns {string[]}
 */
export function chunkText(text, { chunkSize = 600, overlap = 90 } = {}) {
  const sentences = splitSentences(text);
  if (!sentences.length) return [];

  const chunks = [];
  let cur = [];
  let curLen = 0;

  const flush = () => {
    if (!cur.length) return;
    chunks.push(cur.join(' '));
    if (overlap > 0) {
      // 꼬리에서 overlap 글자만큼 문장 단위로 유지
      let tail = [];
      let tailLen = 0;
      for (let i = cur.length - 1; i >= 0; i--) {
        tailLen += cur[i].length;
        tail.unshift(cur[i]);
        if (tailLen >= overlap) break;
      }
      // 목표치 과도 초과 방지 (한 문장이 길 때)
      while (tail.length > 1 && tailLen > overlap * 1.5) {
        tailLen -= tail[0].length;
        tail.shift();
      }
      // 전체가 다시 들어가는 무한루프 방지: 최소 1문장은 새로 시작
      if (tail.length >= cur.length) tail = cur.slice(-1);
      cur = [...tail];
      curLen = cur.reduce((a, s) => a + s.length, 0);
    } else {
      cur = [];
      curLen = 0;
    }
  };

  for (const s of sentences) {
    // 한 문장이 청크보다 길면 강제 절단
    if (s.length > chunkSize) {
      flush();
      for (let i = 0; i < s.length; i += chunkSize) {
        chunks.push(s.slice(i, i + chunkSize));
      }
      cur = [];
      curLen = 0;
      continue;
    }
    if (curLen + s.length + 1 > chunkSize && cur.length) flush();
    cur.push(s);
    curLen += s.length + 1;
  }
  if (cur.length) chunks.push(cur.join(' '));

  // 마지막 청크가 overlap 꼬리와 동일한 중복인 경우 제거
  if (chunks.length >= 2) {
    const last = chunks[chunks.length - 1];
    const prev = chunks[chunks.length - 2];
    if (prev.endsWith(last)) chunks.pop();
  }
  return chunks.filter((c) => c.trim().length >= 20); // 지나치게 짧은 조각 제거
}
