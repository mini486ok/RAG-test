// ═══════════════════════════════════════════════
// 전역 설정 · 기본값 · 프롬프트 템플릿
// ═══════════════════════════════════════════════

export const DEFAULTS = {
  ollamaUrl: 'http://localhost:11434',
  chatModel: 'exaone3.5:7.8b',
  embedModel: 'bge-m3:latest',
  temperature: 0.3,
  numCtx: 8192,
  numPredict: 1024,

  vector: {
    chunkSize: 600,     // 자 (한국어 문서 500~800자가 통용 최적)
    overlap: 90,        // 자 (~15%)
    topK: 4,
    minScore: 0.3,
    mmr: true,
    mmrLambda: 0.6,
    embedBatch: 12,
  },

  graph: {
    maxTriples: 12,
    hops: 2,
    topKEntities: 6,
    maxCtxTriples: 30,
    chunkTopK: 2,
    gleaning: 0,
    extractChunkSize: 1200, // 그래프 추출용 청크(벡터 청크보다 크게)
    entityTypes: ['시설/설비', '차량', '기술/시스템', '규정/법령', '조직/기관', '직무/인물', '사고/장애', '절차/업무', '개념/용어', '위치/노선'],
  },
};

export const MODES = {
  basic: {
    key: 'basic',
    name: 'Basic LLM',
    desc: '검색 없이 모델 자체 지식으로 답변',
  },
  vector: {
    key: 'vector',
    name: 'Vector RAG',
    desc: '임베딩 유사도 검색으로 찾은 문서 청크를 근거로 답변',
  },
  graph: {
    key: 'graph',
    name: 'Graph RAG',
    desc: '지식그래프(개체·관계) 탐색 결과를 근거로 답변',
  },
};

export const EXAMPLE_QUESTIONS = [
  'ATS와 ATC, ATP의 차이점을 설명해줘',
  '선로전환기는 어떤 설비와 연동되어 동작하는가?',
  '선로전환기 장애 발생 시 조치 절차는?',
  '관제사와 기관사는 비상 시 어떤 지휘·보고 관계인가?',
  '철도안전법상 철도종사자의 음주 기준은?',
  '전차선 단전 시 기관사가 취해야 할 조치는?',
  'KTX와 SRT의 공통점과 차이점은?',
  '철도차량 일상검사와 정기검사는 어떻게 다른가?',
];

// 철도 약어·동의어 별칭 사전 — Graph 검색 시 질의 확장에 사용
export const RAIL_ALIASES = [
  ['ATS', '열차자동정지장치', '자동열차정지장치'],
  ['ATC', '열차자동제어장치', '자동열차제어장치'],
  ['ATP', '열차자동방호장치', '자동열차방호장치'],
  ['ATO', '열차자동운전장치', '자동열차운전장치'],
  ['판토그래프', '팬터그래프', '집전장치'],
  ['선로전환기', '분기기', '포인트'],
  ['EMU', '동력분산식'],
  ['CBM', '상태기반 유지보수', '상태기반유지보수'],
  ['SMS', '안전관리체계', '철도안전관리체계'],
  ['KTX', 'Korea Train eXpress', '한국고속철도'],
  ['궤도회로', '트랙서킷'],
  ['연동장치', '인터로킹', 'interlocking'],
  ['전차선', '가공전차선', '카테너리'],
  ['ETCS', '유럽열차제어시스템'],
  ['UTO', '무인운전'],
];

// ── 프롬프트 템플릿 ──────────────────────────────

export const SYSTEM_PROMPT_BASE =
  '당신은 철도교통 분야(철도 시스템, 차량, 신호통신, 시설, 운영, 안전 규정)에 정통한 전문 AI 어시스턴트입니다. ' +
  '정확하고 신뢰할 수 있는 정보를 한국어로, 간결하고 구조적으로 답변하십시오. ' +
  '확실하지 않은 내용은 추측하지 말고 불확실하다고 명시하십시오.';

export function buildBasicMessages(sysPrompt, query) {
  return [
    { role: 'system', content: sysPrompt },
    { role: 'user', content: query },
  ];
}

export function buildVectorMessages(sysPrompt, query, ctxChunks) {
  const ctx = ctxChunks
    .map((c, i) => `[근거 ${i + 1}] (출처: ${c.docName}, 유사도 ${c.score.toFixed(3)})\n${c.text}`)
    .join('\n\n');
  const user =
    `다음은 질문과 관련하여 지식 문서에서 검색된 근거 자료입니다.\n` +
    `--- 근거 자료 시작 ---\n${ctx}\n--- 근거 자료 끝 ---\n\n` +
    `위 근거 자료를 우선적으로 활용하여 아래 질문에 답하십시오. ` +
    `근거를 인용할 때는 [근거 N] 형태로 표시하고, 근거 자료에 없는 내용은 일반 지식임을 밝히십시오.\n\n` +
    `질문: ${query}`;
  return [
    { role: 'system', content: sysPrompt },
    { role: 'user', content: user },
  ];
}

export function buildGraphMessages(sysPrompt, query, triples, ctxChunks) {
  const tripleText = triples
    .map((t, i) => `${i + 1}. (${t.source}) -[${t.relation}]-> (${t.target})${t.desc ? ' : ' + t.desc : ''}`)
    .join('\n');
  const chunkText = ctxChunks.length
    ? `\n\n--- 관련 원문 청크 ---\n` +
      ctxChunks.map((c, i) => `[원문 ${i + 1}] (출처: ${c.docName})\n${c.text}`).join('\n\n')
    : '';
  const user =
    `다음은 질문과 관련하여 지식그래프에서 탐색된 개체·관계 정보입니다.\n` +
    `--- 지식그래프 정보 시작 ---\n${tripleText}${chunkText}\n--- 지식그래프 정보 끝 ---\n\n` +
    `위 지식그래프의 관계 정보를 우선적으로 활용하여 아래 질문에 답하십시오. ` +
    `개체 간 관계와 맥락을 종합하여 설명하고, 그래프에 없는 내용은 일반 지식임을 밝히십시오.\n\n` +
    `질문: ${query}`;
  return [
    { role: 'system', content: sysPrompt },
    { role: 'user', content: user },
  ];
}

// 지식그래프 추출 프롬프트
export function buildExtractionPrompt(text, entityTypes, maxTriples) {
  return (
    `당신은 철도교통 도메인 지식그래프 구축 전문가입니다. 아래 텍스트에서 핵심 개체(entity)와 관계(relation)를 추출하십시오.\n\n` +
    `[개체 유형 목록]\n${entityTypes.join(', ')}\n\n` +
    `[규칙]\n` +
    `1. 개체명은 텍스트에 등장하는 간결한 명사구로 작성하고 조사(은/는/이/가 등)는 제거 (예: "ATS", "선로전환기", "철도안전법")\n` +
    `2. 관계는 최대 ${maxTriples}개까지, 텍스트에 명시된 사실만 추출\n` +
    `3. relation은 짧은 한국어 서술어로 작성 (예: "구성요소이다", "규정한다", "담당한다")\n` +
    `4. 유형 판정 기준: 물리적 장치·설비는 "시설/설비", 제어 방식·표준·소프트웨어는 "기술/시스템", 법률·규칙·기준은 "규정/법령"으로 일관되게 분류\n` +
    `5. 반드시 아래 JSON 형식으로만 출력 (설명·주석 금지)\n\n` +
    `[출력 형식]\n` +
    `{"entities":[{"name":"개체명","type":"개체유형","description":"한 문장 설명"}],` +
    `"relations":[{"source":"개체명","target":"개체명","relation":"관계","description":"한 문장 설명"}]}\n\n` +
    `[텍스트]\n${text}`
  );
}

export function buildGleaningPrompt() {
  return (
    `방금 추출에서 누락된 개체나 관계가 있는지 다시 확인하십시오. ` +
    `새로 발견된 것만 동일한 JSON 형식으로 출력하고, 없으면 {"entities":[],"relations":[]}를 출력하십시오.`
  );
}

export const LS_KEYS = {
  settings: 'raglab.settings.v1',
  history: 'raglab.history.v1',
  theme: 'raglab.theme',
};
