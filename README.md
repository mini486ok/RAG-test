# 🚄 RAIL·RAG LAB — 철도교통 특화 RAG 비교 관제실

**Basic LLM vs Vector RAG vs Graph RAG**를 한 화면에서 실시간으로 비교하는 철도교통 특화 실험 웹앱입니다.
서버 없이 **브라우저 + 로컬 [Ollama](https://ollama.com)** 만으로 동작하며, 모든 데이터는 브라우저(IndexedDB)에만 저장됩니다.

**▶ 바로 사용하기: <https://mini486ok.github.io/RAG-test/>**

![architecture](https://img.shields.io/badge/100%25-Client--Side-blue) ![ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-green) ![pages](https://img.shields.io/badge/Hosting-GitHub%20Pages-black)

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **3-way 실시간 비교** | 하나의 질문에 대해 Basic / Vector RAG / Graph RAG가 스트리밍으로 응답. 기본은 **순차 실행**(GPU 경합 없이 공정한 시간 측정), 옵션으로 동시 실행 가능 |
| **성능 계기판** | 총 응답시간 · 첫 토큰 지연(TTFT, 모델 로드 제외) · 검색 시간 · 입력/생성 토큰 · 생성 속도(tok/s) 비교 차트 |
| **문서 업로드 → 자동 DB 구축** | PDF · DOCX · PPTX · HWPX · MD · TXT 지원, 브라우저에서 직접 파싱 |
| **Vector DB** | 문장 인식 청킹 → bge-m3 임베딩 → 코사인 Top-K + MMR 재순위화 |
| **Graph DB** | LLM 기반 개체·관계 추출 → 지식그래프 구축 → 시드 개체 탐색 + n-hop 확장 |
| **파라미터 튜닝** | 청크 크기/중첩/Top-K/MMR, 트리플 수/홉/시드 개체 수 등 실시간 조정 |
| **DB 시각화** | 임베딩 PCA 산점도, 지식그래프 포스 다이렉티드 네트워크 (팬/줌/검색/상세) |
| **실험 기록** | 질의별 모드 성능 히스토리 자동 저장 |

## 빠른 시작

### 1. Ollama 준비 (필수)

```bash
# 모델 설치 (권장 조합)
ollama pull exaone3.5:7.8b   # 응답 생성 (한국어 특화)
ollama pull bge-m3           # 임베딩 (다국어 검색)
```

### 2. CORS 허용 (웹 배포판 사용 시 필수)

브라우저가 `https://…github.io`에서 `http://localhost:11434`로 직접 접속하므로 Ollama에 허용 오리진을 설정해야 합니다. **보안을 위해 특정 오리진만 허용하는 방식을 권장합니다.**

**Windows (PowerShell):**
```powershell
setx OLLAMA_ORIGINS "https://mini486ok.github.io"
# 이후 트레이 아이콘에서 Ollama 완전 종료 후 재실행
```

**macOS:**
```bash
launchctl setenv OLLAMA_ORIGINS "https://mini486ok.github.io"
# Ollama 재시작
```

**Linux (systemd):**
```bash
sudo systemctl edit ollama.service
# [Service] 섹션에 추가:
# Environment="OLLAMA_ORIGINS=https://mini486ok.github.io"
sudo systemctl restart ollama
```

> `"*"`로 설정하면 모든 웹사이트가 로컬 Ollama에 접근할 수 있게 되므로(무단 추론·모델 열람 위험) 임시 테스트 용도로만 쓰세요.
> 로컬 실행(`python -m http.server`) 시에는 localhost 오리진이 기본 허용되어 이 설정이 필요 없습니다.

**지원 브라우저**: Chrome/Edge 최신 버전 권장. Chrome의 "사설 네트워크 접근(PNA)" 경고가 뜨면 허용을 선택하세요. Safari 등 일부 브라우저는 https→localhost 연결이 제한될 수 있으며, 이 경우 로컬 실행 방식을 사용하세요.

### 3. 접속

<https://mini486ok.github.io/RAG-test/> 접속 → 우측 상단 연결 표시등이 **녹색**이면 준비 완료.

1. **[02 문서 · DB 구축]** 탭에서 문서 업로드 (또는 `🚄 철도 샘플 문서 로드` 클릭)
2. **전체 구축** 버튼으로 Vector + Graph DB 생성
3. **[01 비교 실험]** 탭에서 질문 입력 → 3개 방식의 응답과 성능을 실시간 비교
4. **[03 DB 시각화]** 탭에서 구축된 DB 구조 확인

### 로컬 실행 (선택)

```bash
git clone https://github.com/mini486ok/RAG-test.git
cd RAG-test
python -m http.server 8000
# http://localhost:8000 접속
```
> `file://`로 직접 열면 ES 모듈 제약으로 동작하지 않습니다. 반드시 HTTP 서버로 서빙하세요.
> Windows는 동봉된 `start.bat`을 더블클릭해도 됩니다.

## 아키텍처

```
┌────────────────────────── Browser (GitHub Pages) ──────────────────────────┐
│  UI (관제실 테마)                                                           │
│  ├─ parser.js      PDF/DOCX/PPTX/HWPX/MD/TXT → 텍스트 (pdf.js·mammoth·JSZip)│
│  ├─ chunker.js     문장 경계 인식 청킹                                       │
│  ├─ vectorstore.js 임베딩 검색 (코사인+MMR) · PCA 투영                       │
│  ├─ graphstore.js  LLM 개체·관계 추출 → 지식그래프 · n-hop 탐색              │
│  ├─ viz-*.js       Canvas 산점도 · 포스 그래프                              │
│  └─ IndexedDB      문서·청크·노드·엣지 영속화 (전부 로컬)                     │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ HTTP (localhost)
                    ┌──────────▼──────────┐
                    │   Ollama (로컬)      │
                    │  /api/chat  (생성)   │
                    │  /api/embed (임베딩) │
                    └─────────────────────┘
```

- **Basic**: 검색 없이 모델 자체 지식으로 답변
- **Vector RAG**: 질의 임베딩 → 유사 청크 Top-K(+MMR) → 근거 주입 답변
- **Graph RAG**: 질의 ↔ 개체 임베딩 매칭 + 어휘 부스트로 시드 선정 → n-hop 서브그래프 트리플 + 연계 원문 주입 답변

## 개인정보/보안

- 업로드한 문서, 생성된 DB, 실험 기록은 **모두 사용자 브라우저에만** 저장됩니다(IndexedDB/localStorage).
- 외부 서버로 전송되는 데이터는 없으며, LLM 호출도 로컬 Ollama로만 이루어집니다.
- CDN 라이브러리는 버전 고정 + SRI(무결성 해시) 검증으로 로드됩니다.
- 지표 해석 참고: `입력 토큰`은 Ollama의 `prompt_eval_count`로, 동일 접두부 프롬프트는 KV 캐시 재사용으로 실제보다 작게 보고될 수 있습니다. 같은 실행 내 3개 모드의 상대 비교 용도로 활용하세요.

## 문제 해결

| 증상 | 조치 |
|---|---|
| 연결 표시등이 빨간색 | Ollama 실행 여부, `OLLAMA_ORIGINS` 설정 후 재시작 확인 |
| 브라우저가 localhost 접근 차단 | Chrome/Edge 최신 버전 사용, 사설망 접근 허용 선택 |
| Graph 구축이 느림 | 정상입니다(청크마다 LLM 추출). 문서 수를 줄이거나 `청크당 최대 트리플`을 낮춰보세요 |
| HWP 파일이 안 열림 | 구버전 `.hwp` 바이너리는 미지원. 한글에서 `.hwpx`로 다시 저장하세요 |
| 임베딩 모델 변경 후 검색 이상 | 임베딩 모델을 바꾸면 Vector DB를 다시 구축해야 합니다 |

## 기술 스택

Vanilla JS (ES Modules) · Canvas 2D · IndexedDB · pdf.js · mammoth · JSZip · marked + DOMPurify · Ollama API

---
🤖 Generated with [Claude Code](https://claude.com/claude-code)
