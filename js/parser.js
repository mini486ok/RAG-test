// ═══════════════════════════════════════════════
// 문서 파서: PDF / DOCX / PPTX / HWPX / MD / TXT
//  - 브라우저 100% 클라이언트 사이드 파싱
//  - pdf.js, mammoth, JSZip (CDN 전역 객체) 사용
// ═══════════════════════════════════════════════

let pdfWorkerReady = false;

function ensurePdfWorker() {
  if (!pdfWorkerReady && window.pdfjsLib) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';
    pdfWorkerReady = true;
  }
}

export function detectType(fileName) {
  const ext = fileName.split('.').pop().toLowerCase();
  if (['pdf'].includes(ext)) return 'pdf';
  if (['docx'].includes(ext)) return 'docx';
  if (['pptx'].includes(ext)) return 'pptx';
  if (['hwpx'].includes(ext)) return 'hwpx';
  if (['md', 'markdown'].includes(ext)) return 'md';
  if (['txt'].includes(ext)) return 'txt';
  return null;
}

/** File → 순수 텍스트. 지원하지 않으면 throw */
export async function parseFile(file) {
  const type = detectType(file.name);
  if (!type) throw new Error(`지원하지 않는 형식입니다: ${file.name}`);

  switch (type) {
    case 'pdf': return parsePdf(file);
    case 'docx': return parseDocx(file);
    case 'pptx': return parsePptx(file);
    case 'hwpx': return parseHwpx(file);
    case 'md':
    case 'txt': return readAsText(file);
    default: throw new Error(`지원하지 않는 형식입니다: ${type}`);
  }
}

function readAsText(file) {
  return file.text();
}

async function parsePdf(file) {
  if (!window.pdfjsLib) throw new Error('PDF 라이브러리가 로드되지 않았습니다. 네트워크를 확인하세요.');
  ensurePdfWorker();
  const buf = await file.arrayBuffer();
  const pdf = await window.pdfjsLib.getDocument({ data: buf }).promise;
  if (pdf.numPages > 300) {
    console.warn(`대용량 PDF (${pdf.numPages}쪽): 파싱에 시간이 걸릴 수 있습니다.`);
  }
  const parts = [];
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    // 줄 단위 병합: y좌표가 바뀌면 줄바꿈
    let lastY = null;
    let line = [];
    const lines = [];
    for (const item of content.items) {
      const y = Math.round(item.transform[5]);
      if (lastY !== null && Math.abs(y - lastY) > 2) {
        if (line.length) lines.push(line.join(''));
        line = [];
      }
      line.push(item.str);
      lastY = y;
    }
    if (line.length) lines.push(line.join(''));
    parts.push(lines.join('\n'));
  }
  return parts.join('\n\n');
}

async function parseDocx(file) {
  if (!window.mammoth) throw new Error('DOCX 라이브러리가 로드되지 않았습니다. 네트워크를 확인하세요.');
  const buf = await file.arrayBuffer();
  const result = await window.mammoth.extractRawText({ arrayBuffer: buf });
  return result.value || '';
}

/** PPTX: ppt/slides/slideN.xml 의 <a:t> 텍스트 추출 */
async function parsePptx(file) {
  if (!window.JSZip) throw new Error('압축 라이브러리가 로드되지 않았습니다. 네트워크를 확인하세요.');
  const zip = await window.JSZip.loadAsync(await file.arrayBuffer());
  const slideNames = Object.keys(zip.files)
    .filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))
    .sort((a, b) => {
      const na = parseInt(a.match(/slide(\d+)/)[1], 10);
      const nb = parseInt(b.match(/slide(\d+)/)[1], 10);
      return na - nb;
    });
  if (!slideNames.length) throw new Error('PPTX에서 슬라이드를 찾을 수 없습니다.');
  const parts = [];
  for (const name of slideNames) {
    const xml = await zip.files[name].async('string');
    const texts = extractXmlTexts(xml, 'a:t');
    if (texts.length) parts.push(texts.join('\n'));
  }
  return parts.join('\n\n');
}

/** HWPX(OWPML): Contents/section*.xml 의 <hp:t> 텍스트 추출 */
async function parseHwpx(file) {
  if (!window.JSZip) throw new Error('압축 라이브러리가 로드되지 않았습니다. 네트워크를 확인하세요.');
  const zip = await window.JSZip.loadAsync(await file.arrayBuffer());
  const sections = Object.keys(zip.files)
    .filter((n) => /^Contents\/section\d+\.xml$/i.test(n))
    .sort();
  if (!sections.length) {
    throw new Error('HWPX 본문을 찾을 수 없습니다. (구버전 .hwp 바이너리는 지원하지 않으며, 한글에서 .hwpx로 다시 저장해 주세요)');
  }
  const parts = [];
  for (const name of sections) {
    const xml = await zip.files[name].async('string');
    const texts = extractXmlTexts(xml, 'hp:t');
    if (texts.length) parts.push(texts.join('\n'));
  }
  return parts.join('\n\n');
}

/** DOMParser로 특정 태그의 텍스트를 순서대로 수집 */
function extractXmlTexts(xml, tagName) {
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  // 네임스페이스 프리픽스는 getElementsByTagName 문자열 매칭으로 처리
  const nodes = doc.getElementsByTagName(tagName);
  const out = [];
  for (const n of nodes) {
    const t = n.textContent;
    if (t && t.trim()) out.push(t);
  }
  // 폴백: 프리픽스 없이 로컬네임 매칭
  if (!out.length) {
    const local = tagName.split(':').pop();
    const all = doc.getElementsByTagName('*');
    for (const n of all) {
      if (n.localName === local && n.textContent.trim()) out.push(n.textContent);
    }
  }
  return out;
}

/** 공통 텍스트 정리 */
export function cleanText(raw) {
  return raw
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}
