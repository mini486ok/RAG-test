// ═══════════════════════════════════════════════
// UI 헬퍼: 토스트, 마크다운 렌더, DOM 유틸
// ═══════════════════════════════════════════════

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ICONS = { info: 'ℹ', ok: '✓', warn: '⚠', err: '✕' };

export function toast(msg, type = 'info', ms = 3600) {
  const stack = $('#toasts');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-ico">${ICONS[type] || ''}</span><span></span>`;
  el.lastElementChild.textContent = msg;
  stack.appendChild(el);
  setTimeout(() => {
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, ms);
}

/** 마크다운 → 안전한 HTML */
export function renderMarkdown(el, text) {
  if (window.marked && window.DOMPurify) {
    el.innerHTML = window.DOMPurify.sanitize(window.marked.parse(text, { breaks: true }));
  } else {
    el.textContent = text;
  }
}

/** 스트리밍 중 렌더 스로틀러 */
export function makeStreamRenderer(el, intervalMs = 120) {
  let pending = '';
  let timer = null;
  let lastRender = 0;

  const flush = () => {
    timer = null;
    lastRender = performance.now();
    const scroller = el.closest('.engine-body');
    // 사용자가 위로 스크롤해 읽는 중이면 자동 스크롤로 방해하지 않음
    const nearBottom = scroller
      ? scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 60
      : false;
    renderMarkdown(el, pending);
    if (scroller && nearBottom) scroller.scrollTop = scroller.scrollHeight;
  };

  return {
    update(fullText) {
      pending = fullText;
      const now = performance.now();
      if (now - lastRender >= intervalMs) flush();
      else if (!timer) timer = setTimeout(flush, intervalMs - (now - lastRender));
    },
    finish(fullText) {
      pending = fullText ?? pending;
      if (timer) clearTimeout(timer);
      flush();
    },
  };
}

export function fmtMs(ms) {
  if (ms == null) return '–';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function fmtNum(n) {
  if (n == null) return '–';
  return n.toLocaleString('ko-KR');
}

export function fmtBytes(b) {
  if (b == null) return '';
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / 1024 / 1024).toFixed(1)}MB`;
}

export function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export function uid() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}
