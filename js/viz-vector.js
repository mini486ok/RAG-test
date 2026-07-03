// ═══════════════════════════════════════════════
// Vector DB 시각화: PCA 2D 임베딩 산점도 (Canvas)
//  - 팬/줌/호버 툴팁, 문서별 고정 색상
// ═══════════════════════════════════════════════

import { $, escapeHtml } from './ui.js';

// dataviz 검증 카테고리 팔레트 (다크 슬롯 순서 고정)
const DOC_COLORS = ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926'];

export class VectorViz {
  constructor(canvas, tooltip) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltip = tooltip;
    this.points = []; // {x, y, chunk, px, py}
    this.docColor = new Map();
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.hovered = null;
    this._bound = false;
    this._raf = null;
  }

  setData(projected) {
    this.points = projected;
    this.docColor.clear();
    const docs = [...new Set(projected.map((p) => p.chunk.docName))];
    docs.forEach((d, i) => this.docColor.set(d, DOC_COLORS[i % DOC_COLORS.length]));
    this.fit();
    this.bindEvents();
    this.draw();
    this.renderLegend(docs);
  }

  renderLegend(docs) {
    const legend = $('#vectorLegend');
    legend.innerHTML = docs
      .slice(0, 8)
      .map(
        (d) =>
          `<span class="legend-item"><span class="legend-dot" style="background:${this.docColor.get(d)}"></span>${escapeHtml(d.length > 22 ? d.slice(0, 20) + '…' : d)}</span>`
      )
      .join('');
    if (docs.length > 8) legend.innerHTML += `<span class="legend-item">+${docs.length - 8}개 문서</span>`;
  }

  fit() {
    if (!this.points.length) return;
    const xs = this.points.map((p) => p.x);
    const ys = this.points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const { w, h } = this.cssSize();
    const pad = 46;
    const sx = (w - pad * 2) / (maxX - minX || 1);
    const sy = (h - pad * 2) / (maxY - minY || 1);
    this.scale = Math.min(sx, sy);
    this.offsetX = pad + (w - pad * 2 - (maxX - minX) * this.scale) / 2 - minX * this.scale;
    this.offsetY = pad + (h - pad * 2 - (maxY - minY) * this.scale) / 2 - minY * this.scale;
  }

  cssSize() {
    const rect = this.canvas.getBoundingClientRect();
    return { w: rect.width, h: rect.height };
  }

  resizeBacking() {
    const dpr = window.devicePixelRatio || 1;
    const { w, h } = this.cssSize();
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  toScreen(p) {
    return { x: p.x * this.scale + this.offsetX, y: p.y * this.scale + this.offsetY };
  }

  draw() {
    this.resizeBacking();
    const { w, h } = this.cssSize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    const isLight = document.body.dataset.theme === 'light';
    for (const p of this.points) {
      const s = this.toScreen(p);
      p.px = s.x;
      p.py = s.y;
      const color = this.docColor.get(p.chunk.docName) || '#888';
      ctx.beginPath();
      ctx.arc(s.x, s.y, p === this.hovered ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = p === this.hovered ? 1 : 0.82;
      ctx.fill();
      // 2px 서피스 링 (겹침 구분)
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = isLight ? '#f6f8fb' : '#14203a';
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  bindEvents() {
    if (this._bound) return;
    this._bound = true;
    const c = this.canvas;
    let dragging = false;
    let lastX = 0, lastY = 0, moved = 0;

    c.addEventListener('pointerdown', (e) => {
      dragging = true;
      moved = 0;
      lastX = e.clientX;
      lastY = e.clientY;
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', (e) => {
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (dragging) {
        this.offsetX += e.clientX - lastX;
        this.offsetY += e.clientY - lastY;
        moved += Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY);
        lastX = e.clientX;
        lastY = e.clientY;
        this.scheduleDraw();
        return;
      }
      // 호버 검색
      let best = null;
      let bestD = 12 * 12;
      for (const p of this.points) {
        const d = (p.px - mx) ** 2 + (p.py - my) ** 2;
        if (d < bestD) {
          bestD = d;
          best = p;
        }
      }
      if (best !== this.hovered) {
        this.hovered = best;
        this.scheduleDraw();
      }
      this.showTooltip(best, mx, my);
    });
    c.addEventListener('pointerup', () => (dragging = false));
    c.addEventListener('pointerleave', () => {
      dragging = false;
      this.hovered = null;
      this.tooltip.hidden = true;
      this.scheduleDraw();
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.offsetX = mx - (mx - this.offsetX) * factor;
      this.offsetY = my - (my - this.offsetY) * factor;
      this.scale *= factor;
      this.scheduleDraw();
    }, { passive: false });

    const ro = new ResizeObserver(() => this.scheduleDraw());
    ro.observe(c);
  }

  showTooltip(p, mx, my) {
    if (!p) {
      this.tooltip.hidden = true;
      return;
    }
    const text = p.chunk.text.length > 170 ? p.chunk.text.slice(0, 170) + '…' : p.chunk.text;
    this.tooltip.innerHTML =
      `<div class="tt-title">${escapeHtml(p.chunk.docName)}</div>` +
      `<div class="tt-sub">청크 #${p.chunk.seq + 1}</div>` +
      `<div>${escapeHtml(text)}</div>`;
    this.tooltip.hidden = false;
    const wrap = this.canvas.parentElement.getBoundingClientRect();
    const ttw = this.tooltip.offsetWidth;
    const tth = this.tooltip.offsetHeight;
    let x = mx + 14, y = my + 14;
    if (x + ttw > wrap.width - 8) x = mx - ttw - 14;
    if (y + tth > wrap.height - 8) y = my - tth - 14;
    this.tooltip.style.left = Math.max(4, x) + 'px';
    this.tooltip.style.top = Math.max(4, y) + 'px';
  }

  scheduleDraw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = null;
      this.draw();
    });
  }
}
