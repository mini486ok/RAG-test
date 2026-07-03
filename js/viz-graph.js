// ═══════════════════════════════════════════════
// Graph DB 시각화: 포스 다이렉티드 네트워크 (Canvas)
//  - 자체 구현 물리 시뮬레이션 (반발력 + 스프링 + 중심 인력)
//  - 팬/줌/드래그/호버/클릭 상세, 유형별 색상, 검색 하이라이트
// ═══════════════════════════════════════════════

import { $, escapeHtml } from './ui.js';

const TYPE_COLORS = ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926'];

export class GraphViz {
  constructor(canvas, tooltip, detailPanel) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltip = tooltip;
    this.detailPanel = detailPanel;
    this.nodes = [];
    this.edges = [];
    this.nodeByKey = new Map();
    this.typeColor = new Map();
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.hovered = null;
    this.selected = null;
    this.dragNode = null;
    this.searchTerm = '';
    this.running = false;
    this.alpha = 0;
    this._bound = false;
    this.onSelect = null; // (nodeKey) => neighbors[]
  }

  setData({ nodes, edges }) {
    // 결정적 초기 배치 (황금각 나선)
    const cx = 0, cy = 0;
    this.nodes = nodes.map((n, i) => {
      const r = 40 + 22 * Math.sqrt(i);
      const a = i * 2.39996; // 골든 앵글
      return {
        ...n,
        x: cx + r * Math.cos(a),
        y: cy + r * Math.sin(a),
        vx: 0,
        vy: 0,
        radius: Math.min(22, 6 + Math.sqrt(n.degree + n.weight) * 2.2),
      };
    });
    this.nodeByKey = new Map(this.nodes.map((n) => [n.key, n]));
    this.edges = edges
      .map((e) => ({ ...e, a: this.nodeByKey.get(e.source), b: this.nodeByKey.get(e.target) }))
      .filter((e) => e.a && e.b);

    const types = [...new Set(this.nodes.map((n) => n.type))];
    this.typeColor.clear();
    types.forEach((t, i) => this.typeColor.set(t, TYPE_COLORS[i % TYPE_COLORS.length]));
    this.renderLegend(types);

    const { w, h } = this.cssSize();
    this.scale = 1;
    this.offsetX = w / 2;
    this.offsetY = h / 2;
    this.selected = null;
    this.detailPanel.hidden = true;

    this.bindEvents();
    this.start();
  }

  renderLegend(types) {
    const legend = $('#graphLegend');
    legend.innerHTML = types
      .slice(0, 8)
      .map(
        (t) =>
          `<span class="legend-item"><span class="legend-dot" style="background:${this.typeColor.get(t)}"></span>${escapeHtml(t)}</span>`
      )
      .join('');
    if (types.length > 8) legend.innerHTML += `<span class="legend-item">+${types.length - 8}</span>`;
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

  start() {
    this.alpha = 1;
    if (!this.running) {
      this.running = true;
      this.loop();
    }
  }

  stop() {
    this.running = false;
  }

  toggle() {
    if (this.running) this.stop();
    else {
      this.start();
    }
    return this.running;
  }

  loop() {
    if (!this.running) return;
    this.tick();
    this.draw();
    if (this.alpha < 0.003) {
      this.running = false;
      this.draw();
      return;
    }
    requestAnimationFrame(() => this.loop());
  }

  /** 물리 한 스텝: Barnes-Hut 없이 O(n²)이지만 400노드 상한이라 충분 */
  tick() {
    const nodes = this.nodes;
    const n = nodes.length;
    if (!n) return;
    const repulsion = 1800;
    const springLen = 90;
    const springK = 0.015;
    const centerK = 0.0035;
    const damping = 0.82;

    for (let i = 0; i < n; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < n; j++) {
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) d2 = 1;
        if (d2 > 250000) continue; // 원거리 컷오프
        const f = (repulsion / d2) * this.alpha;
        const d = Math.sqrt(d2);
        dx /= d;
        dy /= d;
        a.vx += dx * f;
        a.vy += dy * f;
        b.vx -= dx * f;
        b.vy -= dy * f;
      }
    }
    for (const e of this.edges) {
      const dx = e.b.x - e.a.x;
      const dy = e.b.y - e.a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = springK * (d - springLen) * this.alpha * Math.min(2, Math.sqrt(e.weight));
      e.a.vx += (dx / d) * f;
      e.a.vy += (dy / d) * f;
      e.b.vx -= (dx / d) * f;
      e.b.vy -= (dy / d) * f;
    }
    for (const node of nodes) {
      node.vx -= node.x * centerK * this.alpha;
      node.vy -= node.y * centerK * this.alpha;
      if (node !== this.dragNode) {
        node.vx *= damping;
        node.vy *= damping;
        node.x += node.vx;
        node.y += node.vy;
      }
    }
    this.alpha *= 0.995;
  }

  toScreen(p) {
    return { x: p.x * this.scale + this.offsetX, y: p.y * this.scale + this.offsetY };
  }

  toWorld(sx, sy) {
    return { x: (sx - this.offsetX) / this.scale, y: (sy - this.offsetY) / this.scale };
  }

  draw() {
    this.resizeBacking();
    const { w, h } = this.cssSize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    const isLight = document.body.dataset.theme === 'light';
    const edgeColor = isLight ? 'rgba(40,60,100,0.22)' : 'rgba(150,180,240,0.2)';
    const labelColor = isLight ? '#33405c' : '#c6d2ee';

    const focusKeys = this.focusSet();

    // 엣지
    ctx.lineWidth = 1;
    for (const e of this.edges) {
      const a = this.toScreen(e.a);
      const b = this.toScreen(e.b);
      const dim = focusKeys && !(focusKeys.has(e.a.key) && focusKeys.has(e.b.key));
      ctx.strokeStyle = edgeColor;
      ctx.globalAlpha = dim ? 0.12 : Math.min(1, 0.5 + e.weight * 0.15);
      ctx.lineWidth = Math.min(3, 0.7 + e.weight * 0.3);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // 노드
    for (const node of this.nodes) {
      const s = this.toScreen(node);
      if (s.x < -40 || s.y < -40 || s.x > w + 40 || s.y > h + 40) continue;
      const color = this.typeColor.get(node.type) || '#888';
      const r = node.radius * Math.min(1.6, Math.max(0.6, this.scale));
      const isFocus = !focusKeys || focusKeys.has(node.key);
      const isHl = node === this.hovered || node === this.selected;

      ctx.globalAlpha = isFocus ? 1 : 0.15;
      ctx.beginPath();
      ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = isHl ? 2.5 : 1.5;
      ctx.strokeStyle = isHl ? (isLight ? '#101623' : '#ffffff') : (isLight ? '#f6f8fb' : '#14203a');
      ctx.stroke();

      // 라벨: 확대 시 또는 주요 노드/포커스만
      const showLabel = isHl || (isFocus && (this.scale > 1.1 || node.degree >= 3 || this.nodes.length <= 40));
      if (showLabel) {
        ctx.font = `${isHl ? 600 : 400} ${Math.max(10, 11 * Math.min(1.3, this.scale))}px "IBM Plex Sans KR", sans-serif`;
        ctx.fillStyle = labelColor;
        ctx.textAlign = 'center';
        const label = node.name.length > 14 ? node.name.slice(0, 13) + '…' : node.name;
        ctx.fillText(label, s.x, s.y + r + 13);
      }
    }
    ctx.globalAlpha = 1;
  }

  focusSet() {
    if (this.searchTerm) {
      const t = this.searchTerm.toLowerCase();
      const hit = new Set();
      for (const n of this.nodes) {
        if (n.name.toLowerCase().includes(t)) hit.add(n.key);
      }
      // 히트 노드의 1-hop 이웃 포함
      for (const e of this.edges) {
        if (hit.has(e.a.key)) hit.add(e.b.key);
        else if (hit.has(e.b.key)) hit.add(e.a.key);
      }
      return hit;
    }
    if (this.selected) {
      const set = new Set([this.selected.key]);
      for (const e of this.edges) {
        if (e.a === this.selected) set.add(e.b.key);
        if (e.b === this.selected) set.add(e.a.key);
      }
      return set;
    }
    return null;
  }

  setSearch(term) {
    this.searchTerm = term.trim();
    this.draw();
  }

  nodeAt(sx, sy) {
    const wpt = this.toWorld(sx, sy);
    let best = null;
    let bestD = Infinity;
    for (const n of this.nodes) {
      const d = (n.x - wpt.x) ** 2 + (n.y - wpt.y) ** 2;
      const rr = (n.radius / Math.min(1.6, Math.max(0.6, this.scale)) + 6) ** 2;
      if (d < rr && d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  bindEvents() {
    if (this._bound) return;
    this._bound = true;
    const c = this.canvas;
    let panning = false;
    let lastX = 0, lastY = 0, movedTotal = 0;

    c.addEventListener('pointerdown', (e) => {
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const node = this.nodeAt(mx, my);
      movedTotal = 0;
      if (node) {
        this.dragNode = node;
        this.alpha = Math.max(this.alpha, 0.25);
        if (!this.running) { this.running = true; this.loop(); }
      } else {
        panning = true;
      }
      lastX = e.clientX;
      lastY = e.clientY;
      c.setPointerCapture(e.pointerId);
    });

    c.addEventListener('pointermove', (e) => {
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (this.dragNode) {
        const wpt = this.toWorld(mx, my);
        this.dragNode.x = wpt.x;
        this.dragNode.y = wpt.y;
        this.dragNode.vx = 0;
        this.dragNode.vy = 0;
        movedTotal += Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY);
        lastX = e.clientX;
        lastY = e.clientY;
        if (!this.running) this.draw();
        return;
      }
      if (panning) {
        this.offsetX += e.clientX - lastX;
        this.offsetY += e.clientY - lastY;
        movedTotal += Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY);
        lastX = e.clientX;
        lastY = e.clientY;
        this.draw();
        return;
      }
      const node = this.nodeAt(mx, my);
      if (node !== this.hovered) {
        this.hovered = node;
        c.style.cursor = node ? 'pointer' : 'grab';
        if (!this.running) this.draw();
      }
      this.showTooltip(node, mx, my);
    });

    c.addEventListener('pointerup', (e) => {
      const rect = c.getBoundingClientRect();
      const wasNode = this.dragNode;
      this.dragNode = null;
      panning = false;
      // 클릭(드래그 아님) → 선택
      if (movedTotal < 5) {
        const node = wasNode || this.nodeAt(e.clientX - rect.left, e.clientY - rect.top);
        this.select(node);
      }
    });

    c.addEventListener('pointerleave', () => {
      this.hovered = null;
      this.tooltip.hidden = true;
      if (!this.running) this.draw();
    });

    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.offsetX = mx - (mx - this.offsetX) * factor;
      this.offsetY = my - (my - this.offsetY) * factor;
      this.scale = Math.min(6, Math.max(0.15, this.scale * factor));
      this.draw();
    }, { passive: false });

    if ('ResizeObserver' in window) {
      new ResizeObserver(() => this.draw()).observe(c);
    } else {
      window.addEventListener('resize', () => this.draw());
    }
  }

  select(node) {
    this.selected = node || null;
    if (!node) {
      this.detailPanel.hidden = true;
      this.draw();
      return;
    }
    const rels = this.onSelect ? this.onSelect(node.key) : [];
    const color = this.typeColor.get(node.type) || '#888';
    this.detailPanel.innerHTML =
      `<button class="nd-close" aria-label="닫기">✕</button>` +
      `<h5><span class="legend-dot" style="background:${color}"></span>${escapeHtml(node.name)}</h5>` +
      `<span class="nd-type" style="color:${color};border-color:${color}">${escapeHtml(node.type)}</span>` +
      (node.desc ? `<p class="nd-desc">${escapeHtml(node.desc)}</p>` : '') +
      `<p class="nd-rel-head">관계 ${rels.length}건 · 등장 ${node.weight}회</p>` +
      rels
        .slice(0, 30)
        .map((r) =>
          r.dir === 'out'
            ? `<div class="nd-rel">→ <b>${escapeHtml(r.other)}</b> <small>(${escapeHtml(r.relation)})</small></div>`
            : `<div class="nd-rel">← <b>${escapeHtml(r.other)}</b> <small>(${escapeHtml(r.relation)})</small></div>`
        )
        .join('');
    this.detailPanel.hidden = false;
    this.detailPanel.querySelector('.nd-close').onclick = () => this.select(null);
    this.draw();
  }

  showTooltip(node, mx, my) {
    if (!node || node === this.selected) {
      this.tooltip.hidden = true;
      return;
    }
    this.tooltip.innerHTML =
      `<div class="tt-title">${escapeHtml(node.name)}</div>` +
      `<div class="tt-sub">${escapeHtml(node.type)} · 연결 ${node.degree} · 등장 ${node.weight}회</div>` +
      (node.desc ? `<div>${escapeHtml(node.desc.slice(0, 130))}${node.desc.length > 130 ? '…' : ''}</div>` : '');
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
}
