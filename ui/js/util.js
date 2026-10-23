// 공용 DOM·포맷 도구. 동적 문자열은 항상 textContent 로 넣는다(innerHTML 금지).
export const $ = (id) => document.getElementById(id);

/** el('div', {class:'x', dataset:{a:1}, aria:{label:'..'}, on:{click:fn}, text:'..'}, ...children) */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  applyProps(node, props);
  appendAll(node, children);
  return node;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
export function svgEl(tag, props = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  applyProps(node, props);
  appendAll(node, children);
  return node;
}

function applyProps(node, props) {
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') node.setAttribute('class', v);
    else if (k === 'text') node.textContent = v;
    else if (k === 'dataset') for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
    else if (k === 'aria') for (const [ak, av] of Object.entries(v)) node.setAttribute(`aria-${ak}`, av);
    else if (k === 'on') for (const [ev, fn] of Object.entries(v)) node.addEventListener(ev, fn);
    else if (k === 'style') node.style.cssText = v;
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, String(v));
  }
}

function appendAll(node, children) {
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

export function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export const fmt = {
  n: (v, d = 2) => (Number.isFinite(v) ? Number(v).toFixed(d) : '—'),
  pct: (r, d = 0) => (Number.isFinite(r) ? `${(r * 100).toFixed(d)}%` : '—'),
  bytes: (b) => (!Number.isFinite(b) ? '—' : b < 1024 ? `${b} B` : `${(b / 1024).toFixed(1)} KB`),
  /** ISO 시각이면 시:분:초만, 아니면 그대로. */
  time: (ts) => {
    const m = /T(\d\d:\d\d:\d\d)/.exec(String(ts ?? ''));
    return m ? m[1] : String(ts ?? '');
  },
};

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** 캔버스를 CSS 크기 × devicePixelRatio 로 맞추고, CSS 픽셀 좌표계의 컨텍스트를 돌려준다. */
export function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(0, Math.round(canvas.clientWidth));
  const H = Math.max(0, Math.round(canvas.clientHeight));
  if (!W || !H) return null;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H };
}

export function observeResize(node, fn) {
  if (typeof ResizeObserver === 'undefined') return null;
  const ro = new ResizeObserver(() => fn());
  ro.observe(node);
  return ro;
}

export function storageGet(key, fallback = null) {
  try {
    const v = window.localStorage.getItem(key);
    return v == null ? fallback : v;
  } catch {
    return fallback;
  }
}

export function storageSet(key, value) {
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    /* 저장소를 못 쓰는 환경 — 무시 */
  }
}

/** 수준(ok|warn|bad|crit) → 배지 클래스 */
export const LEVEL_BADGE = { ok: 'badge-ok', warn: 'badge-warn', bad: 'badge-bad', crit: 'badge-crit' };
export const LEVEL_KO = { ok: '정상', warn: '주의', bad: '나쁨', crit: '위험' };

export function levelOf(ratio) {
  if (!Number.isFinite(ratio)) return 'ok';
  return ratio >= 0.6 ? 'ok' : ratio >= 0.3 ? 'warn' : ratio >= 0.1 ? 'bad' : 'crit';
}

export function errText(e) {
  if (typeof e === 'string') return e;
  if (e && typeof e.message === 'string') return e.message;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}
