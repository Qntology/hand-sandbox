// 레이아웃 동작: 리사이저(드래그·방향키·더블클릭 초기화), 접기, 아코디언, 탭, 전역 툴팁.
// endmill-sandbox index.html 2153~2369행의 동작을 모듈로 옮기고 키보드·ARIA 를 더했다.
import { $ } from './util.js';
import { bus, state } from './state.js';

const root = document.documentElement;
// 접기 전에 사용자가 직접 정한 크기(인라인 값). 없으면 CSS 기본값(미디어 쿼리 포함)으로 되돌린다.
const stash = { left: '', right: '', bottom: '' };
function restoreSize(cssVar, key) {
  if (stash[key]) root.style.setProperty(cssVar, stash[key]);
  else root.style.removeProperty(cssVar);
}

const px = (name) => parseInt(getComputedStyle(root).getPropertyValue(name), 10) || 0;

// ── 리사이저 ────────────────────────────────────────────────────
function initResizer(handle, { axis, panel, cssVar, minVar, minFallback, direction, maxFrac, key }) {
  const min = () => px(minVar) || minFallback;
  const max = () => Math.floor((axis === 'x' ? window.innerWidth : window.innerHeight) * maxFrac);
  const size = () => {
    const r = panel.getBoundingClientRect();
    return axis === 'x' ? r.width : r.height;
  };
  const apply = (v) => {
    const clamped = Math.min(Math.max(v, min()), max());
    root.style.setProperty(cssVar, `${Math.round(clamped)}px`);
  };
  let startPos = 0;
  let startSize = 0;

  handle.addEventListener('pointerdown', (e) => {
    if (panel.classList.contains('collapsed')) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    startPos = axis === 'x' ? e.clientX : e.clientY;
    startSize = size();
    handle.classList.add('active');
    document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';
  });
  handle.addEventListener('pointermove', (e) => {
    if (!handle.classList.contains('active')) return;
    const pos = axis === 'x' ? e.clientX : e.clientY;
    apply(startSize + direction * (pos - startPos));
  });
  const end = () => {
    handle.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
  handle.addEventListener('dblclick', () => {
    root.style.removeProperty(cssVar); // CSS 기본값(미디어 쿼리 포함)으로 복귀
    stash[key] = '';
  });
  handle.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 48 : 16;
    const grow = axis === 'x' ? (direction > 0 ? 'ArrowRight' : 'ArrowLeft') : 'ArrowUp';
    const shrink = axis === 'x' ? (direction > 0 ? 'ArrowLeft' : 'ArrowRight') : 'ArrowDown';
    if (e.key === grow) apply(size() + step);
    else if (e.key === shrink) apply(size() - step);
    else if (e.key === 'Home' || e.key === 'Enter') handle.dispatchEvent(new MouseEvent('dblclick'));
    else return;
    e.preventDefault();
  });
}

// ── 접기 ────────────────────────────────────────────────────────
function setLabel(btn, text) {
  btn.title = text;
  btn.setAttribute('aria-label', text);
}

export function panelOpen(id) {
  const panel = $(id);
  if (panel?.classList.contains('collapsed')) document.querySelector(`.panel-toggle-btn[data-toggle="${id}"]`)?.click();
}

function initCollapse() {
  document.querySelectorAll('.panel-toggle-btn[data-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.toggle;
      const panel = $(id);
      if (!panel) return;
      const collapsed = panel.classList.toggle('collapsed');
      btn.setAttribute('aria-expanded', String(!collapsed));
      if (id === 'panelLeft') {
        btn.textContent = collapsed ? '▶' : '◀';
        setLabel(btn, collapsed ? '좌측 패널 펼치기' : '좌측 패널 접기');
        if (collapsed) {
          stash.left = root.style.getPropertyValue('--left-w');
          root.style.setProperty('--left-w', getComputedStyle(root).getPropertyValue('--rail-w').trim());
        } else {
          restoreSize('--left-w', 'left');
        }
      } else if (id === 'panelRight') {
        btn.textContent = collapsed ? '◀' : '▶';
        setLabel(btn, collapsed ? '우측 패널 펼치기' : '우측 패널 접기');
        if (collapsed) {
          stash.right = root.style.getPropertyValue('--right-w');
          root.style.setProperty('--right-w', '0px');
        } else {
          restoreSize('--right-w', 'right');
        }
      } else if (id === 'panelBottom') {
        btn.textContent = collapsed ? '▲' : '▼';
        setLabel(btn, collapsed ? '하단 패널 펼치기' : '하단 패널 접기');
        if (collapsed) {
          stash.bottom = root.style.getPropertyValue('--bottom-h');
          root.style.setProperty('--bottom-h', getComputedStyle(root).getPropertyValue('--bar-h').trim());
        } else {
          restoreSize('--bottom-h', 'bottom');
        }
      }
      bus.emit('layout', { id, collapsed });
    });
  });
  $('reopen-right')?.addEventListener('click', () => panelOpen('panelRight'));
}

// ── 아코디언 ────────────────────────────────────────────────────
export function setSection(id, open) {
  const body = $(id);
  const header = document.querySelector(`.panel-section-header[data-section="${id}"]`);
  if (!body || !header) return;
  body.classList.toggle('collapsed', !open);
  header.classList.toggle('collapsed', !open);
  header.setAttribute('aria-expanded', String(open));
}

export function openSection(id) {
  const header = document.querySelector(`.panel-section-header[data-section="${id}"]`);
  const rail = header?.closest('.panel-left.collapsed');
  if (rail) panelOpen(rail.id);
  setSection(id, true);
}

function initAccordion() {
  const toggle = (header) => {
    const id = header.dataset.section;
    if (id === 'secOutput') {
      document.querySelector('.panel-toggle-btn[data-toggle="panelBottom"]')?.click();
      return;
    }
    const body = $(id);
    if (!body) return;
    if (header.closest('.panel-left.collapsed')) {
      openSection(id);
      return;
    }
    const willOpen = body.classList.contains('collapsed');
    setSection(id, willOpen);
    if (willOpen) body.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  };
  document.querySelectorAll('.panel-section-header[data-section]').forEach((header) => {
    header.addEventListener('click', (e) => {
      if (e.target.closest('.help')) return;
      toggle(header);
    });
    header.addEventListener('keydown', (e) => {
      if (e.target !== header) return;
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggle(header);
      }
    });
  });
}

// ── 탭 ──────────────────────────────────────────────────────────
export function initTablist(list, onSelect) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  function select(name, { focus = false } = {}) {
    for (const t of tabs) {
      const on = t.dataset.tab === name;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      const pane = $(t.getAttribute('aria-controls'));
      if (pane) pane.hidden = !on;
      if (on && focus) t.focus();
    }
    onSelect?.(name);
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t.dataset.tab));
    t.addEventListener('keydown', (e) => {
      const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (to == null) return;
      e.preventDefault();
      select(tabs[(to + tabs.length) % tabs.length].dataset.tab, { focus: true });
    });
  });
  return { select, current: () => tabs.find((t) => t.getAttribute('aria-selected') === 'true')?.dataset.tab };
}

let centerTabs = null;
let bottomTabs = null;

export function selectCenterTab(name) {
  centerTabs?.select(name);
}
export function selectBottomTab(name) {
  panelOpen('panelBottom');
  bottomTabs?.select(name);
}

function initTabs() {
  centerTabs = initTablist($('center-tabs'), (name) => {
    state.centerTab = name;
    document.querySelectorAll('.menu-btn[data-view]').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
    document.querySelectorAll('.world-only').forEach((n) => { n.hidden = name !== 'world'; });
    bus.emit('tab', { group: 'center', name });
  });
  bottomTabs = initTablist($('bottom-tabs'), (name) => bus.emit('tab', { group: 'bottom', name }));
  document.querySelectorAll('.menu-btn[data-view]').forEach((b) => b.addEventListener('click', () => selectCenterTab(b.dataset.view)));
  $('pip')?.addEventListener('click', () => selectCenterTab('side'));
}

// ── 툴팁 ────────────────────────────────────────────────────────
let tipCur = null;
let tipPop = null;

function tipShow(target) {
  tipCur = target;
  if (!tipPop) {
    tipPop = document.createElement('div');
    tipPop.className = 'tip-pop';
    tipPop.setAttribute('role', 'tooltip');
    document.body.append(tipPop);
  }
  tipPop.replaceChildren();
  const title = target.dataset.tipTitle;
  if (title) {
    const b = document.createElement('b');
    b.textContent = title;
    tipPop.append(b);
  }
  tipPop.append(document.createTextNode(target.dataset.tip ?? ''));
  const r = target.getBoundingClientRect();
  const w = tipPop.offsetWidth;
  const h = tipPop.offsetHeight;
  const x = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
  let y = r.bottom + 8;
  if (y + h > window.innerHeight - 8) y = Math.max(8, r.top - h - 8);
  tipPop.style.left = `${x}px`;
  tipPop.style.top = `${y}px`;
  tipPop.classList.add('show');
}

function tipHide() {
  tipCur = null;
  tipPop?.classList.remove('show');
}

/** data-tip 이 바뀐 뒤 열려 있는 툴팁 문구를 갱신한다. */
export function refreshTooltip() {
  if (tipCur?.isConnected) tipShow(tipCur);
  else if (tipCur) tipHide();
}

function initTooltips() {
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (t && t !== tipCur) tipShow(t);
    else if (!t && tipCur) tipHide();
  });
  document.addEventListener('focusin', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (t) tipShow(t);
  });
  document.addEventListener('focusout', tipHide);
  document.addEventListener('scroll', tipHide, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') tipHide(); });
}

function initWindowClamp() {
  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    for (const [v, key, frac] of [['--left-w', 'left', 0.45], ['--right-w', 'right', 0.45]]) {
      if (root.style.getPropertyValue(v) && px(v) > w * frac && px(v) > 0 && !$(key === 'left' ? 'panelLeft' : 'panelRight').classList.contains('collapsed')) {
        root.style.setProperty(v, `${Math.floor(w * frac)}px`);
      }
    }
  });
}

export function initLayout() {
  initResizer($('resizerLeft'), { axis: 'x', panel: $('panelLeft'), cssVar: '--left-w', minVar: '--left-min', minFallback: 220, direction: 1, maxFrac: 0.45, key: 'left' });
  initResizer($('resizerRight'), { axis: 'x', panel: $('panelRight'), cssVar: '--right-w', minVar: '--right-min', minFallback: 260, direction: -1, maxFrac: 0.45, key: 'right' });
  initResizer($('resizerBottom'), { axis: 'y', panel: $('panelBottom'), cssVar: '--bottom-h', minVar: '--bottom-min', minFallback: 140, direction: -1, maxFrac: 0.7, key: 'bottom' });
  initCollapse();
  initAccordion();
  initTabs();
  initTooltips();
  initWindowClamp();
}
