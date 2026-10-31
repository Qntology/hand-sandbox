// 공용 화면 동작: 탭 목록(키보드·ARIA 포함)과 전역 툴팁.
import { $ } from './util.js';

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

export function initLayout() {
  initTooltips();
}
