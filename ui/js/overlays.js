// 떠 있는 판 관리: 좌상단 월드 팝오버, 하단 Drop-up 두 장, 기록·분석 모달, CCTV 확대 창.
// 여닫기 · 바깥 클릭 · ESC 순서를 한곳에서 정해 모듈마다 같은 규칙을 쓴다. 판은 id 로 부르고 hidden 속성으로 여닫는다.
import { bus } from './state.js';
import { $ } from './util.js';

// ESC 는 맨 위 판부터 하나씩 닫고, 닫을 판이 없으면 유닛 선택을 푼다('escape' 이벤트, unit-select.js).
const STACK = ['cctv-zoom', 'analysis-modal', 'world-popover', 'preset-dropup', 'quest-dropup'];
const DROPUPS = ['preset-dropup', 'quest-dropup'];
const OPENER = { 'world-popover': 'world-pill', 'quest-dropup': 'quest-bar', 'preset-dropup': 'preset-toggle' };

export const isOpen = (id) => !!$(id) && !$(id).hidden;

function setOpen(id, open) {
  const panel = $(id);
  if (!panel || panel.hidden === !open) return;
  panel.hidden = !open;
  $(OPENER[id])?.setAttribute('aria-expanded', String(open));
  bus.emit('panel', { id, open });
}

export function openPanel(id) {
  if (DROPUPS.includes(id)) for (const d of DROPUPS) if (d !== id) setOpen(d, false);
  setOpen(id, true);
}

export function closePanel(id) {
  setOpen(id, false);
}

export function togglePanel(id) {
  if (isOpen(id)) closePanel(id);
  else openPanel(id);
}

export function initOverlays() {
  document.addEventListener('click', (e) => {
    const close = e.target.closest?.('[data-close]');
    if (close) closePanel(close.dataset.close);
  });
  $('world-pill').addEventListener('click', () => togglePanel('world-popover'));
  document.querySelectorAll('.modal-back').forEach((back) => {
    back.addEventListener('click', (e) => { if (e.target === back) closePanel(back.id); });
  });
  // 월드 팝오버는 바깥을 누르면 닫는다. Drop-up 은 다른 유닛으로 옮겨 가도 열려 있어야 하므로 빈 땅 클릭에서만 닫는다.
  document.addEventListener('pointerdown', (e) => {
    if (isOpen('world-popover') && !e.target.closest?.('#world-popover, #world-pill')) closePanel('world-popover');
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    const top = STACK.find(isOpen);
    if (top) closePanel(top);
    else bus.emit('escape');
  });
}
