// 왼쪽 ③ 월드: 시드·크기·재생성, 현재 월드 요약, 손·발 구성이 달라졌다는 안내.
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { newWorld } from './sim.js';

const KIND_KO = {
  tree: '나무', boulder: '바위', coal_ore: '석탄 광석', iron_ore_node: '철 광석',
  rock: '바위', iron_ore: '철 광석', chest: '상자',
};

function kv(k, v) {
  return [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: v })];
}

let summaryBase = '';

function renderInfo(snap) {
  const box = $('world-info');
  clear(box);
  if (!snap) {
    box.append(...kv('상태', '월드 없음'));
    summaryBase = '';
    return;
  }
  const counts = {};
  for (const n of snap.nodes) counts[n.kind] = (counts[n.kind] ?? 0) + 1;
  box.append(...kv('시드', String(snap.seed)));
  box.append(...kv('경계', `${snap.bounds[0]} × ${snap.bounds[1]} m`));
  box.append(...kv('발', snap.agent.base));
  box.append(...kv('웨이포인트', `${snap.waypoints.length}개`));
  for (const [k, n] of Object.entries(counts)) box.append(...kv(KIND_KO[k] ?? k, `${n}개`));
  box.append(...kv('드롭', `${snap.drops.length}개`));
  summaryBase = `시드 ${snap.seed} · ${snap.bounds[0]}×${snap.bounds[1]} m`;
}

function updateDirty() {
  const dirty = state.worldConfigKey != null && state.worldConfigKey !== JSON.stringify(state.config);
  $('world-dirty').hidden = !dirty;
  // ③이 접혀 있어도 알 수 있게 머리글 요약에 표시한다.
  $('sum-world').textContent = dirty ? `${summaryBase} · 재생성 필요` : summaryBase;
}

export function initWorldPanel() {
  const run = async () => {
    const btn = $('world-new');
    btn.disabled = true;
    try {
      const seed = Math.max(0, Math.floor(Number($('world-seed').value) || 0));
      await newWorld(seed, Number($('world-size').value));
    } finally {
      btn.disabled = false;
    }
  };
  $('world-new').addEventListener('click', run);
  $('world-seed').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  bus.on('snapshot', (snap) => { renderInfo(snap); updateDirty(); });
  bus.on('config', updateDirty);
  bus.on('world-new', updateDirty);
  renderInfo(null);
  return run;
}
