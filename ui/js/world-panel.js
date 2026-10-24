// 왼쪽 ③ 월드: 시드·크기·지형·재생성, 월드 리셋 · 신체 상태 리셋, 현재 월드 요약, 손·발 구성이 달라졌다는 안내.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { applyReset, newWorld, setPlaying } from './sim.js';
import { setStatus, toast } from './toast.js';

const KIND_KO = {
  tree: '나무', boulder: '바위', coal_ore: '석탄 광석', iron_ore_node: '철 광석',
  rock: '바위', iron_ore: '철 광석', chest: '상자',
};
const TERRAIN_KO = { open: '열린 평지', rugged: '복합 지형' };

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
  box.append(...kv('지형', TERRAIN_KO[snap.terrain] ?? snap.terrain ?? '—'));
  box.append(...kv('경계', `${snap.bounds[0]} × ${snap.bounds[1]} m`));
  box.append(...kv('발', snap.agent.base));
  box.append(...kv('웨이포인트', `${snap.waypoints.length}개`));
  for (const [k, n] of Object.entries(counts)) box.append(...kv(KIND_KO[k] ?? k, `${n}개`));
  box.append(...kv('드롭', `${snap.drops.length}개`));
  summaryBase = `시드 ${snap.seed} · ${snap.bounds[0]}×${snap.bounds[1]} m${snap.terrain === 'rugged' ? ' · 복합' : ''}`;
}

function updateDirty() {
  const dirty = state.worldConfigKey != null && state.worldConfigKey !== JSON.stringify(state.config);
  $('world-dirty').hidden = !dirty;
  // ③이 접혀 있어도 알 수 있게 머리글 요약에 표시한다.
  $('sum-world').textContent = dirty ? `${summaryBase} · 재생성 필요` : summaryBase;
}

/** 월드 리셋: 같은 시드의 처음 상태로 (신체 상태는 유지). */
export async function resetWorld() {
  const btn = $('world-reset');
  btn.disabled = true;
  try {
    setPlaying(false);
    const r = await call('world_reset', {});
    applyReset(r, 'world');
    toast('월드 리셋', '자원·드롭·웨이포인트 더미·퀘스트를 처음 상태로 되돌렸습니다. 신체 상태는 그대로입니다.', 'ok');
    setStatus('월드 리셋 완료 (신체 상태 유지)');
  } catch {
    /* api-error 가 표시 */
  } finally {
    btn.disabled = false;
  }
}

/** 신체 상태 리셋: 배터리·관절 발열·채취 피로를 처음으로 (월드·적재함은 유지). */
export async function resetBody() {
  try {
    const r = await call('body_reset', {});
    applyReset(r, 'body');
    toast('신체 상태 리셋', '배터리 100%, 관절 발열·감속기 마모 0으로 돌렸습니다. 월드와 적재함은 그대로입니다.', 'ok');
    setStatus('신체 상태 리셋 완료');
  } catch {
    /* api-error 가 표시 */
  }
}

export function initWorldPanel() {
  state.terrain = $('world-terrain').value;
  const run = async () => {
    const btn = $('world-new');
    btn.disabled = true;
    try {
      const seed = Math.max(0, Math.floor(Number($('world-seed').value) || 0));
      state.terrain = $('world-terrain').value;
      await newWorld(seed, Number($('world-size').value), state.terrain);
    } finally {
      btn.disabled = false;
    }
  };
  $('world-new').addEventListener('click', run);
  $('world-seed').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  $('world-terrain').addEventListener('change', (e) => { state.terrain = e.target.value; });
  $('world-reset').addEventListener('click', resetWorld);
  $('body-reset').addEventListener('click', resetBody);
  bus.on('snapshot', (snap) => { renderInfo(snap); updateDirty(); });
  bus.on('config', updateDirty);
  bus.on('world-new', updateDirty);
  renderInfo(null);
  return run;
}
