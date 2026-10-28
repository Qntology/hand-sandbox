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

function syncCoopButton(snap) {
  const btn = $('tg-coop-agent');
  if (!btn) return;
  const team = snap?.team_size ?? state.teamSize ?? 1;
  const active = snap?.agent?.active_robot ?? state.activeRobot ?? 1;
  const phase = snap?.agent?.co_phase;
  if (team <= 1) {
    btn.textContent = '로봇 1 (단독)';
    btn.classList.remove('active');
    btn.setAttribute('aria-pressed', 'false');
  } else {
    const phaseNote = phase === 1 ? '1차 시뮬레이션' : phase === 2 ? '2차 시뮬레이션(중첩 수렴)' : '순차 협동 대기';
    btn.textContent = `로봇 ${active}/2 (${phaseNote})`;
    btn.classList.add('active');
    btn.setAttribute('aria-pressed', 'true');
  }
}

export function initWorldPanel() {
  state.terrain = $('world-terrain').value;
  state.teamSize = Number($('world-team')?.value || 2);
  const run = async () => {
    const btn = $('world-new');
    btn.disabled = true;
    try {
      const seed = Math.max(0, Math.floor(Number($('world-seed').value) || 0));
      state.terrain = $('world-terrain').value;
      state.teamSize = Number($('world-team')?.value || 2);
      await newWorld(seed, Number($('world-size').value), state.terrain);
    } finally {
      btn.disabled = false;
    }
  };
  $('world-new').addEventListener('click', run);
  $('world-seed').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  $('world-terrain').addEventListener('change', (e) => { state.terrain = e.target.value; });
  $('world-team')?.addEventListener('change', (e) => {
    state.teamSize = Number(e.target.value) || 1;
    call('set_team_mode', { teamSize: state.teamSize, activeRobot: state.activeRobot }, { quiet: true }).catch(() => {});
    syncCoopButton(state.snapshot);
    toast('로봇 운용 변경', state.teamSize === 2 ? '순차 협동(2대 · 중첩 물리) 모드가 켜졌습니다.' : '단독 운용(1대) 모드로 전환했습니다.');
  });
  $('tg-coop-agent')?.addEventListener('click', async () => {
    if (state.teamSize <= 1) {
      state.teamSize = 2;
      if ($('world-team')) $('world-team').value = '2';
    } else {
      state.activeRobot = state.activeRobot === 1 ? 2 : 1;
    }
    try {
      const r = await call('set_team_mode', { teamSize: state.teamSize, activeRobot: state.activeRobot }, { quiet: true });
      if (r?.snapshot) bus.emit('snapshot', r.snapshot);
    } catch {}
    syncCoopButton(state.snapshot);
    setStatus(`활성 시뮬레이션 주체: 로봇 ${state.activeRobot} (팀 ${state.teamSize}대)`);
  });
  $('world-reset').addEventListener('click', resetWorld);
  $('body-reset').addEventListener('click', resetBody);
  bus.on('snapshot', (snap) => {
    if (snap?.agent?.active_robot) state.activeRobot = snap.agent.active_robot;
    renderInfo(snap);
    updateDirty();
    syncCoopButton(snap);
  });
  bus.on('config', updateDirty);
  bus.on('world-new', updateDirty);
  renderInfo(null);
  syncCoopButton(null);
  return run;
}
