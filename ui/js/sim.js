// 시뮬레이션 진행: world_new, 재생 중 sim_step 반복 호출, 퀘스트·신체 상태 갱신.
// Tauri 이벤트(listen)는 쓰지 않고 setInterval 로 폴링한다. 이전 호출이 끝나지 않았으면 그 주기는 건너뛴다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $ } from './util.js';
import { setStatus, toast } from './toast.js';

const STEP_MS = 125;
let timer = null;
let pending = false;

function applySnapshot(snap, reset = false) {
  state.snapshot = snap;
  if (reset) state.trail = [];
  const [x, y] = snap.agent.pos;
  const last = state.trail[state.trail.length - 1];
  if (!last || Math.hypot(last[0] - x, last[1] - y) > 0.15) {
    state.trail.push([x, y]);
    if (state.trail.length > 600) state.trail.shift();
  }
  const t = $('sim-time');
  t.textContent = `틱 ${snap.tick} · ${snap.time_s.toFixed(1)} s`;
  $('sb-tick').textContent = String(snap.tick);
  $('sb-seed').textContent = String(snap.seed);
  bus.emit('snapshot', snap);
}

export async function refreshQuests() {
  try {
    state.quests = await call('quest_list', {}, { quiet: true });
    bus.emit('quests', state.quests);
  } catch {
    /* 폴링 실패는 조용히 */
  }
}

export async function refreshBody() {
  try {
    state.body = await call('body_status', {}, { quiet: true });
    bus.emit('body', state.body);
  } catch {
    /* 폴링 실패는 조용히 */
  }
}

export async function newWorld(seed, size) {
  setPlaying(false);
  try {
    const snap = await call('world_new', { seed, size, config: state.config, base: state.config.base, body: state.bodyPlan ?? null });
    state.worldConfigKey = JSON.stringify(state.config);
    state.detections = [];
    applySnapshot(snap, true);
    bus.emit('world-new', snap);
    await Promise.allSettled([refreshQuests(), refreshBody()]);
    setStatus(`월드를 만들었습니다 (시드 ${snap.seed}, ${snap.bounds[0]} × ${snap.bounds[1]} m)`);
    return true;
  } catch (e) {
    bus.emit('world-error', e.message);
    return false;
  }
}

export async function step(ticks) {
  const r = await call('sim_step', { ticks });
  applySnapshot(r.snapshot);
  state.quests = r.quests;
  state.body = r.body;
  bus.emit('quests', r.quests);
  bus.emit('body', r.body);
  bus.emit('sim-events', r.events ?? []);
  return r;
}

async function tick() {
  if (pending) return;
  pending = true;
  try {
    await step(state.speed);
  } catch {
    setPlaying(false);
  } finally {
    pending = false;
  }
}

export function setPlaying(on) {
  if (on === state.playing) return;
  if (on && !state.snapshot) {
    toast('월드가 아직 없습니다', '왼쪽 ③ 월드에서 재생성하세요.', 'warn');
    return;
  }
  state.playing = on;
  clearInterval(timer);
  timer = on ? setInterval(tick, STEP_MS) : null;
  const btn = $('sim-play');
  btn.textContent = on ? '⏸ 일시정지' : '⏵ 재생';
  btn.setAttribute('aria-pressed', String(on));
  const s = $('sim-state');
  s.classList.toggle('running', on);
  s.lastElementChild.textContent = on ? '재생 중' : '정지';
  bus.emit('playing', on);
}

export function initSim() {
  $('sim-play').addEventListener('click', () => setPlaying(!state.playing));
  $('sim-step').addEventListener('click', async () => {
    if (state.playing || pending || !state.snapshot) return;
    pending = true;
    try {
      await step(state.speed);
    } catch {
      /* api-error 가 표시 */
    } finally {
      pending = false;
    }
  });
  $('sim-speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); });
  // 통합 테스트는 월드를 직접 진행시킨다. 재생 루프가 끼어들지 않게 멈춘다.
  bus.on('test-start', () => setPlaying(false));
  state.speed = Number($('sim-speed').value);
}
