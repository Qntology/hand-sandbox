// 깊이 · 카메라 탭: 머리 카메라가 지금 보는 화면에 깊이 모델(TIPSv2)을 돌린 결과를 렌더러의 정답 깊이와 같은 색 척도로 나란히 보이고,
// 머리 카메라의 높이·피치·팬·흔들림·모션 블러·IMU 추정 오차를 보인다(depth_view, world_frame 의 camera).
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';

const AUTO_MS = 2000;
let running = false;
let autoTimer = null;
let visible = false;

const SHAKE_KO = { sway: '몸통 흔들림 (걸음)', heel_strike: '발뒤꿈치 착지 충격', footfall: '울퉁불퉁한 발 디딤', rolling: '바퀴·트랙 구름 진동', still: '없음 (정지)' };

function img(b64, alt) {
  return el('img', { src: `data:image/webp;base64,${b64}`, alt, class: 'depth-img' });
}

function kv(rows) {
  return el('div', { class: 'pv-kv depth-kv' }, rows.flatMap(([k, v]) => [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: v })]));
}

/** 카메라 상태 표(world_frame·depth_view 의 camera). */
export function cameraRows(c) {
  if (!c) return [];
  const s = c.shake ?? {};
  return [
    ['눈 높이', `${fmt.n(c.eye_m, 2)} m (서 있을 때 ${fmt.n(c.standing_eye_m, 2)} m)`],
    ['카메라 피치', `${fmt.n(c.pitch_deg, 1)}° = 장착 ${fmt.n(c.mount_deg, 0)}° − 몸·목 ${fmt.n(c.head_pitch_deg, 1)}° + 흔들림 ${fmt.n(s.pitch_deg, 2)}°`],
    ['차체 기울기 (지면)', c.terrain_tilt_deg ? `피치 ${fmt.n(c.terrain_tilt_deg[0], 1)}° · 롤 ${fmt.n(c.terrain_tilt_deg[1], 1)}° (바퀴·트랙만, 다리형은 몸통을 수평으로 유지)` : '—'],
    ['목 팬 · 롤', `팬 ${fmt.n(c.pan_deg, 1)}° (왼쪽 +) · 롤 ${fmt.n(c.roll_deg, 2)}°`],
    ['흔들림', `${SHAKE_KO[s.source] ?? s.source ?? '—'} · 상하 ${fmt.n(s.dz_mm, 1)} mm · 각속도 ${(s.rate_dps ?? []).map((r) => fmt.n(r, 0)).join(' / ')} °/s (요·피치·롤)`],
    ['모션 블러', `${fmt.n(c.blur_px, 2)} px (노출 ${fmt.n(c.sensor?.exposure_ms, 2)} ms @ ${c.sensor?.scene_lux ?? '—'} lx · 롤링 셔터 ${fmt.n(c.sensor?.readout_ms, 1)} ms)`],
    ['로봇의 추정 오차', `피치 ${fmt.n(c.est_err_deg?.pitch, 3)}° · 롤 ${fmt.n(c.est_err_deg?.roll, 3)}° (자이로 ${fmt.n(c.sensor?.gyro_arw_deg, 3)} °/√s)`],
    ['센서 · 지면', `${c.sensor?.name ?? '—'} · RGB ${c.sensor?.rgb ?? '—'} · 지면 ${c.ground ?? '—'}`],
  ];
}

function render(d) {
  const box = clear($('depth-view'));
  if (!d) {
    box.append(el('p', { class: 'empty-note', text: '깊이 추정 실행을 누르면 지금 머리 카메라 화면에 깊이 모델을 돌립니다.' }));
    return;
  }
  const m = d.metrics;
  $('depth-meta').textContent = `틱 ${d.tick} · ${d.mode} · ${d.device} · 추론 ${fmt.n(d.infer_ms, 0)} ms${d.max_depth_m ? ` · 최대 ${fmt.n(d.max_depth_m, 0)} m` : ''}`;
  const [lo, hi] = d.scale_m ?? [0, 1];
  box.append(
    el('div', { class: 'depth-grid' },
      el('figure', {}, img(d.rgb_webp_base64, '머리 카메라 원본'), el('figcaption', { text: '머리 카메라 (흔들림·블러 포함)' })),
      el('figure', {}, img(d.pred_webp_base64, '깊이 모델 추정'), el('figcaption', { text: `추정 깊이 — ${d.mode}` })),
      el('figure', {}, img(d.truth_webp_base64, '렌더러 정답 깊이'), el('figcaption', { text: '정답 깊이 (렌더러)' }))),
    el('div', { class: 'depth-legend' },
      el('span', { text: `${fmt.n(lo, 1)} m 가까움` }),
      el('span', { class: 'depth-ramp', role: 'img', aria: { label: '색 척도: 빨강 가까움, 파랑 멂 (역깊이)' } }),
      el('span', { text: `멂 ${fmt.n(hi, 1)} m` })),
    kv([
      ['AbsRel', m ? fmt.n(m.absrel, 3) : '—'],
      ['δ1 (< 1.25)', m ? fmt.pct(m.delta1, 1) : '—'],
      ['RMSE', m ? `${fmt.n(m.rmse_m, 2)} m` : '—'],
      ['중앙 축척 (정답/추정)', m ? fmt.n(m.median_scale, 2) : '—'],
      ...cameraRows(d.camera),
    ]),
  );
}

export async function runDepth() {
  if (running) return;
  running = true;
  $('depth-run').disabled = true;
  try {
    const d = await call('depth_view', { quality: 80 });
    state.depthView = d;
    render(d);
  } catch {
    /* api-error 가 표시 */
  } finally {
    running = false;
    $('depth-run').disabled = false;
  }
}

function setAuto(on) {
  clearInterval(autoTimer);
  autoTimer = on ? setInterval(() => { if (visible && state.playing) runDepth(); }, AUTO_MS) : null;
}

export function initDepthView() {
  $('depth-run').addEventListener('click', runDepth);
  $('depth-auto').addEventListener('change', (e) => setAuto(e.target.checked));
  bus.on('tab', ({ group, name }) => {
    if (group !== 'bottom') return;
    visible = name === 'depth';
    if (visible && !state.depthView) runDepth();
  });
  render(null);
}
