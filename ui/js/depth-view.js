// 비전 · 깊이 탭의 깊이 칸과 CCTV 확대 창: 머리 카메라가 지금 보는 화면에 깊이 모델(TIPSv2)을 돌린 결과를 렌더러의 정답 깊이와 같은 색 척도로 나란히 보이고,
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

/** depth_view 응답을 그림·범례·지표로 만든다. core 이면 왼쪽 카메라 · 추정 · 정답 세 장만 (확대 창). */
export function depthBlocks(d, { core = false } = {}) {
  const m = d.metrics;
  const r = d.metrics_raw;
  const [lo, hi] = d.scale_m ?? [0, 1];
  const st = d.stereo;
  const figs = [
    el('figure', {}, img(d.rgb_webp_base64, '머리 카메라 원본'), el('figcaption', { text: st ? '왼쪽 카메라 (기준 · SigLIP2 · SAM이 봄)' : '머리 카메라 (흔들림·블러 포함)' })),
    !core && st && d.right_webp_base64 ? el('figure', {}, img(d.right_webp_base64, '오른쪽 카메라'), el('figcaption', { text: `오른쪽 카메라 (기선 ${fmt.n(st.baseline_m * 100, 1)} cm 옆)` })) : null,
    el('figure', {}, img(d.pred_webp_base64, '깊이 추정'), el('figcaption', { text: st ? `SGBM 깊이 — Z = f·b/d, 매칭 못 한 화소는 비움(어둡게) · ${st.grid[0]}×${st.grid[1]}` : d.align ? `추정 깊이 — 지면 정렬 · 하늘 제외 (${d.mode})` : `추정 깊이 — ${d.mode}` })),
    el('figure', {}, img(d.truth_webp_base64, '렌더러 정답 깊이'), el('figcaption', { text: '정답 깊이 (렌더러)' })),
  ].filter(Boolean);
  if (!core && d.raw_webp_base64) figs.push(el('figure', {}, img(d.raw_webp_base64, '모델 원출력'), el('figcaption', { text: '모델 원출력 (실내 NYU 척도, 정렬 전)' })));
  if (!core && d.seg_webp_base64) figs.push(el('figure', {}, img(d.seg_webp_base64, 'ADE20K 분할'), el('figcaption', { text: `분할 (ADE20K) — 하늘 ${fmt.pct(d.sky?.frac ?? 0, 0)} 깊이에서 제외` })));
  if (!core && d.normals_webp_base64) figs.push(el('figure', {}, img(d.normals_webp_base64, '표면 법선'), el('figcaption', { text: `표면 법선 — 지면 아닌 후보 ${d.align?.off_ground_normals ?? 0}개를 정렬에서 뺌` })));
  const fit = d.align;
  return [
    el('div', { class: 'depth-grid' }, ...figs),
    el('div', { class: 'depth-legend' },
      el('span', { text: `${fmt.n(lo, 1)} m 가까움` }),
      el('span', { class: 'depth-ramp', role: 'img', aria: { label: '색 척도: 빨강 가까움, 파랑 멂 (역깊이)' } }),
      el('span', { text: `멂 ${fmt.n(hi, 1)} m` })),
    kv([
      ['AbsRel (정렬 · 원출력)', m ? `${fmt.n(m.absrel, 3)} · ${r ? fmt.n(r.absrel, 3) : '—'}` : '—'],
      ['δ1 (< 1.25)', m ? `${fmt.pct(m.delta1, 1)} · ${r ? fmt.pct(r.delta1, 1) : '—'}` : '—'],
      ['RMSE', m ? `${fmt.n(m.rmse_m, 2)} m · ${r ? `${fmt.n(r.rmse_m, 2)} m` : '—'}` : '—'],
      ['중앙 축척 (정답/추정)', m ? fmt.n(m.median_scale, 2) : '—'],
      st
        ? ['SGBM (카메라 2대)', `유효 ${fmt.pct(st.valid, 0)} · 사거리 밖 ${fmt.pct(st.far, 0)} · 비움: 민무늬 ${fmt.pct(st.rejected.no_texture, 0)} · 고유성 ${fmt.pct(st.rejected.not_unique, 0)} · 좌우 불일치 ${fmt.pct(st.rejected.lr_mismatch, 0)} · 반사 ${fmt.pct(st.rejected.saturated, 0)} · 얼룩 ${fmt.pct(st.rejected.speckle, 0)} · 가장자리 ${fmt.pct(st.rejected.border, 0)} · 시차 ${st.disparities} · 초점 ${fmt.n(st.focal_px, 0)} px · ${fmt.n(st.ms, 0)} ms`]
        : ['지면 정렬', fit ? `1/Z = ${fmt.n(fit.a, 3)}/Z모델 + ${fmt.n(fit.b, 3)} · 내층 ${fit.inliers}/${fit.candidates} · 잔차 중앙 ${fmt.pct(fit.median_rel_err, 1)} · 지면 ${fmt.n(fit.span_m?.[0], 1)}–${fmt.n(fit.span_m?.[1], 1)} m` : '맞춤 없음 (지면이 충분히 보이지 않음)'],
      ...(st ? [['지면 정렬', '건너뜀 — 스테레오 깊이는 이미 미터 (1/Z = a/Z + b 정렬 안 함)']] : []),
      ...cameraRows(d.camera),
    ]),
  ];
}

function render(d) {
  const box = clear($('depth-view'));
  if (!d) {
    box.append(el('p', { class: 'empty-note', text: '깊이 추정 실행을 누르면 지금 머리 카메라 화면에 깊이 모델을 돌립니다.' }));
    return;
  }
  const tta = d.flip_tta ? ' · 좌우 반전 평균' : '';
  $('depth-meta').textContent = `틱 ${d.tick} · ${d.mode} · ${d.device} · 추론 ${fmt.n(d.infer_ms, 0)} ms${tta}${d.max_depth_m ? ` · 모델 상한 ${fmt.n(d.max_depth_m, 0)} m` : ''}`;
  box.append(...depthBlocks(d));
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
    if (group !== 'analysis') return;
    visible = name === 'vision';
    if (visible && !state.depthView) runDepth();
  });
  render(null);
}
