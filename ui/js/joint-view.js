// 3인칭 '관절 물리' 오버레이: world_frame(view: 'chase:<궤도>')가 돌려주는 overlay 를 그린다.
// 관절 점(토크/피크 비율 색, 포화 고리, 멈춤 X), 무게중심과 그 수직선, 동적 ZMP, 지지 다각형, 지면 반력 화살표,
// 그리고 오른쪽 아래 패널에 이번 자세 변화의 관절별 토크/피크 시계열과 관절 표. 계산은 모두 Rust(hand_core::dynamics)에서 한다.
import { $, clear, el } from './util.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** 토크/피크 비율의 색: 0.5 미만 정상, 0.8 미만 주의 시작, 1 미만 주의, 1 이상 위험. */
function ratioColor(r) {
  if (!Number.isFinite(r)) return css('--text-muted');
  if (r >= 1) return css('--danger');
  if (r >= 0.8) return '#e8944f';
  if (r >= 0.5) return css('--warning');
  return css('--success');
}

const KO = {
  waist_yaw: '허리 요', waist_pitch: '허리 피치', neck: '목', shoulder_r: '오른 어깨', elbow_r: '오른 팔꿈치', shoulder_l: '왼 어깨', elbow_l: '왼 팔꿈치',
  hip_pitch_r: '오른 고관절 굽힘', hip_roll_r: '오른 고관절 벌림', knee_r: '오른 무릎', ankle_r: '오른 발목',
  hip_pitch_l: '왼 고관절 굽힘', hip_roll_l: '왼 고관절 벌림', knee_l: '왼 무릎', ankle_l: '왼 발목', drive: '구동(바퀴)',
};
export const jointKo = (n) => KO[n] ?? n;

function fmt(x, d = 1) {
  return Number.isFinite(x) ? Number(x).toFixed(d) : '—';
}

/** 오버레이를 프레임 위 캔버스에 그린다. 좌표는 프레임 픽셀이다. */
export function drawJointOverlay(o, width, height) {
  const cv = $('joint-layer');
  if (cv.width !== width || cv.height !== height) {
    cv.width = width;
    cv.height = height;
  }
  const g = cv.getContext('2d');
  g.clearRect(0, 0, width, height);
  if (!o) return;
  const s = Math.max(1, width / 480);
  // 지지 다각형
  const sup = (o.support ?? []).filter((p) => p);
  if (sup.length === 4) {
    g.beginPath();
    sup.forEach((p, i) => (i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y)));
    g.closePath();
    g.fillStyle = o.margin >= 0 ? 'rgba(95, 185, 140, 0.18)' : 'rgba(224, 108, 108, 0.22)';
    g.fill();
    g.lineWidth = 1.5 * s;
    g.strokeStyle = o.margin >= 0 ? css('--success') : css('--danger');
    g.stroke();
  }
  // 지면 반력: ZMP 에서 반력 방향으로
  if (o.zmp && o.grf_tip) {
    g.strokeStyle = 'rgba(250, 250, 250, 0.75)';
    g.lineWidth = 2 * s;
    g.beginPath();
    g.moveTo(o.zmp.x, o.zmp.y);
    g.lineTo(o.grf_tip.x, o.grf_tip.y);
    g.stroke();
  }
  // 무게중심과 그 지면 투영
  if (o.com && o.com_ground) {
    g.setLineDash([4 * s, 3 * s]);
    g.strokeStyle = '#7dd3fc';
    g.lineWidth = 1.2 * s;
    g.beginPath();
    g.moveTo(o.com.x, o.com.y);
    g.lineTo(o.com_ground.x, o.com_ground.y);
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = '#7dd3fc';
    g.beginPath();
    g.arc(o.com.x, o.com.y, 4.5 * s, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = '#0b1220';
    g.beginPath();
    g.arc(o.com.x, o.com.y, 2 * s, 0, Math.PI * 2);
    g.fill();
  }
  if (o.zmp) {
    g.fillStyle = '#f472b6';
    g.fillRect(o.zmp.x - 4 * s, o.zmp.y - 4 * s, 8 * s, 8 * s);
  }
  // 관절
  for (const j of o.joints ?? []) {
    const p = j.at;
    if (!p) continue;
    const r = (j.kind === 'sim' ? 5 : 4) * s;
    g.globalAlpha = p.visible ? 1 : 0.35;
    g.fillStyle = j.kind === 'structure' ? css('--text-muted') : ratioColor(j.ratio);
    g.beginPath();
    g.arc(p.x, p.y, r, 0, Math.PI * 2);
    g.fill();
    g.lineWidth = (j.saturated ? 2.5 : 1) * s;
    g.strokeStyle = j.saturated ? css('--danger') : 'rgba(0, 0, 0, 0.7)';
    g.stroke();
    if (j.stalled) {
      g.strokeStyle = css('--danger');
      g.lineWidth = 2 * s;
      g.beginPath();
      g.moveTo(p.x - r, p.y - r);
      g.lineTo(p.x + r, p.y + r);
      g.moveTo(p.x + r, p.y - r);
      g.lineTo(p.x - r, p.y + r);
      g.stroke();
    }
    if (p.visible && j.kind !== 'structure' && j.ratio >= 0.3) {
      g.font = `${Math.round(9 * s)}px ui-monospace, monospace`;
      g.fillStyle = 'rgba(0, 0, 0, 0.6)';
      const label = `${Math.round(j.ratio * 100)}%`;
      const w = g.measureText(label).width;
      g.fillRect(p.x + r + 1, p.y - 6 * s, w + 4, 11 * s);
      g.fillStyle = ratioColor(j.ratio);
      g.fillText(label, p.x + r + 3, p.y + 3 * s);
    }
    g.globalAlpha = 1;
  }
}

/** 시계열: 관절별 토크/피크(시뮬레이션 관절 실선, 정역학 관절 점선)와 여유(하늘색). */
function drawChart(m) {
  const cv = $('jp-chart');
  const W = cv.width;
  const H = cv.height;
  const g = cv.getContext('2d');
  g.clearRect(0, 0, W, H);
  if (!m?.series?.t?.length) return;
  const ts = m.series.t;
  const t1 = Math.max(ts[ts.length - 1], 1e-3);
  const y = (v) => H - 4 - Math.min(1.2, Math.max(0, v)) / 1.2 * (H - 8);
  const x = (t) => 4 + (t / t1) * (W - 8);
  g.strokeStyle = css('--plot-grid-strong');
  g.lineWidth = 1;
  for (const v of [0.5, 1]) {
    g.beginPath();
    g.moveTo(4, y(v));
    g.lineTo(W - 4, y(v));
    g.stroke();
  }
  const line = (vals, color, dash) => {
    g.setLineDash(dash ? [3, 2] : []);
    g.strokeStyle = color;
    g.beginPath();
    vals.forEach((v, i) => (i ? g.lineTo(x(ts[i]), y(v)) : g.moveTo(x(ts[i]), y(v))));
    g.stroke();
  };
  const top = [...m.series.joints].sort((a, b) => Math.max(...b.ratio) - Math.max(...a.ratio)).slice(0, 5);
  for (const j of top) line(j.ratio, ratioColor(Math.max(...j.ratio)), j.kind !== 'sim');
  line(m.series.margin, '#7dd3fc', false);
  g.setLineDash([]);
  if (Number.isFinite(m.t)) {
    g.strokeStyle = 'rgba(250,250,250,0.6)';
    g.beginPath();
    g.moveTo(x(Math.min(m.t, t1)), 2);
    g.lineTo(x(Math.min(m.t, t1)), H - 2);
    g.stroke();
  }
}

/** 오른쪽 아래 관절 표와 노트. */
export function renderJointPanel(o) {
  const panel = $('joint-panel');
  if (!o) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const m = o.motion;
  $('jp-state').textContent = m
    ? `${m.source_ko} · ${o.plan} · 최소 여유 ${fmt(m.min_margin, 2)}${m.feasible ? '' : ' · 불가'}`
    : o.plan;
  drawChart(m);
  const sum = new Map((m?.summary ?? []).map((s) => [s.name, s]));
  const rows = clear($('jp-rows'));
  const joints = [...(o.joints ?? [])].sort((a, b) => b.ratio - a.ratio);
  for (const j of joints) {
    const s = sum.get(j.name);
    const cls = [j.kind, j.stalled ? 'stall' : '', j.ratio >= 1 ? 'over' : '', j.saturated ? 'sat' : ''].filter(Boolean).join(' ');
    const bar = el('span', { class: 'jp-bar' }, el('i', {}));
    bar.firstChild.style.width = `${Math.min(100, j.ratio * 100)}%`;
    bar.firstChild.style.background = ratioColor(j.ratio);
    rows.append(el('tr', { class: cls, title: s ? `이번 변화 최대 ${Math.round(s.peak_ratio * 100)}% · 포화 ${fmt(s.saturated_s, 2)} s · 멈춤 ${fmt(s.stalled_s, 2)} s · 최대 지연 ${fmt(s.max_lag_deg)}°` : '' },
      el('td', { text: jointKo(j.name) }),
      el('td', { class: 'r', text: j.q_deg == null ? '—' : `${fmt(j.q_deg)}°` }),
      el('td', { class: 'r', text: `${fmt(j.tau_nm)}/${fmt(j.peak_nm, 0)}` }),
      el('td', { class: 'r', text: j.kind === 'structure' ? '—' : `${Math.round(j.ratio * 100)}%` }),
      el('td', {}, bar)));
  }
  const notes = clear($('jp-notes'));
  for (const n of m?.notes ?? []) notes.append(el('li', { text: n }));
  if (m && !m.notes?.length) notes.append(el('li', { text: `모든 관절이 목표에 도달 (정착 ${fmt(m.settle_s, 2)} s, 정적 여유 ${fmt(m.static_margin, 2)})` }));
}
