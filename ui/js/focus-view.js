// 행위 키프레임의 부위 포커스: 키프레임 하나의 FocusFrame(noto_core::act::focus)으로
// 손 모양 · 손 관절 · 발 모양 · 발 관절 · 허리 · 목 관절 · 바퀴 · 트랙을 그린다.
// 타임라인 타일(작게)과 검사기(크게)가 같은 그리기를 쓴다. 숫자는 모두 Rust 가 계산한 값이고 여기서는 그리기만 한다.
import { cssVar, el, fitCanvas } from './util.js';

export const FOCUS_PARTS = [
  { id: 'frame', ko: '1인칭 (원본)' },
  { id: 'auto', ko: '자동 (행위 부위)' },
  { id: 'hand', ko: '손 모양' },
  { id: 'hand_joints', ko: '손 관절' },
  { id: 'foot', ko: '발 모양' },
  { id: 'foot_joints', ko: '발 관절' },
  { id: 'waist', ko: '허리' },
  { id: 'neck', ko: '목 관절' },
  { id: 'wheel', ko: '바퀴' },
  { id: 'track', ko: '트랙' },
];
export const PART_KO = Object.fromEntries(FOCUS_PARTS.map((p) => [p.id, p.ko]));
export const DRAWN_PARTS = ['hand', 'hand_joints', 'foot', 'foot_joints', 'waist', 'neck', 'wheel', 'track'];

export const EVENT_KO = {
  fork_down: '포크 내림', fork_in: '포크 넣음', fork_up: '포크 들어올림', fork_out: '포크 뺌',
  bend_bottom: '굽힘 바닥', contact: '접촉', lift_off: '들림', hold: '들어 올림', slip: '미끄러짐', pivot: '손 안에서 돎', drop: '떨어뜨림',
  stowed: '적재', release: '놓음', impact: '착지 충격', stall: '관절 멈춤', miss: '빈손', reach_fail: '닿지 않음', foot_slip: '발 미끄러짐',
  fall: '넘어짐', breakaway: '밀기 시작', turn_slide: '돌다 미끄러짐', push_lost: '밀던 물체 놓침', collision: '충돌', arrive: '도착', stuck: '헛돎', wheel_slip: '바퀴 미끄러짐',
};

const css = (n) => cssVar(n);

/** 토크/피크 비율의 색 (joint-view 와 같은 단계). */
export function ratioColor(r) {
  if (!Number.isFinite(r)) return css('--text-muted');
  if (r >= 1) return css('--danger');
  if (r >= 0.8) return '#e8944f';
  if (r >= 0.5) return css('--warning');
  return css('--success');
}

/** 미끄럼 여유(마찰 × 쥠 / 하중)의 색: 1 미만이면 미끄러진다. */
function slipColor(s) {
  if (!Number.isFinite(s)) return css('--text-muted');
  return s < 1 ? css('--danger') : s < 1.5 ? css('--warning') : css('--success');
}

function n(x, d = 1) {
  return Number.isFinite(x) ? Number(x).toFixed(d) : '—';
}

/** 이 몸에 그 부위 화면이 있는가 (바퀴형엔 발이, 다리형엔 바퀴가 없다). */
export function partAvailable(f, part) {
  if (!f) return false;
  switch (part) {
    case 'foot': case 'foot_joints': return (f.foot ?? []).length > 0;
    case 'wheel': return (f.wheel ?? []).length > 0;
    case 'track': return !!f.track;
    case 'waist': return true;
    case 'neck': return !!f.neck;
    default: return true;
  }
}

/** '자동'을 실제 부위로: 이 순간의 사건·행위가 다루는 첫 부위 중 이 몸에 있는 것 (없으면 손). */
export function resolvePart(part, f) {
  if (part !== 'auto') return part;
  const cands = (f?.parts ?? []).filter((p) => DRAWN_PARTS.includes(p) && partAvailable(f, p));
  return cands[0] ?? 'hand';
}

/** 부위 하나를 캔버스에 그린다. compact=true 이면 글자를 줄인다(타임라인 타일). */
export function drawFocus(canvas, f, part, { compact = false } = {}) {
  const fit = fitCanvas(canvas);
  if (!fit) return;
  const { ctx, W, H } = fit;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = css('--bg-card');
  ctx.fillRect(0, 0, W, H);
  if (!f) return note(ctx, W, H, '포커스 없음');
  if (!partAvailable(f, part)) return note(ctx, W, H, `${PART_KO[part] ?? part}: 이 몸에 없음`);
  const g = { ctx, W, H, compact, fs: compact ? 8 : 10 };
  switch (part) {
    case 'hand': return drawHand(g, f);
    case 'hand_joints': return drawHandJoints(g, f);
    case 'foot': return drawFoot(g, f);
    case 'foot_joints': return drawFootJoints(g, f);
    case 'waist': return drawWaist(g, f);
    case 'neck': return drawNeck(g, f);
    case 'wheel': return drawWheel(g, f);
    case 'track': return drawTrack(g, f);
    default: return note(ctx, W, H, part);
  }
}

function note(ctx, W, H, text) {
  ctx.fillStyle = css('--text-muted');
  ctx.font = `9px ${css('--font-sans')}`;
  ctx.textAlign = 'center';
  ctx.fillText(text, W / 2, H / 2 + 3, W - 6);
  ctx.textAlign = 'left';
}

function label(g, text, x, y, color = css('--text-bright'), align = 'left') {
  const { ctx, fs } = g;
  ctx.font = `${fs}px ${css('--font-mono')}`;
  ctx.textAlign = align;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.textAlign = 'left';
}

function bar(g, x, y, w, h, ratio, color) {
  const { ctx } = g;
  ctx.fillStyle = css('--bg-input');
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.fillRect(x, y, Math.max(0, Math.min(1, ratio)) * w, h);
  if (ratio > 1) {
    ctx.strokeStyle = css('--danger');
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }
}

function arrow(ctx, x0, y0, x1, y1, color, lw = 1.5) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const a = Math.atan2(y1 - y0, x1 - x0);
  const s = 3 + lw;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - s * Math.cos(a - 0.4), y1 - s * Math.sin(a - 0.4));
  ctx.lineTo(x1 - s * Math.cos(a + 0.4), y1 - s * Math.sin(a + 0.4));
  ctx.closePath();
  ctx.fill();
}

// ── 손 모양: 손 좌표계 측면 (z 손끝 방향 →, y 손바닥 쪽 ↓ — 위에서 잡을 때처럼) ──
function drawHand(g, f) {
  const { ctx, W, H, compact } = g;
  const h = f.hand;
  const palmLen = h.palm?.[1] ?? 0.09;
  let zmin = -palmLen - 0.01, zmax = 0.02, ymin = -0.02, ymax = 0.03;
  for (const fg of h.fingers ?? []) for (const [, y, z] of fg) { zmax = Math.max(zmax, z + 0.01); ymax = Math.max(ymax, y + 0.01); ymin = Math.min(ymin, y - 0.01); }
  const obj = h.object_half;
  if (obj) { ymax = Math.max(ymax, 0.012 + 2 * obj[1] + 0.005); zmin = Math.min(zmin, -palmLen / 2 - obj[2] - 0.005); }
  const top = compact ? 4 : 16;
  const bottom = compact ? 12 : 30;
  const s = Math.min((W - 8) / (zmax - zmin), (H - top - bottom) / (ymax - ymin));
  const X = (z) => 4 + (z - zmin) * s;
  const Y = (y) => top + (y - ymin) * s;
  // 물체 (손바닥 아래, 쥔 정도로 감싸인 상자)
  if (obj) {
    ctx.fillStyle = 'rgba(184, 146, 90, 0.35)';
    ctx.strokeStyle = css('--cv-object');
    ctx.lineWidth = 1.2;
    const x0 = X(-palmLen / 2 - obj[2]);
    const y0 = Y(0.012);
    ctx.fillRect(x0, y0, 2 * obj[2] * s, 2 * obj[1] * s);
    ctx.strokeRect(x0, y0, 2 * obj[2] * s, 2 * obj[1] * s);
  }
  // 손바닥
  ctx.fillStyle = css('--cv-palm');
  ctx.strokeStyle = css('--text-muted');
  ctx.lineWidth = 1;
  ctx.fillRect(X(-palmLen), Y(0), palmLen * s, Math.max(2, 0.012 * s));
  ctx.strokeRect(X(-palmLen), Y(0), palmLen * s, Math.max(2, 0.012 * s));
  // 손가락
  ctx.lineCap = 'round';
  (h.fingers ?? []).forEach((pts, i) => {
    const color = h.thumbs?.[i] ? css('--cv-thumb') : css('--cv-finger');
    ctx.strokeStyle = color;
    ctx.lineWidth = compact ? 2 : 3;
    ctx.beginPath();
    pts.forEach(([, y, z], k) => (k ? ctx.lineTo(X(z), Y(y)) : ctx.moveTo(X(z), Y(y))));
    ctx.stroke();
    ctx.fillStyle = color;
    for (const [, y, z] of pts) { ctx.beginPath(); ctx.arc(X(z), Y(y), compact ? 1.4 : 2.2, 0, Math.PI * 2); ctx.fill(); }
  });
  // 물체가 손에 거는 힘 (아래로 = 무게 + 관성)
  const fz = h.force_n?.[2] ?? 0;
  if (Math.abs(fz) > 0.5 && !compact) {
    const cx = X(-palmLen / 2);
    const len = Math.min(H * 0.3, Math.abs(fz) * 0.6);
    arrow(ctx, cx, Y(0.012) + 2, cx, Y(0.012) + 2 + Math.sign(-fz) * len, 'rgba(250,250,250,0.7)');
  }
  const gr = h.grip;
  if (gr) {
    const txt = compact ? `여유 ${n(gr.slip_ratio, 2)}` : `쥠 ${n(gr.grip_n, 0)} N · 하중 ${n(gr.load_n, 0)} N · 여유 ${n(gr.slip_ratio, 2)} · 접촉 ${gr.contacts}`;
    label(g, txt, 4, H - (compact ? 3 : 16), slipColor(gr.slip_ratio));
    if (!compact) label(g, `마찰 ${n(gr.mu, 2)} (믿음 ${n(gr.mu_est, 2)}) · 비틀림 여유 ${n(gr.twist_margin_nm, 3)} N·m`, 4, H - 4, css('--text-dim'));
  } else {
    label(g, `오므림 ${Math.round((h.closure ?? 0) * 100)}%${h.object_kg ? ` · ${n(h.object_kg, 1)} kg` : ''}`, 4, H - (compact ? 3 : 16), css('--text-dim'));
    if (!compact) label(g, `손바닥 높이 ${n(h.palm_at?.[2], 2)} m`, 4, H - 4, css('--text-dim'));
  }
  if (!compact) label(g, '손 모양 (z 손끝 → · y 손바닥 쪽 ↓)', 4, 11, css('--text-muted'));
}

// ── 손 관절: 손가락 × 관절 막대 (토크 / 최대) ──
function drawHandJoints(g, f) {
  const { W, H, compact } = g;
  const joints = f.hand_joints ?? [];
  const fingers = [...new Set(joints.map((j) => j.finger))];
  const cols = Math.max(1, ...joints.map((j) => j.joint + 1));
  const top = compact ? 3 : 16;
  const left = compact ? 3 : 42;
  const rowH = (H - top - (compact ? 3 : 4)) / Math.max(1, fingers.length);
  const colW = (W - left - 4) / cols;
  const small = { ...g, fs: 8 };
  fingers.forEach((name, r) => {
    const y = top + r * rowH;
    if (!compact) label(g, name.slice(0, 6), 3, y + rowH * 0.65, css('--text-dim'));
    for (const j of joints.filter((x) => x.finger === name)) {
      const ratio = j.max_nm > 0 ? j.torque_nm / j.max_nm : 0;
      const x = left + j.joint * colW;
      // The bar on top, the bend under it when the row has room.
      const text = !compact && rowH >= 17;
      bar(g, x, y + 2, colW - 3, Math.max(2, (text ? 0.38 : 0.55) * rowH), ratio, ratioColor(ratio));
      if (text) label(small, `${n(j.angle_deg, 0)}°`, x + colW - 4, y + rowH - 3, css('--text-muted'), 'right');
    }
  });
  if (!compact) label(g, '손 관절 (막대 = 토크/최대, 숫자 = 굽힘)', 4, 11, css('--text-muted'));
}

// ── 발 모양: 위에서 본 두 발바닥, 압력 중심, 접지력, ZMP ──
function drawFoot(g, f) {
  const { ctx, W, H, compact } = g;
  const feet = f.foot ?? [];
  const sup = f.body?.support;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const ft of feet) { x0 = Math.min(x0, ft.rect[0]); x1 = Math.max(x1, ft.rect[1]); y0 = Math.min(y0, ft.rect[2]); y1 = Math.max(y1, ft.rect[3]); }
  if (sup) { x0 = Math.min(x0, sup[0]); x1 = Math.max(x1, sup[1]); y0 = Math.min(y0, sup[2]); y1 = Math.max(y1, sup[3]); }
  const zmp = f.body?.zmp;
  if (zmp) { x0 = Math.min(x0, zmp[0]); x1 = Math.max(x1, zmp[0]); y0 = Math.min(y0, zmp[1]); y1 = Math.max(y1, zmp[1]); }
  const pad = 0.03;
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const top = compact ? 3 : 15;
  const bottom = compact ? 11 : 30;
  // 화면 위 = 앞(x), 오른쪽 = 로봇의 오른쪽(-y)
  const s = Math.min((W - 8) / (y1 - y0), (H - top - bottom) / (x1 - x0));
  const cxp = W / 2 - ((y0 + y1) / 2) * -s;
  const X = (y) => cxp + -y * s;
  const Y = (x) => top + (x1 - x) * s;
  const weight = (f.body?.mass_kg ?? 1) * 9.81;
  for (const ft of feet) {
    const [rx0, rx1, ry0, ry1] = ft.rect;
    const load = ft.normal_n / weight;
    ctx.fillStyle = ft.in_air ? 'transparent' : `rgba(95, 185, 140, ${0.15 + 0.6 * Math.min(1, load)})`;
    ctx.strokeStyle = ft.in_air ? css('--text-muted') : (ft.rcof > ft.mu ? css('--danger') : css('--success'));
    ctx.setLineDash(ft.in_air ? [3, 2] : []);
    ctx.lineWidth = 1.2;
    const px = Math.min(X(ry0), X(ry1));
    ctx.fillRect(px, Y(rx1), Math.abs(X(ry1) - X(ry0)), (rx1 - rx0) * s);
    ctx.strokeRect(px, Y(rx1), Math.abs(X(ry1) - X(ry0)), (rx1 - rx0) * s);
    ctx.setLineDash([]);
    if (!ft.in_air) {
      ctx.fillStyle = css('--text-bright');
      ctx.beginPath();
      ctx.arc(X(ft.cop[1]), Y(ft.cop[0]), compact ? 1.8 : 2.6, 0, Math.PI * 2);
      ctx.fill();
      if (Math.abs(ft.shear_n) > 1 && !compact) arrow(ctx, X(ft.cop[1]), Y(ft.cop[0]), X(ft.cop[1]), Y(ft.cop[0]) - Math.sign(ft.shear_n) * Math.min(18, Math.abs(ft.shear_n) * 0.3), css('--warning'), 1.2);
    }
  }
  if (!compact) {
    const by = (side) => feet.find((x) => x.side === side);
    const load = (ft) => (!ft ? '—' : ft.in_air ? '공중' : `${n(ft.normal_n, 0)} N`);
    label(g, `오른 ${load(by('r'))} · 왼 ${load(by('l'))}`, 4, H - 18, css('--text-dim'));
  }
  if (sup) {
    ctx.strokeStyle = 'rgba(160,160,170,0.5)';
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(Math.min(X(sup[2]), X(sup[3])), Y(sup[1]), Math.abs(X(sup[3]) - X(sup[2])), (sup[1] - sup[0]) * s);
    ctx.setLineDash([]);
  }
  if (zmp) {
    ctx.fillStyle = '#f472b6';
    ctx.fillRect(X(zmp[1]) - 2.5, Y(zmp[0]) - 2.5, 5, 5);
  }
  const stance = feet.filter((x) => !x.in_air);
  const rcof = Math.max(0, ...stance.map((x) => x.rcof));
  label(g, compact ? `여유 ${n(f.body?.margin, 2)}` : `필요 마찰 ${n(rcof, 2)} / 지면 ${n(feet[0]?.mu, 2)} · 안정 여유 ${n(f.body?.margin, 2)}`, 4, H - (compact ? 2 : 6), rcof > (feet[0]?.mu ?? 1) ? css('--danger') : css('--text-dim'));
  if (!compact) label(g, '발 모양 (위에서 · ↑ 앞 · 분홍 = ZMP · 흰 점 = 압력 중심)', 4, 11, css('--text-muted'));
}

// ── 발 관절: 오른/왼 다리의 고관절 굽힘 · 벌림 · 무릎 · 발목 ──
function drawFootJoints(g, f) {
  const { W, H, compact } = g;
  const order = ['hip_pitch', 'hip_roll', 'knee', 'ankle'];
  const ko = { hip_pitch: '고관절', hip_roll: '벌림', knee: '무릎', ankle: '발목' };
  const js = f.foot_joints ?? [];
  const top = compact ? 3 : 16;
  const left = compact ? 3 : 38;
  const rowH = (H - top - 3) / order.length;
  const colW = (W - left - 4) / 2;
  const small = { ...g, fs: 8 };
  order.forEach((k, r) => {
    const y = top + r * rowH;
    if (!compact) label(g, ko[k], 3, y + rowH * 0.65, css('--text-dim'));
    ['r', 'l'].forEach((side, c) => {
      const j = js.find((x) => x.name === `${k}_${side}`);
      if (!j) return;
      const x = left + c * colW;
      // The bar on top; under it the angle (and the torque where the column is wide).
      const text = !compact && rowH >= 17;
      bar(g, x, y + 2, colW - 4, Math.max(2, (text ? 0.38 : 0.55) * rowH), j.ratio, ratioColor(j.ratio));
      if (text) label(small, colW >= 70 ? `${n(j.angle_deg, 0)}° · ${n(j.torque_nm, 1)} N·m` : `${n(j.angle_deg, 0)}°`, x + colW - 6, y + rowH - 3, j.in_range ? css('--text-muted') : css('--danger'), 'right');
    });
  });
  if (!compact) label(g, '발 관절 (왼쪽 열 = 오른 다리 · 막대 = 토크/피크)', 4, 11, css('--text-muted'));
}

// ── 허리: 몸통 각과 허리 피치 관절 ──
function drawWaist(g, f) {
  const { ctx, W, H, compact } = g;
  const deg = f.body?.trunk_deg ?? 0;
  const cx = W * 0.35;
  const cy = H - (compact ? 12 : 26);
  const L = Math.min(H * 0.62, W * 0.45);
  const a = (deg * Math.PI) / 180;
  ctx.strokeStyle = css('--plot-grid-strong');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx, cy - L);
  ctx.stroke();
  ctx.strokeStyle = css('--text-muted');
  ctx.beginPath();
  ctx.arc(cx, cy, L * 0.35, -Math.PI / 2, -Math.PI / 2 + a, a < 0);
  ctx.stroke();
  // Without a pitch actuator the trunk is held by the structure: no torque limit to show.
  const w = f.waist && !f.waist.structural && f.waist.peak_nm > 0 ? f.waist : null;
  const color = w ? ratioColor(w.ratio) : css('--accent-strong');
  ctx.strokeStyle = color;
  ctx.lineWidth = compact ? 3 : 5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + L * Math.sin(a), cy - L * Math.cos(a));
  ctx.stroke();
  ctx.fillStyle = css('--text-bright');
  ctx.beginPath();
  ctx.arc(cx, cy, compact ? 2.5 : 3.5, 0, Math.PI * 2);
  ctx.fill();
  label(g, `${n(deg, 0)}°`, cx + 6, cy - 4, css('--text-bright'));
  if (w) {
    bar(g, W * 0.62, compact ? H * 0.35 : H * 0.4, W * 0.34, compact ? 4 : 6, w.ratio, ratioColor(w.ratio));
    label(g, compact ? `${Math.round(w.ratio * 100)}%` : `피치 ${n(w.torque_nm, 1)}/${n(w.peak_nm, 0)} N·m`, W * 0.62, (compact ? H * 0.35 : H * 0.4) - 3, css('--text-dim'));
  } else if (!compact) {
    label(g, '허리 피치 관절 없음 (구조)', W * 0.5, H * 0.4, css('--text-muted'));
    if (f.waist) label(g, `${n(f.waist.torque_nm, 1)} N·m`, W * 0.5, H * 0.4 + 12, css('--text-muted'));
  }
  if (!compact) {
    label(g, `CoM 높이 ${n(f.body?.com?.[2], 2)} m · 여유 ${n(f.body?.margin, 2)}${f.body?.limiting ? ` · ${f.body.limiting}` : ''}`, 4, H - 6, f.body?.feasible === false ? css('--danger') : css('--text-dim'));
    label(g, '허리 (몸통 기울기 · 앞 → 오른쪽)', 4, 11, css('--text-muted'));
  }
}

// ── 목 관절: 카메라가 보는 방향 ──
function drawNeck(g, f) {
  const { ctx, W, H, compact } = g;
  const nk = f.neck;
  const pitch = nk?.camera_pitch_deg ?? 0;
  const cx = W * 0.28;
  const cy = H * 0.42;
  const r = Math.min(H, W) * (compact ? 0.18 : 0.16);
  ctx.strokeStyle = css('--plot-grid-strong');
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(W - 4, cy);
  ctx.stroke();
  ctx.fillStyle = css('--bg-active');
  ctx.strokeStyle = nk ? ratioColor(nk.joint.ratio) : css('--text-muted');
  ctx.lineWidth = compact ? 1.5 : 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  const a = (pitch * Math.PI) / 180;
  const len = W * 0.6;
  ctx.strokeStyle = 'rgba(125, 211, 252, 0.85)';
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + len * Math.cos(a), cy + len * Math.sin(a));
  ctx.stroke();
  ctx.setLineDash([]);
  label(g, `${n(pitch, 0)}°`, cx + r + 4, cy - 4, css('--text-bright'));
  if (nk) {
    bar(g, 4, H - (compact ? 7 : 20), W * 0.5, compact ? 3 : 5, nk.joint.ratio, ratioColor(nk.joint.ratio));
    if (!compact) label(g, `목 ${n(nk.joint.angle_deg, 0)}° · ${n(nk.joint.torque_nm, 2)}/${n(nk.joint.peak_nm, 1)} N·m`, 4, H - 6, css('--text-dim'));
  }
  if (!compact) label(g, '목 관절 (카메라 시선 · 수평 아래로 +)', 4, 11, css('--text-muted'));
}

// ── 바퀴: 앞·뒤 축 하중과 구동 미끄럼 ──
function drawWheel(g, f) {
  const { ctx, W, H, compact } = g;
  const ws = f.wheel ?? [];
  const rear = ws.find((w) => w.axle.includes('rear')) ?? ws[0];
  const front = ws.find((w) => w.axle.includes('front')) ?? ws[1];
  const weight = (f.body?.mass_kg ?? 1) * 9.81;
  const cy = H * (compact ? 0.62 : 0.55);
  const r = Math.min(H * 0.2, W * 0.12);
  const xs = [W * 0.28, W * 0.72];
  ctx.strokeStyle = css('--text-muted');
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xs[0], cy - r * 0.9);
  ctx.lineTo(xs[1], cy - r * 0.9);
  ctx.stroke();
  [rear, front].forEach((w, i) => {
    if (!w) return;
    const load = w.normal_n / weight;
    ctx.fillStyle = css('--bg-active');
    ctx.strokeStyle = w.normal_n < 1 ? css('--danger') : ratioColor(Math.abs(w.shear_n) / Math.max(1, w.mu * w.normal_n));
    ctx.lineWidth = compact ? 2 : 3;
    ctx.beginPath();
    ctx.arc(xs[i], cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    const len = Math.min(H * 0.3, load * H * 0.4);
    arrow(ctx, xs[i], cy + r + len + 2, xs[i], cy + r + 2, 'rgba(250,250,250,0.6)', 1.2);
    if (!compact) label(g, `${i ? '앞' : '뒤'} ${n(w.normal_n, 0)} N`, xs[i], cy - r - 5, css('--text-dim'), 'center');
  });
  const slip = (rear ?? front)?.slip ?? 0;
  label(g, compact ? `미끄럼 ${Math.round(slip * 100)}%` : `구동 미끄럼 ${Math.round(slip * 100)}% · 마찰 ${n((rear ?? front)?.mu, 2)}`, 4, H - (compact ? 2 : 6), slip > 0.2 ? css('--warning') : css('--text-dim'));
  if (!compact) label(g, '바퀴 (← 뒤 · 앞 →, 화살표 = 접지력)', 4, 11, css('--text-muted'));
}

// ── 트랙: 접지 길이에 걸친 압력 분포 ──
function drawTrack(g, f) {
  const { ctx, W, H, compact } = g;
  const t = f.track;
  const p = t.pressure_kpa ?? [0, 0];
  const pmax = Math.max(1, p[0], p[1]);
  const x0 = W * 0.1;
  const x1 = W * 0.9;
  const base = H * (compact ? 0.7 : 0.62);
  const top = compact ? 4 : 18;
  const scale = (base - top) / pmax;
  ctx.fillStyle = 'rgba(214, 182, 94, 0.35)';
  ctx.strokeStyle = css('--warning');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, base);
  ctx.lineTo(x0, base - p[0] * scale);
  ctx.lineTo(x1, base - p[1] * scale);
  ctx.lineTo(x1, base);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = css('--bg-active');
  ctx.strokeStyle = css('--text-muted');
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x0 - 4, base, x1 - x0 + 8, compact ? 7 : 10, 5);
  ctx.fill();
  ctx.stroke();
  if (!compact) {
    label(g, `${n(p[0], 1)} kPa`, x0, base - p[0] * scale - 3, css('--text-dim'));
    label(g, `${n(p[1], 1)} kPa`, x1, base - p[1] * scale - 3, css('--text-dim'), 'right');
  }
  label(g, compact ? `미끄럼 ${Math.round((t.slip ?? 0) * 100)}%` : `접지 ${n(t.loaded_len_m, 2)} m · 미끄럼 ${Math.round((t.slip ?? 0) * 100)}% · 마찰 ${n(t.mu, 2)}`, 4, H - (compact ? 2 : 6), (t.slip ?? 0) > 0.2 ? css('--warning') : css('--text-dim'));
  if (!compact) label(g, '트랙 (← 뒤 · 앞 →, 면 = 접지 압력)', 4, 11, css('--text-muted'));
}

/** 검사기용: 근접 영상 셋과 여덟 부위 그림. focusWebp 는 {hand, side, low} → base64. */
export function focusPanel(f, focusWebp, event) {
  const box = el('div', { class: 'fv-panel' });
  if (event) {
    box.append(el('div', { class: `fv-event ${event.ok ? 'ok' : 'fail'}` },
      el('b', { text: `${EVENT_KO[event.kind] ?? event.kind}${event.ok ? '' : ' · 실패'}` }),
      el('span', { text: ` ${event.label}` })));
  }
  if (f) {
    box.append(el('div', { class: 'fv-meta', text: `${f.action_ko ?? '정지'}${f.phase ? ` · ${f.phase}` : ''} · 부위 ${(f.parts ?? []).map((p) => PART_KO[p] ?? p).join(', ') || '—'}` }));
  }
  const cams = [['hand', '손 근접'], ['side', '옆 (허리·목·다리)'], ['low', '낮게 (발·바퀴·트랙)']];
  const pics = cams.filter(([k]) => focusWebp?.[k]);
  if (pics.length) {
    box.append(el('div', { class: 'fv-cams' }, ...pics.map(([k, ko]) => el('figure', {},
      el('img', { alt: ko, src: `data:image/webp;base64,${focusWebp[k]}` }),
      el('figcaption', { text: ko })))));
  }
  if (!f) {
    box.append(el('div', { class: 'tl-empty', text: '이 키프레임에는 부위 포커스가 없습니다.' }));
    return box;
  }
  const grid = el('div', { class: 'fv-grid' });
  const active = new Set(f.parts ?? []);
  for (const part of DRAWN_PARTS) {
    const cv = el('canvas', { class: 'fv-cv', aria: { label: PART_KO[part] } });
    const cell = el('figure', { class: `fv-cell${active.has(part) ? ' on' : ''}${partAvailable(f, part) ? '' : ' off'}` }, cv, el('figcaption', { text: PART_KO[part] }));
    grid.append(cell);
    requestAnimationFrame(() => drawFocus(cv, f, part));
  }
  box.append(grid);
  return box;
}
