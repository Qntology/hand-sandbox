// 모의 프레임 렌더러: 오프스크린 캔버스에 단순한 가짜 3D 장면(하늘 그라데이션, 지면 체크 무늬, 나무·바위·광석 블록)을
// 에이전트 기준으로 그리고 WebP 로 인코딩한다. 실제 world-sim 의 레이캐스터가 아니다.
import { DETECT_LABEL, visibleObjects, world } from './mock-world.js';

// 1×1 무손실 WebP (캔버스를 쓸 수 없는 환경용 대체 프레임)
const BLANK_WEBP = 'UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==';
let cv = null;

const clampInt = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(Number(v) || lo)));
const hash = (s) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
};
const DROP_COLOR = { wood: '#b98a4e', iron_ore: '#c4764a', coal: '#1c1c20', stone: '#9a9ca2' };

function detection(o, bbox, w, h, tick) {
  const [id, ko] = o.drop ? [`${o.item_id}_drop`, `${o.label_ko} (드롭)`] : DETECT_LABEL[o.kind];
  const base = 0.97 - o.dist * 0.03 + (hash(o.id) - 0.5) * 0.08 + Math.sin(tick / 9 + hash(o.id) * 6) * 0.02;
  const score = Math.max(0.12, Math.min(0.99, base));
  const accepted = score >= 0.55;
  const [x0, y0, x1, y1] = bbox;
  return {
    label_id: id, label_ko: ko, score: Number(score.toFixed(3)), accepted,
    reason: accepted ? `점수 ${score.toFixed(2)} ≥ 0.55` : `보류: 점수 ${score.toFixed(2)} < 0.55 (거리 ${o.dist.toFixed(1)} m)`,
    bbox: [Math.max(0, x0 / w), Math.max(0, y0 / h), Math.min(1, x1 / w), Math.min(1, y1 / h)].map((v) => Number(v.toFixed(4))),
  };
}

function drawObject(ctx, o, sx, gy, s) {
  const col = (c) => { ctx.fillStyle = c; };
  if (o.kind === 'tree') {
    col('#5b3d24'); ctx.fillRect(sx - 0.15 * s, gy - 1.4 * s, 0.3 * s, 1.4 * s);
    col('#2f7a3c'); ctx.beginPath(); ctx.arc(sx, gy - 2.2 * s, 1.0 * s, 0, Math.PI * 2); ctx.fill();
    col('#3e9650'); ctx.beginPath(); ctx.arc(sx - 0.25 * s, gy - 2.4 * s, 0.55 * s, 0, Math.PI * 2); ctx.fill();
    return [sx - 1.0 * s, gy - 3.2 * s, sx + 1.0 * s, gy];
  }
  if (o.kind === 'rock' || o.kind === 'iron_ore' || o.kind === 'coal_ore') {
    const body = { rock: '#7d7f85', iron_ore: '#6d6a68', coal_ore: '#3a3a40' }[o.kind];
    col(body); ctx.beginPath();
    ctx.moveTo(sx - 0.6 * s, gy); ctx.lineTo(sx - 0.45 * s, gy - 0.6 * s); ctx.lineTo(sx - 0.1 * s, gy - 0.85 * s);
    ctx.lineTo(sx + 0.4 * s, gy - 0.7 * s); ctx.lineTo(sx + 0.62 * s, gy); ctx.closePath(); ctx.fill();
    if (o.kind !== 'rock') {
      col(o.kind === 'iron_ore' ? '#c4764a' : '#0d0d10');
      for (let i = 0; i < 6; i++) ctx.fillRect(sx + (hash(o.id + i) - 0.5) * 0.9 * s, gy - (0.15 + hash(`${o.id}y${i}`) * 0.55) * s, 0.14 * s, 0.14 * s);
    }
    return [sx - 0.62 * s, gy - 0.85 * s, sx + 0.62 * s, gy];
  }
  if (o.kind === 'chest') {
    col('#8a5a2b'); ctx.fillRect(sx - 0.45 * s, gy - 0.5 * s, 0.9 * s, 0.5 * s);
    col('#d6b65e'); ctx.fillRect(sx - 0.45 * s, gy - 0.3 * s, 0.9 * s, 0.08 * s);
    return [sx - 0.45 * s, gy - 0.5 * s, sx + 0.45 * s, gy];
  }
  if (o.drop) {
    const r = 0.22 * s;
    col(DROP_COLOR[o.item_id] ?? '#ddd');
    ctx.beginPath(); ctx.moveTo(sx, gy - 2 * r); ctx.lineTo(sx + r, gy - r); ctx.lineTo(sx, gy); ctx.lineTo(sx - r, gy - r); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = Math.max(1, s * 0.03); ctx.stroke();
    return [sx - r, gy - 2 * r, sx + r, gy];
  }
  return null;
}

function drawAgentView(ctx, w, h, W, dets) {
  const horizon = Math.round(h * 0.42);
  const f = w * 0.9;
  const camH = 1.0;
  const [ax, ay] = W.agent.pos;
  const yaw = W.agent.yaw;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const sky = ctx.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, '#27456b'); sky.addColorStop(1, '#a5bed8');
  ctx.fillStyle = sky; ctx.fillRect(0, 0, w, horizon);
  ctx.fillStyle = '#40604d';
  for (let x = 0; x < w; x += 4) {
    const ang = yaw + (x - w / 2) / f;
    const hh = 12 + 12 * Math.sin(ang * 3 + W.seed) + 7 * Math.sin(ang * 7.3);
    ctx.fillRect(x, horizon - hh, 4, hh);
  }
  const colors = [[['#42733c', '#3a6836'], ['#6b5a45', '#62523f']], [['#3b6a37', '#34602f'], ['#5f503d', '#584a38']]];
  for (let y = horizon + 1; y < h; y += 2) {
    const d = (camH * f) / (y - horizon);
    const fog = Math.min(1, d / 40);
    for (let x = 0; x < w; x += 8) {
      const ro = ((x + 4 - w / 2) / f) * d;
      const wx = ax + cy * d + sy * ro;
      const wy = ay + sy * d - cy * ro;
      const outside = wx < 0 || wy < 0 || wx > W.bounds[0] || wy > W.bounds[1];
      const parity = (Math.floor(wx / 1.5) + Math.floor(wy / 1.5)) & 1;
      ctx.fillStyle = colors[fog > 0.5 ? 1 : 0][outside ? 1 : 0][parity];
      ctx.fillRect(x, y, 8, 2);
    }
  }
  const haze = ctx.createLinearGradient(0, horizon, 0, horizon + h * 0.12);
  haze.addColorStop(0, 'rgba(165,190,216,0.75)'); haze.addColorStop(1, 'rgba(165,190,216,0)');
  ctx.fillStyle = haze; ctx.fillRect(0, horizon, w, h * 0.12);

  const objs = visibleObjects(28);
  for (const o of objs) {
    if (o.forward < 0.7) continue;
    const s = f / o.forward;
    const sx = w / 2 + (o.right / o.forward) * f;
    const gy = horizon + camH * s;
    if (sx < -3 * s || sx > w + 3 * s) continue;
    ctx.globalAlpha = Math.max(0.4, 1 - o.forward / 36);
    if (o.kind === 'waypoint') {
      ctx.fillStyle = '#e8e8ea'; ctx.fillRect(sx - 0.04 * s, gy - 3 * s, 0.08 * s, 3 * s);
      ctx.fillStyle = '#4aa3df'; ctx.beginPath(); ctx.moveTo(sx, gy - 3 * s); ctx.lineTo(sx + 0.9 * s, gy - 2.7 * s); ctx.lineTo(sx, gy - 2.4 * s); ctx.fill();
      ctx.globalAlpha = 1;
      continue;
    }
    const bbox = drawObject(ctx, o, sx, gy, s);
    ctx.globalAlpha = 1;
    if (bbox && o.forward < 20 && bbox[2] - bbox[0] >= 5 && bbox[2] > 0 && bbox[0] < w) dets.push({ o, bbox });
  }
  // 손(그리퍼) 힌트
  ctx.fillStyle = 'rgba(30,30,34,0.85)';
  ctx.fillRect(w / 2 - w * 0.05, h - h * 0.07, w * 0.1, h * 0.07);
}

function drawTopView(ctx, w, h, W, dets) {
  const span = 28;
  const s = w / span;
  const [ax, ay] = W.agent.pos;
  const X = (x) => w / 2 + (x - ax) * s;
  const Y = (y) => h / 2 - (y - ay) * s;
  ctx.fillStyle = '#2b4430'; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#35553a'; ctx.fillRect(X(0), Y(W.bounds[1]), W.bounds[0] * s, W.bounds[1] * s);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1;
  for (let g = 0; g <= Math.max(W.bounds[0], W.bounds[1]); g += 2) {
    ctx.beginPath(); ctx.moveTo(X(g), 0); ctx.lineTo(X(g), h); ctx.moveTo(0, Y(g)); ctx.lineTo(w, Y(g)); ctx.stroke();
  }
  ctx.font = `${Math.max(10, Math.round(s * 0.45))}px sans-serif`;
  const r = { tree: 0.9, rock: 0.6, iron_ore: 0.55, coal_ore: 0.55, chest: 0.4, drop: 0.25, waypoint: 0.5 };
  const col = { tree: '#2f7a3c', rock: '#7d7f85', iron_ore: '#c4764a', coal_ore: '#25252a', chest: '#d6b65e', waypoint: '#4aa3df' };
  for (const o of visibleObjects(30).reverse()) {
    const x = X(o.pos[0]);
    const y = Y(o.pos[1]);
    if (x < -20 || y < -20 || x > w + 20 || y > h + 20) continue;
    const rad = (r[o.kind] ?? 0.4) * s;
    ctx.fillStyle = o.drop ? (DROP_COLOR[o.item_id] ?? '#ddd') : col[o.kind];
    if (o.kind === 'waypoint') ctx.fillRect(x - rad * 0.6, y - rad * 0.6, rad * 1.2, rad * 1.2);
    else { ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fill(); }
    if (o.kind === 'waypoint') { ctx.fillStyle = '#d4d4d8'; ctx.fillText(o.label_ko, x + rad, y - rad); }
    else if (o.dist < 12) dets.push({ o, bbox: [x - rad, y - rad, x + rad, y + rad] });
  }
  ctx.save(); ctx.translate(w / 2, h / 2); ctx.rotate(-W.agent.yaw);
  ctx.fillStyle = '#fafafa'; ctx.beginPath(); ctx.moveTo(0.7 * s, 0); ctx.lineTo(-0.5 * s, 0.4 * s); ctx.lineTo(-0.5 * s, -0.4 * s); ctx.closePath(); ctx.fill();
  ctx.restore();
}

export function renderFrame({ width, height, quality, view }) {
  const W = world();
  if (!W) throw '월드가 없습니다. world_new 를 먼저 호출하세요';
  const w = clampInt(width, 64, 1024);
  const h = clampInt(height, 48, 768);
  const hits = [];
  let ctx = null;
  if (typeof document !== 'undefined') {
    cv = cv ?? document.createElement('canvas');
    if (cv.width !== w) cv.width = w;
    if (cv.height !== h) cv.height = h;
    try { ctx = cv.getContext('2d'); } catch { ctx = null; }
  }
  let b64 = BLANK_WEBP;
  let encode_ms = 0;
  if (ctx) {
    if (view === 'top') drawTopView(ctx, w, h, W, hits);
    else drawAgentView(ctx, w, h, W, hits);
    const t0 = performance.now();
    const url = cv.toDataURL('image/webp', Math.max(0.05, Math.min(1, (Number(quality) || 75) / 100)));
    encode_ms = performance.now() - t0;
    b64 = url.slice(url.indexOf(',') + 1);
  }
  const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  const detections = hits
    .sort((a, b) => a.o.dist - b.o.dist).slice(0, 12)
    .map(({ o, bbox }) => detection(o, bbox, w, h, W.tick));
  return { webp_base64: b64, width: w, height: h, bytes: Math.floor((b64.length * 3) / 4) - pad, encode_ms, tick: W.tick, detections };
}
