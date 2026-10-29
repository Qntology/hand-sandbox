// 미니맵: 지형(world_terrain: 바다·절벽·높이), WorldSnapshot(경계·노드·드롭·웨이포인트·에이전트)과
// 지나온 경로(trail)를 그린다. 보라 빈 화살표는 로봇이 믿는 자기 위치(오도메트리)다. 웨이포인트 이름 옆 숫자는 그곳 더미에 쌓인 개수다.
// 월드 화면의 '경로' 토글이 이 오버레이를 켜고 끈다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, cssVar, fitCanvas, observeResize } from './util.js';

let terrainImg = null; // 셀 1개 = 1픽셀인 지형 그림 (위쪽 행 = 시뮬레이터 z = 0 = UI 의 y = depth)

async function loadTerrain() {
  terrainImg = null;
  if (state.mock) return;
  try {
    const g = await call('world_terrain', {}, { quiet: true });
    const colors = new Map((g.ground_kinds ?? []).map((k) => [k.code, k.color]));
    const c = document.createElement('canvas');
    c.width = g.cells_w;
    c.height = g.cells_d;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(g.cells_w, g.cells_d);
    const H = (x, z) => g.heights_cm[z * g.cells_w + x];
    // 사이트가 있으면 턱마다 종류(0 턱 벽, 1 계단 디딤, 2 현관·집 바닥, 3 경사로)가 온다: 턱 벽만 노랗게.
    const kinds = g.curb_kind ?? null;
    for (let z = 0; z < g.cells_d; z++) {
      for (let x = 0; x < g.cells_w; x++) {
        const i = z * g.cells_w + x;
        const h = H(x, z);
        const cliff = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dz]) => {
          const nx = x + dx;
          const nz = z + dz;
          return nx >= 0 && nz >= 0 && nx < g.cells_w && nz < g.cells_d && h - H(nx, nz) >= 50;
        });
        let rgb = colors.get(g.ground[i]) ?? [90, 90, 90];
        if (kinds && kinds[i]) {
          // 현관·계단·경사로: 그 바닥 색, 높을수록 밝게 (시뮬레이터의 위에서 본 그림과 같은 규칙).
          const k = 0.8 + 0.4 * Math.min(g.curb_cm[i] / 100, 0.5);
          rgb = rgb.map((v) => Math.min(255, v * k));
        } else if (g.curb_cm[i] > 0) rgb = [232, 196, 48];
        else if (cliff) rgb = [40, 38, 36];
        else {
          const k = 1 - Math.min(h * 0.0035, 0.35);
          rgb = rgb.map((v) => v * k);
        }
        img.data.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
      }
    }
    ctx.putImageData(img, 0, 0);
    terrainImg = c;
  } catch {
    terrainImg = null;
  }
  draw();
}

// 시뮬레이터의 노드 이름(tree·boulder·coal_ore·iron_ore_node)과 모의 백엔드의 이름(rock·iron_ore·chest)을 같이 받는다.
const NODE_COLOR = {
  tree: '#4f9a5b', boulder: '#8b8d94', coal_ore: '#6b6b78', iron_ore_node: '#c4764a',
  rock: '#8b8d94', iron_ore: '#c4764a', chest: '#d6b65e',
};

// 사이트의 고정물 색: 벽·문·선반·포장 테이블 (현관·계단·경사로는 지형 그림에 있다).
const FIXTURE_COLOR = { wall: '#4a4644', door: '#9a6a42', shelf: '#3a6ab0', table: '#6a7c8e' };

// 집·물류센터: 칸마다 0.5 m 네모로 고정물을, 건물 이름을 가운데에 그린다.
function drawSite(ctx, site, X, Y, s) {
  if (!site) return;
  const half = 0.25;
  for (const f of site.fixtures ?? []) {
    const color = FIXTURE_COLOR[f.kind];
    if (!color) continue;
    ctx.fillStyle = color;
    for (const [cx, cy] of f.cells) ctx.fillRect(X(cx - half), Y(cy + half), Math.max(1, 2 * half * s), Math.max(1, 2 * half * s));
  }
  ctx.font = `9px ${cssVar('--font-sans')}`;
  ctx.textAlign = 'center';
  ctx.fillStyle = cssVar('--text-dim');
  const mid = (r) => [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2];
  for (const h of site.houses ?? []) {
    const [mx, my] = mid(h.rect);
    ctx.fillText(h.reachable_wheels ? h.label_ko : `${h.label_ko} (계단만)`, X(mx), Y(my) + 3);
  }
  if (site.warehouse) {
    const [mx] = mid(site.warehouse.rect);
    ctx.fillText(site.warehouse.label_ko, X(mx), Y(site.warehouse.rect[3]) + 10);
  }
}

// 서브 작업자: 휴머노이드는 주황 원, 지게차는 노란 네모와 포크 — 번호를 붙이고, 짐을 들었으면 테두리를 두른다.
// 하차한 탑승자(최초 작업자의 것 포함)는 작은 사람 점으로 따로 그린다.
function drawCrew(ctx, snap, X, Y) {
  ctx.font = `8px ${cssVar('--font-sans')}`;
  ctx.textAlign = 'center';
  const rider = (p) => {
    if (!Array.isArray(p)) return;
    ctx.fillStyle = '#d65c28';
    ctx.strokeStyle = 'rgba(0,0,0,0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(X(p[0]), Y(p[1]), 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  };
  rider(snap.rider);
  for (const c of snap.crew ?? []) {
    const [cx, cy] = c.pos;
    const yaw = ((c.yaw_deg ?? 0) * Math.PI) / 180;
    const loaded = (c.carrying ?? []).length > 0;
    ctx.save();
    ctx.translate(X(cx), Y(cy));
    ctx.rotate(-yaw);
    ctx.lineWidth = loaded ? 1.8 : 1;
    ctx.strokeStyle = loaded ? cssVar('--text-bright') : 'rgba(0,0,0,0.65)';
    if (c.form === 'forklift') {
      ctx.fillStyle = '#e8c028';
      ctx.fillRect(-4.5, -3.2, 7, 6.4);
      ctx.strokeRect(-4.5, -3.2, 7, 6.4);
      ctx.strokeStyle = '#3c3c44';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(2.5, -2);
      ctx.lineTo(7, -2);
      ctx.moveTo(2.5, 2);
      ctx.lineTo(7, 2);
      ctx.stroke();
    } else {
      ctx.fillStyle = '#ec8034';
      ctx.beginPath();
      ctx.arc(0, 0, 3.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(3.3, 0);
      ctx.lineTo(6, 0);
      ctx.stroke();
    }
    ctx.restore();
    rider(c.rider);
    ctx.fillStyle = cssVar('--text-bright');
    ctx.fillText(String(c.id), X(cx), Y(cy) - 6);
  }
}

function draw() {
  const canvas = $('minimap');
  if (!canvas.offsetParent) return;
  const fit = fitCanvas(canvas);
  if (!fit) return;
  const { ctx, W, H } = fit;
  ctx.clearRect(0, 0, W, H);
  const snap = state.snapshot;
  if (!snap) return;
  const pad = 8;
  const [bx, by] = snap.bounds;
  const s = Math.min((W - 2 * pad) / bx, (H - 2 * pad) / by);
  const ox = (W - bx * s) / 2;
  const oy = (H - by * s) / 2;
  const X = (x) => ox + x * s;
  const Y = (y) => H - oy - y * s; // +y 가 위쪽

  ctx.fillStyle = cssVar('--bg-card');
  ctx.fillRect(X(0), Y(by), bx * s, by * s);
  if (terrainImg) {
    ctx.save();
    ctx.globalAlpha = 0.62;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(terrainImg, X(0), Y(by), bx * s, by * s);
    ctx.restore();
  }
  ctx.strokeStyle = cssVar('--border-strong');
  ctx.lineWidth = 1;
  ctx.strokeRect(X(0) + 0.5, Y(by) + 0.5, bx * s - 1, by * s - 1);
  drawSite(ctx, snap.site, X, Y, s);

  for (const n of snap.nodes) {
    ctx.fillStyle = NODE_COLOR[n.kind] ?? '#999';
    ctx.beginPath();
    ctx.arc(X(n.pos[0]), Y(n.pos[1]), n.kind === 'tree' ? 2.2 : 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = cssVar('--text-bright');
  for (const d of snap.drops) ctx.fillRect(X(d.pos[0]) - 1.5, Y(d.pos[1]) - 1.5, 3, 3);
  // 현관·선반·테이블 위의 택배: 받침 위에 있다는 테두리 (높이는 pos[2]).
  ctx.strokeStyle = cssVar('--warning');
  ctx.lineWidth = 1;
  for (const d of snap.drops) if (d.support) ctx.strokeRect(X(d.pos[0]) - 3.5, Y(d.pos[1]) - 3.5, 7, 7);

  // 목표 웨이포인트: 활성 퀘스트의 현재 단계 문구에 이름이 들어 있는 곳
  const q = (state.quests ?? []).find((x) => x.state === 'active');
  const stepLabel = (q?.steps ?? []).find((x) => x.state === 'active')?.label_ko ?? '';
  ctx.font = `9px ${cssVar('--font-sans')}`;
  for (const w of snap.waypoints) {
    const x = X(w.pos[0]);
    const y = Y(w.pos[1]);
    const target = stepLabel.includes(w.label_ko);
    ctx.fillStyle = '#6bb3e8';
    ctx.fillRect(x - 3, y - 3, 6, 6);
    if (target) {
      ctx.strokeStyle = cssVar('--warning');
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.stroke();
    }
    const n = (w.stock ?? []).reduce((a, it) => a + (it.amount ?? it.count ?? 0), 0);
    ctx.fillStyle = cssVar('--text-dim');
    ctx.textAlign = x > W * 0.6 ? 'right' : 'left';
    const label = n > 1e-3 ? `${w.label_ko} ▣${Math.abs(n - Math.round(n)) < 1e-3 ? Math.round(n) : n.toFixed(1)}` : w.label_ko;
    ctx.fillText(label, x + (x > W * 0.6 ? -6 : 6), y - 5);
  }

  if (state.trail.length > 1) {
    ctx.strokeStyle = 'rgba(250,250,250,0.55)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    state.trail.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y))));
    ctx.stroke();
  }
  const [ax, ay] = snap.agent.pos;
  const yaw = (snap.agent.yaw_deg * Math.PI) / 180;
  // Where the robot believes it is (its odometry): a hollow arrow, joined to the true one.
  const belief = snap.agent.belief;
  if (belief && belief.error_m > 0.02) {
    const [bxp, byp] = belief.pos;
    ctx.strokeStyle = cssVar('--violet');
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(X(ax), Y(ay));
    ctx.lineTo(X(bxp), Y(byp));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.save();
    ctx.translate(X(bxp), Y(byp));
    ctx.rotate(-(belief.yaw_deg * Math.PI) / 180);
    ctx.beginPath();
    ctx.moveTo(6, 0);
    ctx.lineTo(-4, 3.6);
    ctx.lineTo(-4, -3.6);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
  drawCrew(ctx, snap, X, Y);
  // 움직임: 후진이면 꽁무니 쪽으로 화살 꼬리를, 3점 회전·제자리 회전이면 몸 둘레에 호를 그린다.

  const motion = snap.agent.motion;
  ctx.save();
  ctx.translate(X(ax), Y(ay));
  ctx.rotate(-yaw);
  ctx.fillStyle = cssVar('--text-bright');
  ctx.beginPath();
  ctx.moveTo(6, 0);
  ctx.lineTo(-4, 3.6);
  ctx.lineTo(-4, -3.6);
  ctx.closePath();
  ctx.fill();
  if (motion?.reversing) {
    ctx.strokeStyle = cssVar('--warning');
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(-5, 0);
    ctx.lineTo(-12, 0);
    ctx.moveTo(-12, 0);
    ctx.lineTo(-9, 2.5);
    ctx.moveTo(-12, 0);
    ctx.lineTo(-9, -2.5);
    ctx.stroke();
  } else if (motion && (motion.kind === 'kturn' || motion.kind === 'pivot')) {
    ctx.strokeStyle = cssVar('--warning');
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(0, 0, 9, -0.9, 0.9);
    ctx.stroke();
  }
  ctx.restore();  $('mm-left').textContent = `${bx} × ${by} m`;
  const moveKo = motion ? { reverse: '후진', kturn: '3점 회전', pivot: '제자리 회전' }[motion.kind] : null;
  const where = belief ? `(${ax.toFixed(1)}, ${ay.toFixed(1)}) · 믿음 오차 ${belief.error_m.toFixed(2)} m` : `(${ax.toFixed(1)}, ${ay.toFixed(1)})`;
  $('mm-right').textContent = moveKo ? `${where} · ${moveKo}` : where;
  $('mm-right').title = motion?.label_ko ?? '';
}

export function initWorldMap() {
  bus.on('world-new', loadTerrain);
  bus.on('snapshot', draw);
  bus.on('quests', draw);
  bus.on('layout', draw);
  observeResize($('minimap'), draw);
  draw();
}
