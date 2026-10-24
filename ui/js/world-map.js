// 미니맵: 지형(world_terrain: 바다·절벽·높이), WorldSnapshot(경계·노드·드롭·웨이포인트·에이전트)과
// 지나온 경로(trail)를 그린다. 웨이포인트 이름 옆 숫자는 그곳 더미에 쌓인 개수다.
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
        if (g.curb_cm[i] > 0) rgb = [232, 196, 48];
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

  for (const n of snap.nodes) {
    ctx.fillStyle = NODE_COLOR[n.kind] ?? '#999';
    ctx.beginPath();
    ctx.arc(X(n.pos[0]), Y(n.pos[1]), n.kind === 'tree' ? 2.2 : 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = cssVar('--text-bright');
  for (const d of snap.drops) ctx.fillRect(X(d.pos[0]) - 1.5, Y(d.pos[1]) - 1.5, 3, 3);

  // 목표 웨이포인트: 활성 퀘스트의 현재 단계 문구에 이름이 들어 있는 곳
  const q = state.quests.find((x) => x.state === 'active');
  const stepLabel = q?.steps.find((x) => x.state === 'active')?.label_ko ?? '';
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
  ctx.restore();
  $('mm-left').textContent = `${bx} × ${by} m`;
  $('mm-right').textContent = `(${ax.toFixed(1)}, ${ay.toFixed(1)})`;
}

export function initWorldMap() {
  bus.on('world-new', loadTerrain);
  bus.on('snapshot', draw);
  bus.on('quests', draw);
  bus.on('layout', draw);
  observeResize($('minimap'), draw);
  draw();
}
