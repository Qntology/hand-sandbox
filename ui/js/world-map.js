// 미니맵: WorldSnapshot(경계·노드·드롭·웨이포인트·에이전트)과 지나온 경로(trail)를 그린다.
// 월드 화면의 '경로' 토글이 이 오버레이를 켜고 끈다.
import { bus, state } from './state.js';
import { $, cssVar, fitCanvas, observeResize } from './util.js';

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
    ctx.fillStyle = cssVar('--text-dim');
    ctx.textAlign = x > W * 0.6 ? 'right' : 'left';
    ctx.fillText(w.label_ko, x + (x > W * 0.6 ? -6 : 6), y - 5);
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
  bus.on('snapshot', draw);
  bus.on('quests', draw);
  bus.on('layout', draw);
  observeResize($('minimap'), draw);
  draw();
}
