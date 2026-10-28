// 모의 월드: 아주 작은 상태 기계. 실제 Rust 백엔드(world-sim, noto-quest)가 아니다.
// 명령 "철광석 3개 캐서 베이스캠프로 가져와" → 퀘스트(탐색→접근→채취→줍기→운반→인도)가 틱마다 진행되고,
// 에이전트가 움직이며 배터리·하중·피로 같은 신체 상태가 변한다.
import { BASES, BASE_TURN_SPECS } from './mock-data.js';
import { levelOf } from './util.js';

const DT = 0.2; // 틱당 초
const CAP_J = 30000; // 배터리 용량(가정)
const LOAD_LIMIT = 20; // 적재 한계 kg(가정)

const ITEMS = {
  parcel: { ko: '택배', node: null, mass: 1.5, names: ['택배', '소포', '상자', '박스'], aliases: ['짐', '배달물'] },
  iron_ore: { ko: '철광석', node: 'iron_ore', mass: 2.5, names: ['철광석', '철광', '철광물'], aliases: ['쇠돌', '철석'] },
  coal: { ko: '석탄', node: 'coal_ore', mass: 1.2, names: ['석탄'], aliases: ['숯'] },
  wood: { ko: '목재', node: 'tree', mass: 3.0, names: ['목재', '나무', '원목'], aliases: ['통나무'] },
  stone: { ko: '돌', node: 'rock', mass: 4.0, names: ['바위', '암석', '돌'], aliases: ['자갈'] },
};
const NODE_KO = { tree: '나무', rock: '바위', iron_ore: '철 광석 노드', coal_ore: '석탄 광석 노드', chest: '상자' };
const NODE_HP = { tree: 60, rock: 90, iron_ore: 120, coal_ore: 80, chest: 20 };
export const DETECT_LABEL = {
  tree: ['tree', '나무'], rock: ['rock', '바위'], iron_ore: ['iron_ore', '철광석'], coal_ore: ['coal_ore', '석탄 광석'], chest: ['chest', '상자'],
};
const WAYPOINTS = [
  { id: 'basecamp', label_ko: '베이스캠프', fx: 0.5, fy: 0.5, names: ['베이스캠프', '베이스', '캠프', '기지'] },
  { id: 'mine', label_ko: '광산 입구', fx: 0.8, fy: 0.3, names: ['광산입구', '광산'] },
  { id: 'forest', label_ko: '숲 가장자리', fx: 0.25, fy: 0.72, names: ['숲가장자리', '숲'] },
  { id: 'depot', label_ko: '창고', fx: 0.72, fy: 0.85, names: ['창고', '저장소'] },
];
const KO_NUM = { 한: 1, 하나: 1, 두: 2, 둘: 2, 세: 3, 셋: 3, 네: 4, 넷: 4, 다섯: 5, 여섯: 6, 일곱: 7, 여덟: 8, 아홉: 9, 열: 10 };

let W = null;

function mulberry32(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const hasWorld = () => W != null;
export const world = () => W;
const need = () => {
  if (!W) throw '월드가 없습니다. world_new 를 먼저 호출하세요';
  return W;
};

// ── 로그 ────────────────────────────────────────────────────────
export function log(cat, level, msg, data) {
  if (!W) return null;
  const line = { seq: W.nextSeq++, ts: new Date().toISOString(), cat, level, msg };
  if (data) line.data = data;
  W.logs.push(line);
  W.fresh.push(line);
  if (W.logs.length > 4000) W.logs.splice(0, 1000);
  return line;
}

export function logTail({ afterSeq, limit, category }) {
  if (!W) return { lines: [], next_seq: afterSeq ?? 0 };
  const lines = W.logs.filter((l) => l.seq > afterSeq && (!category || l.cat === category)).slice(0, Math.max(1, limit));
  return { lines, next_seq: lines.length ? lines[lines.length - 1].seq : afterSeq };
}

// ── 월드 생성 ───────────────────────────────────────────────────
export function worldNew({ seed, size, config, base }) {
  const r = mulberry32(Math.imul(seed + 1, 2654435761));
  const half = size * 0.5;
  const at = (fx, fy) => [half * fx, half * fy, 0];
  const nodes = [];
  const put = (kind, around, spread, n) => {
    for (let i = 0; i < n; i++) {
      const x = Math.min(half - 1, Math.max(1, around[0] + (r() - 0.5) * spread));
      const y = Math.min(half - 1, Math.max(1, around[1] + (r() - 0.5) * spread));
      nodes.push({ id: `n${nodes.length}`, kind, label_ko: NODE_KO[kind], pos: [x, y, 0], hp: NODE_HP[kind], hp0: NODE_HP[kind] });
    }
  };
  const scale = Math.max(1, size / 64);
  put('iron_ore', at(0.8, 0.3), half * 0.5, Math.round(7 * scale));
  put('coal_ore', at(0.75, 0.38), half * 0.55, Math.round(5 * scale));
  put('tree', at(0.25, 0.72), half * 0.6, Math.round(14 * scale));
  put('rock', at(0.5, 0.5), half * 1.4, Math.round(8 * scale));
  put('chest', at(0.5, 0.5), half * 1.2, 2);
  const waypoints = WAYPOINTS.map((w) => ({ id: w.id, label_ko: w.label_ko, pos: at(w.fx, w.fy) }));
  const drops = [];
  for (let i = 0; i < 3; i++) drops.push({ id: `d${i}`, item_id: 'wood', label_ko: '목재', pos: [waypoints[2].pos[0] + (r() - 0.5) * 4, waypoints[2].pos[1] + (r() - 0.5) * 4, 0] });
  for (let i = 0; i < 3; i++) drops.push({ id: `d${i + 3}`, item_id: 'parcel', label_ko: '택배', pos: [waypoints[0].pos[0] + 2.2 + (r() - 0.5) * 3, waypoints[0].pos[1] + (r() - 0.5) * 3, 0] });
  const standingEye = base === 'biped' || base === 'quadruped' ? 1.14 : 1.0;
  W = {
    seed, size, bounds: [half, half], tick: 0, time_s: 0, base: BASES[base] ? base : 'wheel', config,
    rng: r, nextSeq: 1, logs: [], fresh: [], quests: [], rt: new Map(), nextQuest: 1, nextDrop: 100,
    nodes, drops, waypoints,
    agent: {
      pos: [waypoints[0].pos[0] + 1.5, waypoints[0].pos[1], 0],
      yaw: 0,
      speed: 0,
      carrying: [],
      standing_eye_m: standingEye,
      eye_m: standingEye,
      mount_deg: -12.0,
      head_pitch_deg: 0.0,
      pitch_deg: -12.0,
      pan_deg: 0.0,
      roll_deg: 0.0,
    },
    energy: 0, fatigue: 0, heat: 0, visionMargin: 0.72, slipMargin: 0.8, lastMoved: 0, idleTicks: 0,
  };
  log('world', 'info', `월드 생성: 시드 ${seed}, ${half} × ${half} m, 노드 ${nodes.length}개`, { seed, size, base: W.base, nodes: nodes.length });
  log('device', 'info', '모의 백엔드: 실제 Rust 백엔드가 아닙니다 (UI 개발용 가짜 데이터)');
  return snapshot();
}

function stack(list) {
  const merged = new Map();
  for (const c of list) {
    const m = merged.get(c.item_id) ?? { item_id: c.item_id, label_ko: c.label_ko, count: 0, amount: 0, mass_kg: c.mass_kg, total_kg: 0 };
    m.count += 1;
    m.amount += 1;
    m.total_kg += c.mass_kg;
    merged.set(c.item_id, m);
  }
  return [...merged.values()];
}

export function setTeamMode({ teamSize = 2, activeRobot = 1 } = {}) {
  const w = need();
  w.teamSize = teamSize >= 2 ? 2 : 1;
  if (w.agent.activeRobot !== activeRobot && w.teamSize === 2) {
    const prevPos = [...w.agent.pos];
    const prevYaw = w.agent.yaw;
    if (w.partnerPos) {
      w.agent.pos = [...w.partnerPos];
      w.agent.yaw = w.partnerYaw ?? 0;
    } else {
      w.agent.pos = [prevPos[0] - 0.9, prevPos[1] + 0.5];
    }
    w.partnerPos = prevPos;
    w.partnerYaw = prevYaw;
  }
  w.agent.activeRobot = w.teamSize === 2 ? activeRobot : 1;
  return { snapshot: snapshot(), body: bodyStatus() };
}

export function snapshot() {
  const w = need();
  const teamSize = w.teamSize ?? 2;
  const activeRt = w.rt.get(w.quests.find((q) => q.state === 'active')?.id);
  const coPhase = activeRt?.coPhase ?? null;
  const activeRobot = coPhase ?? w.agent.activeRobot ?? 1;
  const partnerPos = w.partnerPos ?? [w.agent.pos[0] - 0.85, w.agent.pos[1] + 0.45];
  const partnerYawDeg = ((w.partnerYaw ?? w.agent.yaw) * 180) / Math.PI;
  return {
    tick: w.tick,
    time_s: w.tick * DT,
    seed: w.seed,
    bounds: w.bounds,
    terrain: 'open',
    team_size: teamSize,
    agent: {
      pos: [...w.agent.pos],
      yaw_deg: (w.agent.yaw * 180) / Math.PI,
      speed_mps: w.agent.speed,
      base: w.base,
      active_robot: activeRobot,
      co_phase: coPhase,
      partner: teamSize === 2 ? {
        robot_index: activeRobot === 1 ? 2 : 1,
        pos: partnerPos,
        yaw_deg: partnerYawDeg,
        holding: coPhase === 2,
      } : null,
      carrying: stack(w.agent.carrying),
      motion: w.agent.maneuver
        ? { kind: 'kturn', label_ko: '3점 회전: 전진·후진·전진', reversing: !!w.agent.reversing }
        : w.agent.reversing
          ? { kind: 'reverse', label_ko: '목표가 뒤쪽: 돌지 않고 후진', reversing: true }
          : null
    },
    waypoints: w.waypoints.map((p) => ({ ...p, stock: stack(p.pile ?? []) })),
    nodes: w.nodes.filter((n) => n.hp > 0).map(({ id, kind, label_ko, pos, hp }) => ({ id, kind, label_ko, pos, hp })),
    drops: w.drops,
  };
}

// ── 시야 안 물체(렌더·인식 공용) ─────────────────────────────────
export function visibleObjects(maxDist = 24) {
  const w = need();
  const [ax, ay] = w.agent.pos;
  const yaw = w.agent.yaw - ((w.agent.pan_deg ?? 0) * Math.PI) / 180;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const eyeM = w.agent.eye_m ?? 1.0;
  const pitchRad = ((w.agent.pitch_deg ?? -12.0) * Math.PI) / 180;
  const halfVfov = (25.0 * Math.PI) / 180;
  const out = [];
  const add = (o) => {
    const dx = o.pos[0] - ax;
    const dy = o.pos[1] - ay;
    const forward = dx * cy + dy * sy;
    const right = dx * sy - dy * cy;
    const dist = Math.hypot(dx, dy);
    if (dist > maxDist) return;
    const objH = o.drop ? (o.pos[2] ?? 0) + 0.12 : 0.8;
    const targetPitch = -Math.atan2(Math.max(0, eyeM - objH), Math.max(0.1, forward));
    const inPitch = forward > 0.15 && targetPitch >= pitchRad - halfVfov && targetPitch <= pitchRad + halfVfov;
    out.push({ ...o, dx, dy, forward, right, dist, target_pitch_deg: (targetPitch * 180) / Math.PI, in_pitch: inPitch });
  };
  for (const n of w.nodes) if (n.hp > 0) add({ ...n, drop: false });
  for (const d of w.drops) add({ ...d, kind: 'drop', drop: true });
  for (const p of w.waypoints) add({ ...p, kind: 'waypoint', drop: false });
  return out.sort((a, b) => b.dist - a.dist);
}

// ── 명령 해석 ───────────────────────────────────────────────────
const MINE = /(캐|캔|채굴|채취|벌목|베어|부숴|부수)/;
const COLLECT = /(줍|주워)/;
const CARRY = /(가져|운반|옮겨|옮겨|나르|갖다)/;

function findItem(t) {
  for (const [key, it] of Object.entries(ITEMS)) for (const n of it.names) if (t.includes(n)) return { key, it, hit: n, alias: false };
  for (const [key, it] of Object.entries(ITEMS)) for (const n of it.aliases) if (t.includes(n)) return { key, it, hit: n, alias: true };
  return null;
}

// 내려놓기(모의): 적재함에서 바로 빼서 가까운 웨이포인트 더미나 땅에 둔다.
function dropNow(key, all) {
  const a = W.agent;
  const pick = a.carrying.filter((c) => !key || c.item_id === key);
  if (!pick.length) return 0;
  const take = all || !key ? pick : pick.slice(0, 1);
  a.carrying = a.carrying.filter((c) => !take.includes(c));
  const wp = W.waypoints.find((p) => dist2(p.pos, a.pos) <= 1.6);
  if (wp) (wp.pile ??= []).push(...take);
  else for (const c of take) W.drops.push({ id: `d${W.nextDrop++}`, item_id: c.item_id, label_ko: c.label_ko, pos: [a.pos[0] + 0.4, a.pos[1], 0] });
  log('world', 'info', `(모의) ${take.length}개 내려놓음${wp ? ` → ${wp.label_ko} 더미` : ' → 땅'}`);
  return take.length;
}

export function inventoryDrop({ item, amount }) {
  need();
  const n = dropNow(item ?? null, amount == null);
  if (!n) throw '적재함에 내려놓을 물건이 없습니다';
  return { ok: true, quest_id: null, quests: questList() };
}

export function worldReset() {
  const w = need();
  const { energy, fatigue, heat } = w;
  worldNew({ seed: w.seed, size: w.size, config: w.config, base: w.base });
  Object.assign(W, { energy, fatigue, heat });
  log('world', 'info', '(모의) 월드 리셋: 신체 상태 유지');
  return { snapshot: snapshot(), quests: questList(), body: bodyStatus() };
}

export function bodyReset() {
  const w = need();
  Object.assign(w, { energy: 0, fatigue: 0, heat: 0 });
  log('world', 'info', '(모의) 신체 상태 리셋');
  return { snapshot: snapshot(), body: bodyStatus() };
}

export function commandSubmit({ text }) {
  need(); // 월드가 없으면 오류를 던진다
  const raw = String(text ?? '').trim();
  const t = raw.replace(/\s+/g, '');
  const trace = [{ stage: 'normalize', text: raw.replace(/\s+/g, ' '), score: null, role: null, accepted: true }];
  const fail = (error) => {
    trace.push({ stage: 'lexical', text: '아이템 후보 없음', score: 0, role: 'item', accepted: false });
    log('search', 'warn', `명령 해석 실패: "${raw}"`, { reason: 'no_item', mode: 'lexical' });
    return { ok: false, parsed: { action: 'unknown', item_id: null, item_label_ko: null, count: null, destination: null, confidence: 0 }, quest_id: null, trace, error, mode: 'lexical' };
  };
  if (!raw) return fail('명령이 비어 있습니다');
  if (/내려놔|내려놓|버려|비워|drop|putdown/i.test(t)) {
    const it = findItem(t);
    const n = dropNow(it?.key ?? null, !/(\d+|한|하나)개/.test(t));
    if (!n) return { ...fail('적재함에 내려놓을 물건이 없습니다'), parsed: { action: 'drop', confidence: 1 } };
    return { ok: true, parsed: { action: 'drop', item_id: it?.key ?? null, item_label_ko: it?.it.ko ?? '적재함 전부', count: n, destination: null, confidence: 1 }, quest_id: null, trace, error: null, mode: 'lexical' };
  }
  const item = findItem(t);
  if (!item) return fail('명령에서 아이템을 찾지 못했습니다. 예: 철광석 3개 캐서 베이스캠프로 가져와');
  trace.push({ stage: item.alias ? 'ngram' : 'lexical', text: item.hit, score: item.alias ? 0.78 : 1, role: 'item', accepted: true });
  const other = Object.values(ITEMS).find((x) => x !== item.it);
  trace.push({ stage: 'ngram', text: other.ko, score: 0.31, role: 'item', accepted: false });

  let count = 1;
  const num = /(\d+)(개|조각|덩이)?/.exec(t);
  const kor = /(하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉|열)(개|조각|덩이)/.exec(t);
  if (num) count = Math.max(1, Math.min(9, Number(num[1])));
  else if (kor) count = KO_NUM[kor[1]];
  if (num || kor) trace.push({ stage: 'lexical', text: (num ?? kor)[0], score: 1, role: 'count', accepted: true });

  const wp = WAYPOINTS.find((p) => p.names.some((n) => t.includes(n)));
  const mine = MINE.test(t);
  const collect = COLLECT.test(t);
  const carry = CARRY.test(t);
  const verb = mine ? '캐' : collect ? '줍' : carry ? '가져' : null;
  if (verb) trace.push({ stage: 'lexical', text: verb, score: 1, role: 'action', accepted: true });
  let dest = wp ? waypointRef(wp.id) : null;
  if (wp) trace.push({ stage: 'lexical', text: wp.names.find((n) => t.includes(n)), score: 1, role: 'destination', accepted: true });
  else if (carry) {
    dest = waypointRef('basecamp');
    trace.push({ stage: 'assign', text: '목적지 없음 → 베이스캠프(기본)', score: 0.5, role: 'destination', accepted: true });
  }
  trace.push({ stage: 'nms', text: '겹치는 후보 제거', score: null, role: null, accepted: true });

  const action = collect && !mine ? 'collect' : dest ? 'mine_and_deliver' : 'mine';
  const confidence = Math.min(0.97, 0.5 + (item.alias ? 0.12 : 0.25) + (num || kor ? 0.1 : 0) + (verb ? 0.08 : 0) + (wp ? 0.07 : 0));
  trace.push({ stage: 'assign', text: `${action}: ${item.it.ko} ×${count}${dest ? ` → ${dest.label_ko}` : ''}`, score: confidence, role: null, accepted: true });
  const parsed = { action, item_id: item.key, item_label_ko: item.it.ko, count, destination: dest, confidence };
  const teamCoop = (W.teamSize ?? 2) >= 2;
  let firstQuest = null;
  if (teamCoop && count > 1) {
    for (let idx = 1; idx <= count; idx++) {
      const sub = createQuest({ ...parsed, count: 1, batchIndex: idx, batchTotal: count });
      if (!firstQuest) firstQuest = sub;
    }
  } else {
    firstQuest = createQuest(parsed);
  }
  const mode = item.alias ? 'embedding' : 'lexical';
  log('search', 'info', `명령 해석: ${item.it.ko} ×${count}${dest ? ` → ${dest.label_ko}` : ''} (신뢰도 ${confidence.toFixed(2)}, ${mode})`, { confidence, mode, action });
  if (mode === 'embedding') log('embed', 'info', `임베딩 보조 일치: "${item.hit}" → ${item.it.ko}`, { score: 0.78 });
  return { ok: true, parsed, quest_id: firstQuest.id, queued: teamCoop && count > 1 ? count - 1 : 0, trace, error: null, mode };
}

function waypointRef(id) {
  const p = W.waypoints.find((x) => x.id === id);
  return { kind: 'waypoint', id: p.id, label_ko: p.label_ko };
}

function createQuest(p) {
  const it = ITEMS[p.item_id];
  const isDropOnly = !it.node || p.action === 'collect';
  const teamCoop = (W.teamSize ?? 2) >= 2;
  const batchTag = p.batchTotal > 1 ? ` [${p.batchIndex}/${p.batchTotal}대기열]` : '';
  const steps = [
    { kind: 'locate', label_ko: `${it.ko} 탐색` },
    { kind: 'approach', label_ko: `${it.ko} 가까이 접근` },
    isDropOnly ? { kind: 'look_down', label_ko: `${it.ko} 내려다보며 확인` } : { kind: 'mine', label_ko: `${it.ko} 채취 ${p.count}개` },
    teamCoop
      ? { kind: 'collect', label_ko: `${it.ko} 순차 협동 줍기 (로봇 1 지지 → 로봇 2 수렴)` }
      : { kind: 'collect', label_ko: `${it.ko} 줍기 ${p.count}개` },
  ];
  if (p.destination) steps.push({ kind: 'carry', label_ko: `${p.destination.label_ko} 협동 운반` }, { kind: 'deliver', label_ko: `${p.destination.label_ko} 인도` });
  const q = {
    id: `q${W.nextQuest++}`,
    title: `${it.ko} ${p.count}개 ${isDropOnly ? '줍기' : '채취'}${p.destination ? ` → ${p.destination.label_ko} 운반` : ''}${batchTag}`,
    state: 'pending',
    created_tick: W.tick,
    steps: steps.map((s) => ({ ...s, state: 'pending', progress: 0 })),
  };
  W.quests.push(q);
  W.rt.set(q.id, { i: 0, it, key: p.item_id, count: p.count, dest: p.destination, work: 0, mined: 0, collected: 0, node: null, energy: 0, target: null });
  log('quest', 'info', `퀘스트 대기열 등록: ${q.title}`, { quest: q.id, steps: steps.length });
  return q;
}

export const questList = () => need().quests.map((q) => structuredCloneSafe(q));
const structuredCloneSafe = (o) => JSON.parse(JSON.stringify(o));

// ── 틱 진행 ─────────────────────────────────────────────────────
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const loadKg = () => W.agent.carrying.reduce((a, c) => a + c.mass_kg, 0);

function wrapAngle(rad) {
  let r = rad % (Math.PI * 2);
  if (r > Math.PI) r -= Math.PI * 2;
  else if (r < -Math.PI) r += Math.PI * 2;
  return r;
}

const turnSpec = () => BASE_TURN_SPECS[W.base] ?? BASE_TURN_SPECS.wheel;
const loadFactor = (k) => 1 - k * Math.min(1, loadKg() / LOAD_LIMIT);

// 제자리에서 목표 쪽으로 돈다: 바퀴·트랙은 바닥을 비비며(pivot_rate), 다리는 발을 디뎌(step turn) 돈다.
function turnToward(targetPos) {
  const a = W.agent;
  const desired = Math.atan2(targetPos[1] - a.pos[1], targetPos[0] - a.pos[0]);
  const diff = wrapAngle(desired - a.yaw);
  const maxTurn = turnSpec().pivot_rate_rad_s * loadFactor(0.25) * DT;
  a.reversing = false;
  if (Math.abs(diff) <= maxTurn) {
    a.yaw = desired;
    return 0;
  }
  a.yaw = wrapAngle(a.yaw + Math.sign(diff) * maxTurn);
  return wrapAngle(desired - a.yaw);
}

function drive(dir, ds, kappa) {
  const a = W.agent;
  a.pos[0] += dir * Math.cos(a.yaw) * ds;
  a.pos[1] += dir * Math.sin(a.yaw) * ds;
  a.yaw = wrapAngle(a.yaw + kappa * dir * ds);
  a.reversing = dir < 0;
  W.lastMoved = ds;
}

// 3점 회전(K-턴): 앞으로 꺾어 돌고, 반대로 꺾어 후진하고, 다시 앞으로. 세 토막 모두 같은 방향으로 몸을 돌린다.
function startKTurn(err) {
  const s = Math.sign(err) || 1;
  const R = Math.max(0.4, turnSpec().min_radius_m);
  const parts = [0.45, 0.35, 0.2].map((f) => Math.abs(err) * f);
  W.agent.maneuver = { kind: 'kturn', seg: 0, segs: [{ dir: 1, kappa: s / R, left: parts[0] }, { dir: -1, kappa: -s / R, left: parts[1] }, { dir: 1, kappa: s / R, left: parts[2] }] };
  log('world', 'info', `(모의) 3점 회전 시작: ${((err * 180) / Math.PI).toFixed(0)}° — 전진·후진·전진 (반경 ${R.toFixed(2)} m)`, { maneuver: 'kturn', turn_deg: Number(((err * 180) / Math.PI).toFixed(1)) });
}

function runManeuver(target, reach) {
  const a = W.agent;
  const m = a.maneuver;
  const seg = m.segs[m.seg];
  const spec = turnSpec();
  const v = (seg.dir > 0 ? BASES[W.base].max_speed_mps * 0.45 : spec.reverse_mps) * loadFactor(0.35);
  const dpsi = Math.min(seg.left, Math.abs(seg.kappa) * v * DT);
  drive(seg.dir, dpsi / Math.abs(seg.kappa), seg.kappa);
  seg.left -= dpsi;
  if (seg.left <= 1e-3) {
    m.seg += 1;
    if (m.seg >= m.segs.length) {
      a.maneuver = null;
      a.reversing = false;
    }
  }
  return Math.max(0, dist2(a.pos, target) - reach);
}

// 목표로 간다. 바퀴·트랙은 차처럼 달리며 반경 안에서 꺾고, 크게 틀어야 하면 3점 회전(트랙) 또는 제자리 회전(바퀴),
// 뒤쪽 가까이 있으면 돌지 않고 후진한다. 다리는 걸으며 휘고(굽은 길에서 느려짐), 크게 틀 때는 발을 디뎌 돈다.
function moveToward(target, reach) {
  const a = W.agent;
  const d = dist2(a.pos, target);
  W.lastMoved = 0;
  if (d <= reach) {
    a.maneuver = null;
    turnToward(target);
    return 0;
  }
  if (a.maneuver) return runManeuver(target, reach);
  const spec = turnSpec();
  const bearing = Math.atan2(target[1] - a.pos[1], target[0] - a.pos[0]);
  const err = wrapAngle(bearing - a.yaw);
  const vmax = BASES[W.base].max_speed_mps * loadFactor(0.35);
  if (spec.holonomic) {
    const step = Math.min(vmax * DT, d - reach * 0.9);
    a.pos[0] += Math.cos(bearing) * step;
    a.pos[1] += Math.sin(bearing) * step;
    a.yaw = wrapAngle(a.yaw + Math.max(-spec.turn_rate_rad_s * DT, Math.min(spec.turn_rate_rad_s * DT, err)));
    W.lastMoved = step;
    return d - step;
  }
  const rearErr = wrapAngle(bearing - (a.yaw + Math.PI));
  if (spec.reverse_max_m > 0 && Math.abs(err) > 2.2 && d <= spec.reverse_max_m) {
    const v = spec.reverse_mps * loadFactor(0.35);
    const ds = Math.min(v * DT, d - reach * 0.9);
    const kmax = 1 / Math.max(0.3, spec.min_radius_m);
    drive(-1, ds, Math.max(-kmax, Math.min(kmax, -rearErr / Math.max(ds, 1e-3))) * 0.5);
    if (!a.reversingLogged) {
      log('world', 'info', `(모의) 목표가 뒤쪽 ${d.toFixed(1)} m — 돌지 않고 후진`, { maneuver: 'reverse' });
      a.reversingLogged = true;
    }
    return Math.max(0, d - ds);
  }
  a.reversingLogged = false;
  if (Math.abs(err) > spec.turn_move_rad) {
    if (spec.kturn && Math.abs(err) > 1.0 && d > 1.2) {
      startKTurn(err);
      return runManeuver(target, reach);
    }
    turnToward(target);
    return d;
  }
  const curveSlow = spec.legs ? Math.max(0.45, Math.cos(err) ** 0.33) : Math.max(0.35, Math.cos(err));
  const v = vmax * curveSlow;
  const omega = Math.min(spec.turn_rate_rad_s, v / Math.max(0.05, spec.min_radius_m));
  const ds = Math.min(v * DT, d - reach * 0.9);
  const dpsi = Math.max(-omega * DT, Math.min(omega * DT, err));
  drive(1, ds, dpsi / Math.max(ds, 1e-3));
  return d - ds;
}

function nearest(list, kind, claimed) {
  let best = null;
  for (const n of list) {
    if ((kind && n.kind !== kind && n.item_id !== kind) || n.hp === 0 || claimed?.has(n.id)) continue;
    const d = dist2(W.agent.pos, n.pos);
    if (!best || d < best.d) best = { n, d };
  }
  return best?.n ?? null;
}

function setStep(q, rt, i, st) {
  q.steps[i].state = st;
  if (st === 'done') q.steps[i].progress = 1;
}

function failQuest(q, rt, note) {
  setStep(q, rt, rt.i, 'failed');
  q.steps[rt.i].note = note;
  q.state = 'failed';
  q.result = { ticks: W.tick - q.created_tick, energy_j: rt.energy, precision: null, summary: `실패: ${note}` };
  log('quest', 'error', `퀘스트 실패: ${q.title} — ${note}`, { quest: q.id, step: q.steps[rt.i].kind });
}

function nextStep(q, rt) {
  setStep(q, rt, rt.i, 'done');
  log('quest', 'info', `단계 완료: ${q.steps[rt.i].label_ko}`, { quest: q.id, step: q.steps[rt.i].kind });
  rt.i += 1;
  rt.work = 0;
  if (rt.i >= q.steps.length) {
    q.state = 'done';
    const precision = Math.min(0.98, 0.78 + W.visionMargin * 0.2);
    const ticks = W.tick - q.created_tick;
    q.result = {
      ticks, energy_j: rt.energy, precision,
      summary: `${rt.it.ko} ${rt.count}개${rt.dest ? `를 ${rt.dest.label_ko}에 인도` : ' 확보'}했습니다 (${ticks}틱, ${(rt.energy / 1000).toFixed(1)} kJ)`,
    };
    log('quest', 'info', `퀘스트 완료: ${q.title}`, { quest: q.id, ticks, energy_kj: Number((rt.energy / 1000).toFixed(2)), precision: Number(precision.toFixed(2)) });
  } else {
    q.steps[rt.i].state = 'active';
  }
}

function applyCameraPose(a, headPitchDeg = 0, crouchM = 0) {
  a.standing_eye_m = a.standing_eye_m ?? 1.0;
  a.mount_deg = a.mount_deg ?? -12.0;
  a.head_pitch_deg = headPitchDeg;
  a.eye_m = Math.max(0.35, a.standing_eye_m - crouchM);
  a.pitch_deg = Math.max(-65.0, Math.min(15.0, a.mount_deg - a.head_pitch_deg));
}

function lookDownAt(a, targetPos) {
  const d = Math.max(0.2, dist2(a.pos, targetPos));
  const targetZ = (targetPos[2] ?? 0) + 0.05;
  const dz = Math.max(0.1, (a.standing_eye_m ?? 1.0) - targetZ);
  const directDeg = -(Math.atan2(dz, d) * 180) / Math.PI;
  const desiredPitch = d <= 1.6 ? Math.max(-60.0, Math.min(-12.0, directDeg)) : -12.0;
  const headPitch = Math.max(0, (a.mount_deg ?? -12.0) - desiredPitch);
  const crouch = headPitch > 35 ? Math.min(0.22, (headPitch - 35) * 0.008) : 0;
  applyCameraPose(a, headPitch, crouch);
}

// ── 사람처럼 줍기: 고개를 먼저 숙여 응시(목) → 비전 확인 → 자세 낮춤(무릎·허리) → 쥐기 → 들어올림 ─────────
// 시선이 손보다 먼저 물건에 가고(Land & Hayhoe 2001, Johansson 외 2001), 몸을 낮추는 동안에도 물건을 화면 가운데에 둔다.
const NECK_MAX_DEG = 45;
const CROUCH_M = { human_foot: 0.42, wheel: 0.22, track: 0.22, mecanum: 0.22 };
export const PICK_STAGES = [
  ['approach', '다가가기'],
  ['look', '고개 숙여 응시'],
  ['check', '비전 확인'],
  ['bend', '자세 낮춤'],
  ['grip', '쥐기'],
  ['lift', '들어올림'],
];

function gazeAt(a, targetPos, neckFrac, crouchFrac) {
  const crouch = (CROUCH_M[W.base] ?? 0.22) * crouchFrac;
  const eye = Math.max(0.35, (a.standing_eye_m ?? 1.0) - crouch);
  const d = Math.max(0.2, dist2(a.pos, targetPos));
  const wantPitch = -(Math.atan2(Math.max(0.05, eye - ((targetPos[2] ?? 0) + 0.06)), d) * 180) / Math.PI;
  const total = Math.max(0, (a.mount_deg ?? -12.0) - wantPitch) * neckFrac;
  a.neck_deg = Math.min(NECK_MAX_DEG, total);
  a.trunk_deg = Math.max(0, total - NECK_MAX_DEG) + 30 * crouchFrac;
  applyCameraPose(a, total, crouch);
}

function startPick(rt, drop) {
  rt.pick = { drop, stage: 'approach', t: 0, check: null };
}

// ── 지게차: 바퀴·메카넘·트랙 몸은 탑승자가 정면을 보고, 포크를 내려 포크 카메라로 확인한 뒤 끼워 들어올린다 ──
// (Rust noto_core::act::fork 와 같은 순서·값의 모의판)
export const FORK_STAGES = [
  ['approach', '다가가기'],
  ['align', '정렬 (정면 응시)'],
  ['lower', '포크 내림'],
  ['check', '포크 카메라 확인'],
  ['insert', '포크 끼우기'],
  ['raise', '들어올림'],
];
const FORK_REST_M = 0.05;
const FORK_CARRY_M = 0.10;
const FORK_MPS = 0.10;
const FORK_STANDOFF_M = 0.58;
const FORK_LOAD_X_M = 0.34;
export const usesForks = () => !!W && ['wheel', 'mecanum', 'track'].includes(W.base);

function forkStep(q, rt, needsCoCarry = false, coPhase = 1) {
  const a = W.agent;
  const p = rt.pick;
  const drop = p.drop;
  a.fork_m = a.fork_m ?? FORK_REST_M;
  a.neck_deg = 0;
  a.trunk_deg = 0;
  applyCameraPose(a, 0, 0);
  if (p.stage !== 'raise' && !W.drops.includes(drop)) {
    rt.pick = null;
    return 'lost';
  }
  switch (p.stage) {
    case 'approach': {
      const left = moveToward(drop.pos, FORK_STANDOFF_M);
      if (left <= 0) Object.assign(p, { stage: 'align', t: 0 });
      break;
    }
    case 'align': {
      const rem = turnToward(drop.pos);
      if (Math.abs(rem) < 0.06) Object.assign(p, { stage: 'lower', t: 0 });
      break;
    }
    case 'lower': {
      a.fork_m = Math.max(0, a.fork_m - FORK_MPS * DT);
      if (a.fork_m <= 0) {
        Object.assign(p, { stage: 'check', t: 0 });
        log('vision', 'info', `[로봇 ${coPhase}] 포크를 바닥까지 내림: ${rt.it.ko} — 탑승자는 정면, 포크 카메라로 확인`, { robot_index: coPhase, fork_m: 0 });
      }
      break;
    }
    case 'check': {
      p.t += 1;
      if (p.t === 1) {
        const range = dist2(a.pos, drop.pos);
        const bearing = Math.abs(wrapAngle(Math.atan2(drop.pos[1] - a.pos[1], drop.pos[0] - a.pos[0]) - a.yaw));
        const seen = range < 1.5 && bearing < 0.5;
        p.check = seen ? { ok: true, label_ko: rt.it.ko, score: Number((0.86 + W.rng() * 0.1).toFixed(2)), range_m: Number(range.toFixed(2)), depth_m: Number((range * (1 + (W.rng() - 0.5) * 0.04)).toFixed(2)) } : { ok: false, reason: '포크 카메라 화면 안에 대상이 없음' };
        log('vision', p.check.ok ? 'info' : 'warn', p.check.ok ? `[로봇 ${coPhase}] 비전 확인 ✓ ${rt.it.ko} ${p.check.range_m} m (깊이 ${p.check.depth_m} m) · 포크 카메라` : `[로봇 ${coPhase}] 비전 확인 실패: ${p.check.reason}`, { robot_index: coPhase, check: p.check });
      }
      if (p.t >= 2) Object.assign(p, p.check?.ok ? { stage: 'insert', t: 0 } : { stage: 'approach', t: 0, check: null });
      break;
    }
    case 'insert': {
      const left = moveToward(drop.pos, FORK_LOAD_X_M);
      if (left <= 0) {
        if (needsCoCarry && coPhase === 1) return 'phase1_ready';
        W.drops.splice(W.drops.indexOf(drop), 1);
        Object.assign(p, { stage: 'raise', t: 0 });
      }
      break;
    }
    case 'raise': {
      a.fork_m = Math.min(FORK_CARRY_M, a.fork_m + FORK_MPS * DT);
      W.energy += (needsCoCarry ? 6 : 12) * DT;
      if (a.fork_m >= FORK_CARRY_M) {
        a.carrying.push({ item_id: rt.key, label_ko: rt.it.ko, mass_kg: rt.it.mass });
        rt.pick = null;
        return 'picked';
      }
      break;
    }
    default:
      break;
  }
  return 'busy';
}

function pickStep(q, rt, needsCoCarry = false, coPhase = 1) {
  if (usesForks()) return forkStep(q, rt, needsCoCarry, coPhase);
  const a = W.agent;
  const p = rt.pick;
  const drop = p.drop;
  if (!['grip', 'lift'].includes(p.stage) && !W.drops.includes(drop)) {
    rt.pick = null;
    return 'lost';
  }
  switch (p.stage) {
    case 'approach': {
      const left = moveToward(drop.pos, 0.75);
      if (dist2(a.pos, drop.pos) <= 1.8) gazeAt(a, drop.pos, 0.6, 0);
      else applyCameraPose(a, 0, 0);
      if (left <= 0) Object.assign(p, { stage: 'look', t: 0 });
      break;
    }
    case 'look': {
      const rem = turnToward(drop.pos);
      p.t += 1;
      gazeAt(a, drop.pos, Math.min(1, 0.6 + 0.2 * p.t), 0);
      if (p.t >= 2 && Math.abs(rem) < 0.06) {
        Object.assign(p, { stage: 'check', t: 0 });
        log('vision', 'info', `[로봇 ${coPhase}] 시선 고정: ${rt.it.ko} — 목 ${a.neck_deg.toFixed(0)}° 숙임, 카메라 피치 ${a.pitch_deg.toFixed(0)}°`, { robot_index: coPhase, neck_deg: Number(a.neck_deg.toFixed(1)), pitch_deg: Number(a.pitch_deg.toFixed(1)) });
      }
      break;
    }
    case 'check': {
      p.t += 1;
      gazeAt(a, drop.pos, 1, 0);
      if (p.t === 1) {
        const seen = visibleObjects(6).find((o) => o.id === drop.id && o.in_pitch && o.forward > 0.15);
        const range = dist2(a.pos, drop.pos);
        p.check = seen ? { ok: true, label_ko: rt.it.ko, score: Number((0.86 + W.rng() * 0.1).toFixed(2)), range_m: Number(range.toFixed(2)), depth_m: Number((range * (1 + (W.rng() - 0.5) * 0.04)).toFixed(2)) } : { ok: false, reason: '숙인 화면 안에 대상이 없음' };
        log('vision', p.check.ok ? 'info' : 'warn', p.check.ok ? `[로봇 ${coPhase}] 비전 확인 ✓ ${rt.it.ko} ${p.check.range_m} m (깊이 ${p.check.depth_m} m)` : `[로봇 ${coPhase}] 비전 확인 실패: ${p.check.reason}`, { robot_index: coPhase, check: p.check });
      }
      if (p.t >= 2) Object.assign(p, p.check?.ok ? { stage: 'bend', t: 0 } : { stage: 'approach', t: 0, check: null });
      break;
    }
    case 'bend': {
      p.t += 1;
      gazeAt(a, drop.pos, 1, Math.min(1, p.t / 4));
      W.energy += 15 * DT;
      if (p.t >= 4) Object.assign(p, { stage: 'grip', t: 0 });
      break;
    }
    case 'grip': {
      p.t += 1;
      gazeAt(a, drop.pos, 1, 1);
      if (p.t >= 2) {
        if (needsCoCarry && coPhase === 1) {
          return 'phase1_ready';
        }
        W.drops.splice(W.drops.indexOf(drop), 1);
        a.carrying.push({ item_id: rt.key, label_ko: rt.it.ko, mass_kg: rt.it.mass });
        Object.assign(p, { stage: 'lift', t: 0, from: { neck: a.neck_deg, total: a.head_pitch_deg, crouch: (a.standing_eye_m ?? 1) - a.eye_m } });
      }
      break;
    }
    case 'lift': {
      p.t += 1;
      const u = Math.min(1, p.t / 4);
      a.neck_deg = p.from.neck * (1 - u);
      a.trunk_deg = Math.max(0, (p.from.total - p.from.neck) * (1 - u));
      applyCameraPose(a, p.from.total * (1 - u), p.from.crouch * (1 - u));
      W.energy += (needsCoCarry ? 12.5 : 25) * DT;
      if (u >= 1) {
        rt.pick = null;
        a.neck_deg = 0;
        a.trunk_deg = 0;
        return 'picked';
      }
      break;
    }
    default:
      break;
  }
  return 'busy';
}

export function procedureView() {
  if (!W) return null;
  const q = W.quests.find((x) => x.state === 'active');
  const rt = q && W.rt.get(q.id);
  const p = rt?.pick;
  if (!p) return null;
  const a = W.agent;
  if (usesForks()) {
    const at = FORK_STAGES.findIndex(([id]) => id === p.stage);
    return {
      act: ['align', 'lower', 'check'].includes(p.stage) ? 'look' : 'pick',
      mechanism: 'fork',
      camera: ['lower', 'check', 'insert'].includes(p.stage) ? 'fork' : 'head',
      target: p.drop.id,
      label_ko: rt.it.ko,
      stage: p.stage,
      stages: FORK_STAGES.map(([id, label_ko], i) => ({ id, label_ko, state: i < at ? 'done' : i === at ? 'active' : 'pending' })),
      fork_m: a.fork_m ?? FORK_REST_M,
      neck_deg: 0,
      trunk_deg: 0,
      eye_m: a.eye_m,
      standing_eye_m: a.standing_eye_m,
      pitch_deg: a.pitch_deg,
      check: p.check,
    };
  }
  const at = PICK_STAGES.findIndex(([id]) => id === p.stage);
  return {
    act: 'pick',
    target: p.drop.id,
    label_ko: rt.it.ko,
    stage: p.stage,
    stages: PICK_STAGES.map(([id, label_ko], i) => ({ id, label_ko, state: i < at ? 'done' : i === at ? 'active' : 'pending' })),
    neck_deg: a.neck_deg ?? 0,
    trunk_deg: a.trunk_deg ?? 0,
    eye_m: a.eye_m,
    standing_eye_m: a.standing_eye_m,
    pitch_deg: a.pitch_deg,
    check: p.check,
  };
}

function runStep(q, rt) {
  const s = q.steps[rt.i];
  const a = W.agent;
  switch (s.kind) {
    case 'locate': {
      applyCameraPose(a, 0, 0);
      rt.work += 1;
      s.progress = rt.work / 12;
      if (rt.work >= 12) {
        rt.node = ITEMS[rt.key].node ? nearest(W.nodes, ITEMS[rt.key].node) : null;
        if (!rt.node && q.steps.every((x) => x.kind !== 'mine')) rt.node = nearest(W.drops, rt.key);
        if (!rt.node) return failQuest(q, rt, `${rt.it.ko} 대상을 찾지 못했습니다 (탐색 시간 초과)`);
        log('vision', 'info', `대상 탐지: ${rt.it.ko} @ (${rt.node.pos[0].toFixed(1)}, ${rt.node.pos[1].toFixed(1)})`, { label: rt.key, score: Number((0.62 + W.rng() * 0.3).toFixed(2)), decision: 'accept' });
        nextStep(q, rt);
      }
      break;
    }
    case 'approach': {
      const total = rt.total ?? (rt.total = Math.max(1, dist2(a.pos, rt.node.pos)));
      const left = moveToward(rt.node.pos, 1.0);
      if (q.steps.some((x) => x.kind === 'look_down') && dist2(a.pos, rt.node.pos) <= 1.6 && !usesForks()) {
        lookDownAt(a, rt.node.pos);
      } else {
        applyCameraPose(a, 0, 0);
      }
      s.progress = 1 - left / total;
      if (left <= 0) {
        rt.total = null;
        nextStep(q, rt);
      }
      break;
    }
    case 'look_down': {
      // 첫 물건: 다가가 고개를 숙여 응시하고 비전으로 확인할 때까지. 몸을 낮춰 드는 것은 다음 단계(줍기)에서.
      const drop = rt.pick?.drop ?? nearest(W.drops, rt.key);
      if (!drop) return failQuest(q, rt, `확인할 ${rt.it.ko} 드롭이 없습니다`);
      if (!rt.pick) startPick(rt, drop);
      if (pickStep(q, rt) === 'lost') break;
      const forks = usesForks();
      const at = (forks ? FORK_STAGES : PICK_STAGES).findIndex(([id]) => id === rt.pick?.stage);
      s.progress = Math.min(1, at / (forks ? 4 : 3));
      if (rt.pick?.stage === (forks ? 'insert' : 'bend')) nextStep(q, rt);
      break;
    }
    case 'mine': {
      applyCameraPose(a, 8, 0);
      if (!rt.node || rt.node.hp <= 0) {
        rt.node = nearest(W.nodes, ITEMS[rt.key].node);
        if (!rt.node) return failQuest(q, rt, `채취할 ${rt.it.ko} 노드가 부족합니다 (${rt.mined}/${rt.count})`);
      }
      if (moveToward(rt.node.pos, 1.1) > 0) break;
      rt.node.hp = Math.max(0, rt.node.hp - 4);
      W.fatigue = Math.min(1, W.fatigue + 0.0025);
      W.heat = Math.min(1, W.heat + 0.004);
      W.energy += 60 * DT;
      rt.energy += 60 * DT;
      s.progress = (rt.mined + 1 - rt.node.hp / rt.node.hp0) / rt.count;
      if (rt.node.hp <= 0) {
        W.drops.push({ id: `d${W.nextDrop++}`, item_id: rt.key, label_ko: rt.it.ko, pos: [rt.node.pos[0] + 0.4, rt.node.pos[1] - 0.3, 0] });
        rt.mined += 1;
        s.note = `${rt.mined}/${rt.count}`;
        log('world', 'info', `${rt.it.ko} 노드 파괴 → 드롭 생성 (${rt.mined}/${rt.count})`, { node: rt.node.id });
        rt.node = null;
        if (rt.mined >= rt.count) nextStep(q, rt);
      }
      break;
    }
    case 'collect': {
      const soloGripFail = !usesForks() && W.config?.fingers <= 2 && rt.it.mass >= 3;
      const soloLoadFail = loadKg() + rt.it.mass > LOAD_LIMIT;
      const needsCoCarry = soloGripFail || soloLoadFail;
      const effectiveMass = needsCoCarry ? rt.it.mass * 0.5 : rt.it.mass;
      const teamLimit = LOAD_LIMIT * 2 * 0.85;

      if (needsCoCarry && loadKg() + effectiveMass > LOAD_LIMIT) {
        return failQuest(q, rt, `협동 분담 한계 초과 (분담 후 ${(loadKg() + effectiveMass).toFixed(1)} > ${LOAD_LIMIT} kg, 팀 한계 ${teamLimit.toFixed(1)} kg)`);
      }

      if (!rt.coPhase) {
        rt.coPhase = 1;
        rt.savedPos = [a.pos[0], a.pos[1]];
        rt.savedYaw = a.yaw;
      }

      if (!rt.pick) {
        const drop = nearest(W.drops, rt.key);
        if (!drop) return failQuest(q, rt, `줍을 ${rt.it.ko} 드롭이 없습니다`);
        if (needsCoCarry && rt.coPhase === 1) {
          log('world', 'info', `협동 모드(중첩 물리): 2번 로봇의 반대편 지지를 가정하여 1번 로봇 일차 시뮬레이션 시작 (유효 하중 ${effectiveMass.toFixed(1)} kg)`, { co_phase: 1, effective_mass_kg: effectiveMass });
        }
        startPick(rt, drop);
      }

      const r = pickStep(q, rt, needsCoCarry, rt.coPhase);

      if (r === 'phase1_ready') {
        rt.coPhase = 2;
        a.pos = [rt. pick.drop.pos[0] - 0.9, rt.pick.drop.pos[1] + 0.4];
        a.yaw = Math.atan2(rt.pick.drop.pos[1] - a.pos[1], rt.pick.drop.pos[0] - a.pos[0]);
        Object.assign(rt.pick, { stage: 'approach', t: 0, check: null });
        log('world', 'info', `1번 로봇 파지 위치 확보 완료 → 2번 로봇 시점으로 전환하여 이차 시뮬레이션 진행`, { co_phase: 2 });
        break;
      }

      if (r === 'picked') {
        if (needsCoCarry) {
          const lastItem = a.carrying[a.carrying.length - 1];
          if (lastItem) lastItem.mass_kg = effectiveMass;
        }
        rt.coPhase = null;
        rt.collected += 1;
        s.note = `${rt.collected}/${rt.count}`;
        log('world', 'info', `${rt.it.ko} ${needsCoCarry ? '순차 협동 ' : ''}줍기 완료 (${rt.collected}/${rt.count}), 분담 적재 ${loadKg().toFixed(1)} kg`, { grasp_margin: needsCoCarry ? 0.85 : 0.8 });
      }

      const stagesNow = usesForks() ? FORK_STAGES : PICK_STAGES;
      const at = rt.pick ? stagesNow.findIndex(([id]) => id === rt.pick.stage) : 0;
      const phaseFrac = needsCoCarry
        ? ((rt.coPhase - 1) * 0.5 + (at / stagesNow.length) * 0.5)
        : (at / stagesNow.length);
      s.progress = (rt.collected + phaseFrac) / rt.count;

      if (rt.collected >= rt.count) {
        applyCameraPose(a, 0, 0);
        nextStep(q, rt);
      }
      break;
    }
    case 'carry': {
      applyCameraPose(a, 0, 0);
      const target = W.waypoints.find((p) => p.id === rt.dest.id).pos;
      const total = rt.total ?? (rt.total = Math.max(1, dist2(a.pos, target)));
      const left = moveToward(target, 1.5);
      s.progress = 1 - left / total;
      if (left <= 0) {
        rt.total = null;
        nextStep(q, rt);
      }
      break;
    }
    case 'deliver': {
      if (usesForks()) {
        // 지게차: 탑승자는 정면, 포크를 내려 짐을 놓고 물러난 뒤 포크를 다시 올린다.
        applyCameraPose(a, 0, 0);
        a.fork_m = rt.work < 4 ? Math.max(0, (a.fork_m ?? FORK_CARRY_M) - FORK_MPS * DT * 1.25) : Math.min(FORK_REST_M, (a.fork_m ?? 0) + FORK_MPS * DT);
      } else {
        applyCameraPose(a, 10, 0.05);
      }
      rt.work += 1;
      s.progress = rt.work / 8;
      if (rt.work >= 8) {
        const wp = rt.dest ? W.waypoints.find((p) => p.id === rt.dest.id) : null;
        if (wp) (wp.pile ??= []).push(...a.carrying.filter((c) => c.item_id === rt.key));
        a.carrying = a.carrying.filter((c) => c.item_id !== rt.key);
        applyCameraPose(a, 0, 0);
        nextStep(q, rt);
      }
      break;
    }
    default:
      nextStep(q, rt);
  }
}

function tickOnce() {
  W.tick += 1;
  W.lastMoved = 0;
  const q = W.quests.find((x) => x.state === 'active') ?? W.quests.find((x) => x.state === 'pending');
  if (q) {
    const rt = W.rt.get(q.id);
    if (q.state === 'pending') {
      q.state = 'active';
      q.steps[0].state = 'active';
      log('quest', 'info', `퀘스트 시작: ${q.title}`, { quest: q.id });
    }
    runStep(q, rt);
    W.idleTicks = 0;
  } else {
    W.idleTicks += 1;
  }
  const moving = W.lastMoved > 0;
  const e = BASES[W.base].power_w * DT * (moving ? 1 : 0.3);
  W.energy += e;
  const active = W.quests.find((x) => x.state === 'active');
  if (active) W.rt.get(active.id).energy += e;
  W.agent.speed = W.agent.speed * 0.6 + (W.lastMoved / DT) * 0.4;
  W.fatigue = Math.max(0, W.fatigue - 0.0004);
  W.heat = Math.max(0, W.heat - (moving ? 0.0005 : 0.003));
  W.visionMargin = Math.min(0.95, Math.max(0.3, W.visionMargin + (W.rng() - 0.5) * 0.03));
  const [x, y] = W.agent.pos;
  W.slipMargin = Math.min(0.95, Math.max(0.2, 0.62 + 0.3 * Math.sin(x * 0.4) * Math.cos(y * 0.3)));
  if (active && W.tick % 30 === 0) {
    const s = 0.5 + W.rng() * 0.4;
    log('sds', 'info', `SDS 갱신: vision|${W.rt.get(active.id).it.ko}|primary`, { n: W.tick, mean: Number(s.toFixed(3)), sd: 0.06, drift_z: Number(((W.rng() - 0.5) * 2).toFixed(2)) });
  }
  if (active && W.tick % 45 === 0) {
    const p50 = 0.55 + W.rng() * 0.3;
    log('ts', 'info', '시계열 판정: 새 관측이 예측 띠 안', { p10: Number((p50 - 0.12).toFixed(2)), p50: Number(p50.toFixed(2)), p90: Number((p50 + 0.12).toFixed(2)), method: 'holt-damped', decision: 'accept' });
  }
}

export function simStep({ ticks }) {
  need();
  W.fresh = [];
  const n = Math.max(1, Math.min(200, Math.floor(ticks) || 1));
  for (let i = 0; i < n; i++) tickOnce();
  return { snapshot: snapshot(), quests: questList(), body: bodyStatus(), events: W.fresh.splice(0) };
}

// ── 신체 상태 ───────────────────────────────────────────────────
export function bodyStatus() {
  const w = need();
  const load = loadKg();
  const fingers = w.config?.fingers ?? 5;
  const battery = Math.max(0, 1 - w.energy / CAP_J);
  const grip = Math.min(0.98, 0.35 + fingers * 0.12);
  const driveRatio = Math.max(0.05, Math.min(1, 1 - 0.35 * (load / LOAD_LIMIT) - 0.2 * (1 - w.slipMargin)));
  const part = (id, label_ko, ratio, value, max, detail) => ({ id, label_ko, value, max, ratio: Math.max(0, Math.min(1, ratio)), level: levelOf(ratio), detail });
  const parts = [
    part('head', '머리 (비전)', w.visionMargin, w.visionMargin, 1, `최근 탐지 margin ${w.visionMargin.toFixed(2)}, 프레임 지연 ${(8 + w.visionMargin * 4).toFixed(0)} ms`),
    part('thorax', '가슴 (배터리)', battery, (CAP_J - w.energy) / 1000, CAP_J / 1000, `잔량 ${((CAP_J - w.energy) / 1000).toFixed(1)} / ${CAP_J / 1000} kJ`),
    part('stomach', '배 (적재)', 1 - load / LOAD_LIMIT, load, LOAD_LIMIT, `적재 ${load.toFixed(1)} / ${LOAD_LIMIT} kg, 여유 ${Math.round((1 - load / LOAD_LIMIT) * 100)}%`),
    part('arm_l', '손: 파지', grip, grip, 1, `파지 여유 ${grip.toFixed(2)} (손가락 ${fingers}개, 마찰 용량 대비 필요 힘)`),
    part('arm_r', '손: 관절', 1 - w.fatigue, 1 - w.fatigue, 1, `채취 타격 누적 피로 ${(w.fatigue * 100).toFixed(0)}%`),
    part('leg_l', '발: 구동', driveRatio, w.agent.speed, BASES[w.base].max_speed_mps, `현재 ${w.agent.speed.toFixed(2)} / 최고 ${BASES[w.base].max_speed_mps} m/s (경사·하중 보정 ${driveRatio.toFixed(2)})`),
    part('leg_r', '발: 지형', w.slipMargin, w.slipMargin, 1, `다음 구간 경사·턱 한계 여유 ${w.slipMargin.toFixed(2)}`),
  ];
  const fx = (id, label_ko, ratio, detail) => ({ id, label_ko, level: levelOf(ratio), detail });
  const effects = [
    fx('fatigue', '피로', 1 - w.fatigue * 1.4, `손 관절 피로 ${(w.fatigue * 100).toFixed(0)}%`),
    fx('load', '하중', 1 - load / LOAD_LIMIT, `적재 ${load.toFixed(1)} kg`),
    fx('heat', '과열', 1 - w.heat, `구동·채취 발열 ${(w.heat * 100).toFixed(0)}% (가정값)`),
    fx('slip', '미끄러짐', w.slipMargin + 0.15, `경사 여유 ${w.slipMargin.toFixed(2)}`),
  ];
  return {
    parts, effects,
    hand: { fingers, grip_margin: grip },
    foot: { base: w.base, speed_factor: driveRatio, terrain_margin: w.slipMargin },
  };
}

// ── 인식 ────────────────────────────────────────────────────────
export function perceptionScan({ topK }) {
  const w = need();
  const seen = visibleObjects(22).filter((o) => o.forward > 0.25 && o.in_pitch && o.kind !== 'waypoint');
  const best = new Map();
  for (const o of seen) {
    const [id, ko] = o.drop ? [`${o.item_id}_drop`, `${o.label_ko} (드롭)`] : DETECT_LABEL[o.kind];
    const score = Math.max(0.2, Math.min(0.97, 0.96 - o.dist * 0.028 + (w.rng() - 0.5) * 0.06));
    if (!best.has(id) || best.get(id).score < score) best.set(id, { id, label_ko: ko, score });
  }
  const sorted = [...best.values()].sort((a, b) => b.score - a.score);
  const labels = sorted.slice(0, Math.max(1, topK)).map((l, i) => {
    const margin = l.score - (sorted[i + 1]?.score ?? 0.1);
    const accepted = l.score >= 0.55 && margin >= 0.03;
    return {
      ...l, margin, accepted,
      reason: accepted ? `점수 ${l.score.toFixed(2)} ≥ 0.55, margin ${margin.toFixed(2)} ≥ 0.03` : `보류: ${l.score < 0.55 ? `점수 ${l.score.toFixed(2)} < 0.55` : `margin ${margin.toFixed(2)} < 0.03`}`,
    };
  });
  const embed_ms = 9 + w.rng() * 6;
  log('embed', 'info', `프레임 타일 ${12}개 임베딩 ${embed_ms.toFixed(1)} ms (모의)`, { tiles: 12, embed_ms: Number(embed_ms.toFixed(1)) });
  for (const l of labels.slice(0, 3)) {
    log('vision', l.accepted ? 'info' : 'warn', `${l.accepted ? '수락' : '보류'}: ${l.label_ko}`, { label: l.id, score: Number(l.score.toFixed(3)), margin: Number(l.margin.toFixed(3)), decision: l.accepted ? 'accept' : 'hold' });
  }
  return { labels, tiles: 12, embed_ms, device: 'cpu (모의)', mode: 'mock' };
}
