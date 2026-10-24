// 모의 월드: 아주 작은 상태 기계. 실제 Rust 백엔드(world-sim, hand-quest)가 아니다.
// 명령 "철광석 3개 캐서 베이스캠프로 가져와" → 퀘스트(탐색→접근→채취→줍기→운반→인도)가 틱마다 진행되고,
// 에이전트가 움직이며 배터리·하중·피로 같은 신체 상태가 변한다.
import { BASES } from './mock-data.js';
import { levelOf } from './util.js';

const DT = 0.2; // 틱당 초
const CAP_J = 30000; // 배터리 용량(가정)
const LOAD_LIMIT = 20; // 적재 한계 kg(가정)

const ITEMS = {
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
  W = {
    seed, size, bounds: [half, half], tick: 0, time_s: 0, base: BASES[base] ? base : 'wheel', config, rng: r,
    nextSeq: 1, logs: [], fresh: [], quests: [], rt: new Map(), nextQuest: 1, nextDrop: 100,
    nodes, drops, waypoints,
    agent: { pos: [waypoints[0].pos[0] + 1.5, waypoints[0].pos[1], 0], yaw: 0, speed: 0, carrying: [] },
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

export function snapshot() {
  const w = need();
  return {
    tick: w.tick, time_s: w.tick * DT, seed: w.seed, bounds: w.bounds, terrain: 'open',
    agent: { pos: [...w.agent.pos], yaw_deg: (w.agent.yaw * 180) / Math.PI, speed_mps: w.agent.speed, base: w.base, carrying: stack(w.agent.carrying) },
    waypoints: w.waypoints.map((p) => ({ ...p, stock: stack(p.pile ?? []) })),
    nodes: w.nodes.filter((n) => n.hp > 0).map(({ id, kind, label_ko, pos, hp }) => ({ id, kind, label_ko, pos, hp })),
    drops: w.drops,
  };
}

// ── 시야 안 물체(렌더·인식 공용) ─────────────────────────────────
export function visibleObjects(maxDist = 24) {
  const w = need();
  const [ax, ay] = w.agent.pos;
  const cy = Math.cos(w.agent.yaw);
  const sy = Math.sin(w.agent.yaw);
  const out = [];
  const add = (o) => {
    const dx = o.pos[0] - ax;
    const dy = o.pos[1] - ay;
    const forward = dx * cy + dy * sy;
    const right = dx * sy - dy * cy;
    const dist = Math.hypot(dx, dy);
    if (dist <= maxDist) out.push({ ...o, dx, dy, forward, right, dist });
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
  const q = createQuest(parsed);
  const mode = item.alias ? 'embedding' : 'lexical';
  log('search', 'info', `명령 해석: ${item.it.ko} ×${count}${dest ? ` → ${dest.label_ko}` : ''} (신뢰도 ${confidence.toFixed(2)}, ${mode})`, { confidence, mode, action });
  if (mode === 'embedding') log('embed', 'info', `임베딩 보조 일치: "${item.hit}" → ${item.it.ko}`, { score: 0.78 });
  return { ok: true, parsed, quest_id: q.id, trace, error: null, mode };
}

function waypointRef(id) {
  const p = W.waypoints.find((x) => x.id === id);
  return { kind: 'waypoint', id: p.id, label_ko: p.label_ko };
}

function createQuest(p) {
  const it = ITEMS[p.item_id];
  const steps = [
    { kind: 'locate', label_ko: `${it.ko} 탐색` },
    { kind: 'approach', label_ko: `${it.ko} 노드로 접근` },
  ];
  if (p.action !== 'collect') steps.push({ kind: 'mine', label_ko: `${it.ko} 채취 ${p.count}개` });
  steps.push({ kind: 'collect', label_ko: `${it.ko} 줍기 ${p.count}개` });
  if (p.destination) steps.push({ kind: 'carry', label_ko: `${p.destination.label_ko} 운반` }, { kind: 'deliver', label_ko: `${p.destination.label_ko} 인도` });
  const q = {
    id: `q${W.nextQuest++}`,
    title: `${it.ko} ${p.count}개 ${p.action === 'collect' ? '줍기' : '채취'}${p.destination ? ` → ${p.destination.label_ko} 운반` : ''}`,
    state: 'pending', created_tick: W.tick,
    steps: steps.map((s) => ({ ...s, state: 'pending', progress: 0 })),
  };
  W.quests.push(q);
  W.rt.set(q.id, { i: 0, it, key: p.item_id, count: p.count, dest: p.destination, work: 0, mined: 0, collected: 0, node: null, energy: 0, target: null });
  log('quest', 'info', `퀘스트 생성: ${q.title}`, { quest: q.id, steps: steps.length });
  return q;
}

export const questList = () => need().quests.map((q) => structuredCloneSafe(q));
const structuredCloneSafe = (o) => JSON.parse(JSON.stringify(o));

// ── 틱 진행 ─────────────────────────────────────────────────────
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const loadKg = () => W.agent.carrying.reduce((a, c) => a + c.mass_kg, 0);

function moveToward(target, reach) {
  const a = W.agent;
  const d = dist2(a.pos, target);
  if (d <= reach) return 0;
  const speed = BASES[W.base].max_speed_mps * (1 - 0.35 * Math.min(1, loadKg() / LOAD_LIMIT));
  const step = Math.min(speed * DT, d - reach * 0.9);
  a.yaw = Math.atan2(target[1] - a.pos[1], target[0] - a.pos[0]);
  a.pos[0] += Math.cos(a.yaw) * step;
  a.pos[1] += Math.sin(a.yaw) * step;
  W.lastMoved = step;
  return d - step;
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

function runStep(q, rt) {
  const s = q.steps[rt.i];
  const a = W.agent;
  switch (s.kind) {
    case 'locate': {
      rt.work += 1;
      s.progress = rt.work / 12;
      if (rt.work >= 12) {
        rt.node = nearest(W.nodes, ITEMS[rt.key].node);
        if (!rt.node && q.steps.every((x) => x.kind !== 'mine')) rt.node = nearest(W.drops, rt.key);
        if (!rt.node) return failQuest(q, rt, `${rt.it.ko} 노드를 찾지 못했습니다 (탐색 시간 초과)`);
        log('vision', 'info', `대상 탐지: ${rt.it.ko} @ (${rt.node.pos[0].toFixed(1)}, ${rt.node.pos[1].toFixed(1)})`, { label: rt.key, score: Number((0.62 + W.rng() * 0.3).toFixed(2)), decision: 'accept' });
        nextStep(q, rt);
      }
      break;
    }
    case 'approach': {
      const total = rt.total ?? (rt.total = Math.max(1, dist2(a.pos, rt.node.pos)));
      const left = moveToward(rt.node.pos, 1.0);
      s.progress = 1 - left / total;
      if (left <= 0) { rt.total = null; nextStep(q, rt); }
      break;
    }
    case 'mine': {
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
      if (W.config?.fingers <= 2 && rt.it.mass >= 3) return failQuest(q, rt, `손가락 ${W.config.fingers}개로는 ${rt.it.ko}(${rt.it.mass} kg)를 안정적으로 쥘 수 없습니다 (파지 불안정)`);
      if (loadKg() + rt.it.mass > LOAD_LIMIT) return failQuest(q, rt, `적재 한계 초과 (${(loadKg() + rt.it.mass).toFixed(1)} > ${LOAD_LIMIT} kg)`);
      const drop = nearest(W.drops, rt.key);
      if (!drop) return failQuest(q, rt, `줍을 ${rt.it.ko} 드롭이 없습니다`);
      if (moveToward(drop.pos, 0.8) > 0) break;
      rt.work += 1;
      W.energy += 20 * DT;
      if (rt.work >= 4) {
        rt.work = 0;
        W.drops.splice(W.drops.indexOf(drop), 1);
        a.carrying.push({ item_id: rt.key, label_ko: rt.it.ko, mass_kg: rt.it.mass });
        rt.collected += 1;
        s.note = `${rt.collected}/${rt.count}`;
        log('world', 'info', `${rt.it.ko} 줍기 (${rt.collected}/${rt.count}), 적재 ${loadKg().toFixed(1)} kg`, { grasp_margin: 0.8 });
      }
      s.progress = (rt.collected + rt.work / 4) / rt.count;
      if (rt.collected >= rt.count) nextStep(q, rt);
      break;
    }
    case 'carry': {
      const target = W.waypoints.find((p) => p.id === rt.dest.id).pos;
      const total = rt.total ?? (rt.total = Math.max(1, dist2(a.pos, target)));
      const left = moveToward(target, 1.5);
      s.progress = 1 - left / total;
      if (left <= 0) { rt.total = null; nextStep(q, rt); }
      break;
    }
    case 'deliver': {
      rt.work += 1;
      s.progress = rt.work / 8;
      if (rt.work >= 8) {
        const wp = rt.dest ? W.waypoints.find((p) => p.id === rt.dest.id) : null;
        if (wp) (wp.pile ??= []).push(...a.carrying.filter((c) => c.item_id === rt.key));
        a.carrying = a.carrying.filter((c) => c.item_id !== rt.key);
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
  const seen = visibleObjects(22).filter((o) => o.forward > 0.5 && o.kind !== 'waypoint');
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
