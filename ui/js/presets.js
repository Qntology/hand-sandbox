// 프리셋 Drop-up 의 부품 등급: 머리·가슴·배·팔·다리(섀시) 등급을 고르면 물리 의미(질량, CoM 높이, 들 수 있는 물건,
// 속도·경사·턱·적재 한계, N:N 기준 이탈)를 body_plan_eval 로 바로 계산해 보여 준다. 모습은 그대로지만 물리는 바뀐다.
// 한 줄 요약은 #pp-sum, 자세한 값은 아코디언의 #pp-detail 에 그린다. 적용은 월드 재생성으로 한다(unit-preset.js).
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, debounce, el, fmt } from './util.js';

const SEGS = [
  ['head', '머리 (목 · 카메라)'],
  ['chest', '가슴 (연산 · 배터리)'],
  ['belly', '배 (허리 · 골반)'],
  ['arm', '팔 (어깨 · 팔꿈치)'],
  ['leg', '다리 / 섀시'],
];
const NAMED_KO = { budget: '보급형 전체', mid: '중급형 전체', pro: '고급형 전체', budget_arms_mid_base: '보급 팔·머리 + 중급 몸', pro_arms: '고급 팔 + 중급 몸', battery_low: '중급 + 배터리 골반', wheel_lift: '중급 + 몸통 리프트·고급 허리 (바퀴형 바닥 집기)' };

let presets = null;
let evalSeq = 0;
let lastEval = null;

function plan() {
  // 몸의 크기(build)는 로봇 프리셋이 정한다. 여기서는 등급만 고른다.
  const p = { battery_low: $('pp-battery').checked, lift: $('pp-lift').checked, build: state.bodyPlan?.build ?? 'standard' };
  for (const [id] of SEGS) p[id] = $(`pp-${id}`).value;
  return p;
}

function setPlan(p) {
  for (const [id] of SEGS) $(`pp-${id}`).value = p[id];
  $('pp-battery').checked = !!p.battery_low;
  $('pp-lift').checked = !!p.lift;
}

function r(x, d = 2) {
  return Number.isFinite(x) ? Number(x).toFixed(d) : '—';
}

/** 📊 한 줄 요약: 질량 · CoM 과 체형의 투입 전 검사(rig_validate). */
function renderSummary() {
  const box = clear($('pp-sum'));
  const v = lastEval;
  if (v) box.append(`📊 ${r(v.mass_kg, 1)} kg · CoM ${r(v.com_height_m)} m · 적재 ${r(v.limits_empty?.capacity_kg, 1)} kg`);
  for (const c of state.rigChecks ?? []) {
    box.append(' · ', el('span', { class: c.ok ? 'ok' : 'bad', title: c.detail, text: `${c.label} ${c.ok ? '✓' : '✗'}` }));
  }
}

async function evaluate() {
  const seq = ++evalSeq;
  state.bodyPlan = plan();
  bus.emit('body-plan', state.bodyPlan);
  let v;
  try {
    v = await call('body_plan_eval', { body: state.bodyPlan, base: state.draftBase ?? state.config.base }, { quiet: true });
  } catch (e) {
    $('pp-sum').textContent = `계산 실패: ${e.message}`;
    return;
  }
  if (seq !== evalSeq) return;
  lastEval = v;
  renderSummary();
  const box = clear($('pp-detail'));
  const e0 = v.limits_empty;
  const e8 = v.limits_loaded;
  box.append(el('div', { class: 'pv-kv' },
    ...[
      ['질량 · CoM 높이', `${r(v.mass_kg, 1)} kg · ${r(v.com_height_m)} m`],
      ['배터리 · 부품비', `${v.battery_wh} Wh · $${Math.round(v.cost_usd).toLocaleString()}`],
      ['속도 (빈 / 8 kg)', `${r(e0.max_speed_mps)} / ${r(e8.max_speed_mps)} m/s`],
      ['경사 · 턱 (빈)', `${r(e0.max_slope_deg, 1)}° · ${r(e0.max_step_m * 100, 0)} cm`],
      ['상시 적재', `${r(e0.capacity_kg, 1)} kg`],
    ].flatMap(([k, val]) => [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: val })])));
  box.append(el('div', { class: 'pp-lifts' }, ...v.lifts.map((l) => el('div', { class: `pp-lift ${!l.ok ? 'no' : l.two_arms ? 'two' : 'ok'}`, title: l.limiting ?? '' },
    el('b', { text: `${l.label_ko} ${r(l.mass_kg, 1)} kg · ${!l.ok ? '못 듦' : l.two_arms ? '두 팔' : '한 팔'}` }),
    el('small', { text: l.ok ? `${l.worst} ${fmt.pct(l.worst_ratio)}` : (l.limiting ?? '') })))));
  const off = (v.nn ?? []).filter((c) => c.dev != null && Math.abs(c.dev) > 1).sort((a, b) => Math.abs(b.dev) - Math.abs(a.dev)).slice(0, 6);
  if (off.length) {
    box.append(el('div', { class: 'ct-sec', text: 'N:N 기준 이탈 (사람 기준 대비, 큰 순)' }));
    box.append(el('ul', { class: 'pp-reasons' }, ...off.map((c) => el('li', { title: c.cite, text: `${c.source} → ${c.target} ${c.axis}: ${r(c.value)} (기준 ${r(c.reference)} ± ${r(c.tolerance)}, 편차 ${r(c.dev, 1)})` }))));
  }
  const reasons = [...new Set([...(e0.reasons ?? []), ...(e8.reasons ?? []).map((x) => `8 kg: ${x}`)])].slice(0, 6);
  if (reasons.length) box.append(el('ul', { class: 'pp-reasons' }, ...reasons.map((x) => el('li', { text: x }))));
  $('sum-presets').textContent = v.id;
}

const later = debounce(evaluate, 150);

/** 런타임이 지금 쓰는 몸(world_body 의 plan)을 폼과 월드 기준값으로 맞춘다. */
export function setPlanFrom(p) {
  setPlan(p);
  state.bodyPlan = { ...state.bodyPlan, build: p.build ?? 'standard' };
  state.bodyPlan = plan();
  state.worldPlanKey = JSON.stringify(state.bodyPlan);
  later();
}

export async function initPresets() {
  state.bodyPlan = { head: 'mid', chest: 'mid', belly: 'mid', arm: 'mid', leg: 'mid', battery_low: false, lift: false, build: 'standard' };
  for (const [id] of SEGS) $(`pp-${id}`).addEventListener('change', () => { $('pp-named').value = ''; later(); });
  $('pp-battery').addEventListener('change', later);
  $('pp-lift').addEventListener('change', () => { $('pp-named').value = ''; later(); });
  $('pp-named').addEventListener('change', (e) => {
    const n = e.target.value;
    if (!n || !presets) return;
    const map = {
      budget: { head: 'budget', chest: 'budget', belly: 'budget', arm: 'budget', leg: 'budget', battery_low: false },
      mid: { head: 'mid', chest: 'mid', belly: 'mid', arm: 'mid', leg: 'mid', battery_low: false },
      pro: { head: 'pro', chest: 'pro', belly: 'pro', arm: 'pro', leg: 'pro', battery_low: false },
      budget_arms_mid_base: { head: 'budget', chest: 'mid', belly: 'mid', arm: 'budget', leg: 'mid', battery_low: false },
      pro_arms: { head: 'mid', chest: 'mid', belly: 'mid', arm: 'pro', leg: 'mid', battery_low: false },
      battery_low: { head: 'mid', chest: 'mid', belly: 'mid', arm: 'mid', leg: 'mid', battery_low: true },
      wheel_lift: { head: 'mid', chest: 'mid', belly: 'pro', arm: 'mid', leg: 'mid', battery_low: false, lift: true },
    };
    if (map[n]) setPlan(map[n]);
    later();
  });
  bus.on('config', later);
  bus.on('draft-base', later);
  bus.on('rig-checks', renderSummary);
  try {
    const r0 = await call('body_presets', {}, { quiet: true });
    presets = r0.presets;
    const fill = (sel, tiers) => {
      clear(sel);
      for (const t of tiers) sel.append(el('option', { value: t.id, text: t.label_ko }));
    };
    for (const [id] of SEGS) fill($(`pp-${id}`), id === 'leg' ? presets.leg_tiers : presets.tiers);
    clear($('pp-named')).append(el('option', { value: '', text: '직접 조합' }), ...presets.named.map((n) => el('option', { value: n, text: NAMED_KO[n] ?? n })));
    setPlan(r0.current);
    state.bodyPlan.build = r0.current.build ?? 'standard';
    $('pp-named').value = r0.current_id in NAMED_KO ? r0.current_id : '';
  } catch {
    for (const [id] of SEGS) {
      const sel = $(`pp-${id}`);
      clear(sel);
      for (const t of ['budget', 'mid', 'pro']) sel.append(el('option', { value: t, text: t }));
      sel.value = 'mid';
    }
  }
  evaluate();
}
