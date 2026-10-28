// 줍기 절차 HUD: 사람처럼 "고개를 숙여 응시 → 비전 확인 → 자세 낮춤 → 쥐기 → 들어올림" 순서를 1인칭 화면 위에 보인다.
// world_frame 의 procedure(Rust 런타임 또는 모의 월드)를 그대로 그리며, 비전 확인 결과가 있으면 대상에 조준점을 겹친다.
import { $, clear, el, fmt } from './util.js';

const STATE_KO = { done: '완료', active: '진행', pending: '대기', failed: '실패' };

function stageRow(s) {
  return el('li', { class: `ph-stage ${s.state}`, title: s.note ?? '' },
    el('span', { class: 'ph-dot', aria: { hidden: 'true' } }),
    el('span', { class: 'ph-nm', text: s.label_ko }),
    el('span', { class: 'ph-st', text: STATE_KO[s.state] ?? s.state }));
}

function checkLine(c) {
  if (!c) return el('div', { class: 'ph-check wait', text: '비전 확인 대기 — 고개를 숙여 대상을 화면 가운데에 둡니다' });
  if (!c.ok) return el('div', { class: 'ph-check fail', text: `비전 확인 실패 · ${c.reason ?? '대상이 보이지 않음'}` });
  const depth = c.depth_m != null ? ` · 깊이 모델 ${fmt.n(c.depth_m, 2)} m` : '';
  return el('div', { class: 'ph-check ok', text: `비전 확인 ✓ ${c.label_ko ?? ''} ${c.score != null ? Math.round(c.score * 100) : '—'} · 거리 ${fmt.n(c.range_m, 2)} m${depth}` });
}

function pose(p) {
  const bits = [];
  if (p.pitch_deg != null) bits.push(`카메라 ${fmt.n(p.pitch_deg, 0)}°`);
  if (p.neck_deg != null) bits.push(`목 ${fmt.n(p.neck_deg, 0)}°`);
  if (p.trunk_deg != null) bits.push(`몸통 ${fmt.n(p.trunk_deg, 0)}°`);
  if (p.eye_m != null) bits.push(`눈 ${fmt.n(p.eye_m, 2)} m${p.standing_eye_m ? ` (선 자세 ${fmt.n(p.standing_eye_m, 2)})` : ''}`);
  return bits.join(' · ');
}

function reticle(c) {
  const layer = $('det-layer');
  layer.querySelectorAll('.gaze-reticle').forEach((n) => n.remove());
  if (!c?.bbox || c.bbox.length < 4) return;
  const [x0, y0, x1, y1] = c.bbox;
  const r = el('div', { class: `gaze-reticle${c.ok ? ' ok' : ' fail'}`, aria: { hidden: 'true' } });
  r.style.left = `${((x0 + x1) / 2) * 100}%`;
  r.style.top = `${((y0 + y1) / 2) * 100}%`;
  layer.append(r);
}

/** world_frame 응답의 procedure 를 HUD 로 그린다. 없으면 숨긴다. */
export function renderPickHud(p) {
  const hud = $('pick-hud');
  if (!hud) return;
  const show = !!p && p.act != null;
  hud.hidden = !show;
  reticle(show ? p.check : null);
  if (!show) return;
  $('ph-title').textContent = `${p.act === 'look' ? '응시' : '줍기'} · ${p.label_ko ?? p.target ?? ''}`;
  $('ph-pose').textContent = pose(p);
  const list = clear($('ph-stages'));
  for (const s of p.stages ?? []) list.append(stageRow(s));
  const box = clear($('ph-check'));
  box.append(checkLine(p.check));
}
