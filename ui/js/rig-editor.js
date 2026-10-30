// '로봇 설계' 화면: 로봇 프리셋을 고르고, 내장한 SolidCraft 편집기에서 고치고, 시뮬레이션에 넣는다.
// SolidCraft 웹 패키지는 이 앱의 ui/vendor/solidcraft 에서 같은 출처로 제공되는 페이지이고, 이 앱이 iframe 으로 품는다.
// window.solidcraft 명령 API 로 설계를 만들고, URDF 와 메시를 OPFS 에서 읽어 런타임의 rig_import 로 넘긴다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { toast } from './toast.js';
import { selectCenterTab } from './layout.js';
import { newWorld } from './sim.js';

const SC_URL = 'vendor/solidcraft/index.html';
const EXPORT_NAME = 'noto_rig';

let frame = null;
let sc = null;
let list = [];
let selected = null;
let lastImport = null;

function say(text, bad = false) {
  const s = $('rig-status');
  s.textContent = text;
  s.classList.toggle('bad', bad);
}

function preset() {
  return list.find((p) => p.id === selected) ?? null;
}

async function waitFor(check, ms, what) {
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    const v = await check();
    if (v) return v;
    await new Promise((r) => setTimeout(r, 120));
  }
  throw new Error(`${what}: 시간 초과`);
}

/** 편집기 페이지를 처음 쓸 때 불러온다. 패키지가 없으면 받는 방법을 알려 준다. */
async function editor() {
  if (sc) return sc;
  const page = await fetch(SC_URL).catch(() => null);
  if (!page?.ok) {
    $('rig-missing').hidden = false;
    throw new Error('SolidCraft 웹 패키지가 없습니다');
  }
  $('rig-missing').hidden = true;
  if (!frame) {
    frame = el('iframe', { id: 'rig-frame', title: 'SolidCraft 편집기', src: SC_URL });
    $('rig-stage').append(frame);
  }
  say('편집기를 불러오는 중입니다… (처음 한 번 수십 초 걸릴 수 있습니다)');
  sc = await waitFor(() => frame.contentWindow?.solidcraft, 120000, '편집기 시작');
  // 패키지의 CJK 글꼴은 일본어용이라 한글이 □ 로 나온다. 한글 글꼴을 넣기 전까지는 영어로 띄운다.
  await sc.request('ui.prefs', { language: 'en' }).catch(() => {});
  await sc.request('ui.home', { open: false }).catch(() => {});
  say('편집기 준비 완료');
  return sc;
}

const mm = (p) => p.map((v) => v * 1000);
const hex = (c) => `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;

function volume(s) {
  return s.shape === 'box' ? s.size[0] * s.size[1] * s.size[2] : Math.PI * s.radius * s.radius * s.height;
}

/** noto 의 로봇 설명을 SolidCraft 명령으로 만든다: 링크 하나가 부품 하나, 관절 이름이 그대로 관절 이름. */
async function buildDesign(api, d) {
  await api.execute('file.new', { name: d.name });
  await api.request('ui.home', { open: false }).catch(() => {});
  const occ = {};
  let first = true;
  for (const c of d.components) {
    const r = await api.execute('component.create', { name: c.name, parent: 'root', grounded: first, activate: true });
    occ[c.name] = r.occurrence;
    first = false;
    const bodies = [];
    for (const [i, s] of c.solids.entries()) {
      const body = `${c.name}_${i}`;
      bodies.push(body);
      if (s.shape === 'box') {
        await api.execute('solid.box', { length: s.size[0] * 1000, width: s.size[1] * 1000, height: s.size[2] * 1000, center: mm(s.center), body_name: body });
      } else {
        await api.execute('solid.cylinder', { radius: s.radius * 1000, height: s.height * 1000, base: mm(s.base), axis: s.axis, body_name: body });
      }
    }
    const litres = c.solids.reduce((n, s) => n + volume(s) * 1000, 0);
    const density = Math.min(1000, Math.max(0.001, c.mass_kg / Math.max(litres, 1e-6)));
    const material = `noto_${c.name}`;
    await api.execute('material.create', { name: material, density_g_cm3: density, color: hex(c.color) }).catch(() => {});
    await api.execute('material.assign', { bodies, material });
  }
  for (const j of d.joints) {
    const origin = (name) => ({ occurrence: occ[name], point: mm(j.at), z: j.axis });
    const p = { type: j.kind, a: origin(j.a), b: origin(j.b), flip: true, name: j.name };
    if (j.limits) p.limits = [j.kind === 'slider' ? j.limits.map((v) => v * 1000) : j.limits];
    await api.execute('joint.create', p);
  }
  await api.request('ui.camera', { fit: true }).catch(() => {});
}

/** 편집기는 시작할 때 OPFS 를 읽으므로, 밖에서 넣은 설계 파일은 페이지를 다시 불러와야 보인다. */
async function reloadEditor() {
  sc = null;
  frame.contentWindow.location.reload();
  await new Promise((r) => setTimeout(r, 500));
  return editor();
}

async function openPreset() {
  if (!selected) return;
  try {
    let api = await editor();
    say(`${preset()?.name_ko ?? selected} 불러오는 중…`);
    const r = await call('rig_design', { id: selected });
    const file = `noto_${selected}.sc`;
    const path = `/opfs/${file}`;
    const here = await (await opfs()).getFileHandle(file).then(() => true, () => false);
    if (here) {
      await api.execute('doc.open', { path });
    } else if (r.sc) {
      await writeOpfs(file, r.sc);
      api = await reloadEditor();
      await api.execute('doc.open', { path });
    } else {
      // 문서 이름은 편집기 글꼴로 그려지므로 프리셋 id 를 쓴다.
      await buildDesign(api, { ...r.design, name: selected });
    }
    say(`${preset()?.name_ko ?? selected}: 편집기에 열었습니다. 부품 이름과 관절 이름은 그대로 두고 고쳐 주세요.`);
  } catch (e) {
    say(e.message, true);
  }
}

async function opfs() {
  return navigator.storage.getDirectory();
}

async function writeOpfs(name, b64) {
  const root = await opfs();
  const f = await root.getFileHandle(name, { create: true });
  const w = await f.createWritable();
  await w.write(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
  await w.close();
}

function toBase64(buf) {
  const bytes = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

async function readFresh(root, name, since) {
  return waitFor(async () => {
    try {
      const f = await (await root.getFileHandle(name)).getFile();
      return f.lastModified >= since && f.size > 0 ? f : null;
    } catch {
      return null;
    }
  }, 15000, `${name} 읽기`);
}

/** 편집기의 지금 설계를 내보내 런타임으로 넘긴다. */
async function bringBack() {
  const name = $('rig-name').value.trim() || '새 로봇';
  try {
    const api = await editor();
    say('편집기에서 내보내는 중…');
    const since = Date.now() - 1500;
    const out = await api.execute('file.export', { path: `/opfs/${EXPORT_NAME}.urdf`, effort: 1000, velocity: 10 });
    const root = await opfs();
    const meshes = {};
    let text = '';
    for (const path of out.files) {
      const file = path.split('/').pop();
      const f = await readFresh(root, file, since);
      if (file.endsWith('.urdf')) text = await f.text();
      else meshes[file] = toBase64(await f.arrayBuffer());
    }
    lastImport = await call('rig_import', { name, urdf: text, meshes });
    // 다시 고칠 수 있게 설계 파일도 함께 둔다 (프리셋마다 하나).
    const file = `noto_${lastImport.id}.sc`;
    await api.execute('file.save_as', { path: `/opfs/${file}`, overwrite: true });
    const saved = await readFresh(root, file, since);
    await call('rig_keep_design', { id: lastImport.id, design: toBase64(await saved.arrayBuffer()) });
    await refresh(lastImport.id);
    show(lastImport);
    say(`${name}: 프리셋으로 저장했습니다 (${out.warnings?.length ? `내보내기 경고 ${out.warnings.length}건` : '경고 없음'}).`);
  } catch (e) {
    say(e.message, true);
  }
}

function show(v) {
  const box = clear($('rig-summary'));
  if (!v) return;
  const n = (x, d = 2) => (Number.isFinite(x) ? Number(x).toFixed(d) : '—');
  const rows = [
    ['질량', `${n(v.mass_kg, 1)} kg`],
    ['무게중심 높이', `${n(v.com_m)} m`],
    ['눈 높이', `${n(v.eye_m)} m`],
    [v.base === 'human_foot' ? '발 간격 · 길이' : '폭 · 축거', `${n(v.width_m)} · ${n(v.length_m)} m`],
    ['균형 여유', n(v.margin)],
  ];
  if (v.dof != null) rows.push(['자유도', String(v.dof)]);
  if (v.weakest) rows.push(['가장 힘든 관절', `${v.weakest.name} ${Math.round(v.weakest.ratio * 100)}%`]);
  box.append(el('div', { class: 'pv-kv' }, ...rows.flatMap(([k, val]) => [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: val })])));
  if (v.checks?.length) {
    box.append(el('div', { class: 'ct-sec', text: '투입 전 검사' }));
    box.append(el('ul', { class: 'rig-checks' }, ...v.checks.map((c) => el('li', { class: c.ok ? 'ok' : 'no', title: c.detail }, el('b', { text: `${c.ok ? '✓' : '✗'} ${c.label}` }), el('small', { text: c.detail })))));
  }
  if (v.warnings?.length) {
    box.append(el('div', { class: 'ct-sec', text: '가져오기 경고' }));
    box.append(el('ul', { class: 'pp-reasons' }, ...v.warnings.map((w) => el('li', { text: w }))));
  }
}

async function validate() {
  if (!selected) return;
  try {
    show(await call('rig_validate', { id: selected }));
  } catch (e) {
    say(e.message, true);
  }
}

async function apply() {
  if (!selected) return;
  try {
    const r = await call('rig_apply', { id: selected });
    state.bodyPlan = r.plan;
    state.config.base = r.base;
    bus.emit('config', state.config);
    bus.emit('rig-applied', r);
    const ok = await newWorld(Number($('world-seed').value) || 0, Number($('world-size').value) || 64);
    if (ok) {
      toast('로봇 적용', `${preset()?.name_ko ?? selected}(으)로 월드를 다시 만들었습니다.`);
      $('sum-rig').textContent = preset()?.name_ko ?? '';
      selectCenterTab('world');
    }
  } catch (e) {
    say(e.message, true);
  }
}

async function remove() {
  const p = preset();
  if (!p || p.builtin) return;
  try {
    await call('rig_delete', { id: p.id });
    await refresh();
  } catch (e) {
    say(e.message, true);
  }
}

async function refresh(pick) {
  const r = await call('rig_presets', {}, { quiet: true });
  list = r.presets;
  selected = pick ?? selected ?? r.current ?? list[0]?.id ?? null;
  for (const id of ['rig-preset', 'rig-preset-side']) {
    const sel = clear($(id));
    const group = (label, items) => {
      if (!items.length) return;
      const g = el('optgroup', { label });
      g.append(...items.map((p) => el('option', { value: p.id, text: p.name_ko })));
      sel.append(g);
    };
    group('기본 프리셋', list.filter((p) => p.builtin));
    group('내가 만든 로봇', list.filter((p) => !p.builtin));
    sel.value = selected ?? '';
  }
  $('rig-delete').disabled = preset()?.builtin !== false;
  $('rig-note').textContent = preset()?.note_ko ?? '';
  $('sum-rig').textContent = list.find((p) => p.id === r.current)?.name_ko ?? '';
  if (!$('rig-name').value && preset()) $('rig-name').value = preset().builtin ? '새 로봇' : preset().name_ko;
}

/** 편집기를 앱 창 가득 펼치거나 되돌린다. 앱 창 안에서 넓어질 뿐, 주인은 그대로 이 앱이다. */
function toggleWide() {
  const wide = $('cpane-rig').classList.toggle('wide');
  const b = $('rig-wide');
  b.textContent = wide ? '원래 크기' : '크게 보기';
  b.setAttribute('aria-pressed', String(wide));
}

export async function initRigEditor() {
  const pick = (e) => {
    selected = e.target.value;
    $('rig-name').value = '';
    refresh(selected).then(validate);
  };
  $('rig-preset').addEventListener('change', pick);
  $('rig-preset-side').addEventListener('change', pick);
  $('rig-open').addEventListener('click', openPreset);
  $('rig-back').addEventListener('click', bringBack);
  $('rig-check').addEventListener('click', validate);
  $('rig-apply').addEventListener('click', apply);
  $('rig-apply-side').addEventListener('click', apply);
  $('rig-delete').addEventListener('click', remove);
  $('rig-wide').addEventListener('click', toggleWide);
  $('rig-edit').addEventListener('click', () => selectCenterTab('rig'));
  bus.on('tab', ({ group, name }) => {
    if (group === 'center' && name === 'rig') editor().catch((e) => say(e.message, true));
  });
  try {
    await refresh();
    await validate();
  } catch {
    say('로봇 프리셋은 개발 서버나 Tauri 앱에서 쓸 수 있습니다.', true);
  }
}
