// 화면에 보이지 않는 SolidCraft: 유닛 외형을 AI 로 고칠 때 설계를 CAD 로 조립해 URDF · 메시 · .sc 도면으로 내보낸다.
// 편집기 UI 는 어디에도 나타나지 않는다. ui/vendor/solidcraft 페이지를 화면 밖 iframe 으로 처음 쓸 때 불러오고
// (WebGL 이 필요해 실제 크기로 둔다, display:none 금지), window.solidcraft 명령 API 로 조립한 뒤 OPFS 에서 결과를 읽는다.
// 패키지가 없거나 어디서든 실패하면 null 로 끝난다: 런타임이 URDF 를 직접 만드는 길이 따로 있다.
import { el } from './util.js';

const SC_URL = 'vendor/solidcraft/index.html';
const READY_MS = 120000;
const FILE_MS = 15000;

let frame = null;
let sc = null;
let queue = Promise.resolve(); // 조립은 한 번에 하나씩: 문서가 하나뿐이다

async function waitFor(check, ms, what) {
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    const v = await check();
    if (v) return v;
    await new Promise((r) => setTimeout(r, 120));
  }
  throw new Error(`${what}: 시간 초과`);
}

/** 숨은 CAD 를 처음 쓸 때 불러온다. 패키지가 없으면 null. */
async function cad() {
  if (sc) return sc;
  const page = await fetch(SC_URL).catch(() => null);
  if (!page?.ok) return null;
  if (!frame) {
    frame = el('iframe', {
      title: 'SolidCraft (숨김)', src: SC_URL, tabindex: -1, aria: { hidden: 'true' },
      style: 'position:fixed;left:-10000px;top:0;width:1280px;height:800px;border:0',
    });
    document.body.append(frame);
  }
  sc = await waitFor(() => frame.contentWindow?.solidcraft, READY_MS, 'SolidCraft 시작');
  // 패키지의 CJK 글꼴은 일본어용이라 한글이 □ 로 나온다. 한글 글꼴을 넣기 전까지는 영어로 띄운다.
  await sc.request('ui.prefs', { language: 'en' }).catch(() => {});
  await sc.request('ui.home', { open: false }).catch(() => {});
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
  }, FILE_MS, `${name} 읽기`);
}

/** 설계를 조립해 내보낸다. 문서 이름은 편집기 글꼴로 그려지므로 id 를 쓴다. */
async function exportDesign(api, design, id) {
  await buildDesign(api, { ...design, name: `noto_${id}` });
  const since = Date.now() - 1500;
  const out = await api.execute('file.export', { path: `/opfs/noto_${id}.urdf`, effort: 1000, velocity: 10 });
  const root = await navigator.storage.getDirectory();
  const meshes = {};
  let urdf = '';
  for (const path of out.files) {
    const file = path.split('/').pop();
    const f = await readFresh(root, file, since);
    if (file.endsWith('.urdf')) urdf = await f.text();
    else meshes[file] = toBase64(await f.arrayBuffer());
  }
  // 다시 고칠 수 있게 설계 파일도 함께 둔다.
  const scFile = `noto_${id}.sc`;
  await api.execute('file.save_as', { path: `/opfs/${scFile}`, overwrite: true });
  const saved = await readFresh(root, scFile, since);
  return { urdf, meshes, sc: toBase64(await saved.arrayBuffer()) };
}

/**
 * rig_design 모양의 설계를 CAD 로 조립해 {urdf, meshes, sc} 를 돌려준다.
 * 패키지가 없거나 실패하면 null (호출자는 URDF 없이 rig_commit 을 불러 런타임의 길로 간다).
 */
export function headlessCad(design, id) {
  const run = async () => {
    try {
      const api = await cad();
      return api ? await exportDesign(api, design, id) : null;
    } catch (e) {
      console.warn('헤드리스 CAD 실패, 런타임 URDF 로 넘어갑니다:', e);
      return null;
    }
  };
  const job = queue.then(run, run);
  queue = job;
  return job;
}
