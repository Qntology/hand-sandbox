// 모의 백엔드: Tauri 없이 브라우저(python3 -m http.server)에서 UI 를 돌려 보기 위한 가짜 구현.
// api.js 가 window.__TAURI__ 를 찾지 못할 때만 불러온다. 반환 형태는 계약(docs/PLAN.md 8장)과 같고,
// 거부(reject)는 Tauri 처럼 문자열로 던진다. 실제 월드·인식·SDS·TTM 이 아니라 데모용 숫자다.
import { DICTIONARY } from './mock-data.js';
import * as hand from './mock-hand.js';
import * as mw from './mock-world.js';
import { renderFrame } from './mock-render.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hash = (s) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
};

function sdsStatus() {
  const w = mw.world();
  const tick = w?.tick ?? 0;
  const scopes = [
    ['vision|철광석|primary', 'margin', 'welford'], ['vision|나무|primary', 'margin', 'welford'], ['vision|바위|primary', 'score', 'welford'],
    ['search|item|primary', 'score', 'decay_shape'], ['motion|mine|primary', 'success', 'welford'],
  ];
  const axes = scopes.map(([scope, axis, kind], k) => {
    const mean = 0.5 + 0.25 * hash(scope);
    return {
      scope, axis, kind, n: 8 + Math.floor(tick / (5 + k)), mean, sd: 0.04 + 0.05 * hash(axis + scope),
      recent: mean + 0.08 * Math.sin(tick / 17 + k), drift_z: 1.7 * Math.sin(tick / 40 + k * 1.3),
    };
  });
  const sum = (f) => axes.filter(f).reduce((a, x) => a + x.n, 0);
  return {
    dir: '(모의) ~/.local/share/noto-sandbox/sds',
    recipe: 'vision-v1|margin-z|mock',
    ledgers: [
      { ledger: 'vision', scopes: 3, observations: sum((x) => x.scope.startsWith('vision')), run_observations: tick },
      { ledger: 'search', scopes: 1, observations: sum((x) => x.scope.startsWith('search')), run_observations: Math.floor(tick / 30) },
      { ledger: 'motion', scopes: 1, observations: sum((x) => x.scope.startsWith('motion')), run_observations: Math.floor(tick / 20) },
      { ledger: 'ts', scopes: 2, observations: 24 + Math.floor(tick / 10), run_observations: Math.floor(tick / 45) },
    ],
    axes,
    log: (w?.logs ?? []).filter((l) => l.cat === 'sds' || l.cat === 'ts').slice(-6).map((l) => `${l.ts.slice(11, 19)} ${l.msg}`),
  };
}

function tsForecast({ scope, axis }) {
  const w = mw.world();
  const seed = hash(`${scope}/${axis}`);
  const n = 40;
  const m = 16;
  const phase = Math.floor((w?.tick ?? 0) / 8);
  const base = 0.45 + 0.3 * seed;
  const history = Array.from({ length: n }, (_, i) => {
    const t = i + phase;
    return Number((base + 0.12 * Math.sin(t / 6 + seed * 9) + 0.05 * Math.sin(t / 2.3) + 0.02 * (hash(`${scope}${t}`) - 0.5) * 2).toFixed(4));
  });
  const last = history[n - 1];
  const trend = (history[n - 1] - history[n - 6]) / 5;
  const point = [];
  const p10 = [];
  const p90 = [];
  let v = last;
  let tr = trend;
  for (let h = 1; h <= m; h++) {
    tr *= 0.85;
    v += tr;
    const spread = 0.035 + 0.013 * Math.sqrt(h);
    point.push(Number(v.toFixed(4)));
    p10.push(Number((v - spread).toFixed(4)));
    p90.push(Number((v + spread).toFixed(4)));
  }
  mw.log('ts', 'info', `예측 조회: ${scope}/${axis} (모의, 융합)`, { method: 'fused', history: n, horizon: m });
  return {
    history, point, p10, p90, method: 'fused:ttm-r3+holt-damped',
    weights: [{ method: 'ttm-r3', weight: 0.62 }, { method: 'holt-damped', weight: 0.38 }],
  };
}

function worldNew(a) {
  if (!Number.isFinite(a.seed) || a.seed < 0) throw 'seed 는 0 이상의 숫자여야 합니다';
  if (!(a.size >= 16 && a.size <= 256)) throw 'size 는 16~256 셀이어야 합니다';
  if (!a.config?.fingers) throw 'config(NotoConfig)가 필요합니다';
  return mw.worldNew(a);
}

const handlers = {
  presets: hand.presets,
  hand_geometry: hand.handGeometry,
  grasp_test: hand.graspTest,
  drive_courses: hand.driveCourses,
  dictionary: () => DICTIONARY,
  devices: hand.devices,
  select_device: hand.selectDevice,
  classify_image: async (a) => {
    if (typeof a.imageBase64 !== 'string') throw 'imageBase64 가 필요합니다';
    await sleep(350);
    return hand.classifyImage(a);
  },
  default_paths: hand.paths,
  synthesize: async (a) => { await sleep(500); return hand.synthesize(a); },
  load_dictionary: async () => { await sleep(150); return hand.loadDictionary(); },
  dataset_summary: hand.datasetSummary,
  world_new: worldNew,
  sim_step: mw.simStep,
  world_frame: renderFrame,
  command_submit: mw.commandSubmit,
  quest_list: mw.questList,
  perception_scan: async (a) => { await sleep(120); return mw.perceptionScan(a); },
  body_status: mw.bodyStatus,
  sds_status: sdsStatus,
  ts_forecast: tsForecast,
  log_tail: mw.logTail,
  models_status: async () => ({
    recognizer: 'mock', embedding: false, ttm: false, parser_embedding: false,
    device: { backend: 'cpu', ordinal: 0 },
    siglip_files: { dir: '(브라우저 모의)', present: false }, ttm_files: { dir: '(브라우저 모의)', present: false },
  }),
  recognizer_use: async () => { throw '브라우저 모의 백엔드에서는 SigLIP2 를 불러올 수 없습니다. Tauri 앱에서 사용하세요'; },
  // 모의 백엔드는 모델을 쓰지 않으므로 필수 모델 게이트를 걸지 않는다(required: false).
  models_catalog: async () => ({
    required: false, mock: true, ready: false, all_present: false, loading: null, load_error: null,
    device: { backend: 'cpu', ordinal: 0 },
    download: { running: false, phase: '', done: 0, total: 0, rate_bps: 0, queue: [] },
    models: [
      ['siglip2', 'SigLIP2 base (patch16, NaFlex)', '비전 인식 · 의미 위치 · 명령 임베딩', 'google/siglip2-base-patch16-naflex', 1535341857],
      ['ttm', 'Granite TTM-R3 (52-16)', '시계열 예측 (정밀도 게이트)', 'ibm-granite/granite-timeseries-ttm-r3', 5888229],
      ['tips', 'TIPSv2-B/14 DPT', '깊이 추정 (키프레임)', 'google/tipsv2-b14-dpt', 632074338],
      ['sam', 'MobileSAM (TinyViT)', '마스크 추적', 'lmz/candle-sam', 40613024],
    ].map(([id, label, role_ko, repo, bytes]) => ({ id, label, role_ko, repo, revision: '(모의)', license: 'Apache-2.0', dir: '(브라우저 모의)', files: [], bytes, have: 0, present: false })),
  }),
  models_download: async () => { throw '모델 받기는 Tauri 앱에서만 됩니다'; },
  depth_view: async () => { throw '깊이 · 카메라 탭은 실제 Rust 런타임(개발 서버 또는 Tauri)에서만 동작합니다'; },
  models_cancel: async () => { throw '모델 받기는 Tauri 앱에서만 됩니다'; },
  models_delete: async () => { throw '모델 삭제는 Tauri 앱에서만 됩니다'; },
  models_activate: async () => { throw '모델 불러오기는 Tauri 앱에서만 됩니다'; },
  test_status: async () => ({ active: false }),
  test_start: async () => { throw '통합 테스트는 실제 Rust 런타임(개발 서버 또는 Tauri)에서만 동작합니다'; },
  body_presets: async () => { throw '부품 프리셋 물리는 실제 Rust 런타임에서 계산합니다'; },
  body_plan_eval: async () => { throw '부품 프리셋 물리는 실제 Rust 런타임에서 계산합니다'; },
  ttm_use: async () => { throw '브라우저 모의 백엔드에서는 Granite TTM 을 불러올 수 없습니다. Tauri 앱에서 사용하세요'; },
  test_cancel: async () => ({ active: false }),
  inventory_drop: mw.inventoryDrop,
  world_reset: mw.worldReset,
  body_reset: mw.bodyReset,
  set_team_mode: mw.setTeamMode,
  crew_add: mw.crewAdd,
  crew_clear: mw.crewClear,
  crew_mode: mw.crewMode,
  crew_status: mw.crewStatusCall,
  camera_rig: mw.cameraRig,
  world_terrain: async () => { throw '모의 백엔드에는 지형 격자가 없습니다'; },
};

export async function invoke(cmd, args = {}) {
  const h = handlers[cmd];
  if (!h) throw `unknown command ${cmd}`;
  const out = await h(args ?? {});
  if (cmd === 'world_frame') return out; // 문자열이 큰 응답은 복사하지 않는다
  return out === undefined ? null : JSON.parse(JSON.stringify(out));
}
