// 앱 전체가 공유하는 상태와 간단한 이벤트 버스.
const target = new EventTarget();

export const bus = {
  on(name, fn) {
    const h = (e) => fn(e.detail);
    target.addEventListener(name, h);
    return () => target.removeEventListener(name, h);
  },
  emit(name, detail) {
    target.dispatchEvent(new CustomEvent(name, { detail }));
  },
};

export const state = {
  presets: null,
  dict: [],
  dictById: new Map(),
  paths: null,
  config: {
    fingers: 5,
    joints: {
      segments: 3,
      base_kind: 'universal',
      distal_kind: 'hinge',
      actuation: { mode: 'coupled', ratio: 0.67 },
    },
    base: 'wheel',
  },
  geometry: null,
  grasp: null,
  topPrediction: null,

  // 서브 작업자가 있으면 2 (모의 백엔드의 옛 순차 협동 모형이 읽는다); crew 는 crew_status 그대로.
  teamSize: 1,
  activeRobot: 1,
  crew: null,
  mock: false,
  playing: false,
  speed: 1,
  toggles: { cctv: true, boxes: true, path: true, oracle: false },

  terrain: 'rugged',
  worldConfigKey: null,
  worldPlanKey: null,
  snapshot: null,
  frame: null,
  detections: [],
  procedure: null,
  quests: [],
  body: null,
  bodyPlan: null,
  perception: null,
  lastCommand: null,
  sds: null,
  forecast: null,
  depthView: null,
  device: null,
  models: null,
  modelCatalog: null,

  selectedUnit: null, // 월드에서 고른 유닛 id (없으면 null)
  units: [], // world_frame.units: 탑뷰 프레임 위 로봇 (u, v 는 프레임 안 0..1)
  unitInfo: new Map(), // units_list: id → 배터리·체형까지 든 항목
  draftBase: null, // 프리셋 Drop-up 에서 고른 발 (적용 전)
  rigChecks: [], // 프리셋 Drop-up 에서 고른 체형의 투입 전 검사
};
