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

  teamSize: 2,
  activeRobot: 1,
  centerTab: 'world',
  mock: false,
  playing: false,
  speed: 1,
  toggles: { boxes: true, path: true, oracle: false, body: true, quest: true, pip: true },

  terrain: 'rugged',
  worldConfigKey: null,
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
};
