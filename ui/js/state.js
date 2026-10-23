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
  // 손·발 구성 (hand_geometry / grasp_test / world_new 에 그대로 전달되는 HandConfig)
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

  // 월드
  snapshot: null,
  quests: [],
  body: null, // body_status (신체 상태 HUD)
  bodyPlan: null, // 부품 프리셋 BodyPlan (world_new · test_start 에 그대로 전달)
  detections: [],
  frame: null, // {bytes, encode_ms, tick, width, height}
  worldConfigKey: null, // 현재 월드를 만들 때 쓴 손·발 구성(JSON)
  trail: [], // 에이전트가 지나온 위치 [x, y]

  // 분석
  perception: null,
  lastCommand: null,
  sds: null,
  forecast: null,
  device: null,
  models: null, // models_status: {recognizer, embedding, ttm, parser_embedding, device, siglip_files, ttm_files}

  // 화면
  centerTab: 'world',
  mock: false,
  playing: false,
  speed: 1,
  toggles: { boxes: true, path: true, oracle: false, body: true, quest: true, pip: true },
};
