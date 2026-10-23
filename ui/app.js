// Logistics robot hand simulator UI. Talks to the Rust backend through Tauri commands.
"use strict";

const { invoke } = window.__TAURI__.core;
const $ = (id) => document.getElementById(id);

const COURSE_LABELS = [
  "평지 10m",
  "경사 5m (10°)",
  "턱 5cm",
  "경사 5m (20°)",
  "계단 15cm",
  "평지 5m",
];

const state = {
  presets: null,
  dict: [],
  dictById: new Map(),
  paths: null,
  config: {
    fingers: 5,
    joints: {
      segments: 3,
      base_kind: "universal",
      distal_kind: "hinge",
      actuation: { mode: "coupled", ratio: 0.67 },
    },
    base: "wheel",
  },
  closure: 0,
  geometry: null,
  grasp: null,
  pending: null,
};

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.style.color = isError ? "var(--bad)" : "";
}

async function call(cmd, args) {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    setStatus(`${cmd}: ${e}`, true);
    throw e;
  }
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// ---------- hand configuration ----------

function renderFingerChoices() {
  const wrap = $("finger-count");
  wrap.replaceChildren();
  for (const f of state.presets.fingers) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `${f.count}지`;
    b.title = f.label_ko;
    b.setAttribute("aria-pressed", String(f.count === state.config.fingers));
    b.addEventListener("click", () => {
      state.config.fingers = f.count;
      syncFingerChoices();
      requestRefresh();
    });
    wrap.append(b);
  }
  syncFingerChoices();
}

function syncFingerChoices() {
  const preset = state.presets.fingers.find((f) => f.count === state.config.fingers);
  for (const b of $("finger-count").children) {
    b.setAttribute("aria-pressed", String(b.textContent === `${state.config.fingers}지`));
  }
  $("finger-desc").textContent = preset ? `${preset.label_ko}: ${preset.description_ko}` : "";
}

function renderJointSelects() {
  for (const id of ["base-kind", "distal-kind"]) {
    const sel = $(id);
    sel.replaceChildren();
    for (const k of state.presets.joint_kinds) {
      const o = document.createElement("option");
      o.value = k.id;
      o.textContent = k.label_ko;
      sel.append(o);
    }
  }
}

function renderBaseChoices() {
  const wrap = $("base-choices");
  wrap.replaceChildren();
  for (const b of state.presets.bases) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "base";
    input.value = b.base.kind;
    input.checked = b.base.kind === state.config.base;
    input.addEventListener("change", () => {
      state.config.base = input.value;
      requestRefresh();
      refreshDrive();
    });
    label.append(input, document.createTextNode(` ${b.label_ko}`));
    wrap.append(label);
  }
}

function readJointControls() {
  const segments = Number($("segments").value);
  const mode = $("actuation").value;
  $("ratio-row").hidden = mode !== "coupled";
  state.config.joints = {
    segments,
    base_kind: $("base-kind").value,
    distal_kind: $("distal-kind").value,
    actuation: mode === "coupled"
      ? { mode: "coupled", ratio: Number($("ratio").value) }
      : { mode: "direct" },
  };
  $("ratio-out").textContent = Number($("ratio").value).toFixed(2);
}

function applyJointControls() {
  const j = state.config.joints;
  $("segments").value = String(j.segments);
  $("base-kind").value = j.base_kind;
  $("distal-kind").value = j.distal_kind;
  $("actuation").value = j.actuation.mode;
  if (j.actuation.mode === "coupled") $("ratio").value = String(j.actuation.ratio);
  readJointControls();
}

// ---------- object ----------

function currentEntry() {
  return state.dictById.get($("label").value);
}

function payload() {
  const e = currentEntry();
  return {
    name: e ? e.label_en : "object",
    size_m: e ? e.size_m : [0.1, 0.1, 0.1],
    mass_kg: Number($("mass").value),
    friction: Number($("friction").value),
  };
}

function setup() {
  return {
    z_offset: Number($("zoff").value),
    contact_margin_m: 0.004,
    safety_factor: 2.0,
  };
}

function populateDictionary() {
  const sel = $("label");
  sel.replaceChildren();
  for (const e of state.dict) {
    state.dictById.set(e.id, e);
    const o = document.createElement("option");
    o.value = e.id;
    o.textContent = `${e.label_ko} · ${e.label_en}${e.graspable ? "" : " (파지 제외)"}`;
    sel.append(o);
  }
  sel.value = "carton_medium";
  applyEntryDefaults();
}

function applyEntryDefaults() {
  const e = currentEntry();
  if (!e) return;
  const mass = Math.min(5, Math.max(0.1, e.typical_mass_kg));
  $("mass").value = String(mass);
  const [w, h, d] = e.size_m.map((v) => Math.round(v * 1000));
  $("size-note").textContent =
    `${w} × ${h} × ${d} mm, 사전 질량 ${e.typical_mass_kg} kg` +
    (e.graspable ? "" : " — 이 대상은 파지 대상이 아닙니다 (사전 graspable=false).") +
    ` 출처: ${e.source}`;
  updateReadouts();
}

function updateReadouts() {
  $("mass-out").textContent = Number($("mass").value).toFixed(2);
  $("friction-out").textContent = Number($("friction").value).toFixed(2);
  $("zoff-out").textContent = Number($("zoff").value).toFixed(3);
  $("closure-out").textContent = Number($("closure").value).toFixed(2);
}

// ---------- refresh ----------

const requestRefresh = debounce(() => refreshAll(), 60);
const requestGrasp = debounce(() => refreshGrasp(), 120);

async function refreshAll() {
  readJointControls();
  updateReadouts();
  await refreshGeometry();
  if ($("show-grasp").checked) await refreshGrasp();
}

async function refreshGeometry() {
  try {
    state.geometry = await call("hand_geometry", {
      config: state.config,
      closure: Number($("closure").value),
    });
    $("geom-note").textContent =
      `손가락 최대 도달 ${(state.geometry.reach_m * 100).toFixed(1)} cm · ` +
      `손바닥 ${(state.geometry.palm[0] * 100).toFixed(0)} × ${(state.geometry.palm[1] * 100).toFixed(0)} mm`;
    draw();
  } catch {
    state.geometry = null;
    draw();
  }
}

async function refreshGrasp() {
  if (!$("show-grasp").checked) {
    state.grasp = null;
    renderGrasp();
    draw();
    return;
  }
  try {
    state.grasp = await call("grasp_test", {
      config: state.config,
      payload: payload(),
      setup: setup(),
    });
    renderGrasp();
    draw();
  } catch {
    state.grasp = null;
  }
}

async function refreshDrive() {
  const rows = await call("drive_courses", { course: null });
  const table = $("drive-table");
  table.replaceChildren();
  const head = table.insertRow();
  for (const h of ["발", "통과", "막힌 구간", "시간 s", "에너지 kJ"]) {
    const th = document.createElement("th");
    th.textContent = h;
    head.append(th);
  }
  for (const row of rows) {
    const r = table.insertRow();
    if (row.base.kind === state.config.base) r.style.fontWeight = "600";
    r.insertCell().textContent = row.label_ko;
    const pass = r.insertCell();
    pass.textContent = row.report.passable ? "통과" : "실패";
    pass.className = row.report.passable ? "ok" : "bad";
    const at = row.report.failed_at;
    r.insertCell().textContent = at == null ? "—" : COURSE_LABELS[at];
    const t = r.insertCell();
    t.className = "num";
    t.textContent = row.report.time_s.toFixed(1);
    const e = r.insertCell();
    e.className = "num";
    e.textContent = (row.report.energy_j / 1000).toFixed(2);
  }
}

// ---------- results ----------

const GRASP_TYPE_KO = { none: "없음", pinch: "핀치 (2지 이하)", power: "파워 (3지 이상)" };

function renderGrasp() {
  const dl = $("grasp-results");
  dl.replaceChildren();
  const g = state.grasp;
  if (!g) {
    addRow(dl, "상태", "파지 시뮬레이션을 켜면 결과가 표시됩니다");
    return;
  }
  const r = g.result;
  const fingerContacts = r.contacts.filter((c) => c.source.source === "finger").length;
  const palm = r.contacts.some((c) => c.source.source === "palm");
  addRow(dl, "접촉 (손가락 / 전체)", `${fingerContacts} / ${r.contacts.length}`);
  addRow(dl, "파지 유형", GRASP_TYPE_KO[r.grasp_type] ?? r.grasp_type);
  addRow(dl, "손바닥 지지", palm ? "예" : "아니오");
  addRow(dl, "힘 닫힘 (진단)", r.force_closed ? "예" : "아니오");
  addRow(dl, "마찰 용량", `${r.friction_capacity_n.toFixed(1)} N`);
  addRow(dl, "필요 힘 (질량·g·안전계수)", `${r.required_n.toFixed(1)} N`);
  addRow(dl, "최대 파지 질량", `${r.max_payload_kg.toFixed(2)} kg`);
  const verdict = document.createElement("span");
  verdict.textContent = r.stable ? "안정 파지" : "불안정";
  verdict.className = r.stable ? "ok" : "bad";
  addRow(dl, "판정", verdict);
}

function addRow(dl, label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  if (value instanceof Node) {
    dd.append(value);
  } else {
    dd.textContent = value;
  }
  dl.append(dt, dd);
}

// ---------- canvas ----------

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function bounds(geom, grasp) {
  let zmin = -0.16, zmax = 0.06, ymin = -0.18, ymax = 0.12;
  if (geom) {
    zmin = Math.min(zmin, -geom.palm[1]);
    for (const f of geom.fingers) {
      for (const [, y, z] of f.points) {
        zmax = Math.max(zmax, z + 0.01);
        ymax = Math.max(ymax, y + 0.01);
      }
    }
  }
  if (grasp) {
    const o = grasp.object;
    zmin = Math.min(zmin, o.center[2] - o.half[2]);
    zmax = Math.max(zmax, o.center[2] + o.half[2]);
    ymax = Math.max(ymax, o.center[1] + o.half[1]);
  }
  return { zmin, zmax, ymin, ymax };
}

function draw() {
  const canvas = $("hand-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!state.geometry) return;

  const b = bounds(state.geometry, state.grasp);
  const pad = 36;
  const scale = Math.min((W - 2 * pad) / (b.zmax - b.zmin), (H - 2 * pad) / (b.ymax - b.ymin));
  const X = (z) => pad + (z - b.zmin) * scale;
  const Y = (y) => H - pad - (y - b.ymin) * scale;

  // axes
  ctx.strokeStyle = cssVar("--line");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(X(0), Y(b.ymin));
  ctx.lineTo(X(0), Y(b.ymax));
  ctx.moveTo(X(b.zmin), Y(0));
  ctx.lineTo(X(b.zmax), Y(0));
  ctx.stroke();

  drawBase(ctx, state.geometry.base.kind, X, Y, scale);

  // palm
  const palmLen = state.geometry.palm[1];
  ctx.fillStyle = cssVar("--palm");
  ctx.strokeStyle = cssVar("--muted");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.rect(X(-palmLen), Y(0.012), palmLen * scale, 0.012 * scale);
  ctx.fill();
  ctx.stroke();

  // object
  if (state.grasp) {
    const o = state.grasp.object;
    ctx.fillStyle = "rgba(201,164,106,0.35)";
    ctx.strokeStyle = cssVar("--object");
    ctx.lineWidth = 2;
    const x0 = X(o.center[2] - o.half[2]);
    const y1 = Y(o.center[1] + o.half[1]);
    const w = (2 * o.half[2]) * scale;
    const h = (2 * o.half[1]) * scale;
    ctx.fillRect(x0, y1, w, h);
    ctx.strokeRect(x0, y1, w, h);
  }

  // fingers
  state.geometry.fingers.forEach((f) => {
    const color = f.is_thumb ? cssVar("--thumb") : cssVar("--finger");
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    f.points.forEach(([, y, z], k) => (k === 0 ? ctx.moveTo(X(z), Y(y)) : ctx.lineTo(X(z), Y(y))));
    ctx.stroke();
    ctx.fillStyle = color;
    f.points.forEach(([, y, z], k) => {
      ctx.beginPath();
      ctx.arc(X(z), Y(y), k === f.points.length - 1 ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  // contacts from the grasp test
  if (state.grasp) {
    for (const c of state.grasp.result.contacts) {
      const { y, z } = c.point;
      const palm = c.source.source === "palm";
      ctx.fillStyle = palm ? cssVar("--warn") : cssVar("--ok");
      ctx.beginPath();
      ctx.arc(X(z), Y(y), 5, 0, Math.PI * 2);
      ctx.fill();
      const { y: ny, z: nz } = c.normal;
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(z), Y(y));
      ctx.lineTo(X(z + nz * 0.03), Y(y + ny * 0.03));
      ctx.stroke();
    }
  }

  ctx.fillStyle = cssVar("--muted");
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("손끝 방향 →  z (m)", W - 170, H - 12);
  ctx.fillText("↑ y 손바닥 법선", 12, 18);
}

function drawBase(ctx, kind, X, Y, scale) {
  ctx.strokeStyle = cssVar("--muted");
  ctx.fillStyle = cssVar("--panel");
  ctx.lineWidth = 2;
  const wheel = (z, y, r) => {
    ctx.beginPath();
    ctx.arc(X(z), Y(y), r * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  };
  if (kind === "wheel" || kind === "mecanum") {
    ctx.beginPath();
    ctx.moveTo(X(-0.11), Y(-0.1));
    ctx.lineTo(X(0.0), Y(-0.1));
    ctx.stroke();
    wheel(-0.11, -0.11, 0.03);
    wheel(0.0, -0.11, 0.03);
    if (kind === "mecanum") {
      ctx.beginPath();
      ctx.moveTo(X(-0.125), Y(-0.11));
      ctx.lineTo(X(-0.095), Y(-0.11));
      ctx.stroke();
    }
  } else if (kind === "track") {
    ctx.beginPath();
    ctx.roundRect(X(-0.15), Y(-0.1), 0.16 * scale, 0.05 * scale, 0.025 * scale);
    ctx.fill();
    ctx.stroke();
  } else if (kind === "human_foot") {
    ctx.beginPath();
    ctx.moveTo(X(-0.1), Y(-0.02));
    ctx.lineTo(X(-0.1), Y(-0.12));
    ctx.lineTo(X(0.03), Y(-0.12));
    ctx.lineTo(X(0.03), Y(-0.1));
    ctx.lineTo(X(-0.04), Y(-0.02));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

// ---------- vision ----------

function renderScores(list) {
  const ol = $("scores");
  ol.replaceChildren();
  for (const p of list) {
    const li = document.createElement("li");
    const top = document.createElement("div");
    top.textContent = `${p.label_ko} (${p.label_en}) — ${(p.score * 100).toFixed(1)}%`;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.round(p.score * 100)}%`;
    bar.append(fill);
    li.append(top, bar);
    ol.append(li);
  }
  $("use-top").disabled = list.length === 0;
  state.topPrediction = list[0] ?? null;
}

async function onImageChosen(file) {
  if (!file) return;
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  $("preview").src = dataUrl;
  $("preview-wrap").hidden = false;
  setStatus("SigLIP2 분류 중… (첫 실행은 모델을 불러오므로 시간이 걸립니다)");
  try {
    const list = await call("classify_image", { imageBase64: dataUrl, topK: 5 });
    renderScores(list);
    setStatus(`분류 완료: ${list[0].label_ko} ${(list[0].score * 100).toFixed(1)}%`);
  } catch {
    renderScores([]);
  }
}

function useTopPrediction() {
  const p = state.topPrediction;
  if (!p) return;
  $("label").value = p.id;
  applyEntryDefaults();
  if (p.graspable === false) setStatus(`${p.label_ko}: 파지 대상이 아닌 물체입니다`, true);
  requestGrasp();
}

// ---------- data ----------

async function refreshDataset() {
  const rows = await call("dataset_summary", { storeDir: null });
  const table = $("table-status");
  table.replaceChildren();
  const head = table.insertRow();
  for (const h of ["테이블", "상태", "행 수"]) {
    const th = document.createElement("th");
    th.textContent = h;
    head.append(th);
  }
  const names = {
    samples: "samples (에피소드)",
    contacts: "contacts (접촉점)",
    joint_states: "joint_states (관절)",
    dictionary: "dictionary (사전)",
    point_cloud: "point_cloud",
    vision: "vision (RGB·깊이·마스크)",
    tactile: "tactile (촉각)",
    embeddings: "embeddings (SigLIP2)",
  };
  for (const row of rows) {
    const r = table.insertRow();
    r.insertCell().textContent = names[row.name] ?? row.name;
    r.insertCell().textContent = row.status === "recorded" ? "기록" : "예정";
    const n = r.insertCell();
    n.className = "num";
    n.textContent = row.status === "recorded" ? row.rows.toLocaleString("ko-KR") : "—";
  }
}

async function generateDataset() {
  const count = Number($("count").value);
  const seed = Number($("seed").value);
  const label = $("label").value;
  $("synth").disabled = true;
  setStatus(`${count}개 에피소드 시뮬레이션 후 Lance에 기록 중…`);
  try {
    const s = await call("synthesize", { count, seed, label, storeDir: null });
    setStatus(`기록 완료: ${s.episodes}개 에피소드 중 안정 파지 ${s.stable}개 → ${s.store_dir}`);
    await refreshDataset();
  } catch {
    /* status already shows the error */
  } finally {
    $("synth").disabled = false;
  }
}

async function loadDictionary() {
  try {
    const n = await call("load_dictionary", { storeDir: null });
    setStatus(`사전 ${n}개 항목을 Lance dictionary 테이블에 기록했습니다`);
    await refreshDataset();
  } catch {
    /* shown in status */
  }
}

// ---------- devices ----------

async function refreshDevices() {
  const d = await call("devices");
  const sel = $("device-select");
  sel.replaceChildren();
  for (const b of d.backends) {
    const o = document.createElement("option");
    o.value = `${b.backend}:0`;
    o.disabled = !b.available;
    o.title = b.detail;
    o.textContent = `${b.backend.toUpperCase()}${b.available ? "" : " (사용 불가)"}`;
    sel.append(o);
  }
  sel.value = `${d.selected}:${d.ordinal}`;
  $("device-chip").textContent = `연산: ${d.selected.toUpperCase()} #${d.ordinal}`;
  $("device-chip").title = d.backends.map((b) => `${b.backend}: ${b.detail}`).join("\n");
}

async function onDeviceChanged() {
  const [backend, ordinal] = $("device-select").value.split(":");
  try {
    await call("select_device", { backend, ordinal: Number(ordinal) });
    await refreshDevices();
    setStatus(`연산 장치를 ${backend.toUpperCase()}로 바꿨습니다. 다음 분류에서 모델을 다시 불러옵니다`);
  } catch {
    await refreshDevices();
  }
}

// ---------- start ----------

function bind() {
  $("segments").addEventListener("change", requestRefresh);
  $("base-kind").addEventListener("change", requestRefresh);
  $("distal-kind").addEventListener("change", requestRefresh);
  $("actuation").addEventListener("change", requestRefresh);
  $("ratio").addEventListener("input", requestRefresh);
  $("closure").addEventListener("input", () => {
    updateReadouts();
    refreshGeometry();
  });
  $("show-grasp").addEventListener("change", () => refreshGrasp());
  $("label").addEventListener("change", () => {
    applyEntryDefaults();
    requestGrasp();
  });
  for (const id of ["mass", "friction", "zoff"]) {
    $(id).addEventListener("input", () => {
      updateReadouts();
      requestGrasp();
    });
  }
  $("image-file").addEventListener("change", (e) => onImageChosen(e.target.files[0]));
  $("use-top").addEventListener("click", useTopPrediction);
  $("synth").addEventListener("click", generateDataset);
  $("load-dict").addEventListener("click", loadDictionary);
  $("device-select").addEventListener("change", onDeviceChanged);
}

async function init() {
  bind();
  try {
    state.presets = await call("presets");
    state.dict = await call("dictionary");
    state.paths = await call("default_paths");
    renderFingerChoices();
    renderJointSelects();
    renderBaseChoices();
    applyJointControls();
    populateDictionary();
    $("store-note").textContent = `Lance 저장 위치: ${state.paths.store_dir}`;
    await refreshDevices();
    await refreshDrive();
    await refreshDataset();
    await refreshAll();
    setStatus("준비됨. 손가락 수와 관절 옵션을 바꾸면 손 모양이 즉시 갱신됩니다");
  } catch {
    /* status already shows the error */
  }
}

document.addEventListener("DOMContentLoaded", init);
