//! Robot presets: the built-in bodies and those made by the AI edits (`mcp`). A made robot arrives as a URDF and its
//! meshes, from a CAD package run out of sight in the web view or from the runtime's own export; it is measured into a
//! body here (`noto_core::rig`) and kept in the runtime folder, so a saved robot is a preset like the others: it can be
//! chosen, simulated and edited again.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use noto_core::act::{fork, grasp, material::ItemMaterial};
use noto_core::balance::Body;
use noto_core::dims::Build;
use noto_core::kin::{self, KinLoads, KinPose};
use noto_core::rig::{self, urdf, RigForm, RigSpec};
use noto_core::{BaseKind, BodyPlan, FingerCount, NotoConfig, Surface};
use serde_json::{json, Value};

use crate::runtime::Runtime;

/// A robot saved from the editor.
#[derive(Clone, Debug)]
pub struct SavedRig {
    pub id: String,
    pub name: String,
    pub rig_id: u32,
    pub form: RigForm,
    pub dir: Option<PathBuf>,
}

#[derive(Default)]
pub struct RigShelf {
    pub saved: Vec<SavedRig>,
    pub dir: Option<PathBuf>,
    /// What each unit's next body would be, between a prompt and its commit.
    pub pending: HashMap<u32, crate::mcp::PendingRig>,
}

/// The base models.
const BUILT_IN: [(&str, &str, &str, Build, BaseKind); 6] = [
    ("humanoid", "휴머노이드 (1.65 m · 표준)", "사람 키 · 문과 선반에 맞는 크기", Build::Standard, BaseKind::HumanFoot),
    ("forklift", "지게차 (1톤 · 바퀴)", "실내 전동 지게차 크기", Build::Standard, BaseKind::Wheel),
    ("forklift_mecanum", "지게차 (1톤 · 메카넘)", "옆으로도 움직이는 바퀴", Build::Standard, BaseKind::Mecanum),
    ("forklift_track", "지게차 (1톤 · 트랙)", "험지용 궤도", Build::Standard, BaseKind::Track),
    ("compact_humanoid", "소형 휴머노이드 (1.21 m)", "기존 벤치마크의 몸", Build::Compact, BaseKind::HumanFoot),
    ("compact_truck", "소형 지게차 (축거 0.30 m)", "기존 벤치마크의 몸", Build::Compact, BaseKind::Wheel),
];

fn base_id(b: BaseKind) -> Value {
    serde_json::to_value(b).unwrap_or(Value::Null)
}

fn r3(x: f32) -> f64 {
    ((x as f64) * 1000.0).round() / 1000.0
}

fn slug(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).take(40).collect();
    if s.trim_matches('_').is_empty() {
        "robot".into()
    } else {
        s
    }
}

/// The key a made robot's statistics go under: its id and a hash of its design, so a changed design starts afresh.
pub(crate) fn sds_key(name: &str, design: &str) -> String {
    let hash = design.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3));
    format!("scraft:{}:{:08x}", slug(name), hash as u32)
}

pub(crate) fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len() / 3 * 4 + 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

fn body_for(build: Build, base: BaseKind) -> Body {
    Body::new(&BodyPlan { build, morph: noto_core::Morphology::humanoid(base), ..BodyPlan::standard() }, base, FingerCount::Five)
}

pub(crate) fn body_of(build: Build, base: BaseKind) -> Body {
    body_for(build, base)
}

/// What the world asks of a body before it is sent out: does it stand, does it fit the doors, can it take a log.
pub fn checks(body: &Body) -> Vec<Value> {
    let mut v = Vec::new();
    let mut add = |label: &str, ok: bool, detail: String| v.push(json!({"label": label, "ok": ok, "detail": detail}));
    let stand = kin::solve(body, &KinPose::stand(body, 0.0), &KinLoads::default());
    add("기립 균형", stand.margin > 0.1 && stand.feasible, format!("균형 여유 {:.2} · 무게중심 {:.2} m", stand.margin, stand.com[2]));
    let eye = kin::head_camera(body, &KinPose::stand(body, 0.0)).pos[2];
    add("카메라 높이", (0.4..2.2).contains(&eye), format!("눈 {eye:.2} m"));
    let log = world_sim::ItemKind::Log.spec();
    if fork::applies(body) {
        let roof = body.dims.roof_m.unwrap_or(1.3);
        add("창고 문 통과", body.support_width_m < 2.5 && roof < 3.0, format!("폭 {:.2} m · 높이 {roof:.2} m (문 2.50 × 3.00 m)", body.support_width_m));
        match fork::check(body, log.mass_kg, log.size_m, 0.0, 0.0) {
            Ok(f) => add("포크로 원목 들기", true, format!("정격 {:.0} kg · 전도 여유 {:.1}배", f.capacity_kg, f.tip_ratio)),
            Err(e) => add("포크로 원목 들기", false, e),
        }
        let p = fork::pick(body, body.dims.fork_rest_m, fork::standoff_x(body, log.size_m[0]), log.mass_kg, log.size_m, 0.0, 0.0, 0.0);
        add("포크 동작", p.outcome.ok, p.outcome.reason_ko.unwrap_or_else(|| format!("{:.1} s", p.duration_s)));
    } else {
        let top = stand.boxes.iter().map(|b| b.center[2] + b.half[2]).fold(0.0, f32::max);
        let wide = 2.0 * (body.dims.shoulder_half_m + 0.06 * body.dims.k);
        add("집 문 통과", top < 2.03 && wide < 0.914, format!("키 {top:.2} m · 어깨 폭 {wide:.2} m (문 0.91 × 2.03 m)"));
        let limits = body.drive_limits(&body.base.spec(), 0.0);
        add("보행", limits.max_speed_mps > 0.3, format!("최고 {:.2} m/s", limits.max_speed_mps));
        add("원목 운반", limits.capacity_kg >= log.mass_kg, format!("상시 적재 {:.1} kg · 원목 {:.1} kg", limits.capacity_kg, log.mass_kg));
        match NotoConfig::preset(FingerCount::Five).build() {
            Ok(h) => {
                let hand = h.scaled(body.plan.k());
                let r = grasp::grasp(&grasp::GraspInput::front(body, &hand, ItemMaterial::Wood, log.mass_kg, log.size_m, Surface::Firm, true, 0.0, 0.01, 3));
                add("바닥의 원목 집기", r.outcome.ok, r.outcome.reason_ko.unwrap_or_else(|| format!("{:.1} s", r.duration_s)));
            }
            Err(e) => add("바닥의 원목 집기", false, e.to_string()),
        }
    }
    v
}

fn summary(body: &Body, rig: Option<&RigSpec>) -> Value {
    let stand = kin::solve(body, &KinPose::stand(body, 0.0), &KinLoads::default());
    let eye = kin::head_camera(body, &KinPose::stand(body, 0.0)).pos[2];
    json!({
        "mass_kg": r3(stand.mass_kg), "com_m": r3(stand.com[2]), "eye_m": r3(eye), "margin": r3(stand.margin),
        "joints": stand.joints.len(), "dof": rig.map(RigSpec::dof),
        "width_m": r3(body.support_width_m), "length_m": r3(body.support_len_m),
        "weakest": stand.worst().map(|j| json!({"name": j.name, "ratio": r3(j.ratio), "actuator": j.actuator})),
        "warnings": rig.map_or(Vec::new(), |r| r.warnings.clone()),
        "checks": checks(body),
    })
}

fn load_folder(dir: &Path) -> Option<(String, String, HashMap<String, Vec<u8>>)> {
    let urdf_text = std::fs::read_to_string(dir.join("robot.urdf")).ok()?;
    let name = std::fs::read_to_string(dir.join("name.txt")).ok()?;
    let mut files = HashMap::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "stl") {
            files.insert(p.file_name()?.to_string_lossy().to_string(), std::fs::read(&p).ok()?);
        }
    }
    Some((name.trim().to_string(), urdf_text, files))
}

impl RigShelf {
    /// Loads the robots saved in `dir` (the runtime folder's `rigs`).
    pub fn open(dir: Option<PathBuf>) -> RigShelf {
        let mut shelf = RigShelf { saved: Vec::new(), dir, pending: HashMap::new() };
        let Some(root) = shelf.dir.clone() else { return shelf };
        let Ok(entries) = std::fs::read_dir(&root) else { return shelf };
        for e in entries.flatten() {
            let path = e.path();
            let Some((name, text, files)) = load_folder(&path) else { continue };
            let Ok(u) = urdf::parse(&text) else { continue };
            let Ok(mut spec) = RigSpec::from_urdf(&u, &name, &files) else { continue };
            spec.sds_key = std::fs::read_to_string(path.join("sds_key.txt")).ok().map(|k| k.trim().to_string());
            let form = spec.form;
            let id = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            shelf.saved.push(SavedRig { id, name, rig_id: rig::register(spec), form, dir: Some(path) });
        }
        shelf.saved.sort_by(|a, b| a.name.cmp(&b.name));
        shelf
    }
}

impl Runtime {
    pub(crate) fn rig_preset(&self, id: &str) -> Result<(Build, BaseKind), String> {
        if let Some(p) = BUILT_IN.iter().find(|p| p.0 == id) {
            return Ok((p.3, p.4));
        }
        let s = self.rigs.saved.iter().find(|r| r.id == id).ok_or_else(|| format!("로봇 프리셋 '{id}'가 없습니다"))?;
        Ok((Build::Rig(s.rig_id), s.form.base()))
    }

    /// The preset a body is, by id and name.
    pub(crate) fn rig_name_of(&self, build: Build, base: BaseKind) -> (String, String) {
        if let Some(p) = BUILT_IN.iter().find(|p| p.3 == build && p.4 == base) {
            return (p.0.to_string(), p.1.to_string());
        }
        self.rigs.saved.iter().find(|r| Build::Rig(r.rig_id) == build).map_or((String::new(), String::new()), |r| (r.id.clone(), r.name.clone()))
    }

    fn rig_list(&self) -> Value {
        let built: Vec<Value> = BUILT_IN.iter().map(|(id, name, note, build, base)| json!({"id": id, "name_ko": name, "note_ko": note, "build": build, "base": base_id(*base), "builtin": true})).collect();
        let saved: Vec<Value> = self.rigs.saved.iter().map(|r| json!({"id": r.id, "name_ko": r.name, "note_ko": format!("{} · AI 수정", r.form.label_ko()), "build": Build::Rig(r.rig_id), "base": base_id(r.form.base()), "builtin": false})).collect();
        let current = BUILT_IN.iter().find(|p| p.3 == self.cfg.body.build && p.4 == self.cfg.noto.base).map(|p| p.0.to_string()).or_else(|| self.rigs.saved.iter().find(|r| Build::Rig(r.rig_id) == self.cfg.body.build).map(|r| r.id.clone()));
        json!({"presets": built.into_iter().chain(saved).collect::<Vec<_>>(), "current": current})
    }

    fn rig_import(&mut self, args: &Value) -> Result<Value, String> {
        let name = args.get("name").and_then(Value::as_str).filter(|n| !n.trim().is_empty()).unwrap_or("새 로봇").trim().to_string();
        self.rig_import_as(&name, args, None)
    }

    /// Measures a URDF and its meshes into a preset called `name`, keeping its statistics under `sds_key`.
    pub(crate) fn rig_import_as(&mut self, name: &str, args: &Value, sds_key: Option<String>) -> Result<Value, String> {
        let name = name.to_string();
        let text = args.get("urdf").and_then(Value::as_str).ok_or("urdf 가 필요합니다")?;
        let mut files: HashMap<String, Vec<u8>> = HashMap::new();
        if let Some(m) = args.get("meshes").and_then(Value::as_object) {
            for (k, v) in m {
                let b = v.as_str().ok_or("메시는 base64 문자열이어야 합니다")?;
                files.insert(k.rsplit('/').next().unwrap_or(k).to_string(), urdf::base64_decode(b)?);
            }
        }
        let u = urdf::parse(text)?;
        let mut spec = RigSpec::from_urdf(&u, &name, &files)?;
        spec.sds_key = sds_key.clone();
        let form = spec.form;
        let warnings = spec.warnings.clone();
        // Saving under a name that exists replaces that robot.
        let id = slug(&name);
        if let Some(i) = self.rigs.saved.iter().position(|r| r.id == id) {
            let old = self.rigs.saved.remove(i);
            let worn = Build::Rig(old.rig_id) == self.cfg.body.build || self.world.crew.iter().any(|w| w.body.plan.build == Build::Rig(old.rig_id));
            if !worn {
                rig::remove(old.rig_id);
            }
        }
        let rig_id = rig::register(spec);
        let dir = self.rigs.dir.as_ref().map(|d| d.join(&id));
        if let Some(d) = &dir {
            let _ = std::fs::remove_dir_all(d);
            std::fs::create_dir_all(d).map_err(|e| format!("로봇 폴더를 만들 수 없습니다: {e}"))?;
            std::fs::write(d.join("robot.urdf"), text).map_err(|e| e.to_string())?;
            std::fs::write(d.join("name.txt"), &name).map_err(|e| e.to_string())?;
            if let Some(k) = &sds_key {
                std::fs::write(d.join("sds_key.txt"), k).map_err(|e| e.to_string())?;
            }
            for (f, b) in &files {
                std::fs::write(d.join(f), b).map_err(|e| e.to_string())?;
            }
        }
        self.rigs.saved.push(SavedRig { id: id.clone(), name: name.clone(), rig_id, form, dir });
        self.rigs.saved.sort_by(|a, b| a.name.cmp(&b.name));
        let body = body_for(Build::Rig(rig_id), form.base());
        let spec = rig::get(rig_id);
        let mut s = summary(&body, spec.as_deref());
        s["id"] = json!(id);
        s["name_ko"] = json!(name);
        s["form"] = json!(form);
        s["form_ko"] = json!(form.label_ko());
        s["base"] = base_id(form.base());
        s["build"] = json!(Build::Rig(rig_id));
        s["warnings"] = json!(warnings);
        Ok(s)
    }

    pub(crate) fn rig_call(&mut self, cmd: &str, args: &Value) -> Option<Result<Value, String>> {
        let id = args.get("id").and_then(Value::as_str).unwrap_or("");
        Some(match cmd {
            "rig_presets" => Ok(self.rig_list()),
            "rig_design" => (|| {
                let (build, base) = self.rig_preset(id)?;
                let body = body_for(build, base);
                let saved = self.rigs.saved.iter().find(|r| r.id == id).and_then(|r| r.dir.clone()).and_then(|d| std::fs::read(d.join("design.sc")).ok());
                Ok(json!({"design": rig::design_of(&body), "sc": saved.map(|b| base64(&b))}))
            })(),
            "rig_validate" => (|| {
                let (build, base) = self.rig_preset(id)?;
                let body = body_for(build, base);
                let spec = body.rig();
                Ok(summary(&body, spec.as_deref()))
            })(),
            "rig_import" => self.rig_import(args),
            "rig_keep_design" => (|| {
                let r = self.rigs.saved.iter().find(|r| r.id == id).ok_or_else(|| format!("저장된 로봇 '{id}'가 없습니다"))?;
                let sc = args.get("design").and_then(Value::as_str).ok_or("design 이 필요합니다")?;
                if let Some(d) = &r.dir {
                    std::fs::write(d.join("design.sc"), urdf::base64_decode(sc)?).map_err(|e| e.to_string())?;
                }
                Ok(json!({"id": id}))
            })(),
            "rig_apply" => (|| {
                let (build, base) = self.rig_preset(id)?;
                let plan = BodyPlan { build, morph: noto_core::Morphology::humanoid(base), ..self.cfg.body };
                self.cfg.body = plan;
                let mut hand = self.cfg.noto;
                hand.base = base;
                self.reset_with(None, None, Some(hand)).map_err(|e| e.to_string())?;
                Ok(json!({"plan": plan, "base": base_id(base), "presets": self.rig_list()}))
            })(),
            "rig_delete" => (|| {
                let i = self.rigs.saved.iter().position(|r| r.id == id).ok_or_else(|| format!("저장된 로봇 '{id}'가 없습니다"))?;
                if Build::Rig(self.rigs.saved[i].rig_id) == self.cfg.body.build {
                    return Err("지금 월드에서 쓰는 로봇은 지울 수 없습니다".to_string());
                }
                let r = self.rigs.saved.remove(i);
                rig::remove(r.rig_id);
                if let Some(d) = r.dir {
                    let _ = std::fs::remove_dir_all(d);
                }
                Ok(self.rig_list())
            })(),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_goes_there_and_back() {
        for s in [&b""[..], b"a", b"ab", b"abc", b"abcd", &[0u8, 255, 16, 200, 7]] {
            assert_eq!(urdf::base64_decode(&base64(s)).unwrap(), s);
        }
    }

    #[test]
    fn the_standard_bodies_pass_their_checks() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Mecanum, BaseKind::Track] {
            let body = body_for(Build::Standard, base);
            for c in checks(&body) {
                println!("{base:?} {c}");
                assert_eq!(c["ok"], true, "{base:?}: {c}");
            }
        }
    }
}
