use serde::Serialize;

use crate::balance::{Body, G};
use crate::kin::{self, KinLoads, KinPose, KinReport, Rom, ANKLE_H_M, PALM_OFFSET_M, SHANK_M, THIGH_M, WHEEL_R_M, BUMPER_GAP_M};
use crate::morph::{HeadMount, Torso};
use crate::parts::*;

use super::MIN_JERK_PEAK;

pub const PALM_CLEARANCE_M: f32 = 0.02;
pub const BEND_MIN_S: f32 = 1.0;
pub const STANDOFF_GAP_M: f32 = 0.02;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ReachGoal {
    pub palm_z: f32,
    pub half_len_m: f32,
    pub load_kg: f32,
    pub two_arms: bool,
    pub cargo_kg: f32,
}

impl ReachGoal {
    pub fn ground_object(height_m: f32, load_kg: f32, two_arms: bool, cargo_kg: f32) -> ReachGoal {
        ReachGoal { palm_z: height_m + PALM_CLEARANCE_M, half_len_m: 0.5 * height_m, load_kg, two_arms, cargo_kg }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReachMode {
    #[default]
    Front,
    Side,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReachPlan {
    pub ok: bool,
    pub strategy: &'static str,
    pub strategy_ko: &'static str,
    pub approach: &'static str,
    pub pose: KinPose,
    pub empty: KinReport,
    pub loaded: KinReport,
    pub object_x: f32,
    pub object_y: f32,
    pub lowest_palm_z: f32,
    pub reason_ko: Option<String>,
    pub descend_s: f32,
    pub rise_s: f32,
    pub rise_margin: Option<f32>,
    pub evaluated: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}

fn refused(body: &Body, goal: &ReachGoal, approach: &'static str, code: &'static str, why: &str) -> ReachPlan {
    let stand = body.stand(goal.cargo_kg);
    let empty = kin::solve(body, &stand, &KinLoads::default());
    ReachPlan { ok: false, strategy: "none", strategy_ko: "불가", approach, pose: stand, loaded: empty.clone(), empty, object_x: min_object_x(body, goal.half_len_m), object_y: 0.0, lowest_palm_z: f32::NAN, reason_ko: Some(why.to_string()), descend_s: 0.0, rise_s: 0.0, rise_margin: None, evaluated: 0, code: Some(code) }
}

fn form_refusal(body: &Body, goal: &ReachGoal, approach: &'static str) -> Option<ReachPlan> {
    if !body.has_manipulator() {
        return Some(refused(body, goal, approach, "no_manipulator", "조작기(팔·집게)가 없어 물건을 집을 수 없음"));
    }
    if goal.two_arms && body.arm_count() < 2 {
        return Some(refused(body, goal, approach, "one_arm", "팔이 하나뿐이라 두 손으로 들 수 없음"));
    }
    None
}

pub fn arm_ik(shoulder: [f32; 2], target: [f32; 2], trunk_deg: f32) -> Option<(f32, f32)> {
    let (l1, l2) = (UPPER_ARM_M, FOREARM_M + PALM_OFFSET_M);
    let (dx, dz) = (target[0] - shoulder[0], target[1] - shoulder[1]);
    let d = (dx * dx + dz * dz).sqrt();
    if d > l1 + l2 - 1e-4 || d < (l1 - l2).abs() + 1e-4 {
        return None;
    }
    let elbow = std::f32::consts::PI - ((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)).clamp(-1.0, 1.0).acos();
    let toward = dx.atan2(-dz);
    let beta = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0).acos();
    let upper = toward - beta;
    Some((upper.to_degrees() + trunk_deg, elbow.to_degrees()))
}

pub fn leg_angles(pose: &KinPose) -> (f32, f32) {
    let knee = kin::knee_ik(pose.hip, [0.0, ANKLE_H_M]).map(|k| k.0).unwrap_or([0.0, ANKLE_H_M + SHANK_M]);
    (knee[0].atan2(knee[1] - ANKLE_H_M).to_degrees(), (knee[0] - pose.hip[0]).atan2(pose.hip[1] - knee[1]).to_degrees())
}

pub fn hip_at(shank_deg: f32, thigh_deg: f32) -> [f32; 2] {
    let (s, t) = (shank_deg.to_radians(), thigh_deg.to_radians());
    let knee = [SHANK_M * s.sin(), ANKLE_H_M + SHANK_M * s.cos()];
    [knee[0] - THIGH_M * t.sin(), knee[1] + THIGH_M * t.cos()]
}

pub fn blend(body: &Body, a: &KinPose, b: &KinPose, u: f32) -> KinPose {
    let mut p = KinPose::lerp(a, b, u);
    let planted = |q: &KinPose| q.feet.iter().all(|f| f[0].abs() < 1e-4 && f[1].abs() < 1e-4);
    if body.legged() && planted(a) && planted(b) {
        let ((sa, ta), (sb, tb)) = (leg_angles(a), leg_angles(b));
        p.hip = hip_at(sa + (sb - sa) * u, ta + (tb - ta) * u);
    }
    p
}

pub fn shoulder_point(body: &Body, pose: &KinPose) -> [f32; 2] {
    match body.morph.torso {
        Torso::Humanoid => {}
        Torso::Mast => return [0.0, SHOULDER_M - pose.lift_m],
        Torso::None => {
            let s = body.shoulder_at(0, pose.lift_m);
            return [s[0], s[2]];
        }
    }
    let hip = if body.legged() { pose.hip } else { [0.0, HIP_M - pose.lift_m] };
    let p = pose.pelvis_deg.to_radians();
    let t = pose.trunk_deg().to_radians();
    let waist = [hip[0] + kin::PELVIS_M * p.sin(), hip[1] + kin::PELVIS_M * p.cos()];
    [waist[0] + kin::TRUNK_M * t.sin(), waist[1] + kin::TRUNK_M * t.cos()]
}

pub fn min_object_x(body: &Body, half_len_m: f32) -> f32 {
    let front = if body.legged() { 0.75 * body.leg.foot_len_m } else { 0.5 * body.support_len_m + WHEEL_R_M + BUMPER_GAP_M };
    front + half_len_m + STANDOFF_GAP_M
}

struct Cand {
    pose: KinPose,
    strategy: &'static str,
    loaded: KinReport,
    score: f32,
    rise_margin: Option<f32>,
    rise_s: Option<f32>,
}

fn strategy_of(body: &Body, pose: &KinPose) -> &'static str {
    let squat = body.legged() && pose.hip[1] < HIP_M - 0.05;
    let hinge = body.legged() && pose.pelvis_deg > 10.0;
    let stoop = pose.waist_deg > 10.0;
    if !body.legged() && pose.lift_m > 0.05 {
        return if stoop { "lower_stoop" } else { "lower" };
    }
    match (squat, hinge, stoop) {
        (false, false, false) => "stand",
        (false, false, true) => "stoop",
        (false, true, _) => "hinge",
        (true, false, false) => "squat",
        (true, true, _) => "squat_hinge",
        (true, false, true) => "squat_stoop",
    }
}

pub fn strategy_ko(s: &str) -> &'static str {
    match s {
        "stand" => "선 채로",
        "stoop" => "허리 굽힘",
        "hinge" => "고관절 굽힘",
        "squat" => "쪼그려 앉기",
        "squat_hinge" => "쪼그려 앉아 고관절 굽힘",
        "squat_stoop" => "쪼그려 앉아 허리 굽힘",
        "lower" => "몸통을 내림",
        "lower_stoop" => "몸통을 내리고 허리 굽힘",
        _ => "—",
    }
}

struct Miss {
    violation: f32,
    reason: String,
}

fn candidate(body: &Body, goal: &ReachGoal, x_obj: f32, shank_deg: f32, thigh_deg: f32, pelvis_deg: f32, waist_deg: f32, lift_m: f32) -> Option<Cand> {
    candidate_or_miss(body, goal, x_obj, shank_deg, thigh_deg, pelvis_deg, waist_deg, lift_m).ok()
}

fn violation(r: &KinReport) -> f32 {
    r.worst().map_or(0.0, |j| (j.ratio - 1.0).max(0.0)) + (-r.margin).max(0.0) + 0.5 * r.out_of_range.len() as f32
}

fn look_from(body: &Body, sh: [f32; 2]) -> Option<([f32; 2], f32)> {
    match body.camera_mount() {
        HeadMount::Neck => Some(([sh[0], sh[1] + kin::NECK_UP_M + 0.07], kin::NECK_PITCH_MAX_DEG)),
        HeadMount::MastCamera { height_mm, pan_tilt: true } => Some(([body.mast_camera_x() + kin::CAMERA_FWD_M, height_mm as f32 / 1000.0], body.head_range_deg().0)),
        HeadMount::Fixed { tilt_deg } => Some(([sh[0], sh[1] + kin::NECK_UP_M + 0.07], tilt_deg.abs() as f32)),
        _ => None,
    }
}

pub fn look_down_pose(body: &Body, dist_m: f32, target_z_m: f32, cargo_kg: f32) -> KinPose {
    let mut pose = body.stand(cargo_kg);
    let rom = Rom::of(body);
    let sh0 = shoulder_point(body, &pose);
    let eye_z = sh0[1] + kin::NECK_UP_M + 0.07;
    let needed_deg = (eye_z - target_z_m).max(0.05).atan2(dist_m.max(0.15)).to_degrees();
    let neck_max = body.head_range_deg().0;
    if matches!(body.camera_mount(), HeadMount::Neck | HeadMount::MastCamera { pan_tilt: true, .. }) {
        pose.neck_deg = needed_deg.clamp(-neck_max, neck_max);
    }
    let remainder = (needed_deg - pose.neck_deg).max(0.0);
    if remainder > 1.0 {
        if rom.waist_max > 0.0 {
            pose.waist_deg = remainder.min(rom.waist_max);
        } else if body.legged() && rom.hip_flex_max > 0.0 {
            pose.pelvis_deg = remainder.min(rom.hip_flex_max * 0.5);
        }
    }
    pose
}

fn candidate_or_miss(body: &Body, goal: &ReachGoal, x_obj: f32, shank_deg: f32, thigh_deg: f32, pelvis_deg: f32, waist_deg: f32, lift_m: f32) -> Result<Cand, Option<Miss>> {
    let mut pose = body.stand(goal.cargo_kg);
    if body.legged() {
        let (s, t) = (shank_deg.to_radians(), thigh_deg.to_radians());
        let knee = [SHANK_M * s.sin(), ANKLE_H_M + SHANK_M * s.cos()];
        pose.hip = [knee[0] - THIGH_M * t.sin(), knee[1] + THIGH_M * t.cos()];
        pose.pelvis_deg = pelvis_deg;
    } else {
        pose.lift_m = lift_m;
    }
    pose.waist_deg = waist_deg;
    let sh = shoulder_point(body, &pose);
    let (sd, ed) = arm_ik(sh, [x_obj, goal.palm_z], pose.trunk_deg()).ok_or(None)?;
    let rest = body.stand(goal.cargo_kg);
    pose.shoulder_deg = [sd, if goal.two_arms { sd } else { rest.shoulder_deg[1] }];
    pose.elbow_deg = [ed, if goal.two_arms { ed } else { rest.elbow_deg[1] }];
    if let Some((eye, max)) = look_from(body, sh) {
        let look = (eye[1] - goal.palm_z).atan2(x_obj - eye[0]).to_degrees();
        pose.neck_deg = (look - pose.trunk_deg()).clamp(-max, max);
    }
    pose.closure = 0.0;
    let loaded = kin::solve_fast(body, &pose, &KinLoads::holding(goal.load_kg, goal.two_arms, 0.0));
    if !loaded.feasible {
        return Err(Some(Miss { violation: violation(&loaded), reason: loaded.limiting.clone().unwrap_or_default() }));
    }
    let empty = kin::solve_fast(body, &pose, &KinLoads::default());
    if !empty.feasible {
        return Err(Some(Miss { violation: violation(&empty), reason: empty.limiting.clone().unwrap_or_default() }));
    }
    let score = loaded.effort_sum + (1.0 - loaded.margin.clamp(0.0, 1.0)).powi(2);
    Ok(Cand { strategy: strategy_of(body, &pose), pose, loaded, score, rise_margin: None, rise_s: None })
}

fn lowest_palm(body: &Body, rom: &Rom, x_obj: f32) -> f32 {
    let mut best = f32::INFINITY;
    let steps = |max: f32, n: usize| (0..=n).map(move |i| max * i as f32 / n as f32);
    let shank_max = if body.legged() { rom.ankle_dorsi_max } else { 0.0 };
    let thigh_max = if body.legged() { (rom.knee_max - shank_max).max(0.0).min(110.0) } else { 0.0 };
    let pelvis_max = if body.legged() { rom.hip_flex_max } else { 0.0 };
    for s in steps(shank_max, 3) {
        for t in steps(thigh_max, 11) {
            for p in steps(pelvis_max, 12) {
                for w in steps(rom.waist_max, 9) {
                    let mut pose = KinPose::stand(0.0);
                    if body.legged() {
                        let (sr, tr) = (s.to_radians(), t.to_radians());
                        let knee = [SHANK_M * sr.sin(), ANKLE_H_M + SHANK_M * sr.cos()];
                        pose.hip = [knee[0] - THIGH_M * tr.sin(), knee[1] + THIGH_M * tr.cos()];
                        pose.pelvis_deg = p;
                    } else {
                        pose.lift_m = rom.lift_max_m;
                    }
                    pose.waist_deg = w;
                    let sh = shoulder_point(body, &pose);
                    let reach = UPPER_ARM_M + FOREARM_M + PALM_OFFSET_M;
                    let dx = x_obj - sh[0];
                    if dx.abs() <= reach {
                        best = best.min(sh[1] - (reach * reach - dx * dx).sqrt());
                    }
                }
            }
        }
    }
    best
}

fn highest_palm(body: &Body, x_obj: f32) -> Option<f32> {
    let sh = shoulder_point(body, &body.stand(0.0));
    let reach = UPPER_ARM_M + FOREARM_M + PALM_OFFSET_M;
    let dx = x_obj - sh[0];
    (dx.abs() <= reach).then(|| sh[1] + (reach * reach - dx * dx).sqrt())
}

pub fn plan(body: &Body, goal: &ReachGoal) -> ReachPlan {
    if let Some(no) = form_refusal(body, goal, "front") {
        return no;
    }
    let rom = Rom::of(body);
    let x0 = min_object_x(body, goal.half_len_m);
    let xs = [x0, x0 + 0.06, x0 + 0.12, x0 + 0.20];
    let mut evaluated = 0u32;
    let mut best: Option<(Cand, f32)> = None;
    let mut nearest: Option<Miss> = None;
    let consider = |c: Option<Cand>, x: f32, best: &mut Option<(Cand, f32)>| {
        if let Some(c) = c {
            if best.as_ref().map_or(true, |(b, _)| c.score < b.score) {
                *best = Some((c, x));
            }
        }
    };
    let grid = |max: f32, n: usize| -> Vec<f32> { (0..=n).map(|i| max * i as f32 / n.max(1) as f32).collect() };
    let shank_max = if body.legged() { rom.ankle_dorsi_max } else { 0.0 };
    let thigh_max = if body.legged() { (rom.knee_max - shank_max).max(0.0).min(110.0) } else { 0.0 };
    let pelvis_max = if body.legged() { rom.hip_flex_max } else { 0.0 };
    let (ss, ts, ps, ws) = (grid(shank_max, if body.legged() { 3 } else { 0 }), grid(thigh_max, if body.legged() { 11 } else { 0 }), grid(pelvis_max, if body.legged() { 12 } else { 0 }), grid(rom.waist_max, if rom.waist_max > 0.0 { 9 } else { 0 }));
    let ls = grid(rom.lift_max_m, if rom.lift_max_m > 0.0 { 8 } else { 0 });
    for &x in &xs {
        for &s in &ss {
            for &t in &ts {
                for &p in &ps {
                    for &w in &ws {
                        for &lift in &ls {
                            evaluated += 1;
                            match candidate_or_miss(body, goal, x, s, t, p, w, lift) {
                                Ok(c) => consider(Some(c), x, &mut best),
                                Err(Some(m)) => {
                                    if nearest.as_ref().map_or(true, |n| m.violation < n.violation) {
                                        nearest = Some(m);
                                    }
                                }
                                Err(None) => {}
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some((b, x)) = best.as_ref().map(|(c, x)| (c.pose, *x)) {
        let (s0, t0) = if body.legged() {
            let knee = kin::knee_ik(b.hip, [0.0, ANKLE_H_M]).map(|k| k.0).unwrap_or([0.0, ANKLE_H_M + SHANK_M]);
            ((knee[0]).atan2(knee[1] - ANKLE_H_M).to_degrees(), (knee[0] - b.hip[0]).atan2(b.hip[1] - knee[1]).to_degrees())
        } else {
            (0.0, 0.0)
        };
        let d = |max: f32, n: usize| if n == 0 { 0.0 } else { 0.5 * max / n as f32 };
        let (ds, dt, dp, dw, dl) = (d(shank_max, 3), d(thigh_max, 11), d(pelvis_max, 12), d(rom.waist_max, 9), d(rom.lift_max_m, 8));
        let lift_steps: &[f32] = if rom.lift_max_m > 0.0 { &[-1.0, 0.0, 1.0] } else { &[0.0] };
        for k in [-1.0f32, 0.0, 1.0] {
            for l in [-1.0f32, 0.0, 1.0] {
                for m in [-1.0f32, 0.0, 1.0] {
                    for n in [-1.0f32, 0.0, 1.0] {
                        for &q in lift_steps {
                            for dx in [-0.03f32, 0.0, 0.03] {
                                let xx = (x + dx).max(x0);
                                let c = candidate(body, goal, xx, (s0 + k * ds).clamp(0.0, shank_max), (t0 + l * dt).clamp(0.0, thigh_max), (b.pelvis_deg + m * dp).clamp(0.0, pelvis_max), (b.waist_deg + n * dw).clamp(0.0, rom.waist_max), (b.lift_m + q * dl).clamp(0.0, rom.lift_max_m));
                                evaluated += 1;
                                consider(c, xx, &mut best);
                            }
                        }
                    }
                }
            }
        }
    }
    let stand = body.stand(goal.cargo_kg);
    match best {
        Some((c, x)) => {
            let empty = kin::solve(body, &c.pose, &KinLoads::default());
            let loaded = kin::solve(body, &c.pose, &KinLoads::holding(goal.load_kg, goal.two_arms, 0.0));
            let up = kin::solve_fast(body, &stand, &KinLoads::default());
            let lift_s = body.lift.as_ref().map_or(0.0, |l| MIN_JERK_PEAK * c.pose.lift_m / l.speed_mps.max(1e-3));
            let descend_s = move_time(&up, &empty).max(lift_s);
            let rise_s = move_time(&loaded, &kin::solve_fast(body, &KinPose { cargo_kg: goal.cargo_kg, ..stand }, &KinLoads::holding(goal.load_kg, goal.two_arms, 0.0))).max(lift_s);
            let _ = c.loaded;
            ReachPlan { ok: true, strategy: c.strategy, strategy_ko: strategy_ko(c.strategy), approach: "front", pose: c.pose, empty, loaded, object_x: x, object_y: 0.0, lowest_palm_z: goal.palm_z, reason_ko: None, descend_s, rise_s, rise_margin: None, evaluated, code: None }
        }
        None => {
            let low = lowest_palm(body, &rom, x0).max(0.0);
            let why = if low > goal.palm_z + 1e-3 {
                if body.legged() {
                    format!("다리를 굽혀도 손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m)", low, goal.palm_z)
                } else if body.mast_arms() {
                    format!("마스트 리프트를 {:.2} m 내려도 팔이 손바닥을 {:.2} m 아래로 가져가지 못함 (필요 {:.2} m)", rom.lift_max_m, low, goal.palm_z)
                } else if !body.has_torso() && rom.lift_max_m <= 0.0 {
                    format!("{}의 팔이 손바닥을 {:.2} m 아래로 가져가지 못함 (필요 {:.2} m)", if body.deck_arms() { "갑판" } else { "기둥" }, low, goal.palm_z)
                } else if rom.waist_max <= 0.0 {
                    let lift = if rom.lift_max_m > 0.0 { format!("몸통을 {:.2} m 내려도 ", rom.lift_max_m) } else { String::new() };
                    format!("허리 피치 관절이 없어 {lift}손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m) — 바닥의 물건을 집을 수 없음", low, goal.palm_z)
                } else if rom.lift_max_m > 0.0 {
                    format!("몸통을 {:.2} m 내리고 허리를 {:.0}°까지 굽혀도 손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m)", rom.lift_max_m, rom.waist_max, low, goal.palm_z)
                } else {
                    format!("허리를 {:.0}°까지 굽혀도 손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m)", rom.waist_max, low, goal.palm_z)
                }
            } else if let Some(high) = highest_palm(body, x0).filter(|h| *h < goal.palm_z - 1e-3) {
                let place = if body.deck_arms() { "갑판의 팔" } else if body.mast_arms() { "마스트 캐리지 꼭대기의 팔" } else { "선 자세의 팔" };
                format!("{place}로는 손바닥이 {:.2} m까지만 올라감 (필요 {:.2} m)", high, goal.palm_z)
            } else {
                format!("손은 닿지만 굽힌 자세를 버티지 못함{}", nearest.as_ref().map(|m| format!(": {}", m.reason)).unwrap_or_default())
            };
            let empty = kin::solve(body, &stand, &KinLoads::default());
            ReachPlan { ok: false, strategy: "none", strategy_ko: "불가", approach: "front", pose: stand, loaded: empty.clone(), empty, object_x: x0, object_y: 0.0, lowest_palm_z: low, reason_ko: Some(why), descend_s: 0.0, rise_s: 0.0, rise_margin: None, evaluated, code: None }
        }
    }
}

pub const BETWEEN_GAP_M: f32 = STANDOFF_GAP_M + 0.02;
pub const PALM_HALF_M: f32 = 0.035;

pub fn arm_ik3(shoulder: [f32; 3], target: [f32; 3], trunk_deg: f32, side: usize) -> Option<(f32, f32, f32)> {
    let t = trunk_deg.to_radians();
    let (k, v) = ([t.cos(), -t.sin()], [t.sin(), t.cos()]);
    let r = [target[0] - shoulder[0], target[1] - shoulder[1], target[2] - shoulder[2]];
    let (rk, rv, ry) = (r[0] * k[0] + r[2] * k[1], r[0] * v[0] + r[2] * v[1], r[1]);
    let theta = ry.atan2(-rv);
    let u = -(ry * ry + rv * rv).sqrt();
    let sag = [shoulder[0] + rk * k[0] + u * v[0], shoulder[2] + rk * k[1] + u * v[1]];
    let (fd, ed) = arm_ik([shoulder[0], shoulder[2]], sag, trunk_deg)?;
    let abd = if side == 0 { -theta } else { theta }.to_degrees();
    Some((fd, ed, abd))
}

#[derive(Clone, Copy, Debug)]
enum Hands {
    Side { x: f32, y: f32 },
    Between { x: f32, y: f32 },
}

struct Cand3 {
    cand: Cand,
    hands: Hands,
    out_m: f32,
    legs: [f32; 4],
}

fn hip_splayed(shank_deg: f32, thigh_deg: f32, out_m: f32) -> Option<[f32; 2]> {
    let h = hip_at(shank_deg, thigh_deg);
    if out_m <= 0.0 {
        return Some(h);
    }
    let tall = h[1] - ANKLE_H_M;
    (tall > out_m + 1e-3).then(|| [h[0], ANKLE_H_M + (tall * tall - out_m * out_m).sqrt()])
}

pub fn rise_margin(bottom: &KinReport, top: &KinReport, t: f32) -> f32 {
    let shift = |d: f32, z: f32| super::grasp::MIN_JERK_ACC * d / (t * t) * z / G;
    let (dx, dy) = (top.com[0] - bottom.com[0], top.com[1] - bottom.com[1]);
    let start = [bottom.zmp[0] - shift(dx, bottom.com[2]), bottom.zmp[1] - shift(dy, bottom.com[2])];
    let end = [top.zmp[0] + shift(dx, top.com[2]), top.zmp[1] + shift(dy, top.com[2])];
    kin::support_margin(bottom.support, start).min(kin::support_margin(top.support, end))
}

pub fn rise_timing(bottom: &KinReport, top: &KinReport) -> Option<(f32, f32)> {
    let t_act = move_time(bottom, top);
    let want = bottom.margin.min(top.margin).max(0.0) * (1.0 - 1.0 / RISE_ZMP_SAFETY);
    let m = rise_margin(bottom, top, t_act);
    if m >= want {
        return Some((t_act, m));
    }
    let slowest = super::grasp::LIFT_SLOWEST_S.max(t_act);
    let m_slow = rise_margin(bottom, top, slowest);
    if m_slow < want {
        return (m_slow >= 0.0).then_some((slowest, m_slow));
    }
    let (mut lo, mut hi) = (t_act, slowest);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if rise_margin(bottom, top, mid) >= want {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some((hi, rise_margin(bottom, top, hi)))
}

pub const RISE_ZMP_SAFETY: f32 = crate::balance::GRIP_SAFETY;

fn candidate3(body: &Body, goal: &ReachGoal, hands: Hands, out_m: f32, legs: [f32; 4], top: &KinReport) -> Result<Cand, Option<Miss>> {
    let [shank_deg, thigh_deg, pelvis_deg, waist_deg] = legs;
    let mut pose = body.stand(goal.cargo_kg);
    if body.legged() {
        pose.hip = hip_splayed(shank_deg, thigh_deg, out_m).ok_or(None)?;
        pose.pelvis_deg = pelvis_deg;
        pose.stance_out_m = out_m;
    }
    pose.waist_deg = waist_deg;
    let sh = shoulder_point(body, &pose);
    let trunk = pose.trunk_deg();
    let pan_max = Rom::of(body).pan_max;
    let two = match hands {
        Hands::Side { x, y } => {
            let (fd, ed, ad) = arm_ik3([sh[0], body.shoulder_mount(0)[1], sh[1]], [x, y, goal.palm_z], trunk, 0).ok_or(None)?;
            pose.shoulder_deg[0] = fd;
            pose.elbow_deg[0] = ed;
            pose.shoulder_abd_deg = [ad, 0.0];
            if let Some((eye, max)) = look_from(body, sh) {
                let ground = (x - eye[0]).hypot(y);
                pose.neck_pan_deg = y.atan2(x - eye[0]).to_degrees().clamp(-pan_max, pan_max);
                let look = (eye[1] - goal.palm_z).atan2(ground).to_degrees();
                pose.neck_deg = (look - trunk).clamp(-max, max);
            }
            false
        }
        Hands::Between { x, y } => {
            for s in 0..2 {
                let (sy, ty) = (body.shoulder_mount(s)[1], if s == 0 { -y } else { y });
                let (fd, ed, ad) = arm_ik3([sh[0], sy, sh[1]], [x, ty, goal.palm_z], trunk, s).ok_or(None)?;
                pose.shoulder_deg[s] = fd;
                pose.elbow_deg[s] = ed;
                pose.shoulder_abd_deg[s] = ad;
            }
            if let Some((eye, max)) = look_from(body, sh) {
                let look = (eye[1] - goal.palm_z).atan2(x - eye[0]).to_degrees();
                pose.neck_deg = (look - trunk).clamp(-max, max);
            }
            true
        }
    };
    pose.closure = 0.0;
    let loaded = kin::solve_fast(body, &pose, &KinLoads::holding(goal.load_kg, two, 0.0));
    if !loaded.feasible {
        return Err(Some(Miss { violation: violation(&loaded), reason: loaded.limiting.clone().unwrap_or_default() }));
    }
    let empty = kin::solve_fast(body, &pose, &KinLoads::default());
    if !empty.feasible {
        return Err(Some(Miss { violation: violation(&empty), reason: empty.limiting.clone().unwrap_or_default() }));
    }
    let Some((rise_s, rise)) = rise_timing(&loaded, top) else {
        let m = rise_margin(&loaded, top, super::grasp::LIFT_SLOWEST_S);
        return Err(Some(Miss { violation: -m, reason: format!("{:.1} s에 걸쳐 일어서도 균형 여유 {m:.2} < 0 (몸의 가속 포함)", super::grasp::LIFT_SLOWEST_S) }));
    };
    let score = loaded.effort_sum + (1.0 - loaded.margin.min(rise).clamp(0.0, 1.0)).powi(2);
    Ok(Cand { strategy: strategy_of(body, &pose), pose, loaded, score, rise_margin: Some(rise), rise_s: Some(rise_s) })
}

fn top_report(body: &Body, goal: &ReachGoal, two: bool) -> KinReport {
    kin::solve_fast(body, &super::grasp::hold_pose_for(body, two, goal.cargo_kg), &KinLoads::holding(goal.load_kg, two, 0.0))
}

fn search3(body: &Body, goal: &ReachGoal, options: &[(Hands, f32)]) -> (Option<Cand3>, Option<Miss>, u32) {
    let rom = Rom::of(body);
    let top = top_report(body, goal, options.first().is_some_and(|o| matches!(o.0, Hands::Between { .. })));
    let grid = |max: f32, n: usize| -> Vec<f32> { (0..=n).map(|i| max * i as f32 / n.max(1) as f32).collect() };
    let shank_max = if body.legged() { rom.ankle_dorsi_max } else { 0.0 };
    let thigh_max = if body.legged() { (rom.knee_max - shank_max).max(0.0).min(110.0) } else { 0.0 };
    let pelvis_max = if body.legged() { rom.hip_flex_max } else { 0.0 };
    let n = |legged: bool, k: usize| if legged { k } else { 0 };
    let (ss, ts, ps, ws) = (grid(shank_max, n(body.legged(), 3)), grid(thigh_max, n(body.legged(), 11)), grid(pelvis_max, n(body.legged(), 12)), grid(rom.waist_max, if rom.waist_max > 0.0 { 9 } else { 0 }));
    let mut evaluated = 0u32;
    let mut best: Option<Cand3> = None;
    let mut nearest: Option<Miss> = None;
    let take = |c: Result<Cand, Option<Miss>>, hands: Hands, out_m: f32, legs: [f32; 4], best: &mut Option<Cand3>, nearest: &mut Option<Miss>| match c {
        Ok(c) => {
            if best.as_ref().map_or(true, |b| c.score < b.cand.score) {
                *best = Some(Cand3 { cand: c, hands, out_m, legs });
            }
        }
        Err(Some(m)) => {
            if nearest.as_ref().map_or(true, |x| m.violation < x.violation) {
                *nearest = Some(m);
            }
        }
        Err(None) => {}
    };
    for &(hands, out_m) in options {
        for &s in &ss {
            for &t in &ts {
                for &p in &ps {
                    for &w in &ws {
                        evaluated += 1;
                        let legs = [s, t, p, w];
                        take(candidate3(body, goal, hands, out_m, legs, &top), hands, out_m, legs, &mut best, &mut nearest);
                    }
                }
            }
        }
    }
    if let Some((hands, out_m, [s0, t0, p0, w0])) = best.as_ref().map(|b| (b.hands, b.out_m, b.legs)) {
        let d = |max: f32, n: usize| if n == 0 { 0.0 } else { 0.5 * max / n as f32 };
        let (ds, dt, dp, dw) = (d(shank_max, 3), d(thigh_max, 11), d(pelvis_max, 12), d(rom.waist_max, 9));
        for k in [-1.0f32, 0.0, 1.0] {
            for l in [-1.0f32, 0.0, 1.0] {
                for m in [-1.0f32, 0.0, 1.0] {
                    for q in [-1.0f32, 0.0, 1.0] {
                        let legs = [(s0 + k * ds).clamp(0.0, shank_max), (t0 + l * dt).clamp(0.0, thigh_max), (p0 + m * dp).clamp(0.0, pelvis_max), (w0 + q * dw).clamp(0.0, rom.waist_max)];
                        evaluated += 1;
                        take(candidate3(body, goal, hands, out_m, legs, &top), hands, out_m, legs, &mut best, &mut nearest);
                    }
                }
            }
        }
    }
    (best, nearest, evaluated)
}

fn finish3(body: &Body, goal: &ReachGoal, b: Cand3, approach: &'static str, evaluated: u32) -> ReachPlan {
    let two = matches!(b.hands, Hands::Between { .. });
    let (x, y) = match b.hands {
        Hands::Side { x, y } => (x, y),
        Hands::Between { x, .. } => (x, 0.0),
    };
    let empty = kin::solve(body, &b.cand.pose, &KinLoads::default());
    let loaded = kin::solve(body, &b.cand.pose, &KinLoads::holding(goal.load_kg, two, 0.0));
    let stand = body.stand(goal.cargo_kg);
    let up = kin::solve_fast(body, &stand, &KinLoads::default());
    let descend_s = move_time(&up, &empty);
    let rise_s = b.cand.rise_s.unwrap_or_else(|| move_time(&loaded, &kin::solve_fast(body, &stand, &KinLoads::holding(goal.load_kg, two, 0.0))));
    let strategy = b.cand.strategy;
    ReachPlan { ok: true, strategy, strategy_ko: strategy_ko(strategy), approach, pose: b.cand.pose, empty, loaded, object_x: x, object_y: y, lowest_palm_z: goal.palm_z, reason_ko: None, descend_s, rise_s, rise_margin: b.cand.rise_margin, evaluated, code: None }
}

fn failed3(body: &Body, goal: &ReachGoal, approach: &'static str, why: String, x: f32, y: f32, evaluated: u32) -> ReachPlan {
    let stand = body.stand(goal.cargo_kg);
    let empty = kin::solve(body, &stand, &KinLoads::default());
    ReachPlan { ok: false, strategy: "none", strategy_ko: "불가", approach, pose: stand, loaded: empty.clone(), empty, object_x: x, object_y: y, lowest_palm_z: f32::NAN, reason_ko: Some(why), descend_s: 0.0, rise_s: 0.0, rise_margin: None, evaluated, code: None }
}

pub fn side_object_y(body: &Body, half_w_m: f32) -> f32 {
    let edge = if body.legged() { 0.5 * STANCE_WIDTH_M + 0.5 * body.leg.foot_width_m } else { 0.5 * body.support_width_m };
    -(edge + half_w_m + STANDOFF_GAP_M)
}

pub fn plan_side(body: &Body, goal: &ReachGoal, half_w_m: f32) -> ReachPlan {
    if let Some(no) = form_refusal(body, &ReachGoal { two_arms: false, ..*goal }, "side") {
        return no;
    }
    let y0 = side_object_y(body, half_w_m);
    let x0 = if body.legged() { 0.25 * body.leg.foot_len_m } else { 0.0 };
    let mut options = Vec::new();
    for dy in [0.0f32, -0.06, -0.12] {
        for dx in [-0.06f32, 0.0, 0.08, 0.16] {
            options.push((Hands::Side { x: x0 + dx, y: y0 + dy }, 0.0));
        }
    }
    let goal = ReachGoal { two_arms: false, ..*goal };
    let (best, nearest, evaluated) = search3(body, &goal, &options);
    match best {
        Some(b) => finish3(body, &goal, b, "side", evaluated),
        None => {
            let why = match nearest {
                Some(m) => format!("옆으로 손은 닿지만 그 자세를 버티지 못함: {}", m.reason),
                None => format!("팔을 옆으로 뻗어도 오른발 옆 {:.2} m의 바닥까지 손이 닿지 않음", -y0),
            };
            failed3(body, &goal, "side", why, x0, y0, evaluated)
        }
    }
}

pub fn plan_straddle(body: &Body, goal: &ReachGoal, half_w_m: f32) -> ReachPlan {
    let rom = Rom::of(body);
    if let Some(no) = form_refusal(body, &ReachGoal { two_arms: true, ..*goal }, "straddle") {
        return no;
    }
    if !body.legged() {
        return failed3(body, goal, "straddle", "바퀴·트랙은 물건을 발 사이에 두고 설 수 없음".into(), 0.0, 0.0, 0);
    }
    let need = (half_w_m + BETWEEN_GAP_M + 0.5 * body.leg.foot_width_m - 0.5 * STANCE_WIDTH_M).max(0.0);
    if need > rom.stance_out_max_m {
        return failed3(body, goal, "straddle", format!("물건이 넓어 발 사이에 두려면 발을 {need:.2} m 벌려야 함 (한계 {:.2} m)", rom.stance_out_max_m), 0.0, 0.0, 0);
    }
    let fl = body.leg.foot_len_m;
    let mut options = Vec::new();
    for out in [need, (need + 0.03).min(rom.stance_out_max_m)] {
        for x in [0.0f32, 0.25 * fl, 0.5 * fl] {
            options.push((Hands::Between { x: x.clamp(-0.25 * fl + goal.half_len_m, 0.75 * fl - goal.half_len_m), y: half_w_m + PALM_HALF_M }, out));
        }
    }
    let goal = ReachGoal { two_arms: true, ..*goal };
    let (best, nearest, evaluated) = search3(body, &goal, &options);
    match best {
        Some(b) => finish3(body, &goal, b, "straddle", evaluated),
        None => {
            let why = match nearest {
                Some(m) => format!("발 사이에 두고도 버티지 못함: {}", m.reason),
                None => "발 사이의 물건까지 손이 닿지 않음".into(),
            };
            failed3(body, &goal, "straddle", why, 0.25 * fl, 0.0, evaluated)
        }
    }
}

pub fn plan_or_straddle(body: &Body, goal: &ReachGoal, half_w_m: f32) -> ReachPlan {
    let front = plan(body, goal);
    if front.ok || !body.legged() || front.lowest_palm_z > goal.palm_z + 1e-3 {
        return front;
    }
    let wide = plan_straddle(body, goal, half_w_m);
    if wide.ok {
        return wide;
    }
    ReachPlan { reason_ko: Some(format!("{} · {}", front.reason_ko.clone().unwrap_or_default(), wide.reason_ko.unwrap_or_default())), ..front }
}

pub fn move_time(a: &KinReport, b: &KinReport) -> f32 {
    let body_speed = |name: &str, r: &KinReport| r.joint(name).map(|j| (j.angle_deg.unwrap_or(0.0), j.ratio, j.peak));
    let mut t = BEND_MIN_S;
    for ja in &a.joints {
        if ja.structural || ja.angle_deg.is_none() {
            continue;
        }
        let Some((qb, rb, _)) = body_speed(&ja.name, b) else { continue };
        let dq = (qb - ja.angle_deg.unwrap_or(0.0)).abs().to_radians();
        if dq < 1e-3 {
            continue;
        }
        let ratio = ja.ratio.max(rb).min(0.95);
        let w0 = joint_speed(&ja.name, ja);
        let allowed = w0 * (1.0 - ratio);
        t = t.max(MIN_JERK_PEAK * dq / allowed.max(1e-3));
    }
    t
}

fn joint_speed(_name: &str, j: &kin::KinJoint) -> f32 {
    const ALL: [&Actuator; 8] = [&STS3215, &HTD45H, &XM430, &RS01, &RS02, &RS03, &RS04, &BHL6512];
    ALL.iter().find(|a| j.actuator.contains(a.name) || a.name.contains(&j.actuator)).map(|a| a.speed_rad_s).unwrap_or_else(|| {
        ALL.iter().find(|a| j.actuator.contains(a.name.split(' ').last().unwrap_or(""))).map(|a| a.speed_rad_s).unwrap_or(5.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;

    fn body(plan: BodyPlan, base: BaseKind) -> Body {
        Body::new(&plan, base, FingerCount::Five)
    }

    #[test]
    fn the_arm_ik_puts_the_palm_on_the_target() {
        let b = body(BodyPlan::default(), BaseKind::Wheel);
        let mut pose = KinPose::stand(0.0);
        let (sd, ed) = arm_ik([0.0, SHOULDER_M], [0.30, 0.70], 0.0).unwrap();
        pose.shoulder_deg = [sd, 0.0];
        pose.elbow_deg = [ed, 10.0];
        let r = kin::solve(&b, &pose, &KinLoads::default());
        assert!((r.hands[0][0] - 0.30).abs() < 2e-3 && (r.hands[0][2] - 0.70).abs() < 2e-3, "{:?}", r.hands[0]);
        assert!(arm_ik([0.0, SHOULDER_M], [0.0, 0.40], 0.0).is_none(), "below the hanging hand");
    }

    #[test]
    fn a_rigid_trunk_on_wheels_cannot_reach_the_ground() {
        let b = body(BodyPlan::default(), BaseKind::Wheel);
        let p = plan(&b, &ReachGoal::ground_object(0.15, 2.4, false, 0.0));
        assert!(!p.ok, "{p:?}");
        assert!(p.lowest_palm_z > 0.4, "{}", p.lowest_palm_z);
        assert!(p.reason_ko.as_deref().unwrap_or("").contains("허리 피치"), "{:?}", p.reason_ko);
    }

    #[test]
    fn mid_legs_squat_and_hinge_to_reach_a_log() {
        let b = body(BodyPlan::default(), BaseKind::HumanFoot);
        let p = plan(&b, &ReachGoal::ground_object(0.15, 2.4, false, 0.0));
        assert!(p.ok, "{:?} lowest {}", p.reason_ko, p.lowest_palm_z);
        assert!(p.strategy.starts_with("squat") || p.strategy == "hinge", "{}", p.strategy);
        assert!((p.loaded.hands[0][2] - p.loaded.pose.hip[1]).is_finite());
        assert!((p.empty.hands[0][2] - (0.15 + PALM_CLEARANCE_M)).abs() < 3e-3, "palm at {:?}", p.empty.hands[0]);
        assert!(p.descend_s >= BEND_MIN_S && p.rise_s >= BEND_MIN_S);
        assert!(p.loaded.margin >= 0.0 && p.loaded.feasible);
    }

    #[test]
    fn a_pro_waist_pitch_alone_does_not_get_a_wheeled_hand_to_a_stone() {
        let b = body(BodyPlan::uniform(Tier::Pro), BaseKind::Wheel);
        let p = plan(&b, &ReachGoal::ground_object(0.10, 2.6, false, 0.0));
        assert!(!p.ok);
        assert!(p.lowest_palm_z > 0.12 && p.lowest_palm_z < 0.35, "{}", p.lowest_palm_z);
        let t = plan(&b, &ReachGoal { palm_z: 0.40, half_len_m: 0.05, load_kg: 1.0, two_arms: false, cargo_kg: 0.0 });
        assert!(t.ok && t.strategy == "stoop", "{t:?}");
    }

    #[test]
    fn a_torso_lift_and_a_waist_pitch_together_bring_a_wheeled_hand_to_the_floor() {
        let stone = ReachGoal::ground_object(0.10, 2.6, false, 0.0);
        let lift_only = body(BodyPlan { lift: true, ..BodyPlan::uniform(Tier::Mid) }, BaseKind::Wheel);
        let p = plan(&lift_only, &stone);
        assert!(!p.ok && p.reason_ko.as_deref().is_some_and(|r| r.contains("몸통을")), "{:?}", p.reason_ko);
        let b = body(BodyPlan::named("wheel_lift").unwrap(), BaseKind::Wheel);
        let p = plan(&b, &stone);
        assert!(p.ok, "{:?}", p.reason_ko);
        assert!(p.strategy.starts_with("lower") && p.pose.lift_m > 0.2 && p.pose.lift_m <= TORSO_LIFT.stroke_m + 1e-4, "{} {}", p.strategy, p.pose.lift_m);
        let j = p.loaded.joint("torso_lift").expect("the lift is a joint");
        assert!(j.torque > 50.0 && j.ratio < 1.0, "{j:?}");
        assert!(p.descend_s >= MIN_JERK_PEAK * p.pose.lift_m / TORSO_LIFT.speed_mps - 1e-3, "{}", p.descend_s);
        let legs = body(BodyPlan::named("wheel_lift").unwrap(), BaseKind::HumanFoot);
        assert!(kin::solve_fast(&legs, &KinPose::stand(0.0), &KinLoads::default()).joint("torso_lift").is_none());
        let plain = body(BodyPlan::uniform(Tier::Pro), BaseKind::Wheel);
        assert!(kin::solve_fast(&plain, &KinPose::stand(0.0), &KinLoads::default()).joint("torso_lift").is_none());
    }

    #[test]
    #[ignore]
    fn print_straddle_by_strategy() {
        let b = body(BodyPlan::named("mid").unwrap(), BaseKind::HumanFoot);
        let g = ReachGoal { two_arms: true, ..ReachGoal::ground_object(0.10, 4.0, true, 0.0) };
        let top = top_report(&b, &g, true);
        let rom = Rom::of(&b);
        let need = (0.05 + BETWEEN_GAP_M + 0.5 * b.leg.foot_width_m - 0.5 * STANCE_WIDTH_M).max(0.0);
        let mut best: std::collections::BTreeMap<&'static str, (f32, String)> = Default::default();
        let mut misses: std::collections::BTreeMap<String, u32> = Default::default();
        for out in [need, need + 0.03] {
            for x in [0.0f32, 0.25 * b.leg.foot_len_m, 0.5 * b.leg.foot_len_m] {
                for s in [0.0f32, 10.0, 20.0, 30.0] {
                    for t in (0..=11).map(|i| 105.0 * i as f32 / 11.0) {
                        for p in (0..=12).map(|i| rom.hip_flex_max * i as f32 / 12.0) {
                            let r = candidate3(&b, &g, Hands::Between { x, y: 0.05 + PALM_HALF_M }, out, [s, t, p, 0.0], &top);
                            if t > 40.0 && s >= 20.0 {
                                let why = match &r {
                                    Ok(_) => "ok".to_string(),
                                    Err(Some(m)) => m.reason.split(':').next().unwrap_or("").chars().take(40).collect(),
                                    Err(None) => "ik".to_string(),
                                };
                                *misses.entry(why).or_insert(0u32) += 1;
                            }
                            if let Ok(c) = r {
                                let e = best.entry(c.strategy).or_insert((f32::INFINITY, String::new()));
                                if c.score < e.0 {
                                    let j = |n: &str| c.loaded.joint(n).map(|j| format!("{n} {:.0}° r{:.2}", j.angle_deg.unwrap_or(0.0), j.ratio)).unwrap_or_default();
                                    *e = (c.score, format!("effort {:.3} margin {:.2} rise {:?} {:?}s | out {out:.2} x {x:.2} s{s:.0} t{t:.0} p{p:.0} | {} {} {} {} {}", c.loaded.effort_sum, c.loaded.margin, c.rise_margin, c.rise_s, j("knee_r"), j("hip_pitch_r"), j("ankle_r"), j("shoulder_r"), j("elbow_r")));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (k, (sc, d)) in best {
            println!("[STRAT] {k:<12} score {sc:.3} | {d}");
        }
        println!("[STRAT] squat misses {misses:?}");
    }

    #[test]
    #[ignore]
    fn print_v07_plans() {
        for plan_id in ["budget", "mid", "pro"] {
            let b = body(BodyPlan::named(plan_id).unwrap(), BaseKind::HumanFoot);
            let j = |p: &ReachPlan, n: &str| p.loaded.joint(n).map(|j| format!("{n} {:.0}° {:.1}N·m r{:.2}", j.angle_deg.unwrap_or(0.0), j.torque.abs(), j.ratio)).unwrap_or_default();
            for (h, kg, w) in [(0.10f32, 4.0f32, 0.05f32), (0.15, 2.4, 0.075), (0.10, 0.6, 0.10)] {
                let g = ReachGoal::ground_object(h, kg, true, 0.0);
                let f = plan(&b, &g);
                let s = plan_or_straddle(&b, &g, w);
                println!("[V07] {plan_id} {kg}kg front ok={} {} m{:.2} | fallback ok={} {} {} out {:.2} x {:.2} margin {:.2} rise {:?} {:.2}s | {} | {} | {} | {} | {:?}",
                    f.ok, f.strategy, f.loaded.margin, s.ok, s.approach, s.strategy, s.pose.stance_out_m, s.object_x, s.loaded.margin, s.rise_margin, s.rise_s, j(&s, "knee_r"), j(&s, "hip_pitch_r"), j(&s, "hip_roll_r"), j(&s, "shoulder_abd_r"), s.reason_ko);
                let side = plan_side(&b, &ReachGoal::ground_object(h, kg, false, 0.0), w);
                println!("[V07] {plan_id} {kg}kg side ok={} {} x {:.2} y {:.2} margin {:.2} rise {:?} {:.2}s | {} | {} | {} | {} | eval {} | {:?}",
                    side.ok, side.strategy, side.object_x, side.object_y, side.loaded.margin, side.rise_margin, side.rise_s, j(&side, "shoulder_abd_r"), j(&side, "shoulder_r"), j(&side, "knee_r"), j(&side, "hip_pitch_r"), side.evaluated, side.reason_ko);
            }
        }
    }

    #[test]
    #[ignore]
    fn print_plans() {
        for plan_id in ["budget", "mid", "pro", "pro_arms", "wheel_lift"] {
            for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track] {
                let b = body(BodyPlan::named(plan_id).unwrap(), base);
                for (h, kg) in [(0.15f32, 2.4f32), (0.10, 2.6), (0.10, 4.0)] {
                    let p = plan(&b, &ReachGoal::ground_object(h, kg, false, 0.0));
                    let j = |n: &str| p.loaded.joint(n).map(|j| format!("{n} {:.0}° {:.1}N·m r{:.2}", j.angle_deg.unwrap_or(0.0), j.torque.abs(), j.ratio)).unwrap_or_default();
                    let lift = p.loaded.joint("torso_lift").map(|j| format!("lift {:.2} m {:.0} N r{:.2}", p.pose.lift_m, j.torque, j.ratio)).unwrap_or_default();
                    println!("[REACH] {plan_id:<8} {base:?} h{h} {kg}kg ok={} {} x={:.2} low={:.2} down {:.2}s up {:.2}s margin {:.2} eval {} | {} | {} | {} | {} | {} | {lift} | {:?}",
                        p.ok, p.strategy, p.object_x, p.lowest_palm_z, p.descend_s, p.rise_s, p.loaded.margin, p.evaluated,
                        j("knee_r"), j("hip_pitch_r"), j("ankle_r"), j("waist_pitch"), j("shoulder_r"), p.reason_ko);
                }
            }
        }
    }
}
