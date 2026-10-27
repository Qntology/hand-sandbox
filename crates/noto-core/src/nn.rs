//! Composite N:N scenarios: several causes acting together, each N:N cell
//! weighted the way published test batteries weight their items, scored
//! against the human references. Sources and the derivation of every weight
//! are in `docs/research/nn-scenarios.md`.
//!
//! How a scenario is scored (research §3.0):
//!
//! 1. Every cell with a human reference gets `z = (value − ref) / tol` and a
//!    likeness score `100 − 10·|z|` (floor 0; the Gait Deviation Index scale,
//!    10 points per tolerance). Cells without a reference are reported raw.
//! 2. Gates come before any weighting (the DARPA Robotics Challenge rule that
//!    completion outranks everything): the dynamic margin below zero, a joint
//!    over its peak torque, a target not reached, or the camera tilted more
//!    than 20° (Pozzo 1990) gives the phase, and the scenario, a 0.
//! 3. A phase score is the segment-weighted mean of its cells' scores. The
//!    segment weights come from REBA / RULA maximum segment scores or from
//!    equal weights within a BESTest subsystem.
//! 4. The scenario score is the phase scores weighted by the subsystem
//!    shares of BESTest, Mini-BESTest or Brief-BESTest (items × points).
//!    A braking ladder takes its worst rung (the NIOSH composite lifting
//!    index's floor), and the work-shift composite is the NIOSH sequential
//!    lifting index.
//! 5. Every raw cell, its reference and z are reported with the score.
//!
//! What keeps this from becoming a target to tune (research §4): the
//! controller never reads the references (a test scans `balance.rs` and
//! `dynamics.rs` for it), every scenario runs with the same controller
//! parameters, and the held-out scenarios ([`holdout`]) draw their
//! conditions from ranges the development scenarios do not use and are only
//! counted, never used to change anything.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::balance::{reference, Body, BalanceReport, FeedForward, Posture, Source, G};
use crate::dynamics::{simulate, DynCfg, DynDigest};
use crate::parts::Segment;

/// What acts on the robot in a posture (more than its single `Source`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    ArmRaise,
    ArmLift,
    FootLift,
    StepUp,
    Slope,
    CarryFront,
    Start,
    Brake,
}

/// Every cause present in a posture.
pub fn causes_of(p: &Posture) -> BTreeSet<Cause> {
    let mut c = BTreeSet::new();
    if p.source == Source::ArmRaise || p.arm_elev_deg[0] >= 60.0 {
        c.insert(Cause::ArmRaise);
    }
    if p.source == Source::ArmLift || p.hand_load_kg.iter().any(|l| *l > 0.0) {
        c.insert(Cause::ArmLift);
    }
    if p.single_support {
        c.insert(Cause::FootLift);
    }
    if p.step_m > 0.0 {
        c.insert(Cause::StepUp);
    }
    if p.slope_deg.abs() >= 3.0 {
        c.insert(Cause::Slope);
    }
    if p.cargo_kg > 0.0 {
        c.insert(Cause::CarryFront);
    }
    if p.accel_mps2 > 0.0 {
        c.insert(Cause::Start);
    }
    if p.accel_mps2 < 0.0 {
        c.insert(Cause::Brake);
    }
    c
}

/// Segment order of the weight arrays: head, chest, belly, arm, hand, leg, foot.
pub const SEGMENTS: [Segment; 7] = Segment::ALL;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhaseSpec {
    pub id: String,
    /// BESTest subsystem the phase stands for (`anticipatory`, `reactive`, `sensory`, `gait`, `limits`).
    pub subsystem: &'static str,
    pub weight: f32,
    pub posture: Posture,
    /// The causes a world posture must have to count as this phase.
    pub causes: Vec<Cause>,
    /// Rungs of one ladder share this id; the phase takes the worst rung.
    pub ladder: Option<&'static str>,
    /// An outside disturbance (a vehicle braking under the robot), not a motion the
    /// robot chose. Only these may exceed what people withstand without a step.
    pub imposed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioSpec {
    pub id: &'static str,
    pub name_ko: &'static str,
    /// Segment weights (head, chest, belly, arm, hand, leg, foot), summing to 1.
    pub seg_w: [f32; 7],
    /// Where the phase and segment weights come from.
    pub weights_from: &'static str,
    pub needs_legs: bool,
    pub holdout: bool,
    pub phases: Vec<PhaseSpec>,
}

/// One N:N cell of a scenario phase.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cell {
    pub source: Source,
    pub target: Segment,
    pub axis: String,
    pub value: f32,
    pub reference: Option<f32>,
    pub tolerance: Option<f32>,
    pub cite: Option<&'static str>,
    pub z: Option<f32>,
    pub score: Option<f32>,
    /// From the joint dynamics rather than the quasi-static model.
    pub dynamic: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhaseResult {
    pub id: String,
    pub subsystem: &'static str,
    pub weight: f32,
    pub ladder: Option<&'static str>,
    pub cells: Vec<Cell>,
    /// Mean cell score per segment, `None` where no cell has a reference.
    pub seg_scores: [Option<f32>; 7],
    pub score: f32,
    /// Mini-BESTest-style item: 2 all scored cells within one tolerance, 1 within two, else 0.
    pub item: u8,
    pub gates: Vec<String>,
    /// Failures at an intensity where people need a step too (reported, not gated).
    pub beyond_human: Vec<String>,
    pub static_margin: f32,
    pub min_margin: f32,
    pub peak_ratio: f32,
    pub worst_joint: Option<String>,
    /// Largest sustained torque over the continuous rating (for the shift index).
    pub sustained_effort: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub holdout: bool,
    /// `None` when the base cannot do the scenario (a step on wheels).
    pub skipped: Option<String>,
    pub phases: Vec<PhaseResult>,
    pub score: f32,
    pub gated: bool,
    pub gates: Vec<String>,
    /// Every phase item at least 1 and no gate (the held-out pass rule).
    pub passed: bool,
    pub weights_from: &'static str,
}

fn cause_set(c: &[Cause]) -> Vec<Cause> {
    c.to_vec()
}

/// Ten per cent of the body mass, the heavier load of the research's lifting and carrying studies.
fn load10(body: &Body) -> f32 {
    0.10 * body.mass_kg()
}

/// A lift of `kg` with one arm, or two when one cannot (as the world does it).
fn lift_posture(body: &Body, kg: f32, slope: f32) -> Posture {
    let two = !body.solve(&Posture { slope_deg: slope, ..Posture::arm_lift(kg, false, 0.0) }, &FeedForward::default(), 1e3).feasible;
    Posture { slope_deg: slope, ..Posture::arm_lift(kg, two, 0.0) }
}

fn phase(id: &str, subsystem: &'static str, weight: f32, posture: Posture, causes: &[Cause]) -> PhaseSpec {
    PhaseSpec { id: id.into(), subsystem, weight, posture, causes: cause_set(causes), ladder: None, imposed: false }
}

/// The development scenarios (research §3.1, S1–S4 and S6; S5 needs a
/// frequency response this simulator does not measure, S7 is [`shift_index`],
/// S8 is [`holdout`]).
pub fn library(body: &Body) -> Vec<ScenarioSpec> {
    let load = load10(body);
    let walk = body.legged();
    let mut out = Vec::new();
    // S1: lifting on a 10° slope. Mini-BESTest sensory : anticipatory = 6 : 6; segments
    // sensory side head/chest/foot equal, lifting side REBA maxima.
    out.push(ScenarioSpec {
        id: "S1",
        name_ko: "경사 위 들기",
        seg_w: [0.218, 0.253, 0.052, 0.138, 0.103, 0.069, 0.167],
        weights_from: "Mini-BESTest 감각 6 : 예측 6 (Franchignoni 2010) · REBA 분절 최대점 (Hignett & McAtamney 2000)",
        needs_legs: false,
        holdout: false,
        phases: vec![
            phase("slope", "sensory", 0.5, Posture { source: Source::Slope, slope_deg: 10.0, ..Posture::quiet(0.0) }, &[Cause::Slope]),
            phase("lift", "anticipatory", 0.5, lift_posture(body, load, 10.0), &[Cause::ArmLift, Cause::Slope]),
        ],
    });
    // S2: a front load up a 17 cm step (the BESTest step). BESTest APA 18 : gait 21.
    out.push(ScenarioSpec {
        id: "S2",
        name_ko: "앞 짐 들고 턱 오르기",
        seg_w: [0.056, 0.093, 0.210, 0.149, 0.111, 0.228, 0.154],
        weights_from: "BESTest APA 18 : 보행 21 항목 점수 (Horak 2009) · REBA 분절 최대점",
        needs_legs: true,
        holdout: false,
        phases: vec![
            phase("carry_walk", "gait", 0.538, Posture { source: Source::FootLift, single_support: walk, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::FootLift]),
            phase("step", "anticipatory", 0.462, Posture { source: Source::StepUp, step_m: 0.17, single_support: walk, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::StepUp]),
        ],
    });
    // S3: reach forward on one leg. BESTest limits 21 : APA 18.
    out.push(ScenarioSpec {
        id: "S3",
        name_ko: "한 발로 서서 앞으로 뻗기",
        seg_w: [0.0, 0.179, 0.154, 0.179, 0.0, 0.154, 0.334],
        weights_from: "BESTest 안정 한계 21 : APA 18 · 하위체계 안 동일 가중 (Rasch)",
        needs_legs: true,
        holdout: false,
        phases: vec![
            phase("weight_shift", "anticipatory", 0.462, Posture { source: Source::FootLift, single_support: true, ..Posture::quiet(0.0) }, &[Cause::FootLift]),
            phase("reach", "limits", 0.538, Posture { source: Source::ArmRaise, arm_elev_deg: [90.0, 0.0], elbow_deg: [0.0, 10.0], single_support: true, ..Posture::quiet(0.0) }, &[Cause::FootLift, Cause::ArmRaise]),
        ],
    });
    // S4: braking with a front load, a ladder of decelerations (Krašna 2021 pulses).
    // Mini-BESTest reactive 6 : gait 10; the ladder takes its worst rung (NIOSH CLI floor).
    let mut s4 = vec![phase("carry", "gait", 0.625, Posture { source: Source::CarryFront, ..Posture::quiet(load) }, &[Cause::CarryFront])];
    for a in [0.5f32, 1.0, 1.5, 2.5] {
        let mut p = phase(&format!("brake_{a:.1}"), "reactive", 0.375, Posture { source: Source::Accel, accel_mps2: -a, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake]);
        p.ladder = Some("brake");
        p.imposed = true;
        s4.push(p);
    }
    out.push(ScenarioSpec {
        id: "S4",
        name_ko: "짐 들고 제동 (강도 사다리)",
        seg_w: [0.190, 0.233, 0.065, 0.172, 0.129, 0.086, 0.125],
        weights_from: "Mini-BESTest 반응 6 : 보행 10 · 사다리는 NIOSH CLI 의 최악 단계 바닥값 (Waters 1994)",
        needs_legs: false,
        holdout: false,
        phases: s4,
    });
    // S6: strike on a slope, then start off. Brief-BESTest APA 6 : sensory 3.
    // The start is the robot's own: its drive planner picks the acceleration.
    let start = if walk {
        Posture { source: Source::FootLift, single_support: true, slope_deg: 10.0, ..Posture::quiet(0.0) }
    } else {
        let a = body.planned_accel(&body.base.spec(), 10.0, 0.0, crate::surface::Surface::Firm, 1.0, PLANNER_MIN_RAMP_S);
        Posture { source: Source::Accel, accel_mps2: a, slope_deg: 10.0, ..Posture::quiet(0.0) }
    };
    out.push(ScenarioSpec {
        id: "S6",
        name_ko: "경사 위에서 팔 들어 치고 출발",
        seg_w: [0.168, 0.168, 0.140, 0.114, 0.057, 0.130, 0.223],
        weights_from: "Brief-BESTest APA 6 : 감각 3 (Padgett 2012) · RULA 분절 최대점 (McAtamney & Corlett 1993)",
        needs_legs: false,
        holdout: false,
        phases: vec![
            phase("slope", "sensory", 0.333, Posture { source: Source::Slope, slope_deg: 10.0, ..Posture::quiet(0.0) }, &[Cause::Slope]),
            phase("strike", "anticipatory", 0.3335, Posture { slope_deg: 10.0, ..Posture::arm_raise(0.0) }, &[Cause::ArmRaise, Cause::Slope]),
            phase("start", "anticipatory", 0.3335, start, if walk { &[Cause::FootLift, Cause::Slope] } else { &[Cause::Start, Cause::Slope] }),
        ],
    });
    out
}

/// Shortest ramp to cruise speed of the world's drive (`world_sim::ACCEL_TIME_S`).
pub const PLANNER_MIN_RAMP_S: f32 = 0.5;

/// Ranges of the held-out scenarios. None of them contains a condition the
/// development scenarios use (slope 10°, step 17 cm, braking 0.5 / 1.0 /
/// 1.5 / 2.5 m/s², load 10% of the body mass); a test checks it.
pub const HOLDOUT_SLOPE_DEG: (f32, f32) = (12.0, 18.0);
pub const HOLDOUT_STEP_M: (f32, f32) = (0.12, 0.16);
pub const HOLDOUT_BRAKE: [(f32, f32); 2] = [(0.75, 0.95), (1.7, 2.2)];
pub const HOLDOUT_LOAD_FRAC: [(f32, f32); 2] = [(0.05, 0.08), (0.12, 0.15)];

/// Held-out scenarios for one evaluation seed: random combinations of two or
/// three causes inside [`HOLDOUT_SLOPE_DEG`] and friends (DARPA's surprise
/// task; research §4 rules 1–4). Scored pass / fail, never tuned against.
pub fn holdout(body: &Body, seed: u64, count: usize) -> Vec<ScenarioSpec> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let pick = |r: (f32, f32), u: f32| r.0 + (r.1 - r.0) * u;
    let mut out = Vec::new();
    for i in 0..count {
        let slope = pick(HOLDOUT_SLOPE_DEG, next()) * if next() < 0.5 { 1.0 } else { -1.0 };
        let lf = HOLDOUT_LOAD_FRAC[(next() * 2.0) as usize % 2];
        let load = pick(lf, next()) * body.mass_kg();
        let br = HOLDOUT_BRAKE[(next() * 2.0) as usize % 2];
        let brake = pick(br, next());
        let step = pick(HOLDOUT_STEP_M, next());
        let kind = (next() * 4.0) as usize % 4;
        let (name, phases): (&'static str, Vec<PhaseSpec>) = match kind {
            0 => ("경사 + 들기", vec![phase("lift", "anticipatory", 1.0, lift_posture(body, load, slope), &[Cause::ArmLift, Cause::Slope])]),
            1 => ("짐 + 제동", vec![PhaseSpec { imposed: true, ..phase("brake", "reactive", 1.0, Posture { source: Source::Accel, accel_mps2: -brake, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake]) }]),
            2 => ("경사 + 짐 + 제동", vec![PhaseSpec { imposed: true, ..phase("brake", "reactive", 1.0, Posture { source: Source::Accel, accel_mps2: -brake, slope_deg: slope, ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::Brake, Cause::Slope]) }]),
            _ => ("짐 + 턱", vec![phase("step", "anticipatory", 1.0, Posture { source: Source::StepUp, step_m: step, single_support: body.legged(), ..Posture::quiet(load) }, &[Cause::CarryFront, Cause::StepUp])]),
        };
        let id: &'static str = Box::leak(format!("H{}-{i}", seed % 1000).into_boxed_str());
        out.push(ScenarioSpec { id, name_ko: name, seg_w: [1.0 / 7.0; 7], weights_from: "DARPA DRC: 과제당 통과/실패, 동일 가중", needs_legs: kind == 3, holdout: true, phases });
    }
    out
}

/// References of the composite and dynamic cells that the single-cause table
/// (`balance::reference`) does not have. Derivations in the research doc.
pub fn composite_reference(source: Source, target: Segment, axis: &str, accel: f32) -> Option<(f32, f32, &'static str)> {
    match (source, target, axis) {
        // Share of the limit of stability a deceleration (braking pushes the CoM
        // forward) or an acceleration (backward) uses: human LOS 6.25° forward,
        // 4.45° backward (Jbabdi 2008); people take a step at about 93% of it
        // (de Kam 2017 thresholds 1.0 / 0.7 m/s², cited by Krašna 2021).
        (Source::Accel, Segment::Foot, "los_frac") => {
            let lim = if accel < 0.0 { 6.25f32 } else { 4.45 }.to_radians().tan();
            let v = (accel.abs() / G) / lim;
            Some((v, 0.3 * v.max(0.05), "LOS 6.25°/4.45° (Jbabdi 2008), 내딛기 문턱 1.0/0.7 m/s² (de Kam 2017 via Krašna 2021) — 유도"))
        }
        // Peak sideways pelvis acceleration of the weight shift before a step:
        // 0.044 ± 0.018 g (L5 IMU, 32 adults of 69 years, Schlenstedt 2018).
        (Source::FootLift, Segment::Belly, "pelvis_lat_acc") => Some((0.43, 0.18, "보행 시작 골반 좌우 가속 0.044 ± 0.018 g (Schlenstedt 2018)")),
        // A predictable disturbance: anticipation cuts the backward CoP peak from
        // 60 ± 14 to 28 ± 3.6 mm (Santos, Kanekar & Aruin 2010) → dip ratio 0.47.
        (_, Segment::Foot, "ff_dip_ratio") => Some((0.47, 0.14, "예측 가능한 교란의 CoP 최대 28/60 mm (Santos 2010) — 유도")),
        // Free-style lifting on a slope raises the L5/S1 moment 6.8% over flat
        // ground (Mirka & Shin 2004, n = 13; SD not given, tolerance chosen as ±5%).
        (Source::ArmLift, Segment::Belly, "slope_moment_ratio") => Some((1.068, 0.05, "경사 들기 L5/S1 모멘트 +6.8% (Mirka & Shin 2004) — 허용폭 선택")),
        _ => None,
    }
}

/// A cell with a reference is a likeness cell of the shared judge (`crate::bench::judge`); one without is recorded.
fn scored(c: &mut Cell) {
    if let (Some(reference), Some(tolerance)) = (c.reference, c.tolerance) {
        let v = crate::bench::judge(Some(c.value), crate::bench::Rule::Like { reference, tolerance });
        c.z = v.z;
        c.score = v.score;
    }
}

/// The cells of one state: the quasi-static couplings with their references,
/// and the dynamic cells of the transition into it.
pub fn cells(body: &Body, post: &Posture, r: &BalanceReport, d: Option<&DynDigest>) -> Vec<Cell> {
    let ctx = r.ref_ctx(body.mass_kg());
    let mut out: Vec<Cell> = r
        .couplings
        .iter()
        .map(|c| {
            let rf = reference(c.source, c.target, &c.axis, &ctx);
            let mut cell = Cell { source: c.source, target: c.target, axis: c.axis.clone(), value: c.value, reference: rf.as_ref().map(|x| x.value), tolerance: rf.as_ref().map(|x| x.tolerance), cite: rf.as_ref().map(|x| x.cite), z: None, score: None, dynamic: false };
            scored(&mut cell);
            cell
        })
        .collect();
    let mut add = |target: Segment, axis: &str, value: f32| {
        let rf = composite_reference(post.source, target, axis, post.accel_mps2).or_else(|| reference(post.source, target, axis, &ctx).map(|x| (x.value, x.tolerance, x.cite)));
        let mut cell = Cell { source: post.source, target, axis: axis.into(), value, reference: rf.map(|x| x.0), tolerance: rf.map(|x| x.1), cite: rf.map(|x| x.2), z: None, score: None, dynamic: true };
        scored(&mut cell);
        out.push(cell);
    };
    if let Some(d) = d {
        // Hof's b_min is the smallest margin of the extrapolated CoM: the dynamic minimum compares with it.
        add(Segment::Foot, "margin", d.min_margin);
        add(Segment::Head, "tilt_deg", d.head_pitch_max_deg);
        if post.single_support {
            add(Segment::Belly, "pelvis_lat_acc", d.pelvis_lat_acc_max);
        }
    }
    if post.accel_mps2 != 0.0 {
        // Share of the support's edge distance the inertial ZMP shift uses.
        let s = r.support;
        let edge = if post.accel_mps2 < 0.0 { s[1] - r.com[0] } else { r.com[0] - s[0] };
        add(Segment::Foot, "los_frac", r.com[2] * post.accel_mps2.abs() / G / edge.max(1e-3));
    }
    out
}

fn seg_index(s: Segment) -> usize {
    SEGMENTS.iter().position(|x| *x == s).unwrap_or(0)
}

/// Scores one phase from its cells and gates.
pub fn score_phase(spec: &PhaseSpec, seg_w: &[f32; 7], cells: Vec<Cell>, r: &BalanceReport, d: Option<&DynDigest>) -> PhaseResult {
    let mut sums = [0.0f32; 7];
    let mut counts = [0u32; 7];
    let mut worst_z = 0.0f32;
    for c in &cells {
        if let (Some(s), Some(z)) = (c.score, c.z) {
            let i = seg_index(c.target);
            sums[i] += s;
            counts[i] += 1;
            worst_z = worst_z.max(z.abs());
        }
    }
    let seg_scores: [Option<f32>; 7] = std::array::from_fn(|i| (counts[i] > 0).then(|| sums[i] / counts[i] as f32));
    let (num, den) = (0..7).filter_map(|i| seg_scores[i].map(|s| (seg_w[i] * s, seg_w[i]))).fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let mut gates = Vec::new();
    // Where the human reference itself needs a step (an acceleration past
    // the limit of stability: 100% of people stepped at 2.5 m/s² braking,
    // Krašna 2021), a robot that needs one too is not failing a human task.
    let human_steps = spec.imposed && cells.iter().any(|c| c.axis == "los_frac" && c.reference.map_or(false, |v| v >= 1.0));
    let mut beyond = Vec::new();
    if !r.feasible {
        let why = format!("정적 불가: {}", r.limiting.clone().unwrap_or_default());
        if human_steps && r.margin < 0.0 && r.worst().map_or(true, |w| w.peak_ratio <= 1.0) {
            beyond.push(why);
        } else {
            gates.push(why);
        }
    }
    if let Some(d) = d {
        if d.min_margin < 0.0 {
            let why = format!("동적 여유 {:.2} < 0", d.min_margin);
            if human_steps {
                beyond.push(why);
            } else {
                gates.push(why);
            }
        }
        if d.peak_ratio > 1.0 {
            gates.push(format!("{} 피크 {:.0}%", d.worst_joint.clone().unwrap_or_default(), d.peak_ratio * 100.0));
        }
        if !d.reached {
            gates.push("목표 자세 미도달".into());
        }
        if d.head_pitch_max_deg > 20.0 {
            gates.push(format!("머리 기울기 {:.0}° > 20° (Pozzo 1990)", d.head_pitch_max_deg));
        }
    }
    let score = if gates.is_empty() && den > 0.0 { num / den } else if gates.is_empty() { 100.0 } else { 0.0 };
    let item = if !gates.is_empty() { 0 } else if worst_z <= 1.0 { 2 } else if worst_z <= 2.0 { 1 } else { 0 };
    let sustained = r.joints.iter().map(|j| j.effort).fold(0.0f32, f32::max);
    PhaseResult {
        id: spec.id.clone(),
        subsystem: spec.subsystem,
        weight: spec.weight,
        ladder: spec.ladder,
        cells,
        seg_scores,
        score,
        item,
        gates,
        beyond_human: beyond,
        static_margin: r.margin,
        min_margin: d.map_or(r.margin, |d| d.min_margin),
        peak_ratio: d.map_or(r.worst().map_or(0.0, |w| w.peak_ratio), |d| d.peak_ratio),
        worst_joint: d.and_then(|d| d.worst_joint.clone()),
        sustained_effort: sustained,
    }
}

/// Evaluates a scenario: each phase is simulated from the previous one with
/// the given feed-forward (the controller's own setting, the same for every
/// scenario), scored, and the phases combined.
pub fn evaluate(body: &Body, spec: &ScenarioSpec, ff: &FeedForward) -> ScenarioResult {
    let cfg = DynCfg::default();
    if spec.needs_legs && !body.legged() {
        return ScenarioResult { id: spec.id, name_ko: spec.name_ko, holdout: spec.holdout, skipped: Some("다리가 없는 베이스에는 이 동작(17 cm 턱 · 한 발 서기)이 없음".into()), phases: vec![], score: 0.0, gated: false, gates: vec![], passed: false, weights_from: spec.weights_from };
    }
    let mut prev = Posture::quiet(spec.phases.first().map_or(0.0, |p| p.posture.cargo_kg));
    let mut phases = Vec::new();
    for ph in &spec.phases {
        let p = ph.posture;
        let grip = 1e3;
        let report = body.solve(&p, ff, grip);
        let roll = simulate(body, &prev, &p, ff, grip, &cfg);
        let digest = roll.digest();
        let mut cs = cells(body, &p, &report, Some(&digest));
        // Anticipation's share: the same move without the feed-forward.
        if ff.enabled {
            let off = simulate(body, &prev, &p, &FeedForward { enabled: false, gain: 1.0 }, grip, &cfg);
            let (dip_ff, dip_off) = (digest.static_margin - digest.min_margin, off.static_margin - off.min_margin);
            if dip_off > 0.01 {
                let rf = composite_reference(p.source, Segment::Foot, "ff_dip_ratio", 0.0);
                let mut c = Cell { source: p.source, target: Segment::Foot, axis: "ff_dip_ratio".into(), value: (dip_ff / dip_off).max(0.0), reference: rf.map(|x| x.0), tolerance: rf.map(|x| x.1), cite: rf.map(|x| x.2), z: None, score: None, dynamic: true };
                scored(&mut c);
                cs.push(c);
            }
        }
        // Lifting on a slope against the same lift on flat ground (relative cell).
        if p.source == Source::ArmLift && p.slope_deg.abs() >= 3.0 {
            let flat = Posture { slope_deg: 0.0, ..p };
            let f = simulate(body, &Posture { slope_deg: 0.0, ..prev }, &flat, ff, grip, &cfg);
            let moment = |r: &crate::dynamics::Rollout| -> f32 {
                let k = r.joints.iter().position(|j| j.name == "waist_pitch").unwrap_or(0);
                r.samples.iter().map(|s| s.tau_nm[k]).fold(0.0, f32::max)
            };
            let (ms, mf) = (moment(&roll), moment(&f));
            if mf > 1e-3 {
                let rf = composite_reference(Source::ArmLift, Segment::Belly, "slope_moment_ratio", 0.0);
                let mut c = Cell { source: p.source, target: Segment::Belly, axis: "slope_moment_ratio".into(), value: ms / mf, reference: rf.map(|x| x.0), tolerance: rf.map(|x| x.1), cite: rf.map(|x| x.2), z: None, score: None, dynamic: true };
                scored(&mut c);
                cs.push(c);
            }
        }
        phases.push(score_phase(ph, &spec.seg_w, cs, &report, Some(&digest)));
        prev = p;
    }
    combine(spec, phases)
}

/// Combines phase results: ladders take their worst rung, then subsystem weights.
pub fn combine(spec: &ScenarioSpec, phases: Vec<PhaseResult>) -> ScenarioResult {
    let mut parts: Vec<(f32, f32)> = Vec::new();
    let mut ladders: std::collections::BTreeMap<&'static str, (f32, f32)> = std::collections::BTreeMap::new();
    for p in &phases {
        match p.ladder {
            Some(l) => {
                let e = ladders.entry(l).or_insert((p.weight, f32::INFINITY));
                e.1 = e.1.min(p.score);
            }
            None => parts.push((p.weight, p.score)),
        }
    }
    parts.extend(ladders.values().copied());
    let wsum: f32 = parts.iter().map(|x| x.0).sum();
    let gates: Vec<String> = phases.iter().flat_map(|p| p.gates.iter().map(move |g| format!("{}: {g}", p.id))).collect();
    let gated = !gates.is_empty();
    let score = if gated { 0.0 } else { parts.iter().map(|(w, s)| w * s).sum::<f32>() / wsum.max(1e-6) };
    let passed = !gated && phases.iter().all(|p| p.item >= 1);
    ScenarioResult { id: spec.id, name_ko: spec.name_ko, holdout: spec.holdout, skipped: None, phases, score, gated, gates, passed, weights_from: spec.weights_from }
}

/// The NIOSH sequential lifting index (Waters, Lu & Occhipinti 2007) over
/// scenarios done one after another in a 240-minute block: `LI` is a
/// scenario's peak torque over the peak rating (short exposure), `LImax` its
/// sustained torque over the continuous rating (long exposure, the thermal
/// side), `TF` its share of the block. `SLI = LI₁ + (LImax₁ − LI₁)·K`,
/// `K = Σ LImaxᵢ·TFᵢ / LImax₁`, task 1 the one with the largest `LImax`.
/// Above 1 the robot cannot keep the shift up. (The mapping of LI and LImax
/// onto torque ratios is this project's adaptation.)
pub fn shift_index(results: &[(&ScenarioResult, f32)]) -> Option<f32> {
    let mut tasks: Vec<(f32, f32, f32)> = results
        .iter()
        .filter(|(r, _)| r.skipped.is_none() && !r.phases.is_empty())
        .map(|(r, minutes)| {
            let li = r.phases.iter().map(|p| p.peak_ratio).fold(0.0, f32::max);
            let lmax = r.phases.iter().map(|p| p.sustained_effort).fold(0.0, f32::max).max(li);
            (li, lmax, minutes / 240.0)
        })
        .collect();
    if tasks.is_empty() {
        return None;
    }
    tasks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (li1, lmax1, _) = tasks[0];
    let k = tasks.iter().map(|t| t.1 * t.2).sum::<f32>() / lmax1.max(1e-6);
    Some(li1 + (lmax1 - li1) * k)
}

/// The development scenario phases a world posture counts as: same source and
/// every cause of the phase present.
/// Within a scenario the closest phase wins (the ladder rung nearest in
/// acceleration, slope and step).
pub fn matching_phases<'a>(lib: &'a [ScenarioSpec], post: &Posture) -> Vec<(&'a ScenarioSpec, &'a PhaseSpec)> {
    let have = causes_of(post);
    let dist = |p: &PhaseSpec| (p.posture.accel_mps2 - post.accel_mps2).abs() + (p.posture.slope_deg - post.slope_deg).abs() / 10.0 + (p.posture.step_m - post.step_m).abs() * 10.0;
    lib.iter()
        .filter(|s| !s.holdout)
        .filter_map(|s| {
            s.phases
                .iter()
                .filter(|p| p.posture.source == post.source && p.causes.iter().all(|c| have.contains(c)))
                .min_by(|a, b| dist(a).total_cmp(&dist(b)))
                .map(|p| (s, p))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    #[test]
    fn weights_sum_to_one_and_cells_score_on_the_gdi_scale() {
        let b = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
        for s in library(&b) {
            let w: f32 = s.seg_w.iter().sum();
            assert!((w - 1.0).abs() < 2e-3, "{}: {w}", s.id);
            // A ladder counts once (its rungs share one weight).
            let mut ladders = std::collections::BTreeMap::new();
            let mut total = 0.0f32;
            for p in &s.phases {
                match p.ladder {
                    Some(l) => {
                        ladders.insert(l, p.weight);
                    }
                    None => total += p.weight,
                }
            }
            let total = total + ladders.values().sum::<f32>();
            assert!((total - 1.0).abs() < 2e-3, "{}: phase weights {total}", s.id);
        }
        let mut c = Cell { source: Source::Quiet, target: Segment::Foot, axis: "x".into(), value: 1.3, reference: Some(1.0), tolerance: Some(0.1), cite: None, z: None, score: None, dynamic: false };
        scored(&mut c);
        assert!((c.z.unwrap() - 3.0).abs() < 1e-4 && (c.score.unwrap() - 70.0).abs() < 1e-3);
    }

    #[test]
    fn braking_uses_the_forward_limit_and_starting_the_narrower_backward_one() {
        let (b, s) = (composite_reference(Source::Accel, Segment::Foot, "los_frac", -1.0).unwrap(), composite_reference(Source::Accel, Segment::Foot, "los_frac", 1.0).unwrap());
        // 1 m/s² braking is 93% of the human forward limit (de Kam's single-step threshold).
        assert!((b.0 - 0.931).abs() < 0.005, "{}", b.0);
        assert!(s.0 > b.0, "the same acceleration uses more of the backward limit");
    }

    #[test]
    fn world_postures_match_the_phases_with_all_their_causes() {
        let b = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let lib = library(&b);
        let brake = Posture { source: Source::Accel, accel_mps2: -1.2, ..Posture::quiet(3.0) };
        let m = matching_phases(&lib, &brake);
        assert!(m.iter().any(|(s, p)| s.id == "S4" && p.ladder == Some("brake")), "{:?}", m.iter().map(|(s, p)| (s.id, p.id.clone())).collect::<Vec<_>>());
        let empty_brake = Posture { source: Source::Accel, accel_mps2: -1.2, ..Posture::quiet(0.0) };
        assert!(matching_phases(&lib, &empty_brake).iter().all(|(s, _)| s.id != "S4"), "no cargo, no S4");
    }

    #[test]
    fn holdout_ranges_never_touch_the_development_conditions() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        for a in [0.5f32, 1.0, 1.5, 2.5, 0.7] {
            assert!(HOLDOUT_BRAKE.iter().all(|r| a < r.0 || a > r.1), "{a}");
        }
        assert!(10.0 < HOLDOUT_SLOPE_DEG.0 && (0.17 > HOLDOUT_STEP_M.1));
        assert!(HOLDOUT_LOAD_FRAC.iter().all(|r| 0.10 < r.0 || 0.10 > r.1));
        let h = holdout(&b, 20261024, 6);
        assert_eq!(h.len(), 6);
        assert!(h.iter().all(|s| s.holdout && s.phases.len() == 1));
        assert_eq!(holdout(&b, 20261024, 6), h, "a seed gives the same scenarios");
    }
}
