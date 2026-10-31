//! Walking a load over a step. A robot with something in its hands looks at the ground it is about to walk on, a stride
//! or two ahead (0.8 to 1.5 m); the head tips down for that while it carries. What it finds there is a step of some
//! height some way off (`StepSeen`). From a stride out it bends its knees to take the step down with its legs, leans a
//! little and turns its arms back by as much, so what it holds stays level; it comes up again once it is over. What it
//! did not see it takes with straight knees, and the jolt goes up through its arms.
//!
//! The first worker's eyes are the runtime's: it renders the look down and reads the step from the depth model, and
//! reports it here (`note_step`). Sub-workers run without vision, and their look is a probe of the ground along the way.
//! Either way the look starts about a metre out, which leaves a step at its toes unseen; a step its feet have met it
//! keeps in mind, so it does not walk off the same edge twice with its knees straight.

use super::*;
use noto_core::act::carry;

/// Where the look down reaches on the ground ahead (m).
pub const STEP_LOOK_M: [f32; 2] = [0.8, 1.5];
/// A height change smaller than this is not a step (m) [추정: under a hand's thickness].
pub const STEP_MIN_M: f32 = 0.03;
/// How far it walks between looks (m).
pub const STEP_LOOK_EVERY_M: f32 = 0.45;
/// How long the knees take to bend or straighten (s) [추정].
pub const CROUCH_S: f32 = 0.35;
/// How fast a robot walks onto a step it has seen, with a load on (m/s) [추정].
pub const STEP_SPEED_MPS: f32 = 0.45;
/// The noise of one look at a step's height (m) [추정: a depth pixel at 1 m].
pub const STEP_SEEN_SIGMA_M: f32 = 0.008;
/// How many steps a body keeps in mind, and how close to one its way has to pass (m).
const STEPS_KEPT: usize = 32;
const STEP_ON_WAY_M: f32 = 0.35;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StepSeen {
    /// Where the edge is, on the way (world).
    pub at: (f32, f32),
    /// Up is positive (m).
    pub height_m: f32,
    pub source: &'static str,
}

/// A step its feet have met, and which way it went.
#[derive(Clone, Copy, Debug)]
struct Felt {
    step: StepSeen,
    dir: (f32, f32),
}

#[derive(Clone, Debug, Default)]
pub struct Stride {
    pub seen: Option<StepSeen>,
    /// How far the hips are down now (m).
    pub crouch_m: f32,
    since_look_m: f32,
    /// The eyes are the runtime's; the world does not look for itself.
    pub external_eye: bool,
    felt: Vec<Felt>,
}

impl World {
    /// How far ahead the knees start to bend (m): one stride.
    fn crouch_ahead_m(&self) -> f32 {
        self.walk.plan.as_ref().map_or(0.6, |(g, _, _)| 2.0 * g.step_m).max(0.5) + 0.2
    }

    /// What the runtime's look down found: a step `height_m` high, `dist_m` ahead along the way the robot faces.
    pub fn note_step(&mut self, height_m: f32, dist_m: f32, source: &'static str, worker: u32, events: &mut Vec<TimedEvent>) {
        let (s, c) = self.agent.yaw.sin_cos();
        let at = (self.agent.x + c * dist_m, self.agent.z + s * dist_m);
        self.saw_step(StepSeen { at, height_m, source }, dist_m, worker, events);
    }

    fn saw_step(&mut self, seen: StepSeen, dist_m: f32, worker: u32, events: &mut Vec<TimedEvent>) {
        let new = self.stride.seen.map_or(true, |s| (s.at.0 - seen.at.0).hypot(s.at.1 - seen.at.1) > 0.4);
        self.stride.seen = Some(seen);
        if new {
            // A base with wheels has no knees: all it can do for a step is to take it slowly.
            let crouch = if self.body.legged() { carry::crouch_for(seen.height_m, &self.body.dims) } else { 0.0 };
            let then = if self.body.legged() { format!("무릎 {:.0} cm 굽힘 준비", crouch * 100.0) } else { "감속 준비".to_string() };
            let label = format!("전방 {dist_m:.1} m 단차 {:+.0} cm ({}) → {then}", seen.height_m * 100.0, seen.source);
            events.push(self.stamp(WorldEvent::StepSeen { worker, height_m: seen.height_m, dist_m, crouch_m: crouch, source: seen.source.into(), label }));
        }
    }

    /// The world's own look along the way: the first change of ground height 0.8 to 1.5 m on.
    fn probe_step(&mut self, f: &Follow) -> Option<(StepSeen, f32)> {
        let here = PoseCheck::ground_m(&self.terrain, self.agent.x, self.agent.z);
        let n = ((STEP_LOOK_M[1] - STEP_LOOK_M[0]) / 0.05).ceil() as usize;
        let found = (0..=n).map(|i| STEP_LOOK_M[0] + 0.05 * i as f32).map(|d| (d, f.point_at(f.s + d))).find(|(_, p)| (PoseCheck::ground_m(&self.terrain, p.0, p.1) - here).abs() >= STEP_MIN_M)?;
        let u = self.act_rng.next_f32() + self.act_rng.next_f32() + self.act_rng.next_f32() - 1.5;
        let h = PoseCheck::ground_m(&self.terrain, found.1 .0, found.1 .1) - here + u * STEP_SEEN_SIGMA_M;
        Some((StepSeen { at: found.1, height_m: h, source: "지면 프로브" }, found.0))
    }

    /// The nearest step in mind on the way ahead, as far as the knees would start to bend for it, and how far off.
    fn felt_ahead(&self, f: &Follow) -> Option<(StepSeen, f32)> {
        let (x, z) = (self.agent.x, self.agent.z);
        let (sn, cs) = self.agent.yaw.sin_cos();
        let n = (self.crouch_ahead_m() / 0.1).ceil() as usize;
        (0..=n).map(|i| 0.1 * i as f32).find_map(|d| {
            let p = f.point_at(f.s + d);
            let hit = self.stride.felt.iter().find(|k| (k.step.at.0 - p.0).hypot(k.step.at.1 - p.1) <= STEP_ON_WAY_M && (k.step.at.0 - x) * cs + (k.step.at.1 - z) * sn > 0.2)?;
            // Walked the other way a step down is a step up.
            let way = hit.dir.0 * cs + hit.dir.1 * sn;
            Some((StepSeen { height_m: hit.step.height_m * way.signum(), ..hit.step }, d))
        })
    }

    /// One moment of the walk with a load on: looks, and bends or straightens the knees. Called with the distance just
    /// walked.
    pub(super) fn watch_steps(&mut self, f: &Follow, moved: f32, dt: f32, worker: u32, events: &mut Vec<TimedEvent>) {
        let carrying = self.carried_kg() > AMOUNT_EPS;
        let legged = self.body.legged();
        if carrying && !self.stride.external_eye {
            self.stride.since_look_m += moved;
            if self.stride.since_look_m >= STEP_LOOK_EVERY_M {
                self.stride.since_look_m = 0.0;
                if let Some((seen, d)) = self.probe_step(f) {
                    self.saw_step(seen, d, worker, events);
                }
            }
        }
        if carrying && legged && self.stride.seen.is_none() {
            if let Some((seen, d)) = self.felt_ahead(f) {
                self.saw_step(seen, d, worker, events);
            }
        }
        // A step it has walked past without meeting (a misread, or it turned away) is forgotten.
        let (sn, cs) = self.agent.yaw.sin_cos();
        if self.stride.seen.is_some_and(|s| (s.at.0 - self.agent.x) * cs + (s.at.1 - self.agent.z) * sn < -0.3) {
            self.stride.seen = None;
        }
        let want = match self.stride.seen {
            Some(s) if legged && carrying => {
                let d = (s.at.0 - self.agent.x).hypot(s.at.1 - self.agent.z);
                if d <= self.crouch_ahead_m() {
                    carry::crouch_for(s.height_m, &self.body.dims)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        let rate = carry::CROUCH_MAX_M / CROUCH_S;
        let step = (want - self.stride.crouch_m).clamp(-rate * dt, rate * dt);
        self.stride.crouch_m += step;
    }

    /// The speed a robot with a load on keeps to for a step it has seen near ahead (m/s).
    pub(super) fn step_speed_limit(&self) -> Option<f32> {
        let s = self.stride.seen?;
        (self.carried_kg() > AMOUNT_EPS && (s.at.0 - self.agent.x).hypot(s.at.1 - self.agent.z) <= self.crouch_ahead_m()).then_some(STEP_SPEED_MPS)
    }

    /// The base takes the step of `e`: the shock of it, with however far the knees were bent, goes to the load.
    pub(super) fn take_step(&mut self, e: &EdgePlan, worker: u32, events: &mut Vec<TimedEvent>) {
        let h = {
            let top = |c: Cell| self.terrain.height_m(c.0, c.1) + self.terrain.curb_m(c.0, c.1);
            top(e.to) - top(e.from)
        };
        let shock = if self.body.legged() { carry::step_shock(&self.body, h, self.stride.crouch_m) } else { carry::curb_shock(&self.body, h, self.agent.speed_mps) };
        let seen = self.stride.seen.is_some();
        self.stride.seen = None;
        let (a, b) = (crate::terrain::cell_center(e.from.0, e.from.1), crate::terrain::cell_center(e.to.0, e.to.1));
        let at = (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
        let along = (b.0 - a.0).hypot(b.1 - a.1).max(1e-3);
        self.stride.felt.retain(|k| (k.step.at.0 - at.0).hypot(k.step.at.1 - at.1) > 0.1);
        if self.stride.felt.len() >= STEPS_KEPT {
            self.stride.felt.remove(0);
        }
        self.stride.felt.push(Felt { step: StepSeen { at, height_m: h, source: "발로 겪은 턱" }, dir: ((b.0 - a.0) / along, (b.1 - a.1) / along) });
        if self.carried_kg() <= AMOUNT_EPS {
            return;
        }
        self.ride.pending_az = shock.a_z;
        let how = match (self.stride.crouch_m > 0.01, seen, self.body.legged()) {
            (true, ..) => format!("무릎 {:.0} cm 굽혀", self.stride.crouch_m * 100.0),
            (_, false, _) => "보지 못하고".to_string(),
            (_, true, true) => "굽히지 못하고".to_string(),
            (_, true, false) => "감속해".to_string(),
        };
        let label = format!("단차 {:+.0} cm를 {how} 넘음: 수직 충격 {:.1} m/s²", h * 100.0, shock.a_z);
        events.push(self.stamp(WorldEvent::StepTaken { worker, height_m: h, crouch_m: self.stride.crouch_m, a_z: shock.a_z, label }));
    }

    /// Bends a carrying body for the step ahead: hips down, a little lean, arms turned back by the lean so the load stays
    /// level.
    pub(super) fn crouched(&self, mut pose: hkin::KinPose) -> hkin::KinPose {
        let c = self.stride.crouch_m;
        if c > 1e-4 {
            let lean = carry::crouch_lean_deg(c);
            pose.hip[1] -= c;
            pose.pelvis_deg += lean;
            for s in pose.shoulder_deg.iter_mut() {
                *s -= lean;
            }
        }
        pose
    }
}

use noto_core::kin as hkin;

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    /// A walker in front of a raised strip of ground, a step `cm` high a few metres ahead along +x.
    fn at_a_step(cm: u8, down: bool) -> (World, (f32, f32)) {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = BaseKind::HumanFoot;
        let mut w = World::new(WorldConfig { seed: 11, cells: 64, noto: hand, terrain: crate::terrain::TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        w.nodes.clear();
        w.drops.clear();
        w.fixtures.clear();
        let (cx, cz) = w.agent_cell();
        for z in 0..w.terrain.d {
            for x in 0..w.terrain.w {
                w.terrain.set_height_cm(x, z, 0);
                let raised = (x > cx + 6) == !down;
                w.terrain.set_curb_cm(x, z, if raised { cm } else { 0 });
            }
        }
        (w.agent.x, w.agent.z) = cell_center(cx, cz);
        w.agent.yaw = 0.0;
        let goal = cell_center(cx + 12, cz);
        (w, goal)
    }

    fn walk(w: &mut World, goal: (f32, f32)) -> Vec<TimedEvent> {
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        let mut ev = Vec::new();
        for _ in 0..600 {
            ev.extend(w.tick());
            if !w.busy() {
                break;
            }
        }
        ev
    }

    #[test]
    fn a_step_seen_ahead_is_taken_with_bent_knees_and_the_parcel_stays_in_the_hands() {
        let (mut w, goal) = at_a_step(15, true);
        add_stock(&mut w.agent.carried, ItemKind::Parcel, 1.0);
        let mut bent = 0.0f32;
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        let mut ev = Vec::new();
        for _ in 0..600 {
            ev.extend(w.tick());
            bent = bent.max(w.stride.crouch_m);
            if !w.busy() {
                break;
            }
        }
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::StepSeen { .. })), "it saw the step coming");
        let taken = ev.iter().find_map(|e| match &e.event { WorldEvent::StepTaken { crouch_m, a_z, .. } => Some((*crouch_m, *a_z)), _ => None }).expect("it took the step");
        assert!(taken.0 > 0.08, "{taken:?}");
        assert!(bent > 0.08 && bent <= carry::CROUCH_MAX_M + 1e-3, "{bent}");
        assert_eq!(w.carried_count(ItemKind::Parcel), 1, "{ev:?}");
        assert!(w.stride.crouch_m < 0.02, "up again once it is over: {}", w.stride.crouch_m);
    }

    #[test]
    fn a_step_not_seen_jolts_the_load_much_harder() {
        let (mut seen, goal) = at_a_step(15, true);
        add_stock(&mut seen.agent.carried, ItemKind::Log, 1.0);
        let ev = walk(&mut seen, goal);
        let shock = |ev: &[TimedEvent]| ev.iter().find_map(|e| match &e.event { WorldEvent::StepTaken { a_z, .. } => Some(*a_z), _ => None }).unwrap_or(0.0);
        let with_eyes = shock(&ev);
        let (mut blind, goal) = at_a_step(15, true);
        add_stock(&mut blind.agent.carried, ItemKind::Log, 1.0);
        blind.stride.external_eye = true;
        let ev = walk(&mut blind, goal);
        let without = shock(&ev);
        assert!(without > 4.0 * with_eyes, "blind {without} vs seen {with_eyes}");
    }

    #[test]
    fn the_crouch_lowers_the_hips_and_the_hands_stay_level_with_them() {
        let (mut w, _) = at_a_step(15, true);
        add_stock(&mut w.agent.carried, ItemKind::Parcel, 1.0);
        let stand = w.current_kin_pose();
        let up = hkin::solve_fast(&w.body, &stand, &hkin::KinLoads::default());
        w.stride.crouch_m = 0.12;
        let low = w.crouched(stand);
        let down = hkin::solve_fast(&w.body, &low, &hkin::KinLoads::default());
        let lean = carry::crouch_lean_deg(0.12).to_radians();
        assert!(lean > 0.03);
        let fall = up.hands[0][2] - down.hands[0][2];
        assert!((fall - 0.12).abs() < 0.05, "the hands go down with the hips: {fall}");
    }
}
