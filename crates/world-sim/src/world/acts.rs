//! The world's side of the actions (`noto_core::act`): picking up and setting
//! down play their time series instead of a fixed timer, a legged robot's
//! steps are drawn one by one as it walks (slips, falls), and a wheeled or
//! tracked robot can push an item along a route with its bumper.
//!
//! A pick-up used to succeed whenever the static grasp table said the hand
//! could hold the item, with the hand at chest height and the item on the
//! ground. Now the body must bend to reach it (a robot that cannot reach the
//! ground cannot pick anything up off it), the aim is off by the head camera's
//! and the arm's error, the pads' friction is drawn per item, and the lift can
//! slip; the item then falls back where it was. The draws come from their own
//! random stream so the world's layout does not change with them.

use super::*;
use noto_core::act::focus::{self, FocusCtx, FocusFrame};
use noto_core::act::material::{self, ItemMaterial};
use noto_core::act::reach::ReachMode;
use noto_core::act::{gait, grasp as agrasp, push as apush, ActEvent, ActKind, ActRollout};
use noto_core::kin::{self as hkin, KinLoads, KinPose};

/// Time to get back up after a fall, seconds (est: a small humanoid's stand-up routine).
pub const GET_UP_S: f32 = 6.0;
/// How far the robot may be from where a push starts and still begin it, metres
/// (a cell centre is within 0.354 m of any point; the last bit is crept at [`SHUFFLE_MPS`]).
pub const PUSH_START_SLACK_M: f32 = 0.40;
/// Speed of the short shuffle that puts an item at the planned stand-off before a pick, m/s (est).
pub const SHUFFLE_MPS: f32 = 0.2;
/// Farthest a reach to the side shuffles (forward or across, without turning)
/// to bring the item to its place beside the right foot, metres (est: a short side step).
pub const SIDE_SHUFFLE_MAX_M: f32 = 0.5;
/// Mean height of a piece on a waypoint's pile, metres (the 0.10–0.15 m items).
pub const PILE_PIECE_H_M: f32 = 0.12;

/// An action being played out.
#[derive(Clone, Debug)]
pub struct ActPlay {
    pub rollout: ActRollout,
    /// World time it started at.
    pub start_s: f64,
    /// Seconds of the rollout played so far, and up to where its events are out.
    pub elapsed_s: f32,
    pub played_s: f32,
    /// The item it handles, and how much of one item the piece is.
    pub object: Option<(ItemKind, f32)>,
    /// Where the rollout's base frame was in the world: x, z, yaw.
    pub origin: (f32, f32, f32),
}

impl ActPlay {
    fn new(rollout: ActRollout, start_s: f64, object: Option<(ItemKind, f32)>, origin: (f32, f32, f32)) -> ActPlay {
        ActPlay { rollout, start_s, elapsed_s: 0.0, played_s: -1.0, object, origin }
    }

    /// The rollout time now.
    pub fn t(&self) -> f32 {
        self.elapsed_s.clamp(0.0, self.rollout.duration_s)
    }
}

/// Walking state of a legged robot along a move.
#[derive(Clone, Debug, Default)]
pub struct WalkState {
    /// Distance walked since the last foot came down, metres.
    pub since_step_m: f32,
    pub steps: u32,
    pub stance: usize,
    pub slips: u32,
    /// The gait of the edge being walked, with its ground and slope.
    pub plan: Option<(gait::GaitPlan, Ground, i32)>,
    /// The friction drawn for the last step that came down (the foot views show it).
    pub last_mu: Option<f32>,
}

/// A push along a route.
#[derive(Clone, Debug)]
pub struct PushState {
    pub drop: u32,
    /// The item's path, world (x, z), first point where it is now.
    pub path: Vec<(f32, f32)>,
    pub length_m: f32,
    /// Distance the item has moved along the path.
    pub done_m: f32,
    /// How far along the path the push ends (the whole path, or where it failed).
    pub end_m: f32,
    pub lead_m: f32,
}

/// What an item is made of.
pub fn material_of(kind: ItemKind) -> ItemMaterial {
    match kind {
        ItemKind::Log => ItemMaterial::Wood,
        ItemKind::Stone => ItemMaterial::Stone,
        ItemKind::Coal => ItemMaterial::Coal,
        ItemKind::IronOre => ItemMaterial::IronOre,
        ItemKind::Parcel => ItemMaterial::Cardboard,
    }
}

/// The stereo camera of each head preset: focal length (px), baseline (m) and
/// subpixel error (research §1.4: the D435 is 424 px, 50 mm, 0.08 px; the
/// OAK-D Lite's 75 mm baseline at 640 px over 73°, subpixel as the D435's, est).
pub fn camera_of(sensor: &str) -> (f32, f32, f32) {
    if sensor.contains("OAK") {
        (432.0, 0.075, 0.08)
    } else {
        (424.0, 0.050, 0.08)
    }
}

/// Positioning error of the arm's end, metres (est: a backdrivable QDD arm
/// with absolute encoders 5 mm; bus servos with gear backlash 10 mm).
fn arm_error_m(arm: &noto_core::parts::ArmPreset) -> f32 {
    if arm.shoulder.backdrivable {
        0.005
    } else {
        0.010
    }
}

impl World {
    /// 1σ of the hand's aim at an item lying in front of the robot, metres: the
    /// head camera's depth noise at that range and the arm's own error.
    pub fn aim_sigma_m(&self) -> f32 {
        let (f, b, sub) = camera_of(self.body.head.sensor);
        let range = (SHOULDER_RANGE_M * SHOULDER_RANGE_M + 0.3f32 * 0.3).sqrt();
        let depth = material::stereo_depth_sigma(range, f, b, sub);
        (depth * depth + arm_error_m(&self.body.arm).powi(2)).sqrt()
    }

    /// The friction a hand's pads believe they have on an item (the table mean).
    pub fn pad_mu(&self, kind: ItemKind) -> f32 {
        material::pad_on(material_of(kind)).mean
    }

    /// Plays a pick-up out: the reach plan, the grasp, the lift. Fails right
    /// away when the body cannot bring its hand down to the item.
    ///
    /// The robot puts itself and its hand where it believes the item is
    /// (`aim`, its believed frame; the item itself when it sees it): the
    /// shuffle brings that point to the planned stand-off, and what its memory
    /// got wrong is the hand's aim offset. In front it then turns to that
    /// point; beside the right foot (`side`) it does not turn. Returns the
    /// time it takes and the point it turns to face.
    pub(super) fn start_grasp(&mut self, drop: u32, two_arms: bool, aim: Option<(f32, f32)>, side: bool, memory_sigma_m: f32) -> Result<(f32, Option<(f32, f32)>), ActionError> {
        let d = self.drop_item(drop).cloned().ok_or(ActionError::UnknownTarget)?;
        let spec = d.kind.spec();
        let size = spec.piece_size_m(d.amount);
        let seed = self.act_rng.next_u64();
        let ground = self.ground_at(self.terrain.cell_of(d.x, d.z).unwrap_or((0, 0))).surface();
        // Where the robot's body takes "where I believe the item is".
        let at = aim.map(|(x, z)| self.true_of(x, z)).unwrap_or((d.x, d.z));
        let heading = if side { self.agent.yaw } else { (at.1 - self.agent.z).atan2(at.0 - self.agent.x) };
        let (sh, ch) = heading.sin_cos();
        // Believed minus actual, in the frame the robot will reach in (x forward, y left).
        let (ox, oz) = (at.0 - d.x, at.1 - d.z);
        let offset = [ox * ch + oz * sh, ox * sh - oz * ch];
        let mode = if side { ReachMode::Side } else { ReachMode::Front };
        let inp = agrasp::GraspInput {
            body: &self.body,
            hand: &self.hand,
            material: material_of(d.kind),
            mass_kg: d.mass_kg(),
            size_m: size,
            ground,
            two_arms,
            cargo_kg: self.carried_kg(),
            aim_sigma_m: self.aim_sigma_m(),
            seed,
            mode,
            aim_offset_m: offset,
            memory_sigma_m,
        };
        let mut rollout = agrasp::grasp(&inp);
        if matches!(rollout.outcome.code, Some("reach_ground") | Some("reach_side")) {
            let need = size[1] + noto_core::act::reach::PALM_CLEARANCE_M;
            return Err(ActionError::CannotReach { lowest_m: rollout.metric("lowest_palm_z").unwrap_or(0.0), need_m: need, reason: rollout.outcome.reason_ko.clone().unwrap_or_default() });
        }
        // A short shuffle puts the (believed) item at the planned stand-off first.
        let planned = rollout.samples.first().and_then(|s| s.object).map_or([0.3, 0.0, 0.0], |o| o);
        let (step_x, step_z, step) = if side {
            // Forward or across, without turning: the believed item's place in the robot's frame against the plan's.
            let (rx, rz) = (at.0 - self.agent.x, at.1 - self.agent.z);
            let (fwd, left) = (rx * ch + rz * sh, rx * sh - rz * ch);
            let (df, dl) = (fwd - planned[0], left - planned[1]);
            let n = df.hypot(dl);
            if n > SIDE_SHUFFLE_MAX_M {
                return Err(ActionError::OutOfReach { distance_m: n, reach_m: SIDE_SHUFFLE_MAX_M });
            }
            (df * ch + dl * sh, df * sh - dl * ch, n)
        } else {
            let dist = (at.0 - self.agent.x).hypot(at.1 - self.agent.z);
            let k = (dist - planned[0]) / dist.max(1e-3);
            ((at.0 - self.agent.x) * k, (at.1 - self.agent.z) * k, (dist - planned[0]).abs())
        };
        let shuffle = step / SHUFFLE_MPS;
        // The shuffle walks on ground the robot can stand on: within its cell, or
        // onto a neighbour it could drive onto. Otherwise (a rock column, a ledge)
        // the item is out of the hand's reach from here and the robot must go round.
        if shuffle > 0.05 && !self.can_step_to(self.agent.x + step_x, self.agent.z + step_z) {
            let dist = (at.0 - self.agent.x).hypot(at.1 - self.agent.z);
            return Err(ActionError::OutOfReach { distance_m: dist, reach_m: (dist - step).max(0.0) });
        }
        if shuffle > 0.05 {
            for s in rollout.samples.iter_mut() {
                s.t += shuffle;
            }
            for e in rollout.events.iter_mut() {
                e.t += shuffle;
            }
            rollout.duration_s += shuffle;
            rollout.metrics.push(("shuffle_s", shuffle));
            self.agent.x += step_x;
            self.agent.z += step_z;
        }
        let dur = rollout.duration_s;
        self.act = Some(ActPlay::new(rollout, self.time_s, Some((d.kind, d.amount)), (self.agent.x, self.agent.z, heading)));
        Ok((dur, (!side).then_some(at)))
    }

    /// Whether a short step of the body to `(x, z)` stays on ground it can
    /// stand on: its own cell, or a free neighbour it could drive onto now
    /// (the route there is that one edge).
    pub(super) fn can_step_to(&self, x: f32, z: f32) -> bool {
        let Some(to) = self.terrain.cell_of(x, z) else { return false };
        let from = self.agent_cell();
        if to == from {
            return true;
        }
        let near = (to.0 as i32 - from.0 as i32).abs() <= 1 && (to.1 as i32 - from.1 as i32).abs() <= 1;
        near && !self.blocked_cells().contains(&to) && self.plan_route(Goal::Cell(to)).is_some_and(|p| p.edges.len() == 1)
    }

    /// Plays a release out for every piece being set down, one after another; returns the total time.
    pub(super) fn start_release(&mut self, pieces: &[(ItemKind, f32)], waypoint: Option<u32>) -> f32 {
        let mut all: Option<ActRollout> = None;
        let mut cargo = self.carried_kg();
        let pile_n = waypoint.and_then(|w| self.waypoint(w)).map_or(0usize, |w| render::stock_pieces(&w.stock).len());
        let ground = match waypoint.and_then(|w| self.waypoint(w)) {
            Some(w) => self.ground_at(self.terrain.cell_of(w.x, w.z).unwrap_or((0, 0))).surface(),
            None => self.ground_here().surface(),
        };
        for (i, (k, a)) in pieces.iter().enumerate() {
            let mass = k.spec().mass_kg * a;
            cargo = (cargo - mass).max(0.0);
            let surface_z = if waypoint.is_some() { ((pile_n + i) / 4) as f32 * PILE_PIECE_H_M } else { 0.0 };
            let inp = agrasp::ReleaseInput { body: &self.body, material: material_of(*k), mass_kg: mass, size_m: k.spec().piece_size_m(*a), ground, surface_z_m: surface_z, cargo_kg: cargo, aim_sigma_m: self.aim_sigma_m(), seed: self.act_rng.next_u64() };
            let r = agrasp::release(&inp);
            all = Some(match all {
                None => r,
                Some(mut acc) => {
                    let t0 = acc.duration_s;
                    acc.samples.extend(r.samples.into_iter().map(|mut s| {
                        s.t += t0;
                        s
                    }));
                    acc.events.extend(r.events.into_iter().map(|mut e| {
                        e.t += t0;
                        e
                    }));
                    acc.duration_s += r.duration_s;
                    acc.metrics.extend(r.metrics);
                    acc
                }
            });
        }
        let Some(r) = all else { return DROP_TIME_S };
        let dur = r.duration_s;
        let first = pieces.first().copied();
        self.act = Some(ActPlay::new(r, self.time_s, first, (self.agent.x, self.agent.z, self.agent.yaw)));
        dur
    }

    /// Plays the action on by `dt` seconds and puts out its events up to there.
    pub(super) fn act_advance(&mut self, dt: f32, events: &mut Vec<TimedEvent>) {
        let Some(p) = self.act.as_mut() else { return };
        p.elapsed_s += dt;
        let t = p.t();
        let out: Vec<WorldEvent> = p.rollout.events_between(p.played_s, t).map(|e| act_event(p.rollout.kind, e)).collect();
        p.played_s = t;
        for ev in out {
            events.push(self.stamp(ev));
        }
    }

    /// Plays the action out to its end: its work is done, and what is left of
    /// its time is the rounding of the ticks (the last moment sits on the end).
    pub(super) fn act_finish(&mut self, events: &mut Vec<TimedEvent>) {
        let Some(p) = self.act.as_mut() else { return };
        let end = p.rollout.duration_s;
        let out: Vec<WorldEvent> = p.rollout.events_between(p.played_s, end).map(|e| act_event(p.rollout.kind, e)).collect();
        p.elapsed_s = p.elapsed_s.max(end);
        p.played_s = end;
        for ev in out {
            events.push(self.stamp(ev));
        }
    }

    /// Ends the playing action: keeps it for the focus views and returns how it went.
    pub(super) fn end_act(&mut self) -> Option<ActPlay> {
        let p = self.act.take()?;
        self.last_act = Some(p.clone());
        Some(p)
    }

    /// The grasp failed: the item falls back near where it lay (bounced by its restitution).
    pub(super) fn grasp_failed(&mut self, drop: u32, p: &ActPlay) {
        let h = p.rollout.metric("drop_height_m").unwrap_or(0.0);
        if h <= 0.0 {
            return;
        }
        let Some(kind) = p.object.map(|o| o.0) else { return };
        let ground = self.ground_here().surface();
        let e = material::restitution(material_of(kind), ground);
        // The rebound throws it a little to the side: 2·e²·h·sin 2θ with θ up to 30° (as a release).
        let theta = self.act_rng.next_f32() * 30f32.to_radians();
        let dir = self.act_rng.next_f32() * std::f32::consts::TAU;
        let r = 2.0 * e * e * h * (2.0 * theta).sin();
        if let Some(d) = self.drops.iter_mut().find(|d| d.id == drop) {
            d.x += r * dir.cos();
            d.z += r * dir.sin();
        }
        // Bounced off a porch's edge: it now lies on the ground below.
        if let Some(i) = self.drops.iter().position(|d| d.id == drop) {
            let d = &self.drops[i];
            if !d.support.is_ground() && (self.surface_m(d.x, d.z) - d.y_m).abs() > 0.03 {
                self.drops[i].support = crate::site::Support::Ground;
            }
        }
    }

    /// Where a dropped piece lands from a release (the rollout's own landing point, rotated into the world).
    pub(super) fn release_landing(&self, p: &ActPlay, index: usize) -> Option<(f32, f32)> {
        let xs: Vec<f32> = p.rollout.metrics.iter().filter(|(k, _)| *k == "land_x").map(|(_, v)| *v).collect();
        let ys: Vec<f32> = p.rollout.metrics.iter().filter(|(k, _)| *k == "land_y").map(|(_, v)| *v).collect();
        let (lx, ly) = (*xs.get(index)?, *ys.get(index)?);
        let (ox, oz, yaw) = p.origin;
        let (s, c) = yaw.sin_cos();
        // Base frame: x forward, y left; world: z is to the robot's right of +x when yaw = 0.
        Some((ox + c * lx + s * ly, oz + s * lx - c * ly))
    }

    /// A legged robot's steps along the distance it just walked: each step
    /// that comes down draws the ground's friction and a slip; a slip the robot
    /// cannot catch ends the move with a fall.
    pub(super) fn walk_steps(&mut self, walked_m: f32, e: &nav::EdgePlan, events: &mut Vec<TimedEvent>) -> Option<String> {
        if !self.body.legged() || walked_m <= 0.0 {
            return None;
        }
        let ground = self.ground_at(e.to);
        let props = *ground.surface().props();
        let slope = e.slope_deg.round() as i32;
        let need_plan = self.walk.plan.as_ref().map_or(true, |(_, g, s)| *g != ground || *s != slope);
        if need_plan {
            let spec = self.drive_spec();
            let g = gait::plan(&self.body, spec.max_speed_mps, noto_core::surface::available_mu(self.body.base, &props), e.slope_deg);
            self.walk.plan = Some((g, ground, slope));
        }
        let (g, _, _) = self.walk.plan.clone().expect("set above");
        self.walk.since_step_m += walked_m;
        while self.walk.since_step_m >= g.step_m {
            self.walk.since_step_m -= g.step_m;
            self.walk.steps += 1;
            self.walk.stance = 1 - self.walk.stance;
            let mu = draw_lognormal(&mut self.act_rng, noto_core::surface::available_mu(self.body.base, &props), gait::GROUND_MU_COV);
            self.walk.last_mu = Some(mu);
            let margin = mu - g.rcof;
            let p = noto_core::surface::slip_probability(margin);
            if self.act_rng.next_f32() >= p {
                continue;
            }
            self.walk.slips += 1;
            let (fell, slide) = gait::slip_outcome(&g, mu);
            // The foot slid without the legs' kinematics seeing it: the odometry counts it as travel.
            self.odom.foot_slid(slide);
            let side = if self.walk.stance == 0 { "오른" } else { "왼" };
            if fell {
                let label = format!("{side}발이 미끄러져 넘어짐: 마찰 {mu:.2} < 필요 {:.2}, {:.0} mm", g.rcof, slide * 1000.0);
                events.push(self.stamp(WorldEvent::Act { act: "walk".into(), kind: "fall".into(), label: label.clone(), ok: false, value: Some(slide), parts: vec!["foot".into(), "foot_joints".into(), "waist".into()] }));
                self.fallen_until = Some(self.time_s + GET_UP_S as f64);
                return Some(label);
            }
            let label = format!("{side}발 미끄러짐 {:.0} mm → 회복 (마찰 {mu:.2} < 필요 {:.2})", slide * 1000.0, g.rcof);
            events.push(self.stamp(WorldEvent::Act { act: "walk".into(), kind: "foot_slip".into(), label, ok: true, value: Some(slide), parts: vec!["foot".into(), "foot_joints".into()] }));
        }
        None
    }

    /// The whole body at this instant from the actions: the playing action's
    /// pose, a walking robot's step, `None` when neither applies.
    pub(super) fn act_pose(&self) -> Option<(KinPose, KinLoads, Source)> {
        if let Some(p) = &self.act {
            let s = p.rollout.sample_at(p.t())?;
            let src = match p.rollout.kind {
                ActKind::Push | ActKind::Drive => Source::Quiet,
                _ => Source::ArmLift,
            };
            return Some((s.pose, s.loads, src));
        }
        if self.body.legged() && matches!(&self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.0 {
            let (g, _, _) = self.walk.plan.clone()?;
            let u = (self.walk.since_step_m / g.step_m.max(1e-3)).clamp(0.0, 1.0);
            let (pose, loads, _) = gait::pose_at(&self.body, &g, self.walk.stance, u, self.carried_kg());
            return Some((pose, loads, Source::FootLift));
        }
        None
    }

    /// The pose of the actions for the renderer (`None`: the joint dynamics draw it).
    pub(super) fn act_body_pose(&self) -> Option<BodyPose> {
        let (pose, loads, src) = self.act_pose()?;
        let r = hkin::solve(&self.body, &pose, &loads);
        Some(r.body_pose(0.0, src))
    }

    /// Every focus view of this instant: the playing action's sample (or the
    /// last one's end), the walking step, or the robot standing as it is.
    pub fn focus_frame(&self) -> FocusFrame {
        let walking = self.act.is_none() && self.body.legged && matches!(&self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.0;
        let (pose, loads, kind, sample, event, object) = match (&self.act, &self.last_act) {
            (Some(p), _) => {
                let t = p.t();
                let s = p.rollout.sample_at(t);
                let ev = p.rollout.events.iter().filter(|e| e.t <= t + 1e-6).last();
                (s.map_or(KinPose::stand(self.carried_kg()), |s| s.pose), s.map_or(KinLoads::default(), |s| s.loads), Some(p.rollout.kind), s.cloned(), ev.cloned(), p.object)
            }
            // Just after an action, the robot still stands as it ended (for the keyframe that follows it).
            (None, Some(p)) if self.active.is_none() && self.time_s - (p.start_s + p.rollout.duration_s as f64) < 3.0 => {
                let t = p.t();
                let s = p.rollout.sample_at(t);
                let ev = p.rollout.events.last();
                (s.map_or(KinPose::stand(self.carried_kg()), |s| s.pose), s.map_or(KinLoads::default(), |s| s.loads), Some(p.rollout.kind), s.cloned(), ev.cloned(), p.object)
            }
            _ => match self.act_pose() {
                Some((pose, loads, _)) => (pose, loads, Some(ActKind::Walk), None, None, None),
                None => {
                    let moving = matches!(&self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.0;
                    let pose = KinPose { base_pitch_deg: if self.body.legged() { 0.0 } else { self.posture_now().0.slope_deg }, ..KinPose::stand(self.carried_kg()) };
                    (pose, KinLoads::default(), moving.then_some(ActKind::Drive), None, None, None)
                }
            },
        };
        let r = hkin::solve_fast(&self.body, &pose, &loads);
        let props = self.ground_here().surface().props();
        let drive = (!self.body.legged()).then(|| noto_core::act::drive::state(&self.body, props, self.posture_now().0.slope_deg, 0.0, self.carried_kg(), 0.0));
        let obj = object.map(|(k, a)| {
            let s = k.spec().piece_size_m(a);
            ([0.5 * s[0], 0.5 * s[1], 0.5 * s[2]], k.spec().mass_kg * a)
        });
        focus::frame(&FocusCtx { body: &self.body, hand: &self.hand, report: &r, sample: sample.as_ref(), action: kind, event: event.as_ref(), ground_mu: noto_core::surface::available_mu(self.body.base, props), drive, object: obj, step_mu: if walking { self.walk.last_mu } else { None } })
    }

    /// Whether this body brings its palm down to a piece of `amount` items
    /// lying on the ground (the bend plan of `noto_core::act::reach`), and the
    /// reason when it does not.
    pub fn reach_ground(&self, kind: ItemKind, amount: f32) -> Result<(), String> {
        let size = kind.spec().piece_size_m(amount);
        let goal = noto_core::act::reach::ReachGoal::ground_object(size[1], kind.spec().mass_kg * amount, false, self.carried_kg());
        let plan = noto_core::act::reach::plan(&self.body, &goal);
        if plan.ok {
            return Ok(());
        }
        // A load the bend cannot hold with one hand may still go up with two.
        let both = noto_core::act::reach::plan(&self.body, &noto_core::act::reach::ReachGoal { two_arms: true, ..goal });
        if both.ok {
            Ok(())
        } else {
            Err(plan.reason_ko.unwrap_or_else(|| "바닥에 손이 닿지 않음".into()))
        }
    }

    /// Where an item of `kind` (a piece of `amount` items) lies relative to the
    /// robot when its right hand reaches it beside the right foot without
    /// turning (`noto_core::act::reach::plan_side` with the cargo aboard):
    /// forward and left (negative: right), metres. `None` when this body
    /// cannot reach it there. Planned once per item, piece and cargo.
    pub fn side_reach(&self, kind: ItemKind, amount: f32) -> Option<(f32, f32)> {
        let key = (kind, (amount * 1000.0).round() as i32, (self.carried_kg() * 10.0).round() as i32);
        if let Some(v) = self.side_cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return *v;
        }
        let size = kind.spec().piece_size_m(amount);
        let goal = noto_core::act::reach::ReachGoal::ground_object(size[1], kind.spec().mass_kg * amount, false, self.carried_kg());
        let plan = noto_core::act::reach::plan_side(&self.body, &goal, 0.5 * size[2]);
        let v = plan.ok.then_some((plan.object_x, plan.object_y));
        self.side_cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, v);
        v
    }

    /// A pushed item that lies within reach of a waypoint goes onto its pile:
    /// the amount, or `None` when it is not there.
    pub fn stock_pushed(&mut self, drop: u32, waypoint: u32, events: &mut Vec<TimedEvent>) -> Option<f32> {
        let d = self.drop_item(drop)?.clone();
        let w = self.waypoint(waypoint)?;
        if (d.x - w.x).hypot(d.z - w.z) > DELIVER_REACH_M {
            return None;
        }
        self.drops.retain(|x| x.id != drop);
        if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == waypoint) {
            add_stock(&mut w.stock, d.kind, d.amount);
        }
        events.push(self.stamp(WorldEvent::Delivered { waypoint: waypoint_id(waypoint), item: d.kind, amount: d.amount }));
        Some(d.amount)
    }

    // ----- pushing -----

    /// The item's route for a push to near `(x, z)`: cells it can slide over
    /// (no water, no node, no curb), and where the robot lines up behind it
    /// (x, z, heading).
    pub fn push_route(&self, drop: u32, x: f32, z: f32) -> Result<(nav::Plan, (f32, f32, f32)), ActionError> {
        if self.body.legged() {
            return Err(ActionError::CannotPush { reason: "다리형은 범퍼가 없어 바닥의 물체를 밀지 않음 (들어서 옮김)".into() });
        }
        let d = self.drop_item(drop).ok_or(ActionError::UnknownTarget)?;
        if !d.support.is_ground() {
            return Err(ActionError::CannotPush { reason: format!("{}의 물체는 범퍼로 밀 수 없음", d.support.label_ko()) });
        }
        let start = self.terrain.cell_of(d.x, d.z).ok_or(ActionError::UnknownTarget)?;
        let nodes = self.blocked_cells();
        let t = &self.terrain;
        // A pushed item cannot be pushed over a curb: cells with one are walls to it.
        let blocked = |cx: usize, cz: usize| nodes.contains(&(cx, cz)) || t.curb_m(cx, cz) > 0.0;
        let spec = self.drive_spec();
        // The route must carry the item too: its friction and slope share on top of the drive's own need.
        let load = nav::PushLoad { mass_kg: d.kind.spec().mass_kg * d.amount, material: material_of(d.kind) };
        let plan = nav::plan_with(&spec, t, &blocked, start, Goal::Within { x, z, reach_m: 0.5 }, Some(load)).ok_or(ActionError::NoPath)?;
        let side = d.kind.spec().piece_size_m(d.amount)[0];
        let lead = apush::lead_m(&self.body, side);
        let (hx, hz) = plan.cells.get(1).map(|c| cell_center(c.0, c.1)).unwrap_or((x, z));
        let yaw = (hz - d.z).atan2(hx - d.x);
        Ok((plan, (d.x - lead * yaw.cos(), d.z - lead * yaw.sin(), yaw)))
    }

    /// Starts a push: the robot must stand lined up behind the item.
    pub(super) fn start_push(&mut self, drop: u32, x: f32, z: f32) -> Result<PushState, ActionError> {
        let (plan, (lx, lz, yaw)) = self.push_route(drop, x, z)?;
        let off = (lx - self.agent.x).hypot(lz - self.agent.z);
        if off > PUSH_START_SLACK_M {
            return Err(ActionError::OutOfReach { distance_m: off, reach_m: PUSH_START_SLACK_M });
        }
        let d = self.drop_item(drop).cloned().ok_or(ActionError::UnknownTarget)?;
        let size = d.kind.spec().piece_size_m(d.amount);
        let radius = apush::planned_radius_m(&self.body, material_of(d.kind), size[0]);
        // The path: the item's centre now, then the cell centres; legs are straight runs with the turns between them.
        let mut path = vec![(d.x, d.z)];
        path.extend(plan.cells.iter().skip(1).map(|c| cell_center(c.0, c.1)));
        let mut legs: Vec<apush::PushLeg> = Vec::new();
        let mut heading = yaw;
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let h = (b.1 - a.1).atan2(b.0 - a.0);
            let turn = wrap_angle(h - heading).to_degrees();
            heading = h;
            let cell = self.terrain.cell_of(b.0, b.1).unwrap_or((0, 0));
            let rise = self.terrain.height_m(cell.0, cell.1) - self.terrain.cell_of(a.0, a.1).map_or(0.0, |c| self.terrain.height_m(c.0, c.1));
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            let slope = rise.atan2(len.max(1e-3)).to_degrees();
            let ground = self.ground_at(cell).surface();
            match legs.last_mut() {
                Some(l) if turn.abs() < 1.0 && l.ground == ground && (l.slope_deg - slope).abs() < 1.0 => l.len_m += len,
                _ => legs.push(apush::PushLeg { turn_deg: turn, radius_m: if turn.abs() < 1.0 { 0.0 } else { radius }, len_m: len, ground, slope_deg: slope, blocked: false }),
            }
        }
        let inp = apush::PushInput { body: &self.body, material: material_of(d.kind), mass_kg: d.mass_kg(), size_m: size, cargo_kg: self.carried_kg(), legs, aim_sigma_m: self.aim_sigma_m(), seed: self.act_rng.next_u64() };
        let mut rollout = apush::push(&inp);
        // The last creep into line behind the item comes first.
        let creep = off / SHUFFLE_MPS;
        if creep > 0.05 {
            for s in rollout.samples.iter_mut() {
                s.t += creep;
            }
            for e in rollout.events.iter_mut() {
                e.t += creep;
            }
            rollout.duration_s += creep;
            rollout.metrics.push(("line_up_s", creep));
        }
        let length: f32 = path.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum();
        let end_m = if rollout.outcome.ok { length } else { rollout.metric("pushed_m").unwrap_or(0.0).min(length) };
        let lead = apush::lead_m(&self.body, size[0]);
        self.agent.yaw = yaw;
        self.act = Some(ActPlay::new(rollout, self.time_s, Some((d.kind, d.amount)), (self.agent.x, self.agent.z, yaw)));
        Ok(PushState { drop, path, length_m: length, done_m: 0.0, end_m, lead_m: lead })
    }

    /// Advances a push: the item moves along its path at the rollout's pace and the robot follows behind it.
    pub(super) fn advance_push(&mut self, ps: &mut PushState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let Some(p) = self.act.as_ref() else {
            *done = Some((false, Some("push_lost".into())));
            return 0.0;
        };
        let dur = p.rollout.duration_s.max(1e-3);
        let t_now = p.t();
        let dt = remaining.min((dur - t_now).max(0.0));
        let t_next = t_now + dt;
        // The item's progress: the rollout's distance over its time, up to where it ends.
        let pace = if p.rollout.outcome.ok { ps.length_m / dur } else { ps.end_m / dur };
        ps.done_m = (pace * t_next).min(ps.end_m);
        let (ix, iz, h) = point_along(&ps.path, ps.done_m);
        if let Some(d) = self.drops.iter_mut().find(|d| d.id == ps.drop) {
            d.x = ix;
            d.z = iz;
        }
        self.agent.x = ix - ps.lead_m * h.cos();
        self.agent.z = iz - ps.lead_m * h.sin();
        self.agent.yaw = h;
        self.agent.speed_mps = pace;
        let power = self.drive_spec().power_w as f64;
        let push_n = p.rollout.metric("push_peak_n").unwrap_or(0.0) as f64;
        self.add_energy(power * dt as f64 + push_n * pace as f64 * dt as f64);
        // Events of the rollout up to the next instant.
        self.act_advance(dt, events);
        if t_next >= dur - 1e-4 {
            let ok = self.act.as_ref().map_or(false, |p| p.rollout.outcome.ok);
            let code = self.act.as_ref().and_then(|p| p.rollout.outcome.code).map(String::from);
            if !ok {
                // A lost item stays beside the bumper where it slid off.
                let off = self.act.as_ref().and_then(|p| p.rollout.metric("offset_m")).unwrap_or(0.0);
                if let Some(d) = self.drops.iter_mut().find(|d| d.id == ps.drop) {
                    d.x += -h.sin() * off.clamp(-0.4, 0.4);
                    d.z += h.cos() * off.clamp(-0.4, 0.4);
                }
            }
            self.end_act();
            *done = Some((ok, code));
        }
        dt.max(1e-4)
    }
}

/// The camera's range to an item at the feet: the head is about this high above it, metres.
const SHOULDER_RANGE_M: f32 = 1.05;

/// A point along a polyline at distance `s`, and the heading there.
fn point_along(path: &[(f32, f32)], s: f32) -> (f32, f32, f32) {
    let mut left = s.max(0.0);
    let n = path.len().saturating_sub(1);
    for i in 0..n {
        let (a, b) = (path[i], path[i + 1]);
        let len = (b.0 - a.0).hypot(b.1 - a.1);
        let h = (b.1 - a.1).atan2(b.0 - a.0);
        if left <= len || i + 1 == n {
            let u = (left / len.max(1e-6)).clamp(0.0, 1.0);
            return (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u, h);
        }
        left -= len;
    }
    let p = path.last().copied().unwrap_or((0.0, 0.0));
    (p.0, p.1, 0.0)
}

/// An action's event as a world event.
fn act_event(kind: ActKind, e: &ActEvent) -> WorldEvent {
    WorldEvent::Act { act: kind.id().into(), kind: e.kind.into(), label: e.label_ko.clone(), ok: e.ok, value: e.value, parts: e.parts.iter().map(|p| p.id().to_string()).collect() }
}

/// A positive draw with a mean and a coefficient of variation (log-normal).
fn draw_lognormal(rng: &mut Rng, mean: f32, cov: f32) -> f32 {
    let s2 = (1.0 + cov * cov).ln();
    let mu = mean.ln() - 0.5 * s2;
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
    (mu + s2.sqrt() * z).exp()
}
