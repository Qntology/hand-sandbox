//! What rides on a robot rides on it for as long as physics lets it. Each tick the place the load sits is followed
//! through the world, and its acceleration, in the body's own frame, is what the load feels: the brakes, the push of
//! pulling away, the swing of a corner (for a truck, turning about its centre, that is a good way out on the tines), the
//! pull of a slope and the kick of a step. `noto_core::act::carry` says whether that moves the load. A load that goes
//! past the tine tips, off their side, or out of the hands comes off: it falls from where it was, keeps the speed it
//! had, bounces, and lies on the ground as a drop like any other, for the robot to go back for.
//!
//! The friction of a haul is drawn when the load goes on (dust, a damp log, how it sat down), so a careful robot that
//! slows for its corners on the average friction still sometimes has to live with less of it.

use super::*;
use noto_core::act::carry::{self, Holder, LoadSpec, LoadState, Lost};
use noto_core::balance::G;

/// How far the friction of one haul strays from the table's value (coefficient of variation) [추정].
pub const HAUL_MU_COV: f32 = 0.12;
/// The time constant the acceleration is seen through (s): what a tick's difference shows of a move the body smooths.
const ACCEL_TAU_S: f32 = 0.08;
/// A slide is worth saying from this far (m).
const NOTE_SHIFT_M: f32 = 0.03;
/// What a load keeps of its shift when the piece above it comes off.
const SHIFT_KEPT: f32 = 0.4;

#[derive(Clone, Copy, Debug, Default)]
pub struct CargoRide {
    pub state: LoadState,
    /// This haul's friction: a standard normal draw, and the load it was drawn for (kg).
    draw: f32,
    drawn_for_kg: f32,
    last_at: Option<(f32, f32)>,
    last_vel: Option<[f32; 2]>,
    accel: [f32; 2],
    /// The kick of a step, taken at the next tick (m/s²).
    pub pending_az: f32,
    pub use_ratio: f32,
    noted: bool,
}

impl CargoRide {
    /// The base stands: what it does next starts from rest.
    pub fn at_rest(&mut self) {
        self.last_at = None;
        self.last_vel = None;
        self.accel = [0.0, 0.0];
        self.pending_az = 0.0;
    }
}

impl World {
    /// The load on this robot as a load on its support, for this haul.
    pub fn load_spec(&self) -> Option<LoadSpec> {
        let pieces: Vec<(ItemKind, f32)> = self.agent.carried.iter().filter(|(_, n)| *n > AMOUNT_EPS).copied().collect();
        let (top, amount) = *pieces.last()?;
        let mass: f32 = pieces.iter().map(|(k, n)| k.spec().mass_kg * n).sum();
        let layers: f32 = pieces.iter().map(|(_, n)| n.ceil()).sum();
        let size = top.spec().piece_size_m(amount.min(1.0));
        let spec = if noto_core::act::fork::applies(&self.body) {
            LoadSpec::tines(&self.body.dims, material_of(top), mass, size[0].max(size[2]), layers as usize)
        } else {
            let weight = mass * G;
            let hold = 2.0 * self.grip_capacity_for(top, amount.min(1.0));
            LoadSpec::hands(material_of(top), mass, size, hold.max(noto_core::balance::GRIP_SAFETY * weight))
        };
        Some(spec.with_draw(HAUL_MU_COV, self.ride.draw))
    }

    /// How hard this robot may accelerate with its load on, corner or straight (m/s²); `None` with nothing on.
    pub(super) fn load_accel(&self) -> Option<f32> {
        self.load_spec().map(|s| s.safe_accel().max(0.3))
    }

    /// How high the load rides off the ground under the base (m), and how far ahead of the base's centre.
    fn load_seat(&self) -> (f32, f32) {
        let top = self.agent.carried.last().map_or([0.1; 3], |(k, n)| k.spec().piece_size_m(n.min(1.0)));
        if noto_core::act::fork::applies(&self.body) {
            (self.body.dims.fork_carry_m + self.body.dims.tine_t_m, noto_core::act::fork::load_x(&self.body, top[0].max(top[2])))
        } else {
            (noto_core::act::cocarry::carry_hand_z(&self.body), noto_core::act::cocarry::CARRY_REACH_M)
        }
    }

    /// How far the load sits from the point the base turns about (m).
    pub(super) fn load_lever_m(&self) -> f32 {
        self.load_seat().1.abs().max(0.1)
    }

    fn load_point(&self) -> (f32, f32) {
        let (_, ahead) = self.load_seat();
        (self.agent.x + ahead * self.agent.yaw.cos(), self.agent.z + ahead * self.agent.yaw.sin())
    }

    /// One tick of the load riding: `moving` when the base drives itself (an act that holds still does not shake it).
    /// True when something came off.
    pub(super) fn ride_cargo(&mut self, dt: f32, moving: bool, worker: u32, events: &mut Vec<TimedEvent>) -> bool {
        let kg = self.carried_kg();
        if kg <= AMOUNT_EPS {
            self.ride = CargoRide::default();
            return false;
        }
        if (self.ride.drawn_for_kg - kg).abs() > 1e-3 {
            self.ride.draw = normal(&mut self.act_rng);
            self.ride.drawn_for_kg = kg;
        }
        let at = self.load_point();
        if !moving {
            self.ride.at_rest();
            return false;
        }
        let vel = self.ride.last_at.map_or([0.0, 0.0], |p| [(at.0 - p.0) / dt, (at.1 - p.1) / dt]);
        let raw = self.ride.last_vel.map_or([0.0, 0.0], |v| [(vel[0] - v[0]) / dt, (vel[1] - v[1]) / dt]);
        self.ride.last_at = Some(at);
        self.ride.last_vel = Some(vel);
        let keep = (dt / (ACCEL_TAU_S + dt)).clamp(0.0, 1.0);
        self.ride.accel = [self.ride.accel[0] + keep * (raw[0] - self.ride.accel[0]), self.ride.accel[1] + keep * (raw[1] - self.ride.accel[1])];
        let (s, c) = self.agent.yaw.sin_cos();
        let (ax, az) = (self.ride.accel[0], self.ride.accel[1]);
        let slope = self.posture_now().0.slope_deg.to_radians();
        let a = [ax * c + az * s + G * slope.sin(), -ax * s + az * c];
        let a_z = std::mem::take(&mut self.ride.pending_az);
        let Some(spec) = self.load_spec() else { return false };
        let r = carry::ride(&spec, self.ride.state, a, a_z, dt);
        self.ride.state = r.state;
        self.ride.use_ratio = r.use_ratio;
        let shift = r.state.shift[0].hypot(r.state.shift[1]);
        if shift >= NOTE_SHIFT_M && !self.ride.noted {
            self.ride.noted = true;
            let item = self.agent.carried.last().map(|c| c.0);
            if let Some(item) = item {
                let how = if spec.holder == Holder::Tines { "포크 위에서" } else { "손에서" };
                let label = format!("{} {how} {:.0} cm 밀림 (마찰 이용률 {:.0}%)", item.spec().label_ko, shift * 100.0, r.use_ratio * 100.0);
                events.push(self.stamp(WorldEvent::CargoSlip { worker, item, shift_m: shift, use_ratio: r.use_ratio, label }));
            }
        }
        match r.lost {
            Some(why) => {
                self.lose_piece(why, worker, events);
                true
            }
            None => false,
        }
    }

    /// A sudden change of speed the tick cannot see (`dv`, forward and left, over `dt_s`): a bump, a slipping foot.
    pub(super) fn jolt_cargo(&mut self, dv: [f32; 2], dt_s: f32, worker: u32, events: &mut Vec<TimedEvent>) -> bool {
        let Some(spec) = self.load_spec() else { return false };
        let r = carry::jolt(&spec, self.ride.state, dv, dt_s);
        self.ride.state = r.state;
        self.ride.use_ratio = r.use_ratio;
        match r.lost {
            Some(why) => {
                self.lose_piece(why, worker, events);
                true
            }
            None => false,
        }
    }

    /// Something outside the robot shoves its load: a speed change `dv` (forward, left) over `dt_s`. What it does comes
    /// back as events, for whoever watches the world.
    pub fn knock_load(&mut self, dv: [f32; 2], dt_s: f32) -> Vec<TimedEvent> {
        let mut ev = Vec::new();
        self.jolt_cargo(dv, dt_s, 1, &mut ev);
        ev
    }

    /// The robot went down: everything it carried goes with it.
    pub(super) fn spill_all(&mut self, worker: u32, events: &mut Vec<TimedEvent>) {
        let mut guard = 0;
        while self.carried_kg() > AMOUNT_EPS && guard < 32 {
            guard += 1;
            self.lose_piece(Lost::Grip, worker, events);
        }
    }

    /// The top piece of the load comes off.
    fn lose_piece(&mut self, why: Lost, worker: u32, events: &mut Vec<TimedEvent>) {
        let Some(k) = self.agent.carried.iter().rposition(|(_, n)| *n > AMOUNT_EPS) else { return };
        let (item, have) = self.agent.carried[k];
        let amount = have.min(1.0);
        self.agent.carried[k].1 -= amount;
        self.agent.carried.retain(|(_, n)| *n > AMOUNT_EPS);
        let (h, _) = self.load_seat();
        let (lx, lz) = self.load_point();
        let (s, c) = self.agent.yaw.sin_cos();
        let shift = self.ride.state.shift;
        let from = (lx + c * shift[0] - s * shift[1], lz + s * shift[0] + c * shift[1]);
        let t_fall = (2.0 * h.max(0.02) / G).sqrt();
        let v = self.ride.last_vel.unwrap_or([0.0, 0.0]);
        let slid = self.ride.state.vel;
        let (ax, az) = (v[0] + c * slid[0] - s * slid[1], v[1] + s * slid[0] + c * slid[1]);
        let ground = self.ground_here().surface();
        let e = noto_core::act::material::restitution(material_of(item), ground);
        let dir = self.act_rng.next_f32() * std::f32::consts::TAU;
        let bounce = 2.0 * e * e * h * (2.0 * self.act_rng.next_f32() * 30f32.to_radians()).sin();
        let want = (from.0 + ax * t_fall + bounce * dir.cos(), from.1 + az * t_fall + bounce * dir.sin());
        let (x, z) = self.free_spot(want);
        let id = self.next_drop_id();
        self.drops.push(DropItem { amount, ..DropItem::whole(id, item, x, z) });
        self.ride.state.shift = [shift[0] * SHIFT_KEPT, shift[1] * SHIFT_KEPT];
        self.ride.state.vel = [0.0, 0.0];
        self.ride.noted = false;
        let cause = match why {
            Lost::Forward => "앞으로 미끄러져 떨어짐",
            Lost::Side => "옆으로 미끄러져 떨어짐",
            Lost::Grip => "손에서 놓침",
        };
        let label = format!("{} {} {cause} ({:.1}, {:.1})", item.spec().label_ko, amount_text(amount), x, z);
        events.push(self.stamp(WorldEvent::CargoLost { worker, item, amount, drop: drop_id(id), cause: why.id().into(), label }));
    }

    /// The nearest place to `want` a dropped thing can lie: dry land on the map, not inside anything.
    pub(super) fn free_spot(&self, want: (f32, f32)) -> (f32, f32) {
        let map = self.clear_map();
        let blocked = self.blocked_cells();
        let lies = |x: f32, z: f32| self.terrain.cell_of(x, z).is_some_and(|(cx, cz)| !self.terrain.is_water(cx, cz) && !blocked.contains(&(cx, cz))) && map.at(x, z) >= 0.08;
        if lies(want.0, want.1) {
            return want;
        }
        for ring in 1..=10 {
            let r = 0.12 * ring as f32;
            for k in 0..12 {
                let a = k as f32 * std::f32::consts::TAU / 12.0;
                let (x, z) = (want.0 + r * a.cos(), want.1 + r * a.sin());
                if lies(x, z) {
                    return (x, z);
                }
            }
        }
        (self.agent.x, self.agent.z)
    }
}

fn normal(rng: &mut Rng) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    fn world(base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        let mut w = World::new(WorldConfig { seed: 11, cells: 64, noto: hand, terrain: crate::terrain::TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        w.nodes.clear();
        w.drops.clear();
        w.agent.yaw = 0.0;
        w
    }

    fn drive(w: &mut World, to: (f32, f32), ticks: usize) -> Vec<TimedEvent> {
        w.command(Action::MoveTo { x: to.0, z: to.1, reach_m: 0.0 }).unwrap();
        let mut ev = Vec::new();
        for _ in 0..ticks {
            ev.extend(w.tick());
            if !w.busy() {
                break;
            }
        }
        ev
    }

    #[test]
    fn a_careful_drive_keeps_the_load_on() {
        for base in [BaseKind::Wheel, BaseKind::HumanFoot] {
            let mut w = world(base);
            add_stock(&mut w.agent.carried, ItemKind::Log, 1.0);
            let (x, z) = (w.agent.x, w.agent.z);
            let ev = drive(&mut w, (x + 4.0, z + 3.0), 400);
            assert!(!ev.iter().any(|e| matches!(e.event, WorldEvent::CargoLost { .. })), "{base:?}");
            assert_eq!(w.carried_count(ItemKind::Log), 1, "{base:?}");
            assert!(w.ride.state.shift[0].abs() < 0.05, "{base:?} {:?}", w.ride.state);
        }
    }

    #[test]
    fn a_truck_with_a_load_comes_to_a_stop_within_what_the_load_stands() {
        let mut w = world(BaseKind::Wheel);
        add_stock(&mut w.agent.carried, ItemKind::Log, 1.0);
        let (x, z) = (w.agent.x, w.agent.z);
        let limit = w.load_accel().unwrap();
        // It stops 3 m short of the place, not at the end of the route.
        w.command(Action::MoveTo { x: x + 12.0, z, reach_m: 3.0 }).unwrap();
        let (mut v0, mut worst, mut top) = (0.0f32, 0.0f32, 0.0f32);
        for _ in 0..400 {
            w.tick();
            if !w.busy() {
                break;
            }
            let v = w.agent.speed_mps;
            worst = worst.max((v0 - v) / TICK_S);
            top = top.max(v);
            v0 = v;
        }
        assert!(!w.busy(), "it arrived");
        assert!(top > 0.8, "it got up to speed: {top}");
        assert!(v0 <= limit * TICK_S + 0.1, "it was all but stopped before it stopped: {v0}");
        assert!(worst <= 1.3 * limit, "it braked at {worst} m/s² against a limit of {limit}");
        assert_eq!(w.carried_count(ItemKind::Log), 1);
    }

    #[test]
    fn a_load_flies_off_the_tines_when_the_truck_stops_dead_at_speed() {
        let mut w = world(BaseKind::Wheel);
        add_stock(&mut w.agent.carried, ItemKind::Parcel, 1.0);
        let (x, z) = (w.agent.x, w.agent.z);
        let mut ev = Vec::new();
        // At full speed a wall of inertia: the base is stopped dead again and again, as if it hit something soft.
        w.command(Action::MoveTo { x: x + 12.0, z, reach_m: 0.0 }).unwrap();
        for i in 0..80 {
            ev.extend(w.tick());
            if i % 6 == 5 {
                w.jolt_cargo([-3.5, 0.0], 0.03, 1, &mut ev);
            }
        }
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::CargoLost { .. })), "{:?}", w.ride.state);
        assert_eq!(w.carried_count(ItemKind::Parcel), 0);
        let d = w.drops.iter().find(|d| d.kind == ItemKind::Parcel).expect("it lies on the ground");
        assert!(d.x > x, "it went on the way the truck was going: {} vs {x}", d.x);
    }

    #[test]
    fn the_piece_that_comes_off_lies_on_land_and_can_be_picked_up_again() {
        let mut w = world(BaseKind::HumanFoot);
        add_stock(&mut w.agent.carried, ItemKind::Parcel, 2.0);
        let mut ev = Vec::new();
        w.lose_piece(Lost::Side, 1, &mut ev);
        assert_eq!(w.carried_count(ItemKind::Parcel), 1, "one piece at a time");
        let d = w.drops.last().unwrap().clone();
        assert!(!w.terrain.is_water(w.terrain.cell_of(d.x, d.z).unwrap().0, w.terrain.cell_of(d.x, d.z).unwrap().1));
        assert!(matches!(&ev[0].event, WorldEvent::CargoLost { cause, .. } if cause == "side"));
        w.spill_all(1, &mut ev);
        assert_eq!(w.carried_kg(), 0.0);
        assert_eq!(w.drops.len(), 2);
    }
}
