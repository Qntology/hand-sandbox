//! Bodies stop at what they can hit. These run robots over seeded worlds and look at the body at every tick: the box the
//! camera draws must stay out of the boxes the camera draws of the rocks, trunks and walls, and out of a curb its wheels
//! cannot climb, whoever is driving it.

use noto_core::{BaseKind, BodyPlan, FingerCount, NotoConfig};
use world_sim::crew::{Form, Job};
use world_sim::kinds::NodeKind;
use world_sim::terrain::{cell_center, TerrainKind};
use world_sim::world::{Action, Node, World, WorldConfig, WorldEvent};

fn world(seed: u64, base: BaseKind, terrain: TerrainKind) -> World {
    let mut hand = NotoConfig::preset(FingerCount::Five);
    hand.base = base;
    World::new(WorldConfig { seed, cells: 64, noto: hand, body: BodyPlan::standard(), terrain, parcels: Some(0), ..WorldConfig::default() }).unwrap()
}

fn room_now(w: &World) -> f32 {
    w.footprint().room(&w.clear_map(), w.agent.x, w.agent.z, w.agent.yaw)
}

/// Drives the first worker to each `goal` and gives the least room its body had on the way, and how many it reached.
fn tour(w: &mut World, goals: &[(f32, f32)]) -> (f32, usize) {
    let (mut worst, mut reached) = (f32::INFINITY, 0);
    for &(x, z) in goals {
        if w.command(Action::MoveTo { x, z, reach_m: 1.2 }).is_err() {
            continue;
        }
        for _ in 0..4000 {
            w.tick();
            worst = worst.min(room_now(w));
            if !w.busy() {
                break;
            }
        }
        reached += usize::from((w.agent.x - x).hypot(w.agent.z - z) < 1.6);
        while w.fallen_until.is_some_and(|t| w.time_s < t) {
            w.tick();
        }
    }
    (worst, reached)
}

fn spots(w: &World) -> Vec<(f32, f32)> {
    let wh = w.waypoint_by_key("warehouse").unwrap();
    let mut v: Vec<(f32, f32)> = w.nodes.iter().step_by(7).take(5).map(|n| (n.x, n.z)).collect();
    v.push((wh.x, wh.z));
    v
}

#[test]
fn no_body_goes_into_a_rock_a_tree_or_a_curb_on_any_seed() {
    let mut reached = 0;
    for seed in 1..=4u64 {
        for terrain in [TerrainKind::Open, TerrainKind::Rugged] {
            for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track, BaseKind::Mecanum] {
                let mut w = world(seed, base, terrain);
                let goals = spots(&w);
                let (worst, n) = tour(&mut w, &goals);
                reached += n;
                assert!(worst >= -1e-3, "seed {seed} {terrain:?} {base:?}: the body went {:.3} m into something", -worst);
            }
        }
    }
    assert!(reached > 60, "they got to most places: {reached}");
}

#[test]
fn sub_workers_keep_out_of_what_they_can_hit_as_the_first_worker_does() {
    for seed in 1..=3u64 {
        let mut w = world(seed, BaseKind::HumanFoot, TerrainKind::Open);
        let ids = w.add_crew(&[Form::Forklift, Form::Humanoid], 1);
        let goals = spots(&w);
        let mut worst = f32::INFINITY;
        for &id in &ids {
            for &(x, z) in goals.iter().take(4) {
                if w.crew_command(id, Job::Go { x, z, reach_m: 1.6 }).is_err() {
                    continue;
                }
                for _ in 0..3000 {
                    w.tick();
                    let c = w.crew_worker(id).unwrap();
                    let map = w.clear_map();
                    // The clearance of the body against the world's solids: ledges are checked by the worker's own map.
                    let room = c.footprint().room(&map, c.x, c.z, c.yaw);
                    worst = worst.min(room);
                    if w.crew_take_outcome(id).is_some() {
                        break;
                    }
                }
            }
        }
        assert!(worst >= -1e-3, "seed {seed}: a sub-worker went {:.3} m into something", -worst);
    }
}

fn plant(w: &mut World, kind: NodeKind, c: (usize, usize)) {
    let (x, z) = cell_center(c.0, c.1);
    let id = 900 + w.nodes.len() as u32;
    let hp = kind.break_time_s();
    w.nodes.push(Node { id, kind, x, z, hp, max_hp: hp });
}

fn cleared(base: BaseKind) -> World {
    let mut w = world(11, base, TerrainKind::Open);
    w.nodes.clear();
    w.drops.clear();
    let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
    (w.agent.x, w.agent.z) = cell_center(cx, cz);
    w.agent.yaw = 0.0;
    w
}

#[test]
fn a_rock_that_turns_up_in_the_way_stops_the_robot_and_it_goes_round() {
    for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
        let mut w = cleared(base);
        let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
        let goal = cell_center(cx + 14, cz);
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        for _ in 0..12 {
            w.tick();
        }
        assert!(w.agent.speed_mps > 0.1);
        // Right across its path, and too late to have planned for.
        let at = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
        for dz in -2i32..=2 {
            plant(&mut w, NodeKind::Boulder, (at.0 + 5, (at.1 as i32 + dz) as usize));
        }
        let (mut stopped, mut worst, mut events) = (false, f32::INFINITY, Vec::new());
        for _ in 0..3000 {
            events.extend(w.tick());
            stopped |= w.agent.speed_mps < 1e-3;
            worst = worst.min(room_now(&w));
            if !w.busy() {
                break;
            }
        }
        let said: Vec<String> = events.iter().filter_map(|e| match &e.event { WorldEvent::Hold { label, .. } | WorldEvent::Bump { label, .. } => Some(label.clone()), _ => None }).collect();
        assert!(stopped, "{base:?} stopped for it");
        assert!(worst >= -1e-3, "{base:?} went {:.3} m into the rock", -worst);
        assert!(!said.is_empty(), "{base:?} planned again: {said:?}");
        assert!(!w.busy() && (w.agent.x - goal.0).hypot(w.agent.z - goal.1) < 0.3, "{base:?} went round and got there: {:?}", (w.agent.x, w.agent.z));
    }
}

#[test]
fn running_into_a_rock_at_speed_is_a_collision_and_it_does_not_stop_inside_it() {
    let mut w = cleared(BaseKind::HumanFoot);
    let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
    let goal = cell_center(cx + 14, cz);
    w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
    for _ in 0..20 {
        w.tick();
    }
    let v = w.agent.speed_mps;
    assert!(v > 0.5, "{v}");
    // The rock appears against its chest: nothing can brake in that distance.
    let (ax, rz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
    let rx = ax + 4;
    let face = cell_center(rx, rz).0 - 0.24 - w.footprint().front_m;
    while face - w.agent.x > 0.04 {
        w.step(0.01);
    }
    plant(&mut w, NodeKind::Boulder, (rx, rz));
    let mut bump = None;
    for _ in 0..400 {
        for e in w.tick() {
            if let WorldEvent::Bump { peak_n, what, .. } = e.event {
                bump = Some((peak_n, what));
            }
        }
        assert!(room_now(&w) > -0.06, "it stops at the rock: {}", room_now(&w));
        if bump.is_some() {
            break;
        }
    }
    let (peak, what) = bump.expect("it felt the rock");
    assert!(peak > 50.0, "{peak} N");
    assert_eq!(what, "바위");
}

#[test]
fn a_truck_never_rides_up_over_a_curb_it_cannot_climb() {
    let mut w = cleared(BaseKind::Wheel);
    let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
    for z in 0..w.terrain.d {
        w.terrain.set_curb_cm(cx + 8, z, 15);
    }
    let goal = cell_center(cx + 7, cz + 20);
    let hull = |w: &World| w.footprint().hull(w.agent.x, w.agent.z, w.agent.yaw);
    assert!(w.command(Action::MoveTo { x: cell_center(cx + 12, cz).0, z: cell_center(cx + 12, cz).1, reach_m: 0.0 }).is_err(), "no way over the curb for wheels");
    w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
    let line = (cx + 8) as f32 * 0.5;
    for _ in 0..2000 {
        w.tick();
        let h = hull(&w);
        let reach = h.half.0 * h.u.0.abs() + h.half.1 * h.u.1.abs();
        assert!(h.c.0 + reach <= line + 1e-3, "the hull reached the curb: {:?} vs {line}", h);
        if !w.busy() {
            break;
        }
    }
    assert!(!w.busy());
}

#[test]
fn a_walker_steps_over_the_same_curb() {
    let mut w = cleared(BaseKind::HumanFoot);
    let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
    for z in 0..w.terrain.d {
        w.terrain.set_curb_cm(cx + 8, z, 15);
    }
    let goal = cell_center(cx + 12, cz);
    w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
    let mut climbed = false;
    for _ in 0..2000 {
        for e in w.tick() {
            climbed |= matches!(e.event, WorldEvent::CurbClimbed { .. });
        }
        if !w.busy() {
            break;
        }
    }
    assert!(climbed && (w.agent.x - goal.0).abs() < 0.3);
}

#[test]
fn a_riders_walk_to_a_node_goes_round_what_stands_between() {
    let mut w = cleared(BaseKind::Wheel);
    let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap();
    // The node to fell, with a wall of rocks on the straight line from the seat to it.
    plant(&mut w, NodeKind::Tree, (cx + 2, cz + 7));
    let id = w.nodes.last().unwrap().id;
    for dx in -2i32..=3 {
        plant(&mut w, NodeKind::Boulder, ((cx as i32 + dx) as usize, cz + 4));
    }
    let n = w.nodes.iter().find(|n| n.id == id).unwrap().clone();
    assert!(w.command(Action::Mine { node: id }).is_ok());
    let (mut far, mut worst) = (0.0f32, f32::INFINITY);
    let solids = w.clear_map();
    for _ in 0..1600 {
        w.tick();
        if let Some(r) = w.rider.filter(|r| r.off()) {
            far = far.max((r.x - w.agent.x).hypot(r.z - w.agent.z));
            worst = worst.min(solids.at(r.x, r.z));
        }
        if !w.busy() {
            break;
        }
    }
    println!("[RIDER] walked {far:.2} m out, least room {worst:.2} m, node at {:?}", (n.x, n.z));
    assert!(worst >= 0.2, "the rider stayed clear of the rocks: {worst}");
    assert!(w.node(id).is_none(), "and felled the tree");
}
