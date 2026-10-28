use std::collections::HashSet;

use noto_core::{BaseKind, FingerCount, NotoConfig};
use world_sim::explore::{self, Coverage};
use world_sim::fixture::FixtureKind;
use world_sim::nav::{self, Goal};
use world_sim::render::{self, unpack_id, View};
use world_sim::site::{SiteCfg, Support};
use world_sim::terrain::{cell_center, CurbKind, CELL_M};
use world_sim::world::{Action, ActionError, World, WorldConfig};
use world_sim::{Camera, ObjectClass, TerrainKind};

fn world(seed: u64, terrain: TerrainKind, base: BaseKind, site: Option<SiteCfg>) -> World {
    let mut noto = NotoConfig::preset(FingerCount::Five);
    noto.base = base;
    World::new(WorldConfig { seed, cells: 64, noto, terrain, site, ..WorldConfig::default() }).unwrap()
}

fn sited(seed: u64, terrain: TerrainKind, base: BaseKind) -> World {
    world(seed, terrain, base, Some(SiteCfg::default()))
}

const SEEDS: [u64; 8] = [1, 2, 3, 4, 5, 7, 9, 12];

#[test]
fn the_same_seed_builds_the_same_site_and_another_seed_another() {
    let a = sited(7, TerrainKind::Open, BaseKind::Wheel);
    let b = sited(7, TerrainKind::Open, BaseKind::Wheel);
    assert_eq!(a.terrain_grid(), b.terrain_grid());
    assert_eq!(a.fixtures, b.fixtures);
    assert_eq!(a.site, b.site);
    assert!(a.drops.iter().zip(&b.drops).all(|(x, y)| (x.id, x.x, x.z, x.support) == (y.id, y.x, y.z, y.support)));
    let c = sited(8, TerrainKind::Open, BaseKind::Wheel);
    assert_ne!(a.site.as_ref().unwrap().houses, c.site.as_ref().unwrap().houses);
    let d = sited(7, TerrainKind::Open, BaseKind::HumanFoot);
    assert_eq!(a.fixtures, d.fixtures);
}

#[test]
fn without_a_site_the_map_has_no_structures() {
    let w = world(7, TerrainKind::Open, BaseKind::Wheel, None);
    assert!(w.site.is_none() && w.fixtures.is_empty());
    assert!(w.terrain_grid().curb_kind.is_empty());
    assert!(w.snapshot().site.is_none());
    assert!(w.search_prior().is_none());
    assert!(w.drops.iter().all(|d| d.support == Support::Ground));
    let json = serde_json::to_string(&WorldConfig::default()).unwrap();
    assert!(!json.contains("site"), "the default config serialises as before: {json}");
    let back: WorldConfig = serde_json::from_str(&json).unwrap();
    assert!(back.site.is_none());
    let with: WorldConfig = serde_json::from_str(&json.replacen('{', r#"{"site": {}, "#, 1)).unwrap();
    assert_eq!(with.site, Some(SiteCfg::default()));
    let two: WorldConfig = serde_json::from_str(&json.replacen('{', r#"{"site": {"houses": 2, "ramps": false}, "#, 1)).unwrap();
    assert_eq!(two.site, Some(SiteCfg { houses: 2, warehouse: true, ramps: false }));
}

#[test]
fn every_map_gets_houses_and_a_logistics_centre_around_the_warehouse_waypoint() {
    for terrain in [TerrainKind::Open, TerrainKind::Rugged] {
        for seed in SEEDS {
            let w = sited(seed, terrain, BaseKind::Wheel);
            let s = w.site.as_ref().unwrap();
            assert!(s.houses.len() >= 2, "{terrain:?} seed {seed}: {} houses", s.houses.len());
            let wh = s.warehouse.as_ref().expect("a logistics centre");
            let flag = w.waypoint_by_key("warehouse").unwrap();
            let fc = w.terrain.cell_of(flag.x, flag.z).unwrap();
            assert!(wh.rect.contains(fc), "the flag stands inside");
            assert!(!wh.shelves.is_empty() && !wh.tables.is_empty());
            let blocked = w.blocked_cells();
            assert!(!blocked.contains(&fc) && !blocked.contains(&(fc.0 - 1, fc.1)));
            for n in &w.nodes {
                let c = w.terrain.cell_of(n.x, n.z).unwrap();
                assert!(!s.is_reserved(c), "{terrain:?} seed {seed}: a node on the site at {c:?}");
            }
        }
    }
}

#[test]
fn the_site_never_cuts_the_way_between_the_waypoints_on_any_map_size() {
    for cells in [32usize, 64, 96] {
        for terrain in [TerrainKind::Open, TerrainKind::Rugged] {
            for seed in [1u64, 4, 7] {
                for base in [BaseKind::Wheel, BaseKind::HumanFoot] {
                    let make = |site: Option<SiteCfg>| {
                        let mut noto = NotoConfig::preset(FingerCount::Five);
                        noto.base = base;
                        World::new(WorldConfig { seed, cells, noto, terrain, site, ..WorldConfig::default() }).unwrap()
                    };
                    let (bare, w) = (make(None), make(Some(SiteCfg::default())));
                    let s = w.site.as_ref().unwrap();
                    let blocked = w.blocked_cells();
                    let flag = w.waypoint_by_key("warehouse").unwrap();
                    let camp = w.waypoint_by_key("base_camp").unwrap();
                    for p in [flag, camp] {
                        let c = w.terrain.cell_of(p.x, p.z).unwrap();
                        assert!(!blocked.contains(&c), "{cells} {terrain:?} seed {seed}: a fixture on the {} flag", p.key);
                    }
                    let reach = |w: &World| w.plan_route(Goal::Within { x: flag.x, z: flag.z, reach_m: 1.0 }).is_some();
                    if reach(&bare) {
                        assert!(reach(&w), "{cells} {terrain:?} seed {seed} {base:?}: the site cuts the way to the warehouse");
                    }
                    for h in &s.houses {
                        let slots = s.slots.iter().any(|sl| matches!(sl.support, Support::Step { house, .. } if house == h.id));
                        assert_eq!(slots, h.reachable_legs, "{cells} {terrain:?} seed {seed}: house {}", h.id);
                    }
                    if base == BaseKind::Wheel {
                        println!("[SITE] {cells}칸 {terrain:?} 시드 {seed}: 집 {} (다리 {} · 바퀴 {}), 물류센터 {}, 택배 자리 {}", s.houses.len(), s.houses.iter().filter(|h| h.reachable_legs).count(), s.houses.iter().filter(|h| h.reachable_wheels).count(), s.warehouse.is_some(), s.slots.len());
                    }
                }
            }
        }
    }
}

#[test]
fn stairs_have_real_risers_legs_climb_them_and_wheels_take_the_ramp() {
    for terrain in [TerrainKind::Open, TerrainKind::Rugged] {
        for seed in SEEDS {
            let legs = sited(seed, terrain, BaseKind::HumanFoot);
            let wheels = sited(seed, terrain, BaseKind::Wheel);
            let track = sited(seed, terrain, BaseKind::Track);
            let t = &legs.terrain;
            for h in &legs.site.as_ref().unwrap().houses {
                let floor = h.floor_cm as i32;
                let rise = h.riser_mm as i32 / 10;
                assert_eq!(floor, rise * h.risers as i32);
                assert!((15..=18).contains(&rise), "{rise} cm");
                for (k, row) in h.treads.iter().enumerate() {
                    for c in row {
                        assert_eq!((t.curb_m(c.0, c.1) * 100.0).round() as i32, floor - (k as i32 + 1) * rise);
                        assert_eq!(t.curb_kind(c.0, c.1), CurbKind::Step);
                    }
                }
                let porch = Goal::Cell(h.porch[0]);
                assert!(h.reachable_legs, "{terrain:?} seed {seed} house {}: legs reach no porch", h.id);
                assert!(legs.plan_route(porch).is_some());
                let mut closed = legs.blocked_cells();
                closed.extend(h.ramp.iter().copied());
                let top = *h.porch.iter().min_by_key(|c| nav::cell_distance_m(**c, h.stair_foot[0]).to_bits()).unwrap();
                let p = nav::plan(&legs.drive_spec(), t, &|x, z| closed.contains(&(x, z)), h.stair_foot[0], Goal::Cell(top)).expect("legs climb the stair");
                let steps: Vec<f32> = p.edges.iter().map(|e| e.step_m).filter(|s| *s > 0.0).collect();
                assert!(steps.iter().all(|s| (*s - rise as f32 / 100.0).abs() < 1e-3), "{steps:?}");
                assert_eq!(steps.len(), h.risers as usize, "the walk climbs every riser once: {steps:?}");
                assert!(nav::plan(&wheels.drive_spec(), t, &|x, z| closed.contains(&(x, z)), h.stair_foot[0], Goal::Cell(top)).is_none());
                assert!(nav::plan(&track.drive_spec(), t, &|x, z| closed.contains(&(x, z)), h.stair_foot[0], Goal::Cell(top)).is_none());
                assert!(h.reachable_wheels, "{terrain:?} seed {seed} house {}: the ramp is out of a wheel's reach", h.id);
                let stair: HashSet<_> = h.porch.iter().chain(h.treads.iter().flatten()).copied().collect();
                for w in [&wheels, &track] {
                    let p = w.plan_route(porch).expect("up the ramp");
                    let risers: Vec<f32> = p.edges.iter().filter(|e| stair.contains(&e.to) && e.step_m > 0.0).map(|e| e.step_m).collect();
                    assert!(risers.is_empty(), "{risers:?}");
                    if w.drive_spec().max_step_m < 0.05 {
                        assert!(p.edges.iter().all(|e| e.step_m == 0.0), "a wheel climbs no step at all");
                    }
                    let ramp: HashSet<_> = h.ramp.iter().copied().collect();
                    assert!(p.cells.iter().any(|c| ramp.contains(c)));
                    for e in &p.edges {
                        if ramp.contains(&e.to) || ramp.contains(&e.from) {
                            assert!(e.slope_deg < 7.0, "ramp edge {}°", e.slope_deg);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn without_ramps_only_legs_reach_the_porches() {
    let cfg = SiteCfg { ramps: false, ..SiteCfg::default() };
    let legs = world(4, TerrainKind::Open, BaseKind::HumanFoot, Some(cfg));
    let wheels = world(4, TerrainKind::Open, BaseKind::Wheel, Some(cfg));
    let s = wheels.site.as_ref().unwrap();
    assert!(!s.houses.is_empty());
    for h in &s.houses {
        assert!(h.ramp.is_empty() && !h.reachable_wheels);
        assert!(wheels.plan_route(Goal::Cell(h.porch[0])).is_none(), "a wheel cannot climb a {} mm riser", h.riser_mm);
        assert!(legs.plan_route(Goal::Cell(h.porch[0])).is_some());
    }
}

#[test]
fn doors_let_robots_in_and_routes_never_cross_walls_shelves_or_tables() {
    for seed in SEEDS {
        let legs = sited(seed, TerrainKind::Open, BaseKind::HumanFoot);
        let wheels = sited(seed, TerrainKind::Open, BaseKind::Wheel);
        let s = legs.site.as_ref().unwrap();
        let fixtures: HashSet<_> = world_sim::fixture::blocked(&legs.fixtures).collect();
        for h in s.houses.iter().filter(|h| h.reachable_legs) {
            let inside = (h.rect.x0 as usize + 4, h.rect.z0 as usize + 3);
            let inside = if fixtures.contains(&inside) { ((h.rect.x0 + h.rect.x1) as usize / 2, (h.rect.z0 + h.rect.z1) as usize / 2) } else { inside };
            let p = legs.plan_route(Goal::Cell(inside)).unwrap_or_else(|| panic!("seed {seed}: no way into house {}", h.id));
            let door: HashSet<_> = h.door.iter().copied().collect();
            assert!(p.cells.iter().any(|c| door.contains(c)), "seed {seed}: into house {} not through its door", h.id);
        }
        let wh = s.warehouse.as_ref().unwrap();
        let flag = wheels.waypoint_by_key("warehouse").unwrap();
        let p = wheels.plan_route(Goal::Within { x: flag.x, z: flag.z, reach_m: 1.0 }).expect("a wheel reaches the warehouse flag");
        let door: HashSet<_> = wh.door.iter().copied().collect();
        assert!(p.cells.iter().any(|c| door.contains(c)), "seed {seed}: into the logistics centre not through its door");
        for sl in &s.slots {
            let Some(p) = legs.plan_route(Goal::Cell(sl.stand)) else { panic!("seed {seed}: no way to {:?} for {:?}", sl.stand, sl.support) };
            for w in p.cells.windows(2) {
                assert!(!fixtures.contains(&w[1]), "seed {seed}: through a fixture at {:?}", w[1]);
                let (dx, dz) = (w[1].0 as i32 - w[0].0 as i32, w[1].1 as i32 - w[0].1 as i32);
                if dx != 0 && dz != 0 {
                    assert!(!fixtures.contains(&((w[0].0 as i32 + dx) as usize, w[0].1)) && !fixtures.contains(&(w[0].0, (w[0].1 as i32 + dz) as usize)), "seed {seed}: a corner cut past a fixture");
                }
            }
        }
    }
}

#[test]
fn parcels_rest_on_porches_shelves_and_tables_and_the_camp_parcel_stays() {
    let mut on = [0usize; 3];
    for terrain in [TerrainKind::Open, TerrainKind::Rugged] {
        for seed in SEEDS {
            let w = sited(seed, terrain, BaseKind::HumanFoot);
            let bare = world(seed, terrain, BaseKind::HumanFoot, None);
            assert_eq!(w.drops.len(), bare.drops.len(), "the same number of parcels");
            assert_eq!((w.drops[0].x, w.drops[0].z, w.drops[0].support), (bare.drops[0].x, bare.drops[0].z, Support::Ground), "the base camp's parcel");
            let snap = w.snapshot();
            for (d, v) in w.drops.iter().zip(&snap.drops).skip(1) {
                let c = w.terrain.cell_of(d.x, d.z).unwrap();
                let y = match d.support {
                    Support::Ground => panic!("{terrain:?} seed {seed}: a parcel left in the open"),
                    Support::Step { level, .. } => {
                        assert_eq!(level, 0, "on the landing");
                        on[0] += 1;
                        w.terrain.height_m(c.0, c.1) + w.terrain.curb_m(c.0, c.1)
                    }
                    Support::Shelf { fixture, level } => {
                        on[1] += 1;
                        let f = w.fixtures.iter().find(|f| f.id == fixture).unwrap();
                        assert!(f.cells.contains(&c));
                        f.base_m + f.support_levels_m()[level as usize]
                    }
                    Support::Table { fixture } => {
                        on[2] += 1;
                        let f = w.fixtures.iter().find(|f| f.id == fixture).unwrap();
                        assert!(f.cells.contains(&c));
                        f.base_m + f.support_levels_m()[0]
                    }
                };
                assert!((d.y_m - y).abs() < 1e-4 && (v.pos.y - y).abs() < 1e-4, "{:?}: {} vs {y}", d.support, d.y_m);
                assert!(w.support_top_m(d.support).is_some_and(|top| (top - y).abs() < 1e-4), "{:?}", d.support);
                assert_eq!(v.support, d.support);
            }
            let cells: HashSet<_> = w.drops.iter().map(|d| (d.x.to_bits(), d.z.to_bits(), d.y_m.to_bits())).collect();
            assert_eq!(cells.len(), w.drops.len());
            let s = w.site.as_ref().unwrap();
            for h in &s.houses {
                for (k, row) in h.treads.iter().enumerate() {
                    let top = w.support_top_m(Support::Step { house: h.id, level: k as u8 + 1 }).unwrap();
                    assert!(row.iter().all(|c| (w.terrain.height_m(c.0, c.1) + w.terrain.curb_m(c.0, c.1) - top).abs() < 1e-4));
                }
                assert!(w.support_top_m(Support::Step { house: h.id, level: h.risers }).is_none(), "the ground is not a tread");
            }
            assert!(w.support_top_m(Support::Ground).is_none() && w.support_top_m(Support::Table { fixture: 0 }).is_none());
        }
    }
    println!("[SITE] parcels on porches {} · shelves {} · tables {}", on[0], on[1], on[2]);
    assert!(on.iter().all(|&n| n > 0), "{on:?}");
}

fn look_at(from: (f32, f32), eye_y: f32, target: [f32; 3]) -> View {
    let (dx, dy, dz) = (target[0] - from.0, target[1] - eye_y, target[2] - from.1);
    View { origin: [from.0, eye_y, from.1], yaw: dz.atan2(dx), pitch: dy.atan2(dx.hypot(dz)), roll: 0.0 }
}

#[test]
fn every_parcel_slot_shows_from_where_the_robot_stands() {
    let cam = Camera { width: 160, height: 120, ..Camera::default() };
    for seed in [1u64, 4, 7] {
        let mut w = sited(seed, TerrainKind::Open, BaseKind::HumanFoot);
        let eye = w.standing_eye_m();
        let slots = w.site.as_ref().unwrap().slots.clone();
        let mut next = 9000u32;
        for sl in slots {
            w.drops.clear();
            w.drops.push(world_sim::world::DropItem::on(next, world_sim::ItemKind::Parcel, sl.x, sl.z, sl.support, sl.y_m));
            let (sx, sz) = cell_center(sl.stand.0, sl.stand.1);
            let floor = w.terrain.height_m(sl.stand.0, sl.stand.1) + w.terrain.curb_m(sl.stand.0, sl.stand.1);
            let view = look_at((sx, sz), floor + eye, [sl.x, sl.y_m + 0.05, sl.z]);
            let f = render::render_view(&w, &cam, &view, &render::Exposure::default());
            let px = f.ids.iter().filter(|&&v| unpack_id(v) == (ObjectClass::Drop, next)).count();
            assert!(px >= 20, "seed {seed}: the parcel on {:?} shows {px} px from {:?}", sl.support, sl.stand);
            next += 1;
        }
    }
}

#[test]
fn the_site_renders_as_structures_and_furniture_never_as_detections() {
    let mut w = sited(4, TerrainKind::Open, BaseKind::HumanFoot);
    let s = w.site.clone().unwrap();
    let h = &s.houses[0];
    let foot = cell_center(h.stair_foot[0].0, h.stair_foot[0].1);
    let door = cell_center(h.door[0].0, h.door[0].1);
    w.agent.x = foot.0;
    w.agent.z = foot.1;
    w.agent.yaw = (door.1 - foot.1).atan2(door.0 - foot.0);
    let cam = Camera { width: 160, height: 120, pitch_deg: 0.0, ..Camera::default() };
    let f = w.render(&cam);
    let structure = f.ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Structure).count();
    let curb = f.ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Curb).count();
    assert!(structure > 1000, "the house fills the view: {structure} px");
    assert_eq!(curb, 0, "the porch and the stair are drawn as such, not as the yellow kerb");
    assert!(w.visible_objects(&f).iter().all(|o| !matches!(o.class, ObjectClass::Structure | ObjectClass::Fixture)));
    let wh = s.warehouse.as_ref().unwrap();
    let (ax, az) = cell_center((wh.receiving.x1 + 1) as usize, ((wh.rect.z0 + wh.rect.z1) / 2) as usize);
    w.agent.x = ax;
    w.agent.z = az;
    w.agent.yaw = 0.0;
    let f = w.render(&cam);
    let furniture = f.ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Fixture).count();
    assert!(furniture > 500, "shelves in view: {furniture} px");
    assert!(w.visible_objects(&f).iter().all(|o| !matches!(o.class, ObjectClass::Structure | ObjectClass::Fixture)));
    let top = render::render_top(&w, 128, 32.0);
    assert_eq!(top.rgb.len(), 128 * 128 * 3);
}

#[test]
fn a_raised_parcel_is_picked_only_from_its_own_level() {
    let mut w = sited(4, TerrainKind::Open, BaseKind::HumanFoot);
    let s = w.site.clone().unwrap();
    let h = s.houses.iter().find(|h| h.reachable_legs).unwrap();
    let sl = s.slots.iter().find(|sl| matches!(sl.support, Support::Step { house, .. } if house == h.id)).unwrap().clone();
    w.drops.clear();
    w.drops.push(world_sim::world::DropItem::on(800, world_sim::ItemKind::Parcel, sl.x, sl.z, sl.support, sl.y_m));
    let foot = cell_center(h.stair_foot[0].0, h.stair_foot[0].1);
    let (dx, dz) = (sl.x - foot.0, sl.z - foot.1);
    let d = dx.hypot(dz);
    w.agent.x = sl.x - 0.5 * dx / d;
    w.agent.z = sl.z - 0.5 * dz / d;
    if w.agent_ground_m() < 0.05 {
        match w.command(Action::pick(800)) {
            Err(ActionError::CannotReach { need_m, .. }) => assert!((need_m - sl.y_m).abs() < 1e-4),
            other => panic!("{other:?}"),
        }
    }
    let (stx, stz) = cell_center(sl.stand.0, sl.stand.1);
    w.agent.x = stx;
    w.agent.z = stz;
    assert!((w.agent_ground_m() - sl.y_m).abs() < 1e-4);
    assert!(w.command(Action::pick(800)).is_ok());
    let mut w = sited(4, TerrainKind::Open, BaseKind::HumanFoot);
    let sl = w.site.as_ref().unwrap().slots.iter().find(|sl| matches!(sl.support, Support::Shelf { .. })).unwrap().clone();
    w.drops.push(world_sim::world::DropItem::on(801, world_sim::ItemKind::Parcel, sl.x, sl.z, sl.support, sl.y_m));
    let (stx, stz) = cell_center(sl.stand.0, sl.stand.1);
    w.agent.x = stx;
    w.agent.z = stz;
    assert!(matches!(w.command(Action::pick(801)), Err(ActionError::CannotReach { .. })));
    assert!(matches!(w.command(Action::Push { drop: 801, x: stx, z: stz }), Err(ActionError::CannotPush { .. })));
}

#[test]
fn the_search_prior_points_at_porches_and_the_logistics_centre_not_at_parcels() {
    let w = sited(7, TerrainKind::Open, BaseKind::Wheel);
    let prior = w.search_prior().unwrap();
    let s = w.site.as_ref().unwrap();
    let t = &w.terrain;
    let h = &s.houses[0];
    assert!(prior[t.idx(h.porch[0].0, h.porch[0].1)] > 1.0);
    let wh = s.warehouse.as_ref().unwrap();
    assert!(prior[t.idx(wh.aisles[0].0, wh.aisles[0].1)] > 1.0);
    let camp = t.cell_of(w.agent.x, w.agent.z).unwrap();
    assert_eq!(prior[t.idx(camp.0, camp.1)], 1.0);
    let mut moved = w.clone();
    moved.drops.truncate(1);
    assert_eq!(moved.search_prior().unwrap(), prior);
    let cov = Coverage::new(t);
    let blocked = w.blocked_cells();
    let spec = w.drive_spec();
    let start = camp;
    let eye = w.standing_eye_m();
    let plain = explore::next_viewpoint(t, &spec, &|x, z| blocked.contains(&(x, z)), start, &cov, eye, -12.0, 50.0, 6.0, &[]).unwrap();
    let with = explore::next_viewpoint_with(t, &spec, &|x, z| blocked.contains(&(x, z)), start, &cov, eye, -12.0, 50.0, 6.0, &[], Some(&prior)).unwrap();
    assert!((plain.weighted - plain.gain as f32).abs() < 1e-3, "without a prior every cell weighs 1");
    let worth = |v: &explore::Viewpoint| {
        let mut sum = 0.0;
        let e = explore::Eye::at(t, v.cell, eye, -12.0, 50.0);
        explore::viewshed(t, &explore::Eye { range_m: 6.0, ..e }, 0.0, std::f32::consts::TAU, &mut |i| sum += prior[i] - 1.0);
        sum
    };
    assert!(worth(&with) >= worth(&plain), "with {} vs plain {}", worth(&with), worth(&plain));
    let _ = nav::cell_distance_m(start, with.cell);
}

#[test]
fn fixtures_have_the_dimensions_of_their_sources() {
    let w = sited(4, TerrainKind::Open, BaseKind::Wheel);
    for f in &w.fixtures {
        match &f.kind {
            FixtureKind::Shelf { levels_mm, depth_mm, width_mm, height_mm } => {
                assert_eq!((*width_mm, *height_mm, *depth_mm), (1829, 1829, 457));
                assert_eq!(levels_mm, &vec![152, 914, 1676]);
                assert_eq!(f.cells.len(), 4);
                assert!(f.blocks());
            }
            FixtureKind::Table { top_mm } => assert!((864..=940).contains(top_mm)),
            FixtureKind::Door { width_mm, height_mm, .. } => assert!(*width_mm >= 813 && *height_mm >= 1981, "an egress door's clear opening"),
            _ => {}
        }
    }
    let s = w.site.as_ref().unwrap();
    for h in &s.houses {
        assert!(h.porch.len() as f32 * CELL_M * CELL_M >= 0.914 * 0.914);
        assert!(h.porch.iter().all(|c| w.terrain.curb_kind(c.0, c.1) == CurbKind::Floor && (w.terrain.curb_m(c.0, c.1) * 100.0).round() as u8 == h.floor_cm));
    }
}
