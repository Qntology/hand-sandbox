use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::fixture::{self, Fixture, FixtureKind, Owner};
use crate::kinds::Ground;
use crate::nav::Cell;
use crate::rng::Rng;
use crate::snapshot::{FixtureView, HouseView, RampView, SiteView, StairView, WarehouseView, ZoneView};
use crate::terrain::{cell_center, seg_dist, CurbKind, FlatZone, Terrain, CELL_M};

pub const HOUSE_W_CELLS: i32 = 10;
pub const HOUSE_D_CELLS: i32 = 8;
pub const DOOR_CELLS: i32 = 2;
pub const PORCH_W_CELLS: i32 = 4;
pub const PORCH_D_CELLS: i32 = 2;
pub const STAIR_W_CELLS: i32 = 2;
pub const RISER_MAX_MM: u16 = 180;
pub const RISER_MIN_MM: u16 = 150;
pub const TREAD_MM: u16 = 500;
pub const RISERS_MIN: u8 = 2;
pub const RISERS_MAX: u8 = 3;
pub const RAMP_RUN_PER_RISE: f32 = 12.0;
pub const WAREHOUSE_W_CELLS: i32 = 18;
pub const WAREHOUSE_D_CELLS: i32 = 14;
const WAREHOUSE_MIN_CELLS: (i32, i32) = (10, 8);
pub const WAREHOUSE_DOOR_CELLS: i32 = 5;
pub const WAYPOINT_INSIDE_CELLS: i32 = 4;
pub const RECEIVING_CELLS: i32 = 6;
pub const AISLE_CELLS: i32 = 2;
pub const SHELF_UNIT_CELLS: i32 = 4;
pub const SLOT_TOP_M: f32 = 1.162;
pub const RACK_FLOOR_LX: f32 = 150.0;
pub const PACKING_LX: f32 = 300.0;
pub const PRIOR_PORCH: f32 = 4.0;
pub const PRIOR_WAREHOUSE: f32 = 3.0;
const PRIOR_PORCH_RING: i32 = 4;
const KEEP_WAYPOINT_M: f32 = 4.5;
const ZONE_SPACING_M: f32 = 2.0;
const SITE_SALT: u64 = 0x517E_5EED_4A11_C0DE;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SiteCfg {
    pub houses: u8,
    pub warehouse: bool,
    pub ramps: bool,
}

impl Default for SiteCfg {
    fn default() -> Self {
        SiteCfg { houses: 3, warehouse: true, ramps: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Facing {
    PlusX,
    PlusZ,
    MinusX,
    MinusZ,
}

impl Facing {
    pub const ALL: [Facing; 4] = [Facing::PlusX, Facing::PlusZ, Facing::MinusX, Facing::MinusZ];

    pub fn dir(self) -> (i32, i32) {
        match self {
            Facing::PlusX => (1, 0),
            Facing::PlusZ => (0, 1),
            Facing::MinusX => (-1, 0),
            Facing::MinusZ => (0, -1),
        }
    }

    pub fn yaw_deg(self) -> f32 {
        match self {
            Facing::PlusX => 0.0,
            Facing::PlusZ => 90.0,
            Facing::MinusX => 180.0,
            Facing::MinusZ => 270.0,
        }
    }

    fn along(self) -> (i32, i32) {
        let (x, z) = self.dir();
        (-z, x)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "on", rename_all = "snake_case")]
pub enum Support {
    #[default]
    Ground,
    Step { house: u32, level: u8 },
    Shelf { fixture: u32, level: u8 },
    Table { fixture: u32 },
}

impl Support {
    pub fn is_ground(&self) -> bool {
        matches!(self, Support::Ground)
    }

    pub fn label_ko(&self) -> String {
        match self {
            Support::Ground => "바닥".into(),
            Support::Step { level: 0, .. } => "현관 앞".into(),
            Support::Step { level, .. } => format!("계단 {level}단"),
            Support::Shelf { level, .. } => format!("선반 {}단", level + 1),
            Support::Table { .. } => "포장 테이블 위".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    pub support: Support,
    pub x: f32,
    pub z: f32,
    pub y_m: f32,
    pub owner: Owner,
    pub stand: Cell,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x0: i32,
    pub z0: i32,
    pub x1: i32,
    pub z1: i32,
}

impl Rect {
    fn grow(self, n: i32) -> Rect {
        Rect { x0: self.x0 - n, z0: self.z0 - n, x1: self.x1 + n, z1: self.z1 + n }
    }

    #[cfg(test)]
    fn overlaps(&self, o: &Rect) -> bool {
        self.x0 <= o.x1 && o.x0 <= self.x1 && self.z0 <= o.z1 && o.z0 <= self.z1
    }

    pub fn contains(&self, c: Cell) -> bool {
        (self.x0..=self.x1).contains(&(c.0 as i32)) && (self.z0..=self.z1).contains(&(c.1 as i32))
    }

    fn cells(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        (self.z0..=self.z1).flat_map(move |z| (self.x0..=self.x1).map(move |x| (x, z)))
    }

    pub fn metres(&self) -> [f32; 4] {
        [self.x0 as f32 * CELL_M, self.z0 as f32 * CELL_M, (self.x1 + 1) as f32 * CELL_M, (self.z1 + 1) as f32 * CELL_M]
    }

    fn dist_m(&self, p: (f32, f32)) -> f32 {
        let m = self.metres();
        let dx = (m[0] - p.0).max(0.0).max(p.0 - m[2]);
        let dz = (m[1] - p.1).max(0.0).max(p.1 - m[3]);
        dx.hypot(dz)
    }
}

#[derive(Clone, Debug)]
pub struct Keep {
    pub base: (f32, f32),
    pub warehouse: (f32, f32),
    pub wall_cx: usize,
    pub way: Vec<(f32, f32)>,
    pub way_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HousePlan {
    origin: (i32, i32),
    front: Facing,
    flip: bool,
    risers: u8,
    riser_cm: u8,
    ramp: u8,
}

impl HousePlan {
    fn at(&self, u: i32, v: i32) -> (i32, i32) {
        let u = if self.flip { HOUSE_W_CELLS - 1 - u } else { u };
        let (fx, fz) = self.front.dir();
        let (ax, az) = self.front.along();
        (self.origin.0 + u * ax + v * fx, self.origin.1 + u * az + v * fz)
    }

    fn floor_cm(&self) -> u8 {
        self.risers * self.riser_cm
    }

    fn bounds(&self) -> Rect {
        let n = self.risers as i32;
        let u1 = (HOUSE_W_CELLS - 1).max(PORCH_W_CELLS + self.ramp as i32);
        let corners = [(0, -(HOUSE_D_CELLS - 1)), (u1, -(HOUSE_D_CELLS - 1)), (0, PORCH_D_CELLS + n), (u1, PORCH_D_CELLS + n)];
        let pts: Vec<(i32, i32)> = corners.iter().map(|&(u, v)| self.at(u, v)).collect();
        Rect {
            x0: pts.iter().map(|p| p.0).min().unwrap(),
            z0: pts.iter().map(|p| p.1).min().unwrap(),
            x1: pts.iter().map(|p| p.0).max().unwrap(),
            z1: pts.iter().map(|p| p.1).max().unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WarehousePlan {
    rect: Rect,
    waypoint: (i32, i32),
    door_z: (i32, i32),
}

#[derive(Clone, Debug)]
pub struct SitePlan {
    pub cfg: SiteCfg,
    houses: Vec<HousePlan>,
    warehouse: Option<WarehousePlan>,
}

impl SitePlan {
    pub fn zones(&self) -> Vec<FlatZone> {
        let mut out = Vec::new();
        for r in self.reserved() {
            let m = r.metres();
            let (w, d) = (m[2] - m[0], m[3] - m[1]);
            let (nx, nz) = ((w / ZONE_SPACING_M).ceil().max(1.0), (d / ZONE_SPACING_M).ceil().max(1.0));
            let (sx, sz) = (w / nx, d / nz);
            let radius = 0.5 * sx.hypot(sz) + 0.05;
            for i in 0..nx as usize {
                for j in 0..nz as usize {
                    out.push((m[0] + (i as f32 + 0.5) * sx, m[1] + (j as f32 + 0.5) * sz, radius));
                }
            }
        }
        out
    }

    pub fn reserved(&self) -> Vec<Rect> {
        let mut v: Vec<Rect> = self.houses.iter().map(|h| h.bounds().grow(1)).collect();
        if let Some(w) = &self.warehouse {
            v.push(w.rect.grow(1));
        }
        v
    }

    pub fn house_count(&self) -> usize {
        self.houses.len()
    }
}

pub fn plan(seed: u64, cells: usize, cfg: &SiteCfg, keep: &Keep) -> SitePlan {
    let n = cells as i32;
    let mut rng = Rng::new(seed ^ SITE_SALT);
    let warehouse = if cfg.warehouse { plan_warehouse(n, keep) } else { None };
    let mut free = FreeMap::new(n, keep, warehouse.is_some());
    if let Some(w) = &warehouse {
        free.take(&w.rect.grow(1));
    }
    let mut houses = Vec::new();
    for _ in 0..cfg.houses {
        let risers = RISERS_MIN + rng.below((RISERS_MAX - RISERS_MIN + 1) as u32) as u8;
        let riser_mm = RISER_MIN_MM + rng.below((RISER_MAX_MM - RISER_MIN_MM) as u32 / 10 + 1) as u16 * 10;
        let pick = rng.next_u64();
        let riser_cm = (riser_mm / 10) as u8;
        let floor_cm = risers as f32 * riser_cm as f32;
        let ramp = if cfg.ramps { ((floor_cm / (CELL_M * 100.0 / RAMP_RUN_PER_RISE)).ceil() as i32 - 1).max(1) as u8 } else { 0 };
        let mut best: Vec<HousePlan> = Vec::new();
        let mut best_contact = -1i32;
        for front in Facing::ALL {
            for flip in [false, true] {
                for oz in 0..n {
                    for ox in 0..n {
                        let h = HousePlan { origin: (ox, oz), front, flip, risers, riser_cm, ramp };
                        let r = h.bounds().grow(1);
                        if !free.fits(&r) {
                            continue;
                        }
                        let contact = free.contact(&r);
                        if contact > best_contact {
                            best_contact = contact;
                            best.clear();
                        }
                        if contact == best_contact {
                            best.push(h);
                        }
                    }
                }
            }
        }
        if best.is_empty() {
            break;
        }
        let h = best[(pick % best.len() as u64) as usize];
        free.take(&h.bounds().grow(1));
        houses.push(h);
    }
    SitePlan { cfg: *cfg, houses, warehouse }
}

struct FreeMap {
    n: i32,
    blocked: Vec<bool>,
    sum: Vec<i32>,
}

impl FreeMap {
    fn new(n: i32, keep: &Keep, warehouse: bool) -> FreeMap {
        let mut blocked = vec![false; (n * n) as usize];
        for z in 0..n {
            for x in 0..n {
                blocked[(z * n + x) as usize] = !cell_free(x, z, n, keep, warehouse);
            }
        }
        let mut m = FreeMap { n, blocked, sum: Vec::new() };
        m.rebuild();
        m
    }

    fn rebuild(&mut self) {
        let (n, w) = (self.n, (self.n + 1) as usize);
        self.sum = vec![0; w * w];
        for z in 0..n as usize {
            for x in 0..n as usize {
                self.sum[(z + 1) * w + x + 1] = self.blocked[z * n as usize + x] as i32 + self.sum[z * w + x + 1] + self.sum[(z + 1) * w + x] - self.sum[z * w + x];
            }
        }
    }

    fn is_blocked(&self, x: i32, z: i32) -> bool {
        x < 0 || z < 0 || x >= self.n || z >= self.n || self.blocked[(z * self.n + x) as usize]
    }

    fn fits(&self, r: &Rect) -> bool {
        if r.x0 < 0 || r.z0 < 0 || r.x1 >= self.n || r.z1 >= self.n {
            return false;
        }
        let w = (self.n + 1) as usize;
        let (x0, z0, x1, z1) = (r.x0 as usize, r.z0 as usize, r.x1 as usize + 1, r.z1 as usize + 1);
        self.sum[z1 * w + x1] - self.sum[z0 * w + x1] - self.sum[z1 * w + x0] + self.sum[z0 * w + x0] == 0
    }

    fn contact(&self, r: &Rect) -> i32 {
        let ring = r.grow(1);
        let mut c = 0;
        for x in ring.x0..=ring.x1 {
            c += self.is_blocked(x, ring.z0) as i32 + self.is_blocked(x, ring.z1) as i32;
        }
        for z in r.z0..=r.z1 {
            c += self.is_blocked(ring.x0, z) as i32 + self.is_blocked(ring.x1, z) as i32;
        }
        c
    }

    fn take(&mut self, r: &Rect) {
        for (x, z) in r.cells() {
            if (0..self.n).contains(&x) && (0..self.n).contains(&z) {
                self.blocked[(z * self.n + x) as usize] = true;
            }
        }
        self.rebuild();
    }
}

fn plan_warehouse(n: i32, keep: &Keep) -> Option<WarehousePlan> {
    let (wx, wz) = ((keep.warehouse.0 / CELL_M) as i32, (keep.warehouse.1 / CELL_M) as i32);
    let x0 = wx - WAYPOINT_INSIDE_CELLS;
    let x1 = (x0 + WAREHOUSE_W_CELLS - 1).min(n - 2);
    let z0 = (wz - WAREHOUSE_D_CELLS / 2).max(1);
    let z1 = (z0 + WAREHOUSE_D_CELLS - 1).min(n - 2);
    if x0 <= keep.wall_cx as i32 + 2 || x1 - x0 + 1 < WAREHOUSE_MIN_CELLS.0 || z1 - z0 + 1 < WAREHOUSE_MIN_CELLS.1 {
        return None;
    }
    let half = WAREHOUSE_DOOR_CELLS / 2;
    let door_z = ((wz - half).max(z0 + 1), (wz + half).min(z1 - 1));
    Some(WarehousePlan { rect: Rect { x0, z0, x1, z1 }, waypoint: (wx, wz), door_z })
}

fn cell_free(x: i32, z: i32, n: i32, keep: &Keep, warehouse: bool) -> bool {
    if x < 1 || z < 1 || x > n - 2 || z > n - 2 {
        return false;
    }
    if x.abs_diff(keep.wall_cx as i32) <= 2 {
        return false;
    }
    let r = Rect { x0: x, z0: z, x1: x, z1: z };
    if r.dist_m(keep.base) < KEEP_WAYPOINT_M || (!warehouse && r.dist_m(keep.warehouse) < KEEP_WAYPOINT_M) {
        return false;
    }
    let (px, pz) = cell_center(x as usize, z as usize);
    keep.way.windows(2).all(|s| seg_dist(px, pz, s[0], s[1]) > keep.way_m + 0.36)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct House {
    pub id: u32,
    pub rect: Rect,
    pub front: Facing,
    pub floor_cm: u8,
    pub risers: u8,
    pub riser_mm: u16,
    pub tread_mm: u16,
    pub door: Vec<Cell>,
    pub porch: Vec<Cell>,
    pub treads: Vec<Vec<Cell>>,
    pub ramp: Vec<Cell>,
    pub stair_foot: Vec<Cell>,
    pub ramp_foot: Vec<Cell>,
    pub reachable_legs: bool,
    pub reachable_wheels: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Warehouse {
    pub rect: Rect,
    pub door: Vec<Cell>,
    pub door_facing: Facing,
    pub shelves: Vec<u32>,
    pub tables: Vec<u32>,
    pub aisles: Vec<Cell>,
    pub receiving: Rect,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub cfg: SiteCfg,
    pub houses: Vec<House>,
    pub warehouse: Option<Warehouse>,
    pub slots: Vec<Slot>,
    pub reserved: Vec<Rect>,
}

pub fn build(plan: &SitePlan, t: &mut Terrain, next_id: &mut u32) -> (Site, Vec<Fixture>) {
    let mut fixtures = Vec::new();
    let mut slots = Vec::new();
    let mut houses = Vec::new();
    let mut take_id = || {
        let id = *next_id;
        *next_id += 1;
        id
    };
    let cell = |p: (i32, i32)| (p.0 as usize, p.1 as usize);
    for (k, h) in plan.houses.iter().enumerate() {
        let hid = k as u32 + 1;
        let owner = Owner::House(hid);
        let floor = h.floor_cm();
        let n = h.risers as i32;
        let body = Rect {
            x0: h.at(0, 0).0.min(h.at(HOUSE_W_CELLS - 1, -(HOUSE_D_CELLS - 1)).0),
            z0: h.at(0, 0).1.min(h.at(HOUSE_W_CELLS - 1, -(HOUSE_D_CELLS - 1)).1),
            x1: h.at(0, 0).0.max(h.at(HOUSE_W_CELLS - 1, -(HOUSE_D_CELLS - 1)).0),
            z1: h.at(0, 0).1.max(h.at(HOUSE_W_CELLS - 1, -(HOUSE_D_CELLS - 1)).1),
        };
        let raise = |t: &mut Terrain, c: Cell, cm: u8, kind: CurbKind| {
            t.set_height_cm(c.0, c.1, 0);
            t.set_curb_cm(c.0, c.1, cm);
            t.set_curb_kind(c.0, c.1, kind);
            t.set_ground(c.0, c.1, Ground::Floor);
        };
        let mut floor_cells = Vec::new();
        let mut walls = Vec::new();
        let mut door = Vec::new();
        for u in 0..HOUSE_W_CELLS {
            for v in -(HOUSE_D_CELLS - 1)..=0 {
                let c = cell(h.at(u, v));
                raise(t, c, floor, CurbKind::Floor);
                floor_cells.push(c);
                let edge = u == 0 || u == HOUSE_W_CELLS - 1 || v == 0 || v == -(HOUSE_D_CELLS - 1);
                if v == 0 && (1..=DOOR_CELLS).contains(&u) {
                    door.push(c);
                } else if edge {
                    walls.push(c);
                }
            }
        }
        let mut porch = Vec::new();
        for u in 0..PORCH_W_CELLS {
            for v in 1..=PORCH_D_CELLS {
                let c = cell(h.at(u, v));
                raise(t, c, floor, CurbKind::Floor);
                porch.push(c);
            }
        }
        let mut treads = Vec::new();
        for k in 1..n {
            let mut row = Vec::new();
            for u in 1..=STAIR_W_CELLS {
                let c = cell(h.at(u, PORCH_D_CELLS + k));
                raise(t, c, (n - k) as u8 * h.riser_cm, CurbKind::Step);
                row.push(c);
            }
            treads.push(row);
        }
        let stair_foot: Vec<Cell> = (1..=STAIR_W_CELLS).map(|u| cell(h.at(u, PORCH_D_CELLS + n))).collect();
        let l = h.ramp as i32;
        let mut ramp = Vec::new();
        for j in 1..=l {
            let cm = (floor as f32 * (l + 1 - j) as f32 / (l + 1) as f32).round() as u8;
            for v in 1..=PORCH_D_CELLS {
                let c = cell(h.at(PORCH_W_CELLS - 1 + j, v));
                raise(t, c, cm, CurbKind::Ramp);
                ramp.push(c);
            }
        }
        let ramp_foot: Vec<Cell> = if l > 0 { (1..=PORCH_D_CELLS).map(|v| cell(h.at(PORCH_W_CELLS + l, v))).collect() } else { vec![] };
        let yaw = h.front.yaw_deg();
        let base = floor as f32 / 100.0;
        let mut push = |kind: FixtureKind, cells: Vec<Cell>, yaw_deg: f32, base_m: f32| {
            fixtures.push(Fixture { id: take_id(), kind, cells, yaw_deg, base_m, owner });
        };
        push(FixtureKind::Floor, floor_cells, yaw, 0.0);
        push(FixtureKind::Landing, porch.clone(), yaw, 0.0);
        let rise_mm = h.riser_cm as u16 * 10;
        if !treads.is_empty() {
            push(FixtureKind::Step { rise_mm, tread_mm: TREAD_MM }, treads.iter().flatten().copied().collect(), yaw, 0.0);
        }
        if !ramp.is_empty() {
            push(FixtureKind::Ramp { run_per_rise: RAMP_RUN_PER_RISE as u8 }, ramp.clone(), yaw, 0.0);
        }
        let wall_mm = (fixture::HOUSE_WALL_H_M * 1000.0) as u16;
        push(FixtureKind::Wall { height_mm: wall_mm, thick_mm: (fixture::WALL_THICK_M * 1000.0) as u16 }, walls, yaw, base);
        push(FixtureKind::Door { width_mm: fixture::HOUSE_DOOR_W_MM, height_mm: fixture::HOUSE_DOOR_H_MM, wall_height_mm: wall_mm }, door.clone(), yaw, base);
        push(FixtureKind::Roof { height_mm: wall_mm }, body.cells().map(cell).collect(), yaw, base);
        for u in [0, PORCH_W_CELLS - 1] {
            for v in 1..=PORCH_D_CELLS {
                let c = cell(h.at(u, v));
                let stand = cell(h.at(if u == 0 { 1 } else { PORCH_W_CELLS - 2 }, v));
                let (x, z) = cell_center(c.0, c.1);
                slots.push(Slot { support: Support::Step { house: hid, level: 0 }, x, z, y_m: base, owner, stand });
            }
        }
        houses.push(House {
            id: hid,
            rect: body,
            front: h.front,
            floor_cm: floor,
            risers: h.risers,
            riser_mm: rise_mm,
            tread_mm: TREAD_MM,
            door,
            porch,
            treads,
            ramp,
            stair_foot,
            ramp_foot,
            reachable_legs: true,
            reachable_wheels: l > 0,
        });
    }

    let warehouse = plan.warehouse.map(|w| {
        let owner = Owner::Warehouse;
        let r = w.rect;
        let mut walls = Vec::new();
        let mut door = Vec::new();
        for (x, z) in r.cells() {
            let c = cell((x, z));
            t.set_height_cm(c.0, c.1, 0);
            t.set_curb_cm(c.0, c.1, 0);
            t.set_ground(c.0, c.1, Ground::Floor);
            let edge = x == r.x0 || x == r.x1 || z == r.z0 || z == r.z1;
            if x == r.x0 && (w.door_z.0..=w.door_z.1).contains(&z) {
                door.push(c);
            } else if edge {
                walls.push(c);
            }
        }
        let wall_mm = (fixture::WAREHOUSE_WALL_H_M * 1000.0) as u16;
        let door_yaw = Facing::MinusX.yaw_deg();
        let mut push = |kind: FixtureKind, cells: Vec<Cell>, yaw_deg: f32| -> u32 {
            let id = take_id();
            fixtures.push(Fixture { id, kind, cells, yaw_deg, base_m: 0.0, owner });
            id
        };
        push(FixtureKind::Wall { height_mm: wall_mm, thick_mm: (fixture::WALL_THICK_M * 1000.0) as u16 }, walls, door_yaw);
        push(FixtureKind::Door { width_mm: fixture::WAREHOUSE_DOOR_W_MM, height_mm: fixture::WAREHOUSE_DOOR_H_MM, wall_height_mm: wall_mm }, door.clone(), door_yaw);
        push(FixtureKind::Roof { height_mm: wall_mm }, r.cells().map(cell).collect(), door_yaw);
        let inner = Rect { x0: r.x0 + 1, z0: r.z0 + 1, x1: r.x1 - 1, z1: r.z1 - 1 };
        let receiving = Rect { x0: inner.x0, z0: inner.z0, x1: (inner.x0 + RECEIVING_CELLS - 1).min(inner.x1), z1: inner.z1 };
        let keep_free = |c: (i32, i32)| (c.1 >= w.door_z.0 - 1 && c.1 <= w.door_z.1 + 1) || c == w.waypoint || c == (w.waypoint.0 - 1, w.waypoint.1);
        let mut tables = Vec::new();
        for (zs, facing) in [((inner.z0, inner.z0 + 1), Facing::PlusZ), ((inner.z1 - 1, inner.z1), Facing::MinusZ)] {
            let rect = Rect { x0: inner.x0 + 1, z0: zs.0, x1: (inner.x0 + 3).min(receiving.x1), z1: zs.1 };
            if rect.x1 < rect.x0 || rect.cells().any(keep_free) || rect.z0 > rect.z1 {
                continue;
            }
            let cells: Vec<Cell> = rect.cells().map(cell).collect();
            let id = push(FixtureKind::Table { top_mm: fixture::TABLE_TOP_MM }, cells.clone(), facing.yaw_deg());
            let (dx, dz) = facing.dir();
            for c in cells {
                let stand = ((c.0 as i32 + dx) as usize, (c.1 as i32 + dz) as usize);
                if rect.contains(stand) {
                    let s2 = ((stand.0 as i32 + dx) as usize, (stand.1 as i32 + dz) as usize);
                    let (x, z) = cell_center(c.0, c.1);
                    slots.push(Slot { support: Support::Table { fixture: id }, x, z, y_m: fixture::TABLE_TOP_MM as f32 / 1000.0, owner, stand: s2 });
                } else {
                    let (x, z) = cell_center(c.0, c.1);
                    slots.push(Slot { support: Support::Table { fixture: id }, x, z, y_m: fixture::TABLE_TOP_MM as f32 / 1000.0, owner, stand });
                }
            }
            tables.push(id);
        }
        let sx0 = receiving.x1 + 2;
        let sx1 = inner.x1 - 1;
        let units = ((sx1 - sx0 + 1) / SHELF_UNIT_CELLS).max(0);
        let mut rows: Vec<(i32, Facing)> = Vec::new();
        if units > 0 && inner.z1 - inner.z0 + 1 >= 1 + AISLE_CELLS + 1 {
            let mut z = inner.z0;
            rows.push((z, Facing::PlusZ));
            z += 1;
            while z + AISLE_CELLS <= inner.z1 {
                z += AISLE_CELLS;
                let left = inner.z1 - z + 1;
                if left >= 2 + AISLE_CELLS {
                    rows.push((z, Facing::MinusZ));
                    rows.push((z + 1, Facing::PlusZ));
                    z += 2;
                } else {
                    rows.push((inner.z1, Facing::MinusZ));
                    break;
                }
            }
        }
        let mut shelves = Vec::new();
        let mut aisles: HashSet<Cell> = HashSet::new();
        let levels: Vec<u16> = fixture::SHELF_LEVELS_MM.to_vec();
        for &(z, facing) in &rows {
            for u in 0..units {
                let cells: Vec<Cell> = (0..SHELF_UNIT_CELLS).map(|i| cell((sx0 + u * SHELF_UNIT_CELLS + i, z))).collect();
                let id = push(
                    FixtureKind::Shelf { levels_mm: levels.clone(), depth_mm: fixture::SHELF_DEPTH_MM, width_mm: fixture::SHELF_WIDTH_MM, height_mm: fixture::SHELF_HEIGHT_MM },
                    cells.clone(),
                    facing.yaw_deg(),
                );
                let (_, dz) = facing.dir();
                for c in cells {
                    let stand = (c.0, (c.1 as i32 + dz) as usize);
                    aisles.insert(stand);
                    let (x, z) = cell_center(c.0, c.1);
                    for (li, &l) in levels.iter().enumerate() {
                        let y = l as f32 / 1000.0;
                        if y <= SLOT_TOP_M {
                            slots.push(Slot { support: Support::Shelf { fixture: id, level: li as u8 }, x, z, y_m: y, owner, stand });
                        }
                    }
                }
                shelves.push(id);
            }
        }
        let mut aisles: Vec<Cell> = aisles.into_iter().collect();
        aisles.sort_unstable();
        Warehouse { rect: r, door, door_facing: Facing::MinusX, shelves, tables, aisles, receiving }
    });

    let site = Site { cfg: plan.cfg, houses, warehouse, slots, reserved: plan.reserved() };
    (site, fixtures)
}

pub fn choose_slots<'a>(site: &'a Site, n: usize, rng: &mut Rng) -> Vec<&'a Slot> {
    let mut owners: Vec<Owner> = Vec::new();
    for s in &site.slots {
        if !owners.contains(&s.owner) {
            owners.push(s.owner);
        }
    }
    let mut used = vec![false; site.slots.len()];
    let mut out = Vec::new();
    while out.len() < n {
        let open: Vec<Owner> = owners.iter().copied().filter(|o| site.slots.iter().zip(&used).any(|(s, u)| s.owner == *o && !*u)).collect();
        if open.is_empty() {
            break;
        }
        let o = open[rng.below(open.len() as u32) as usize];
        let free: Vec<usize> = (0..site.slots.len()).filter(|&i| site.slots[i].owner == o && !used[i]).collect();
        let i = free[rng.below(free.len() as u32) as usize];
        used[i] = true;
        out.push(&site.slots[i]);
    }
    out
}

impl Site {
    pub fn is_reserved(&self, c: Cell) -> bool {
        self.reserved.iter().any(|r| r.contains(c))
    }

    pub fn house(&self, id: u32) -> Option<&House> {
        self.houses.iter().find(|h| h.id == id)
    }

    pub fn support_top_m(&self, fixtures: &[Fixture], s: Support) -> Option<f32> {
        let fixture = |id: u32| fixtures.iter().find(|f| f.id == id);
        match s {
            Support::Ground => None,
            Support::Step { house, level } => {
                let h = self.house(house)?;
                (level < h.risers).then(|| (h.floor_cm as f32 - level as f32 * h.riser_mm as f32 / 10.0) / 100.0)
            }
            Support::Shelf { fixture: id, level } => {
                let f = fixture(id).filter(|f| matches!(f.kind, FixtureKind::Shelf { .. }))?;
                f.support_levels_m().get(level as usize).map(|l| f.base_m + l)
            }
            Support::Table { fixture: id } => {
                let f = fixture(id).filter(|f| matches!(f.kind, FixtureKind::Table { .. }))?;
                f.support_levels_m().first().map(|l| f.base_m + l)
            }
        }
    }

    pub fn indoor(&self, c: Cell) -> bool {
        self.houses.iter().any(|h| h.rect.contains(c)) || self.warehouse.as_ref().is_some_and(|w| w.rect.contains(c))
    }

    pub fn prior(&self, t: &Terrain) -> Vec<f32> {
        let mut w = vec![1.0f32; t.w * t.d];
        for h in self.houses.iter().filter(|h| h.reachable_legs || h.reachable_wheels) {
            for &(px, pz) in h.porch.iter().chain(h.treads.iter().flatten()) {
                for dz in -PRIOR_PORCH_RING..=PRIOR_PORCH_RING {
                    for dx in -PRIOR_PORCH_RING..=PRIOR_PORCH_RING {
                        let (x, z) = (px as i32 + dx, pz as i32 + dz);
                        if t.in_bounds(x, z) && !h.rect.contains((x as usize, z as usize)) {
                            let i = t.idx(x as usize, z as usize);
                            w[i] = w[i].max(PRIOR_PORCH);
                        }
                    }
                }
            }
        }
        if let Some(wh) = &self.warehouse {
            for (x, z) in wh.rect.cells() {
                let i = t.idx(x as usize, z as usize);
                w[i] = w[i].max(PRIOR_WAREHOUSE);
            }
        }
        w
    }

    pub fn view(&self, fixtures: &[Fixture], t: &Terrain) -> SiteView {
        let at = |c: &Cell| {
            let (x, z) = cell_center(c.0, c.1);
            [x, z]
        };
        let top = |c: &Cell| t.height_m(c.0, c.1) + t.curb_m(c.0, c.1);
        let span = |cells: &[Cell]| -> [f32; 4] {
            let r = Rect {
                x0: cells.iter().map(|c| c.0 as i32).min().unwrap_or(0),
                z0: cells.iter().map(|c| c.1 as i32).min().unwrap_or(0),
                x1: cells.iter().map(|c| c.0 as i32).max().unwrap_or(0),
                z1: cells.iter().map(|c| c.1 as i32).max().unwrap_or(0),
            };
            r.metres()
        };
        let mean = |cells: &[Cell]| -> [f32; 2] {
            let n = cells.len().max(1) as f32;
            let s = cells.iter().map(at).fold([0.0, 0.0], |a, b| [a[0] + b[0], a[1] + b[1]]);
            [s[0] / n, s[1] / n]
        };
        let houses = self
            .houses
            .iter()
            .map(|h| HouseView {
                id: h.id,
                label_ko: format!("집 {}", h.id),
                rect_m: h.rect.metres(),
                front_deg: h.front.yaw_deg(),
                floor_m: h.floor_cm as f32 / 100.0,
                door: mean(&h.door),
                porch_m: span(&h.porch),
                stairs: StairView {
                    risers: h.risers,
                    riser_m: h.riser_mm as f32 / 1000.0,
                    tread_m: h.tread_mm as f32 / 1000.0,
                    width_m: STAIR_W_CELLS as f32 * CELL_M,
                    treads: h.treads.iter().flatten().map(|c| [at(c)[0], at(c)[1], top(c)]).collect(),
                },
                ramp: (!h.ramp.is_empty()).then(|| RampView {
                    cells: h.ramp.iter().map(|c| [at(c)[0], at(c)[1], top(c)]).collect(),
                    rise_m: h.floor_cm as f32 / 100.0,
                    run_m: (h.ramp.len() as f32 / PORCH_D_CELLS as f32 + 1.0) * CELL_M,
                    run_per_rise: RAMP_RUN_PER_RISE,
                }),
                reachable_legs: h.reachable_legs,
                reachable_wheels: h.reachable_wheels,
            })
            .collect();
        let warehouse = self.warehouse.as_ref().map(|w| {
            let fx = |ids: &[u32]| -> Vec<&Fixture> { fixtures.iter().filter(|f| ids.contains(&f.id)).collect() };
            let mut zones = vec![ZoneView { kind: "packing".into(), label_ko: "입고·포장".into(), rect_m: w.receiving.metres(), em_lx: PACKING_LX }];
            if !w.shelves.is_empty() {
                let r = Rect { x0: w.receiving.x1 + 1, z0: w.receiving.z0, x1: w.rect.x1 - 1, z1: w.receiving.z1 };
                zones.push(ZoneView { kind: "rack".into(), label_ko: "선반 통로".into(), rect_m: r.metres(), em_lx: RACK_FLOOR_LX });
            }
            WarehouseView {
                label_ko: "물류센터".into(),
                rect_m: w.rect.metres(),
                door: mean(&w.door),
                door_width_m: fixture::WAREHOUSE_DOOR_W_MM as f32 / 1000.0,
                door_front_deg: w.door_facing.yaw_deg(),
                aisle_m: AISLE_CELLS as f32 * CELL_M,
                shelves: fx(&w.shelves).len(),
                tables: fx(&w.tables).len(),
                zones,
            }
        });
        let fixtures = fixtures
            .iter()
            .filter(|f| !matches!(f.kind, FixtureKind::Floor | FixtureKind::Roof { .. }))
            .map(|f| FixtureView {
                id: f.id,
                kind: f.kind_id().into(),
                label_ko: f.label_ko().into(),
                cells: f.cells.iter().map(at).collect(),
                yaw_deg: f.yaw_deg,
                base_m: f.base_m,
                top_m: f.base_m + f.height_m(),
                levels_m: f.support_levels_m().iter().map(|l| f.base_m + l).collect(),
                blocks: f.blocks(),
            })
            .collect();
        SiteView { houses, warehouse, fixtures, slots: self.slots.len() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep() -> Keep {
        Keep { base: (4.48, 4.48), warehouse: (24.96, 21.12), wall_cx: 29, way: vec![(4.48, 4.48), (4.48, 28.0), (14.75, 28.0), (24.96, 21.12)], way_m: 2.2 }
    }

    #[test]
    fn a_plan_is_deterministic_and_keeps_clear_of_the_way_and_the_waypoints() {
        let cfg = SiteCfg::default();
        let a = plan(5, 64, &cfg, &keep());
        let b = plan(5, 64, &cfg, &keep());
        assert_eq!(a.houses, b.houses);
        assert!(a.warehouse.is_some());
        let k = keep();
        for r in a.reserved() {
            assert!(r.x0 >= 0 && r.x1 < 64 && r.z0 >= 0 && r.z1 < 64);
            for (x, z) in r.cells() {
                let (px, pz) = cell_center(x as usize, z as usize);
                assert!(((px - k.base.0).hypot(pz - k.base.1)) > 3.5, "a site on the base camp's pad");
            }
        }
        let rs = a.reserved();
        for i in 0..rs.len() {
            for j in i + 1..rs.len() {
                assert!(!rs[i].overlaps(&rs[j]), "{:?} {:?}", rs[i], rs[j]);
            }
        }
        let zones = a.zones();
        for r in &rs {
            for (x, z) in r.cells() {
                let (px, pz) = cell_center(x as usize, z as usize);
                assert!(zones.iter().any(|&(zx, zz, zr)| (px - zx).hypot(pz - zz) <= zr), "({x},{z}) not flattened");
            }
        }
    }

    #[test]
    fn stairs_follow_the_riser_rules_and_the_ramp_the_gradient() {
        for seed in 1..=8u64 {
            let p = plan(seed, 64, &SiteCfg::default(), &keep());
            for h in &p.houses {
                let r = h.riser_cm as u16 * 10;
                assert!((RISER_MIN_MM..=RISER_MAX_MM).contains(&r), "{r}");
                assert!((RISERS_MIN..=RISERS_MAX).contains(&h.risers));
                let per = h.floor_cm() as f32 / (h.ramp as f32 + 1.0);
                assert!(per <= CELL_M * 100.0 / RAMP_RUN_PER_RISE + 1e-3, "{per} cm per cell");
            }
        }
    }
}
