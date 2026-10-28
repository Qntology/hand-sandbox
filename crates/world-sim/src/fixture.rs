use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::nav::Cell;
use crate::render::ObjectClass;
use crate::terrain::{Terrain, CELL_M};

pub const HOUSE_WALL_H_M: f32 = 2.4;
pub const WAREHOUSE_WALL_H_M: f32 = 3.6;
pub const WALL_THICK_M: f32 = 0.2;
pub const ROOF_THICK_M: f32 = 0.15;
pub const HOUSE_DOOR_W_MM: u16 = 914;
pub const HOUSE_DOOR_H_MM: u16 = 2032;
pub const WAREHOUSE_DOOR_W_MM: u16 = 2500;
pub const WAREHOUSE_DOOR_H_MM: u16 = 3000;
pub const SHELF_WIDTH_MM: u16 = 1829;
pub const SHELF_HEIGHT_MM: u16 = 1829;
pub const SHELF_DEPTH_MM: u16 = 457;
pub const SHELF_POST_MM: u16 = 38;
pub const SHELF_DECK_MM: u16 = 54;
pub const SHELF_LEVELS_MM: [u16; 3] = [152, 914, 1676];
pub const TABLE_TOP_MM: u16 = 900;
pub const TABLE_TOP_THICK_MM: u16 = 30;
pub const TABLE_LEG_MM: u16 = 40;
const COVER_EPS_M: f32 = 0.001;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "site", content = "id", rename_all = "snake_case")]
pub enum Owner {
    House(u32),
    Warehouse,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FixtureKind {
    Wall { height_mm: u16, thick_mm: u16 },
    Door { width_mm: u16, height_mm: u16, wall_height_mm: u16 },
    Roof { height_mm: u16 },
    Floor,
    Landing,
    Step { rise_mm: u16, tread_mm: u16 },
    Ramp { run_per_rise: u8 },
    Shelf { levels_mm: Vec<u16>, depth_mm: u16, width_mm: u16, height_mm: u16 },
    Table { top_mm: u16 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fixture {
    pub id: u32,
    pub kind: FixtureKind,
    pub cells: Vec<Cell>,
    pub yaw_deg: f32,
    pub base_m: f32,
    pub owner: Owner,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixtureBox {
    pub cell: Cell,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub color: [u8; 3],
    pub class: ObjectClass,
    pub entity: u32,
}

impl Fixture {
    pub fn blocks(&self) -> bool {
        matches!(self.kind, FixtureKind::Wall { .. } | FixtureKind::Shelf { .. } | FixtureKind::Table { .. })
    }

    pub fn kind_id(&self) -> &'static str {
        match self.kind {
            FixtureKind::Wall { .. } => "wall",
            FixtureKind::Door { .. } => "door",
            FixtureKind::Roof { .. } => "roof",
            FixtureKind::Floor => "floor",
            FixtureKind::Landing => "landing",
            FixtureKind::Step { .. } => "step",
            FixtureKind::Ramp { .. } => "ramp",
            FixtureKind::Shelf { .. } => "shelf",
            FixtureKind::Table { .. } => "table",
        }
    }

    pub fn label_ko(&self) -> &'static str {
        match self.kind {
            FixtureKind::Wall { .. } => "벽",
            FixtureKind::Door { .. } => "문",
            FixtureKind::Roof { .. } => "지붕",
            FixtureKind::Floor => "바닥",
            FixtureKind::Landing => "현관 앞 층계참",
            FixtureKind::Step { .. } => "계단",
            FixtureKind::Ramp { .. } => "경사로",
            FixtureKind::Shelf { .. } => "선반",
            FixtureKind::Table { .. } => "포장 테이블",
        }
    }

    pub fn support_levels_m(&self) -> Vec<f32> {
        match &self.kind {
            FixtureKind::Shelf { levels_mm, .. } => levels_mm.iter().map(|&l| l as f32 / 1000.0).collect(),
            FixtureKind::Table { top_mm } => vec![*top_mm as f32 / 1000.0],
            _ => vec![],
        }
    }

    pub fn height_m(&self) -> f32 {
        match &self.kind {
            FixtureKind::Wall { height_mm, .. } => *height_mm as f32 / 1000.0,
            FixtureKind::Door { wall_height_mm, .. } => *wall_height_mm as f32 / 1000.0,
            FixtureKind::Roof { height_mm } => *height_mm as f32 / 1000.0 + ROOF_THICK_M,
            FixtureKind::Shelf { height_mm, .. } => *height_mm as f32 / 1000.0,
            FixtureKind::Table { top_mm } => *top_mm as f32 / 1000.0,
            FixtureKind::Floor | FixtureKind::Landing | FixtureKind::Step { .. } | FixtureKind::Ramp { .. } => 0.0,
        }
    }

    fn class(&self) -> ObjectClass {
        match self.kind {
            FixtureKind::Shelf { .. } | FixtureKind::Table { .. } => ObjectClass::Fixture,
            _ => ObjectClass::Structure,
        }
    }

    pub fn facing(&self) -> (f32, f32) {
        let a = self.yaw_deg.to_radians();
        (a.cos(), a.sin())
    }

    fn push_boxes(&self, t: &Terrain, joins: &HashSet<(Owner, Cell)>, out: &mut Vec<FixtureBox>) {
        let class = self.class();
        let mut add = |cell: Cell, min: [f32; 3], max: [f32; 3], color: [u8; 3]| {
            if max[0] > min[0] && max[1] > min[1] && max[2] > min[2] {
                out.push(FixtureBox { cell, min, max, color, class, entity: self.id });
            }
        };
        let house = matches!(self.owner, Owner::House(_));
        let b = self.base_m;
        match &self.kind {
            FixtureKind::Floor | FixtureKind::Landing | FixtureKind::Step { .. } | FixtureKind::Ramp { .. } => {
                let color = match self.kind {
                    FixtureKind::Floor => [196, 190, 180],
                    FixtureKind::Landing => [188, 182, 172],
                    FixtureKind::Step { .. } => [172, 168, 160],
                    _ => [158, 160, 166],
                };
                for &(cx, cz) in &self.cells {
                    let (h, c) = (t.height_m(cx, cz), t.curb_m(cx, cz));
                    if c <= 0.0 {
                        continue;
                    }
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    let e = COVER_EPS_M;
                    add((cx, cz), [x0 - e, h, z0 - e], [x0 + CELL_M + e, h + c + e, z0 + CELL_M + e], color);
                }
            }
            FixtureKind::Wall { height_mm, thick_mm } => {
                let color = if house { [228, 216, 196] } else { [168, 178, 190] };
                let (top, th) = (b + *height_mm as f32 / 1000.0, *thick_mm as f32 / 1000.0);
                for &(cx, cz) in &self.cells {
                    let has = |dx: i32, dz: i32| {
                        let (nx, nz) = (cx as i32 + dx, cz as i32 + dz);
                        nx >= 0 && nz >= 0 && joins.contains(&(self.owner, (nx as usize, nz as usize)))
                    };
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    let (xc, zc) = (x0 + 0.5 * CELL_M, z0 + 0.5 * CELL_M);
                    let (xr, zr) = (has(-1, 0) || has(1, 0), has(0, -1) || has(0, 1));
                    if xr || !zr {
                        let lo = if has(-1, 0) { x0 } else { xc - 0.5 * th };
                        let hi = if has(1, 0) { x0 + CELL_M } else { xc + 0.5 * th };
                        add((cx, cz), [lo, b, zc - 0.5 * th], [hi, top, zc + 0.5 * th], color);
                    }
                    if zr {
                        let lo = if has(0, -1) { z0 } else { zc - 0.5 * th };
                        let hi = if has(0, 1) { z0 + CELL_M } else { zc + 0.5 * th };
                        add((cx, cz), [xc - 0.5 * th, b, lo], [xc + 0.5 * th, top, hi], color);
                    }
                }
            }
            FixtureKind::Door { width_mm, height_mm, wall_height_mm } => {
                let color = if house { [120, 84, 56] } else { [96, 104, 112] };
                let (door_top, top) = (b + *height_mm as f32 / 1000.0, b + *wall_height_mm as f32 / 1000.0);
                let th = WALL_THICK_M;
                let along_x = self.cells.iter().all(|c| c.1 == self.cells[0].1);
                let mut cells = self.cells.clone();
                cells.sort_by_key(|c| if along_x { c.0 } else { c.1 });
                let jamb = ((cells.len() as f32 * CELL_M - *width_mm as f32 / 1000.0) * 0.5).clamp(0.0, 0.5 * CELL_M);
                let n = cells.len();
                for (i, &(cx, cz)) in cells.iter().enumerate() {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    let (xc, zc) = (x0 + 0.5 * CELL_M, z0 + 0.5 * CELL_M);
                    let (r0, r1) = if along_x { (x0, x0 + CELL_M) } else { (z0, z0 + CELL_M) };
                    let across = |lo: f32, hi: f32, y0: f32, y1: f32| -> ([f32; 3], [f32; 3]) {
                        if along_x {
                            ([lo, y0, zc - 0.5 * th], [hi, y1, zc + 0.5 * th])
                        } else {
                            ([xc - 0.5 * th, y0, lo], [xc + 0.5 * th, y1, hi])
                        }
                    };
                    let (lo, hi) = across(r0, r1, door_top, top);
                    add((cx, cz), lo, hi, color);
                    if i == 0 && jamb > 0.0 {
                        let (lo, hi) = across(r0, r0 + jamb, b, door_top);
                        add((cx, cz), lo, hi, color);
                    }
                    if i + 1 == n && jamb > 0.0 {
                        let (lo, hi) = across(r1 - jamb, r1, b, door_top);
                        add((cx, cz), lo, hi, color);
                    }
                }
            }
            FixtureKind::Roof { height_mm } => {
                let color = if house { [150, 70, 56] } else { [118, 126, 136] };
                let y0 = b + *height_mm as f32 / 1000.0;
                for &(cx, cz) in &self.cells {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    add((cx, cz), [x0, y0, z0], [x0 + CELL_M, y0 + ROOF_THICK_M, z0 + CELL_M], color);
                }
            }
            FixtureKind::Shelf { levels_mm, depth_mm, width_mm, height_mm } => {
                let (post_c, deck_c) = ([52, 92, 170], [146, 150, 158]);
                let (fx, fz) = self.facing();
                let along_x = fx.abs() < 0.5;
                let mut cells = self.cells.clone();
                cells.sort_by_key(|c| if along_x { c.0 } else { c.1 });
                let (first, last) = (cells[0], cells[cells.len() - 1]);
                let run0 = if along_x { first.0 as f32 * CELL_M } else { first.1 as f32 * CELL_M };
                let run1 = if along_x { (last.0 + 1) as f32 * CELL_M } else { (last.1 + 1) as f32 * CELL_M };
                let mid = 0.5 * (run0 + run1);
                let half = (0.5 * *width_mm as f32 / 1000.0).min(0.5 * (run1 - run0));
                let (u0, u1) = (mid - half, mid + half);
                let (depth, top, post, deck) = (*depth_mm as f32 / 1000.0, b + *height_mm as f32 / 1000.0, SHELF_POST_MM as f32 / 1000.0, SHELF_DECK_MM as f32 / 1000.0);
                for &(cx, cz) in &cells {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    let (d0, d1) = if along_x {
                        if fz > 0.0 { (z0, z0 + depth) } else { (z0 + CELL_M - depth, z0 + CELL_M) }
                    } else if fx > 0.0 {
                        (x0, x0 + depth)
                    } else {
                        (x0 + CELL_M - depth, x0 + CELL_M)
                    };
                    let (s0, s1) = if along_x { (x0, x0 + CELL_M) } else { (z0, z0 + CELL_M) };
                    let (a0, a1) = (s0.max(u0), s1.min(u1));
                    let bx = |r0: f32, r1: f32, q0: f32, q1: f32, y0: f32, y1: f32| -> ([f32; 3], [f32; 3]) {
                        if along_x {
                            ([r0, y0, q0], [r1, y1, q1])
                        } else {
                            ([q0, y0, r0], [q1, y1, r1])
                        }
                    };
                    for &l in levels_mm {
                        let y = b + l as f32 / 1000.0;
                        let (lo, hi) = bx(a0, a1, d0, d1, y - deck, y);
                        add((cx, cz), lo, hi, deck_c);
                    }
                    for (r0, r1) in [(u0, u0 + post), (u1 - post, u1)] {
                        if r0 >= s0 - 1e-4 && r1 <= s1 + 1e-4 {
                            for (q0, q1) in [(d0, d0 + post), (d1 - post, d1)] {
                                let (lo, hi) = bx(r0, r1, q0, q1, b, top);
                                add((cx, cz), lo, hi, post_c);
                            }
                        }
                    }
                }
            }
            FixtureKind::Table { top_mm } => {
                let (top_c, leg_c) = ([86, 104, 118], [70, 70, 76]);
                let top = b + *top_mm as f32 / 1000.0;
                let (slab, leg) = (TABLE_TOP_THICK_MM as f32 / 1000.0, TABLE_LEG_MM as f32 / 1000.0);
                let x_lo = self.cells.iter().map(|c| c.0).min().unwrap_or(0);
                let x_hi = self.cells.iter().map(|c| c.0).max().unwrap_or(0);
                let z_lo = self.cells.iter().map(|c| c.1).min().unwrap_or(0);
                let z_hi = self.cells.iter().map(|c| c.1).max().unwrap_or(0);
                for &(cx, cz) in &self.cells {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    add((cx, cz), [x0, top - slab, z0], [x0 + CELL_M, top, z0 + CELL_M], top_c);
                    let lx = if cx == x_lo { Some(x0 + 0.02) } else if cx == x_hi { Some(x0 + CELL_M - 0.02 - leg) } else { None };
                    let lz = if cz == z_lo { Some(z0 + 0.02) } else if cz == z_hi { Some(z0 + CELL_M - 0.02 - leg) } else { None };
                    if let (Some(lx), Some(lz)) = (lx, lz) {
                        add((cx, cz), [lx, b, lz], [lx + leg, top - slab, lz + leg], leg_c);
                    }
                }
            }
        }
    }
}

pub fn render_boxes(fixtures: &[Fixture], t: &Terrain) -> Vec<FixtureBox> {
    let joins: HashSet<(Owner, Cell)> = fixtures
        .iter()
        .filter(|f| matches!(f.kind, FixtureKind::Wall { .. } | FixtureKind::Door { .. }))
        .flat_map(|f| f.cells.iter().map(move |&c| (f.owner, c)))
        .collect();
    let mut out = Vec::new();
    for f in fixtures {
        f.push_boxes(t, &joins, &mut out);
    }
    out
}

pub fn blocked(fixtures: &[Fixture]) -> impl Iterator<Item = Cell> + '_ {
    fixtures.iter().filter(|f| f.blocks()).flat_map(|f| f.cells.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Terrain {
        Terrain::generate(1, 16, 16, &[(4.0, 4.0, 16.0)])
    }

    #[test]
    fn every_box_stays_inside_its_cell_and_covers_overhang_by_a_millimetre_at_most() {
        let mut t = flat();
        for x in 4..8 {
            t.set_curb_cm(x, 4, 17);
        }
        let shelf = Fixture {
            id: 3,
            kind: FixtureKind::Shelf { levels_mm: SHELF_LEVELS_MM.to_vec(), depth_mm: SHELF_DEPTH_MM, width_mm: SHELF_WIDTH_MM, height_mm: SHELF_HEIGHT_MM },
            cells: vec![(4, 8), (5, 8), (6, 8), (7, 8)],
            yaw_deg: 90.0,
            base_m: 0.0,
            owner: Owner::Warehouse,
        };
        let step = Fixture { id: 4, kind: FixtureKind::Step { rise_mm: 170, tread_mm: 500 }, cells: (4..8).map(|x| (x, 4)).collect(), yaw_deg: 0.0, base_m: 0.0, owner: Owner::House(1) };
        let table = Fixture { id: 5, kind: FixtureKind::Table { top_mm: TABLE_TOP_MM }, cells: vec![(10, 10), (11, 10), (10, 11), (11, 11)], yaw_deg: 90.0, base_m: 0.0, owner: Owner::Warehouse };
        let boxes = render_boxes(&[shelf.clone(), step, table], &t);
        assert!(!boxes.is_empty());
        for b in &boxes {
            let (x0, z0) = (b.cell.0 as f32 * CELL_M, b.cell.1 as f32 * CELL_M);
            let eps = COVER_EPS_M + 1e-5;
            assert!(b.min[0] >= x0 - eps && b.max[0] <= x0 + CELL_M + eps && b.min[2] >= z0 - eps && b.max[2] <= z0 + CELL_M + eps, "{b:?}");
        }
        let decks = boxes.iter().filter(|b| b.entity == 3 && b.color == [146, 150, 158]).count();
        assert_eq!(decks, 12);
        let posts: Vec<_> = boxes.iter().filter(|b| b.entity == 3 && b.color == [52, 92, 170]).collect();
        assert_eq!(posts.len(), 4);
        assert!(posts.iter().all(|p| p.cell == (4, 8) || p.cell == (7, 8)));
        let span = posts.iter().map(|p| p.max[0]).fold(0.0f32, f32::max) - posts.iter().map(|p| p.min[0]).fold(f32::MAX, f32::min);
        assert!((span - 1.829).abs() < 1e-3, "{span}");
        let d = boxes.iter().find(|b| b.entity == 3 && b.color == [146, 150, 158]).unwrap();
        assert!((d.min[2] - 8.0 * CELL_M).abs() < 1e-5 && (d.max[2] - d.min[2] - 0.457).abs() < 1e-5);
        let s = boxes.iter().find(|b| b.entity == 4).unwrap();
        assert!((s.max[1] - 0.171).abs() < 1e-4 && s.class == ObjectClass::Structure);
        assert!(shelf.blocks() && shelf.support_levels_m() == vec![0.152, 0.914, 1.676]);
        assert_eq!(boxes.iter().filter(|b| b.entity == 5 && b.color == [70, 70, 76]).count(), 4, "a leg under each corner");
    }

    #[test]
    fn wall_runs_join_at_corners_and_reach_into_the_door() {
        let t = flat();
        let wall = Fixture { id: 1, kind: FixtureKind::Wall { height_mm: 2400, thick_mm: 200 }, cells: vec![(2, 2), (3, 2), (2, 3), (2, 4)], yaw_deg: 0.0, base_m: 0.0, owner: Owner::House(1) };
        let door = Fixture { id: 2, kind: FixtureKind::Door { width_mm: 914, height_mm: 2032, wall_height_mm: 2400 }, cells: vec![(4, 2), (5, 2)], yaw_deg: 270.0, base_m: 0.0, owner: Owner::House(1) };
        let boxes = render_boxes(&[wall, door], &t);
        let corner: Vec<_> = boxes.iter().filter(|b| b.cell == (2, 2)).collect();
        assert_eq!(corner.len(), 2);
        assert!(corner.iter().any(|b| (b.max[0] - 1.5).abs() < 1e-5));
        assert!(corner.iter().any(|b| (b.max[2] - 1.5).abs() < 1e-5));
        assert!(boxes.iter().any(|b| b.cell == (3, 2) && (b.max[0] - 2.0).abs() < 1e-5));
        let jambs: Vec<_> = boxes.iter().filter(|b| b.entity == 2 && b.min[1] == 0.0).collect();
        assert_eq!(jambs.len(), 2);
        let open = jambs.iter().map(|b| b.min[0]).fold(0.0f32, f32::max) - jambs.iter().map(|b| b.max[0]).fold(f32::MAX, f32::min);
        assert!((open - 0.914).abs() < 1e-4, "{open}");
    }
}
