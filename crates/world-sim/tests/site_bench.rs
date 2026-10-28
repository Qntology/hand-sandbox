use noto_core::{BaseKind, FingerCount, NotoConfig};
use world_sim::explore::{self, Coverage, Eye};
use world_sim::fixture::FixtureKind;
use world_sim::site::{SiteCfg, Support};
use world_sim::world::{World, WorldConfig};
use world_sim::{ItemKind, TerrainKind};

#[derive(Clone, Copy, Debug)]
enum Rule {
    AtMost { limit: f32 },
    AtLeast { limit: f32 },
    Within { lo: f32, hi: f32 },
    Raw,
}

struct Cell {
    name: &'static str,
    value: f32,
    rule: Rule,
    cite: &'static str,
}

fn cell(name: &'static str, value: f32, rule: Rule, cite: &'static str) -> Cell {
    Cell { name, value, rule, cite }
}

fn score(c: &Cell) -> Option<f32> {
    let x = c.value;
    Some(match c.rule {
        Rule::AtMost { limit } => if x <= limit { 100.0 } else { 100.0 * (1.0 - (x - limit) / limit.abs().max(1e-6)).max(0.0) },
        Rule::AtLeast { limit } => if x >= limit { 100.0 } else { 100.0 * (x / limit.max(1e-6)).max(0.0) },
        Rule::Within { lo, hi } => {
            let d = if x < lo { lo - x } else if x > hi { x - hi } else { 0.0 };
            100.0 * (1.0 - d / (hi - lo).max(1e-6)).max(0.0)
        }
        Rule::Raw => return None,
    })
}

fn print_scenario(id: &str, title: &str, cells: &[Cell]) -> f32 {
    let mut scores = Vec::new();
    for c in cells {
        let s = score(c);
        let tail = s.map(|s| format!(" · 점수 {s:.0}")).unwrap_or_default();
        println!("[SX] {id}/{} {:.3} ({:?}){tail} — {}", c.name, c.value, c.rule, c.cite);
        scores.extend(s);
    }
    if scores.is_empty() {
        println!("[SX-SUM] {id} {title} 기록만 (비교할 기준 없음)");
        return f32::NAN;
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    println!("[SX-SUM] {id} {title} 점수 {mean:.1}");
    mean
}

fn sited(seed: u64, base: BaseKind) -> World {
    let mut noto = NotoConfig::preset(FingerCount::Five);
    noto.base = base;
    World::new(WorldConfig { seed, cells: 64, noto, terrain: TerrainKind::Open, site: Some(SiteCfg::default()), ..WorldConfig::default() }).unwrap()
}

const SEEDS: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

const CITE_ADK: &str = "영국 Approved Document K 사설 계단: 단 높이 ≤ 220, 디딤 220–300, 2R+G 550–700 mm, 기울기 ≤ 42° (BWF 2011 재수록)";
const CITE_AU: &str = "호주 시공자 설문: 79%가 주택 계단 단 높이 150–180 mm를 씀 (Di Mazio 2022, ABCB, 요약만)";
const CITE_KR15: &str = "건축물의 피난·방화구조 등의 기준에 관한 규칙 제15조: 중·고교 단 높이 ≤ 180, 디딤 ≥ 260 mm";
const CITE_HOFFMANN: &str = "랙 다음 단까지 여유 약 100 mm (Hoffmann 991202 목록)";
const CITE_TENNSCO: &str = "72 in 볼트리스 선반 5단 균등 배치 시 단 사이 약 330 mm (Tennsco Z-Line 판매처 자료) — 이 선반은 3단";
const CITE_WATERS: &str = "개정 들기 식의 수직 계수 VM = 1 − 0.003·|V − 75 cm| (Waters 1993) — 들기 높이의 좋은 정도";
const CITE_UCDAVIS: &str = "가벼운 서서 하는 일: 손이 팔꿈치보다 약 5 cm 아래, 팔꿈치 100–110 cm → 작업면 95–105 cm (UC Davis, 요약만)";
const CITE_KOSH: &str = "산업안전보건기준에 관한 규칙 제8조: 보통 작업 ≥ 150 lx, 그 밖의 작업 ≥ 75 lx";
const CITE_OSHA: &str = "29 CFR 1926.56: 창고 ≥ 5 fc (54 lx), 작업 중인 저장실 ≥ 10 fc (108 lx)";
const CITE_TOYOTA: &str = "지게차 통로 폭: VNA 1.52–2.13 m (Toyota 통로 폭 안내) — 로봇·보행 피킹 통로는 기준 밖";

#[test]
fn stairs_shelves_tables_and_light_against_other_sources() {
    let mut risers = Vec::new();
    let mut treads = Vec::new();
    let mut deck_pitch = Vec::new();
    let mut tables = Vec::new();
    let mut slot_heights = Vec::new();
    let mut aisle = 0.0;
    let mut lux = (f32::MAX, f32::MAX);
    for seed in SEEDS {
        let w = sited(seed, BaseKind::HumanFoot);
        let s = w.site.as_ref().unwrap();
        for h in &s.houses {
            risers.push(h.riser_mm as f32);
            treads.push(h.tread_mm as f32);
        }
        for f in &w.fixtures {
            match &f.kind {
                FixtureKind::Shelf { levels_mm, .. } => {
                    for p in levels_mm.windows(2) {
                        deck_pitch.push((p[1] - p[0]) as f32);
                    }
                }
                FixtureKind::Table { top_mm } => tables.push(*top_mm as f32),
                _ => {}
            }
        }
        for d in w.drops.iter().filter(|d| !d.support.is_ground()) {
            slot_heights.push(d.y_m);
        }
        let v = w.snapshot().site.unwrap();
        let wh = v.warehouse.unwrap();
        aisle = wh.aisle_m;
        for z in &wh.zones {
            match z.kind.as_str() {
                "rack" => lux.0 = lux.0.min(z.em_lx),
                _ => lux.1 = lux.1.min(z.em_lx),
            }
        }
    }
    let max = |v: &[f32]| v.iter().copied().fold(f32::MIN, f32::max);
    let min = |v: &[f32]| v.iter().copied().fold(f32::MAX, f32::min);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (r_max, t_min) = (max(&risers), min(&treads));
    let two_r_g = mean(&risers.iter().zip(&treads).map(|(r, t)| 2.0 * r + t).collect::<Vec<_>>());
    let pitch = max(&risers.iter().zip(&treads).map(|(r, t)| (r / t).atan().to_degrees()).collect::<Vec<_>>());
    let typical = risers.iter().filter(|r| (150.0..=180.0).contains(*r)).count() as f32 / risers.len() as f32;
    let s1 = vec![
        cell("riser_max_mm", r_max, Rule::AtMost { limit: 220.0 }, CITE_ADK),
        cell("going_min_mm", t_min, Rule::Within { lo: 220.0, hi: 300.0 }, CITE_ADK),
        cell("two_r_plus_g_mm", two_r_g, Rule::Within { lo: 550.0, hi: 700.0 }, CITE_ADK),
        cell("pitch_max_deg", pitch, Rule::AtMost { limit: 42.0 }, CITE_ADK),
        cell("riser_150_180_share", typical, Rule::AtLeast { limit: 0.79 }, CITE_AU),
        cell("riser_max_school_mm", r_max, Rule::AtMost { limit: 180.0 }, CITE_KR15),
        cell("tread_min_school_mm", t_min, Rule::AtLeast { limit: 260.0 }, CITE_KR15),
    ];
    let a = print_scenario("S1-stairs", "현관 계단 (8 시드의 모든 집)", &s1);

    let parcel_h_mm = ItemKind::Parcel.spec().piece_size_m(1.0)[1] * 1000.0;
    let clearance = min(&deck_pitch) - world_sim::fixture::SHELF_DECK_MM as f32 - parcel_h_mm;
    let vm = mean(&slot_heights.iter().map(|y| 1.0 - 0.003 * (y * 100.0 - 75.0).abs()).collect::<Vec<_>>());
    let s2 = vec![
        cell("clearance_above_parcel_mm", clearance, Rule::AtLeast { limit: 100.0 }, CITE_HOFFMANN),
        cell("deck_pitch_mm", min(&deck_pitch), Rule::Raw, CITE_TENNSCO),
        cell("table_top_mm", mean(&tables), Rule::Within { lo: 950.0, hi: 1050.0 }, CITE_UCDAVIS),
        cell("raised_parcel_vm", vm, Rule::Raw, CITE_WATERS),
        cell("raised_parcel_height_m", mean(&slot_heights), Rule::Raw, CITE_WATERS),
        cell("picking_aisle_m", aisle, Rule::Raw, CITE_TOYOTA),
    ];
    let b = print_scenario("S2-warehouse", "선반 · 포장 테이블 · 통로", &s2);

    let s3 = vec![
        cell("rack_floor_lx", lux.0, Rule::AtLeast { limit: 150.0 }, CITE_KOSH),
        cell("packing_lx", lux.1, Rule::AtLeast { limit: 150.0 }, CITE_KOSH),
        cell("rack_floor_lx_us", lux.0, Rule::AtLeast { limit: 108.0 }, CITE_OSHA),
    ];
    let c = print_scenario("S3-light", "물류센터 작업면 조도 (설계값)", &s3);
    assert!(a.is_finite() && b.is_finite() && c.is_finite());
    assert!(r_max <= 180.0 && t_min >= 260.0 && clearance > 0.0);
}

fn search(w: &World, prior: Option<&[f32]>, budget_s: f32) -> Vec<Option<f32>> {
    const SMALL_ITEM_SEE_M: f32 = 3.0;
    const VFOV_DEG: f32 = 50.0;
    const PITCH_DEG: f32 = -12.0;
    let t = &w.terrain;
    let blocked = w.blocked_cells();
    let spec = w.drive_spec();
    let eye = w.standing_eye_m();
    let parcels: Vec<(usize, usize)> = w.drops.iter().skip(1).filter_map(|d| t.cell_of(d.x, d.z)).collect();
    let mut found: Vec<Option<f32>> = vec![None; parcels.len()];
    let mut cov = Coverage::new(t);
    let mut at = t.cell_of(w.agent.x, w.agent.z).unwrap();
    let mut time = 0.0f32;
    let look = |cov: &mut Coverage, c| cov.look(t, &Eye { range_m: SMALL_ITEM_SEE_M, ..Eye::at(t, c, eye, PITCH_DEG, VFOV_DEG) }, 0.0, 360.0);
    look(&mut cov, at);
    loop {
        for (k, c) in parcels.iter().enumerate() {
            if found[k].is_none() && cov.is_seen(c.0, c.1) {
                found[k] = Some(time);
            }
        }
        if time > budget_s || found.iter().all(Option::is_some) {
            break;
        }
        let Some(v) = explore::next_viewpoint_with(t, &spec, &|x, z| blocked.contains(&(x, z)), at, &cov, eye, PITCH_DEG, VFOV_DEG, SMALL_ITEM_SEE_M, &[], prior) else { break };
        time += v.time_s + explore::LOOK_COST_S;
        at = v.cell;
        look(&mut cov, at);
    }
    found
}

#[test]
fn the_search_prior_reaches_the_parcels_sooner() {
    const BUDGET_S: f32 = 240.0;
    let mut rows: Vec<(&str, f32, f32)> = Vec::new();
    for (label, site, use_prior) in [("open_map", false, false), ("site_no_prior", true, false), ("site_prior", true, true)] {
        let (mut first, mut share, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for seed in SEEDS {
            let w = if site {
                sited(seed, BaseKind::HumanFoot)
            } else {
                let mut noto = NotoConfig::preset(FingerCount::Five);
                noto.base = BaseKind::HumanFoot;
                World::new(WorldConfig { seed, cells: 64, noto, ..WorldConfig::default() }).unwrap()
            };
            let prior = if use_prior { w.search_prior() } else { None };
            let found = search(&w, prior.as_deref(), BUDGET_S * 4.0);
            let times: Vec<f32> = found.iter().flatten().copied().collect();
            first += times.iter().copied().fold(BUDGET_S * 4.0, f32::min);
            share += found.iter().filter(|f| f.is_some_and(|t| t <= BUDGET_S)).count() as f32 / found.len().max(1) as f32;
            n += 1.0;
        }
        rows.push((label, first / n, share / n));
    }
    let cite = "같은 탐색기, 같은 3 m 인식 거리 · 시점마다 한 바퀴 (계획기 수준, 벽은 가리지 않음)";
    let mut cells = Vec::new();
    for (label, first, share) in &rows {
        let (a, b): (&'static str, &'static str) = match *label {
            "open_map" => ("open_map_first_parcel_s", "open_map_found_240s_share"),
            "site_no_prior" => ("site_no_prior_first_parcel_s", "site_no_prior_found_240s_share"),
            _ => ("site_prior_first_parcel_s", "site_prior_found_240s_share"),
        };
        cells.push(cell(a, *first, Rule::Raw, cite));
        cells.push(cell(b, *share, Rule::Raw, cite));
    }
    print_scenario("S4-search", "택배 찾기: 열린 지형 · 장소 사전 지식 없이 · 있이 (A/B)", &cells);
    let (no, yes) = (rows[1], rows[2]);
    assert!(yes.2 >= no.2, "the prior never finds fewer parcels in time: {yes:?} vs {no:?}");
    assert!(yes.1 <= no.1 + 1e-3, "nor later: {yes:?} vs {no:?}");
}

#[test]
fn every_raised_parcel_is_on_a_support_the_site_built() {
    for seed in SEEDS {
        let w = sited(seed, BaseKind::Wheel);
        for d in w.drops.iter().skip(1) {
            let ok = match d.support {
                Support::Ground => false,
                Support::Step { house, .. } => w.site.as_ref().unwrap().house(house).is_some(),
                Support::Shelf { fixture, .. } | Support::Table { fixture } => w.fixtures.iter().any(|f| f.id == fixture),
            };
            assert!(ok, "seed {seed}: {:?}", d.support);
        }
    }
}

#[test]
fn the_site_engine_never_names_a_site_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/world-sim/src/site.rs",
        "crates/world-sim/src/fixture.rs",
        "crates/world-sim/src/terrain.rs",
        "crates/world-sim/src/nav.rs",
        "crates/world-sim/src/explore.rs",
        "crates/world-sim/src/world.rs",
        "crates/world-sim/src/snapshot.rs",
        "crates/world-sim/src/render.rs",
    ];
    let banned = [
        "Approved Document K", "BWF", "British Woodworking", "Di Mazio", "ABCB", "Australian Building Codes", "피난·방화", "Schadler", "Ninja Van", "SUTD",
        "산업안전보건기준", "1926.56", "OSHA", "UC Davis", "Waters", "Racklify", "EHS Today", "Toyota", "BigRentz", "Hoffmann", "Tennsco", "IndustrialShelving",
        "Patla", "Matthis", "Sparrow",
    ];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}
