//! Daylight benchmark: where the engine's sun stands, how much light its
//! clear sky gives and what colour it has, against references that are not
//! engine inputs (research `docs/research/lighting.md` §6, group B). The
//! engine (`world_sim::light`) uses the NOAA series, the IESNA daylight model
//! and a Rayleigh + Ångström atmosphere; nothing here feeds back into it
//! (`the_engine_never_names_a_light_benchmark_source` checks its files). The
//! lines it prints are `[LX]` cells and `[LX-SUM]` scenarios, scored as the
//! other benchmarks: `like` scores `100 − 10·|z|` against a reference and its
//! spread, `holds` scores 100 when a measured direction holds and 0 when it
//! does not, a `raw` cell is recorded only.
//!
//! Colour temperature is not something the engine computes: the bench turns
//! the engine's channel colours into a chromaticity with the sRGB matrices
//! (IEC 61966-2-1), the CIE 1931 observer in the analytic fit of Wyman, Sloan
//! & Shirley (2013) and McCamy's (1992) CCT formula, taking the camera's white
//! (the sun at the zenith) as a 5778 K black body under one air mass of the
//! engine's own atmosphere. The fit's coefficients were written from memory;
//! `the_colour_tools_find_a_black_body_temperature` checks them.

use world_sim::light::{air_mass, rayleigh_depth, solar_position, sun_colour, Daylight, ANGSTROM_ALPHA, ANGSTROM_BETA};

// ----- references (benchmark side only; the engine never reads these) -----

/// NREL SPA reference example (Reda & Andreas 2004, as encoded in the
/// pvlib-python tests): 2003-10-17 12:30:30 local time (UTC−7) at
/// 39.742476° N, 105.1786° W: topocentric zenith 50.11162° with refraction,
/// azimuth 194.340241°, declination −9.31434°, hour angle 11.10590°.
const SPA_LAT_DEG: f32 = 39.742476;
const SPA_LON_DEG: f32 = -105.1786;
const SPA_TZ_H: f32 = -7.0;
/// 17 October 2003 (not a leap year): 273 + 17.
const SPA_DAY: u32 = 290;
const SPA_HOUR: f32 = 12.0 + 30.0 / 60.0 + 30.0 / 3600.0;
const SPA_ELEVATION_DEG: f32 = 90.0 - 50.11162;
const SPA_AZIMUTH_DEG: f32 = 194.340241;
const SPA_DECLINATION_DEG: f32 = -9.31434;
const SPA_HOUR_ANGLE_DEG: f32 = 11.10590;
/// The research pack's tolerance for a NOAA-series position (its own
/// estimate of that series' error, about 0.3°, rounded up).
const SPA_TOL_DEG: f32 = 0.5;
const CITE_SPA: &str = "NREL SPA 예제 2003-10-17 12:30:30 (UTC−7), 39.74° N 105.18° W: 천정각 50.112° · 방위 194.340° · 적위 −9.314° · 시간각 11.106° (Reda & Andreas 2004, pvlib 시험)";

/// Measured clear-sky global horizontal illuminance at a solar altitude of
/// about 80°: 110.5 klx (Golmohammadi, Shekari & Mahjub 2009, their own
/// measurements at three Iranian stations); spread: the SD of all their
/// measurements, 14.3 klx; measured range 53–129 klx over altitudes 41.7–79.8°.
const LUX_80_REF: f32 = 110_500.0;
const LUX_SD: f32 = 14_300.0;
const LUX_RANGE: (f32, f32) = (53_000.0, 129_000.0);
const CITE_LUX: &str = "맑은 날 수평 전천 조도: 태양 고도 약 80°에서 110.5 klx, SD 14.3 klx, 고도 41.7–79.8°에서 53–129 klx (Golmohammadi 2009 측정값)";

/// Direct sunlight: about 3,000 K at sunrise and sunset, about 6,000 K at
/// midday (DIAL GmbH 2022, a manufacturer's article without a protocol).
const BEAM_CCT_LOW_K: f32 = 3_000.0;
const BEAM_CCT_NOON_K: f32 = 6_000.0;
const CITE_BEAM: &str = "직사광 색온도: 해 뜰 때·질 때 약 3,000 K, 한낮 약 6,000 K (DIAL 2022, 제조사 글)";
/// Global daylight: about 5,750 K around midday with a small SD, bluer and
/// more spread just after sunrise and before sunset (Ratnasingam, Collins &
/// Hernández-Andrés 2013; Hernández-Andrés et al. 2001: 5,700 K typifies it);
/// spectra leave the daylight locus in twilight (Spitschan et al. 2016).
const GLOBAL_CCT_NOON_K: f32 = 5_750.0;
const CITE_GLOBAL: &str = "전천 일광 색온도: 한낮 약 5,750 K (Ratnasingam 2013) · 대표 5,700 K (Hernández-Andrés 2001); 해 뜬 직후·지기 직전엔 더 푸름 (Ratnasingam 2013, Spitschan 2016)";

// ----- scoring -----

#[derive(Clone, Copy, Debug)]
enum Rule {
    Like { reference: f32, tolerance: f32 },
    Holds,
    Raw,
}

struct Cell {
    axis: String,
    value: Option<f32>,
    rule: Rule,
    cite: &'static str,
    score: Option<f32>,
}

fn cell(axis: impl Into<String>, value: Option<f32>, rule: Rule, cite: &'static str) -> Cell {
    let score = match (value, rule) {
        (None, Rule::Raw) => None,
        (None, _) => Some(0.0),
        (Some(v), Rule::Like { reference, tolerance }) => Some((100.0 - 10.0 * ((v - reference) / tolerance.max(1e-6)).abs()).max(0.0)),
        (Some(v), Rule::Holds) => Some(if v > 0.5 { 100.0 } else { 0.0 }),
        (Some(_), Rule::Raw) => None,
    };
    Cell { axis: axis.into(), value, rule, cite, score }
}

fn holds(axis: &str, ok: bool, cite: &'static str) -> Cell {
    cell(axis, Some(f32::from(u8::from(ok))), Rule::Holds, cite)
}

fn print_scenario(id: &str, name: &str, cells: &[Cell]) -> f32 {
    for c in cells {
        let v = c.value.map_or("미구현".to_string(), |v| format!("{v:.3}"));
        let s = c.score.map_or(String::new(), |s| format!(" · 점수 {s:.0}"));
        println!("[LX] {id}/{} {v} ({:?}){s} — {}", c.axis, c.rule, c.cite);
    }
    let scored: Vec<f32> = cells.iter().filter_map(|c| c.score).collect();
    if scored.is_empty() {
        println!("[LX-SUM] {id} {name} 기록만 (비교할 기준 없음)");
        return f32::NAN;
    }
    let score = scored.iter().sum::<f32>() / scored.len() as f32;
    println!("[LX-SUM] {id} {name} 점수 {score:.1}");
    score
}

// ----- colour tools (benchmark side: chromaticity and CCT) -----

/// Linear sRGB (BT.709 primaries, D65) to CIE XYZ (IEC 61966-2-1).
const RGB_TO_XYZ: [[f64; 3]; 3] = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
/// Its inverse (IEC 61966-2-1).
const XYZ_TO_RGB: [[f64; 3]; 3] = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
/// Effective black-body temperature of the sun outside the atmosphere, K
/// (the research pack's spectral model).
const SUN_BLACKBODY_K: f64 = 5778.0;

fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

/// One lobe of the Wyman–Sloan–Shirley fit: a Gaussian with a different
/// width on each side of its peak.
fn lobe(l: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let t = (l - mu) / if l < mu { s1 } else { s2 };
    (-0.5 * t * t).exp()
}

/// CIE 1931 2° colour matching functions at `l` nm (Wyman, Sloan & Shirley
/// 2013, multi-lobe fit).
fn cmf(l: f64) -> [f64; 3] {
    [
        1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2),
        0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1),
        1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8),
    ]
}

/// Planck's law at `l` nm and `t` K, up to a constant.
fn planck(l: f64, t: f64) -> f64 {
    let lm = l * 1e-9;
    1.0 / (lm.powi(5) * ((1.4388e-2 / (lm * t)).exp() - 1.0))
}

/// XYZ of a spectrum given at every nanometre from 380 to 780 nm.
fn xyz_of(spectrum: impl Fn(f64) -> f64) -> [f64; 3] {
    let mut acc = [0.0f64; 3];
    for l in 380..=780 {
        let (l, s) = (l as f64, spectrum(l as f64));
        let c = cmf(l);
        for k in 0..3 {
            acc[k] += s * c[k];
        }
    }
    acc
}

/// Correlated colour temperature of an XYZ colour, K (McCamy 1992).
fn cct_k(xyz: [f64; 3]) -> f32 {
    let sum = xyz[0] + xyz[1] + xyz[2];
    let (x, y) = (xyz[0] / sum, xyz[1] / sum);
    let n = (x - 0.3320) / (0.1858 - y);
    (449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33) as f32
}

/// The engine's vertical optical depth at `l` nm: its Rayleigh term plus its
/// Ångström aerosol term (the same as `light::channel_depth` at the three
/// channel wavelengths).
fn engine_depth(l: f64) -> f64 {
    let um = (l / 1000.0) as f32;
    (rayleigh_depth(um) + ANGSTROM_BETA * um.powf(-ANGSTROM_ALPHA)) as f64
}

/// The direct beam's spectrum after `m` air masses of the engine's atmosphere.
fn beam_spectrum(m: f64) -> impl Fn(f64) -> f64 {
    move |l| planck(l, SUN_BLACKBODY_K) * (-m * engine_depth(l)).exp()
}

/// The camera's white (the sun at the zenith) in linear sRGB.
fn camera_white() -> [f64; 3] {
    mul(&XYZ_TO_RGB, xyz_of(beam_spectrum(1.0)))
}

/// CCT of a light the engine renders with relative channel colour `rgb`
/// (white = the zenith sun), K.
fn rendered_cct_k(rgb: [f32; 3]) -> f32 {
    let w = camera_white();
    cct_k(mul(&RGB_TO_XYZ, [0, 1, 2].map(|k| rgb[k] as f64 * w[k])))
}

/// The light on a sunlit horizontal face (beam and sky) as the engine colours it.
fn global_rgb(elevation_deg: f32) -> [f32; 3] {
    Daylight::new(elevation_deg, 180.0, 0.0).face_light([0.0, 1.0, 0.0], true)
}

#[test]
fn the_colour_tools_find_a_black_body_temperature() {
    // McCamy is good to a few kelvin on the Planckian locus between about 2,800 and 6,500 K.
    for t in [2856.0, 3000.0, 4000.0, 5000.0, 6504.0] {
        let got = cct_k(xyz_of(|l| planck(l, t)));
        assert!((got as f64 - t).abs() < 0.02 * t, "{t} K black body read as {got} K");
    }
    // The fit's luminance curve peaks near 1 at 555 nm.
    assert!((cmf(555.0)[1] - 1.0).abs() < 0.01);
    // The camera white (zenith sun) is a near-neutral warm white in sRGB.
    let w = camera_white();
    assert!(w.iter().all(|&v| v > 0.0) && w[0] > w[2]);
}

#[test]
fn light_benchmark_prints_its_cells() {
    // L1: the NOAA series against the SPA reference instant.
    let p = solar_position(SPA_LAT_DEG, SPA_LON_DEG, SPA_TZ_H, SPA_DAY, SPA_HOUR);
    let l1 = vec![
        cell("elevation_deg", Some(p.elevation_deg), Rule::Like { reference: SPA_ELEVATION_DEG, tolerance: SPA_TOL_DEG }, CITE_SPA),
        cell("azimuth_deg", Some(p.azimuth_deg), Rule::Like { reference: SPA_AZIMUTH_DEG, tolerance: SPA_TOL_DEG }, CITE_SPA),
        cell("declination_deg", Some(p.declination_deg), Rule::Like { reference: SPA_DECLINATION_DEG, tolerance: SPA_TOL_DEG }, CITE_SPA),
        cell("hour_angle_deg", Some(p.hour_angle_deg), Rule::Like { reference: SPA_HOUR_ANGLE_DEG, tolerance: SPA_TOL_DEG }, CITE_SPA),
    ];
    let s1 = print_scenario("L1", "해의 위치 (NOAA 급수, SPA 예제 시각)", &l1);

    // L2: the clear sky's global horizontal illuminance.
    let lux = |h: f32| Daylight::new(h, 180.0, 0.0).global_lx;
    let l2 = vec![
        cell("global_lux_at_80deg", Some(lux(80.0)), Rule::Like { reference: LUX_80_REF, tolerance: LUX_SD }, CITE_LUX),
        holds("in_measured_range_41.7_to_79.8deg", lux(41.7) >= LUX_RANGE.0 && lux(79.8) <= LUX_RANGE.1, CITE_LUX),
        cell("global_lux_at_41.7deg", Some(lux(41.7)), Rule::Raw, CITE_LUX),
        cell("sky_share_at_80deg", Some(Daylight::new(80.0, 180.0, 0.0).sky_lx / lux(80.0)), Rule::Raw, CITE_LUX),
    ];
    let s2 = print_scenario("L2", "맑은 하늘 수평 조도 (IESNA 모델)", &l2);

    // L3: colour temperature of the beam and of the global light as rendered.
    let beam = |h: f32| rendered_cct_k(sun_colour(h));
    let spectral = |h: f32| cct_k(xyz_of(beam_spectrum(air_mass(h) as f64)));
    let global = |h: f32| rendered_cct_k(global_rgb(h));
    let (b_low, b_noon) = (beam(5.0), beam(60.0));
    let (g_low, g_noon) = (global(5.0), global(60.0));
    let l3 = vec![
        holds("beam_warmer_at_5deg_than_60deg", b_low < b_noon, CITE_BEAM),
        cell("beam_cct_K_5deg", Some(b_low), Rule::Raw, CITE_BEAM),
        cell("beam_cct_K_60deg", Some(b_noon), Rule::Raw, CITE_BEAM),
        cell("beam_cct_over_dial_5deg", Some(b_low / BEAM_CCT_LOW_K), Rule::Raw, CITE_BEAM),
        cell("beam_cct_over_dial_60deg", Some(b_noon / BEAM_CCT_NOON_K), Rule::Raw, CITE_BEAM),
        cell("beam_cct_K_5deg_spectral", Some(spectral(5.0)), Rule::Raw, CITE_BEAM),
        holds("global_bluer_at_5deg_than_60deg", g_low > g_noon, CITE_GLOBAL),
        cell("global_cct_K_60deg", Some(g_noon), Rule::Raw, CITE_GLOBAL),
        cell("global_cct_over_measured_60deg", Some(g_noon / GLOBAL_CCT_NOON_K), Rule::Raw, CITE_GLOBAL),
        cell("global_cct_K_5deg", Some(g_low), Rule::Raw, CITE_GLOBAL),
    ];
    let s3 = print_scenario("L3", "직사광과 전천 일광의 색온도", &l3);
    assert!(s1.is_finite() && s2.is_finite() && s3.is_finite());
}

/// What a sun costs the renderer (a shadow ray per lit pixel): wall-clock
/// time of the head camera's frame, flat against sun. Timing only, so it is
/// not a benchmark cell; run with `-- --ignored --nocapture`.
#[test]
#[ignore]
fn sun_render_cost() {
    use world_sim::light::Lighting;
    use world_sim::{Camera, World, WorldConfig};
    let mut w = World::new(WorldConfig::default()).unwrap();
    let cam = Camera::default();
    let time = |w: &World| {
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            std::hint::black_box(w.render(&cam));
        }
        t0.elapsed().as_secs_f64() * 100.0
    };
    let flat = time(&w);
    w.set_lighting(Lighting::preset("morning").unwrap());
    let sun = time(&w);
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        std::hint::black_box(w.shadow_casters());
    }
    let casters = t0.elapsed().as_secs_f64() * 100.0;
    println!("render {}x{}: flat {flat:.1} ms, sun {sun:.1} ms ({:.2}x), of which the robot's pose for its shadow {casters:.1} ms", cam.width, cam.height, sun / flat);
}

#[test]
fn the_engine_never_names_a_light_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/world-sim/src/light.rs",
        "crates/world-sim/src/render.rs",
        "crates/world-sim/src/iso.rs",
        "crates/world-sim/src/world/eye.rs",
        "crates/noto-runtime/src/views.rs",
        "crates/noto-runtime/src/chase.rs",
    ];
    // Golmohammadi et al. 2009 is both: the engine cites the IESNA coefficients
    // as they reproduce them, the bench uses only their own measurements.
    let banned = ["Reda", "Andreas", "NREL", "pvlib", "Ratnasingam", "Hernández", "Hernandez", "DIAL GmbH", "dial.de", "Spitschan", "Wyman", "McCamy"];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}
