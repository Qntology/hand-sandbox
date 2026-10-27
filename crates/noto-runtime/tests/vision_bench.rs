//! Spatial-memory benchmark: the robot's intermittent vision and the memory
//! it acts on in between, against people's and other robots' (research
//! `docs/research/perception.md` §4, group B). None of these sources is an
//! engine input (`the_engine_never_names_a_vision_benchmark_source` checks the
//! engine's files). The lines it prints are `[VX]` cells and `[VX-SUM]` scenarios.
//!
//! Scoring as the other benchmarks: `like` scores `100 − 10·|z|` against a
//! reference and its spread, `no_worse` scores 100 at or below a reference
//! error and `100 − 10·z` above it, `at_most` gates the scenario when a limit is
//! broken, and a `raw` cell is recorded only. The references and cells live
//! in `camvx/vx.rs` (shared with the cross-area report `uncanny_runtime.rs`).

mod camvx;

use camvx::{print_scenario, vx};
use noto_core::BaseKind;

#[test]
fn vision_benchmark_prints_its_cells() {
    let mut total = Vec::new();
    for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
        total.push(print_scenario("VX", &vx::v1("mid", base)));
    }
    total.push(print_scenario("VX", &vx::v2("mid")));
    // V3 has no reference to score against: the cost and the success of the two schedules are recorded.
    assert!(print_scenario("VX", &vx::v3("pro")).is_nan());
    assert!(total.iter().all(|s| s.is_finite()));
}

#[test]
fn the_engine_never_names_a_vision_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/noto-runtime/src/vision.rs",
        "crates/noto-runtime/src/driver.rs",
        "crates/noto-runtime/src/mldrive.rs",
        "crates/noto-runtime/src/precision.rs",
        "crates/noto-runtime/src/runtime.rs",
        "crates/world-sim/src/odometry.rs",
        "crates/world-sim/src/world.rs",
        "crates/world-sim/src/world/acts.rs",
        "crates/world-sim/src/render.rs",
        "crates/world-sim/src/light.rs",
        "crates/world-sim/src/iso.rs",
        "crates/world-sim/src/site.rs",
        "crates/world-sim/src/fixture.rs",
        "crates/world-sim/src/explore.rs",
        "crates/noto-core/src/act/reach.rs",
        "crates/noto-core/src/act/grasp.rs",
        "crates/noto-core/src/kin.rs",
        "crates/noto-core/src/morph.rs",
        "crates/noto-vision/src/decide.rs",
        // v1.0 ML fixes (their own benchmark names are checked in ml_ab.rs).
        "crates/noto-vision/src/align.rs",
        "crates/noto-vision/src/calib.rs",
        "crates/noto-runtime/src/ml.rs",
        "crates/noto-runtime/src/embedding.rs",
        "crates/noto-runtime/src/perception.rs",
        "crates/noto-vision/src/locate.rs",
        "crates/noto-vision/src/tips.rs",
    ];
    let banned = ["Loomis", "Rieser", "Thomson", "Steenhuis", "Elliott", "Madalena", "Westwood", "Hesse", "Patla", "Hollands", "Laurent", "Assaiante", "TidyBot", "Dusty", "Deyle", "Ramanan", "Holbein", "Wysog", "King, Judge", "Toussaint", "van der Burg"];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}
