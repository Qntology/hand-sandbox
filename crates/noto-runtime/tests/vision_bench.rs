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
