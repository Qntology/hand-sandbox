//! Head-camera benchmark: how the robot's camera shakes against what people's
//! heads and other robots' chassis measured (research `docs/research/camera.md`
//! §7, group B). None of these sources is an engine input
//! (`the_engine_never_names_a_camera_benchmark_source` checks the engine's
//! files), and no engine constant is set from them. The lines it prints are
//! `[CAM]` cells and `[CAM-SUM]` scenarios, scored as the other benchmarks:
//! `like` scores `100 − 10·|z|` against a reference and its spread, a `raw`
//! cell is recorded only. The references and cells live in `camvx/cam.rs`
//! (shared with the cross-area report `uncanny_runtime.rs`).

mod camvx;

use camvx::{cam, print_scenario};

#[test]
fn camera_benchmark_prints_its_cells() {
    // C1: a mid wheeled base's chassis on the three grounds; C2: the heel
    // strike on grass against a hard floor, mid legs at a walk (`camvx::cam`).
    let s1 = print_scenario("CAM", &cam::c1());
    let s2 = print_scenario("CAM", &cam::c2());
    assert!(s1.is_finite() && s2.is_finite());
}

#[test]
fn the_engine_never_names_a_camera_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/world-sim/src/world/eye.rs",
        "crates/world-sim/src/render.rs",
        "crates/world-sim/src/light.rs",
        "crates/world-sim/src/iso.rs",
        "crates/world-sim/src/fixture.rs",
        "crates/world-sim/src/site.rs",
        "crates/noto-core/src/kin.rs",
        "crates/noto-core/src/morph.rs",
        "crates/noto-core/src/act/reach.rs",
        "crates/noto-runtime/src/vision.rs",
        "crates/noto-runtime/src/record.rs",
        "crates/noto-runtime/src/views.rs",
        "crates/noto-runtime/src/chase.rs",
    ];
    let banned = ["Milella", "Reina", "Moon, Leary", "Moon 2025", "Waite", "Menz", "Kavanagh"];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}
