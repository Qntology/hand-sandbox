mod camvx;

use camvx::{cam, print_scenario};

#[test]
fn camera_benchmark_prints_its_cells() {
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
