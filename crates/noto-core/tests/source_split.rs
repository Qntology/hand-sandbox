//! The engine / benchmark source split over the central registry
//! (`noto_core::bench`): every engine file — found by pattern, so a new
//! module is covered the day it lands — is checked against every area's
//! benchmark sources, and the runtime's decision files against every way of
//! reading a reference. The per-benchmark checks (`action_bench`,
//! `effects_bench`, `nn_scenarios`, the runtime's `camera_bench` and
//! `vision_bench`) stay in place; this is their superset.

use std::path::PathBuf;

use noto_core::bench::{engine_files, engine_text, reference_reads, source_hits, BENCH_SOURCES, DECISION_FILES, NOT_ENGINE, SOURCE_EXCEPTIONS};

fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn the_registry_finds_the_engine_and_leaves_the_benchmarks_out() {
    let files = engine_files(&root());
    for f in [
        "crates/noto-core/src/kin.rs",
        "crates/noto-core/src/balance.rs",
        "crates/noto-core/src/act/grasp.rs",
        "crates/noto-core/src/act/gait.rs",
        "crates/world-sim/src/world.rs",
        "crates/world-sim/src/world/acts.rs",
        "crates/world-sim/src/world/eye.rs",
        "crates/world-sim/src/render.rs",
        "crates/noto-vision/src/decide.rs",
        "crates/noto-runtime/src/driver.rs",
        "crates/noto-runtime/src/vision.rs",
    ] {
        assert!(files.iter().any(|x| x == f), "{f} is engine: {files:?}");
    }
    for f in ["crates/noto-core/src/actbench.rs", "crates/noto-core/src/bench.rs", "crates/noto-core/src/effects.rs", "crates/noto-core/src/nn.rs", "crates/noto-core/src/lib.rs", "crates/noto-runtime/src/actsds.rs", "crates/noto-runtime/src/api.rs"] {
        assert!(!files.iter().any(|x| x == f), "{f} is not engine");
    }
    // Every file named in the exclusions and exceptions exists (a renamed file must not silently drop a rule).
    for (f, _) in NOT_ENGINE {
        assert!(root().join(f).is_file(), "{f}");
    }
    for (f, _, _) in SOURCE_EXCEPTIONS {
        assert!(root().join(f).is_file(), "{f}");
    }
    // The balance file's reference table is cut off; its controller stays.
    let balance = read("crates/noto-core/src/balance.rs");
    let engine = engine_text("crates/noto-core/src/balance.rs", &balance);
    assert!(engine.len() < balance.len() && engine.contains("impl Body") && !engine.contains("pub fn reference("));
    println!("[SRC] 엔진 파일 {}개 · 영역 {}개의 채점 출처 {}개", files.len(), BENCH_SOURCES.len(), BENCH_SOURCES.iter().map(|s| s.names.len()).sum::<usize>());
}

#[test]
fn the_engine_never_names_any_benchmark_source() {
    let mut hits = Vec::new();
    for f in engine_files(&root()) {
        let text = read(&f);
        for (area, name) in source_hits(&f, &text) {
            hits.push(format!("{f} names the {area} benchmark source `{name}`"));
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn decisions_never_read_reference_axes() {
    // The SDS keeps references beside the engine's numbers for the reports;
    // the files that decide must never look at them (actsds.rs, mlsds.rs,
    // api.rs and testrun.rs record and show them).
    for f in DECISION_FILES {
        let reads = reference_reads(&read(f));
        assert!(reads.is_empty(), "{f} reads {reads:?}");
    }
}

#[test]
fn every_area_lists_distinct_sources() {
    let mut areas: Vec<&str> = BENCH_SOURCES.iter().map(|s| s.area).collect();
    areas.sort();
    areas.dedup();
    assert_eq!(areas.len(), BENCH_SOURCES.len(), "one row per area");
    for s in BENCH_SOURCES {
        assert!(!s.names.is_empty(), "{}", s.area);
        let mut n = s.names.to_vec();
        n.sort();
        n.dedup();
        assert_eq!(n.len(), s.names.len(), "{} lists a name twice", s.area);
    }
}
