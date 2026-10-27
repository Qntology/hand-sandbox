//! The cross-area "uncanniness" report with every area: noto-core's (`AX`,
//! `FX`, `NN`, and what registers in `noto_core::bench::core_areas`) on the
//! shared robot matrix, and the runtime's head camera (`CAM`) and spatial
//! memory (`VX`), whose cells are the camera and vision benches' own
//! (`camvx`). Same line format as `noto-core/tests/uncanny_report.rs`
//! (`[UX]`, `[UX-CELL]`, `[UX-SUM]`, research `docs/research/uncanny.md`);
//! a record, never a threshold.

mod camvx;

use noto_core::bench::{core_areas, forms, spread, summarize, ux_lines, Summary};

#[test]
fn cross_area_deviation_report_with_camera_and_vision() {
    let forms = forms();
    let mut summaries: Vec<Summary> = core_areas().iter().map(|a| summarize(a.id, a.name_ko, &(a.run)(&forms))).collect();
    let cam = camvx::cam::scenarios();
    let vx = camvx::vx::scenarios();
    assert!(cam.iter().chain(vx.iter()).all(|s| s.cells.iter().all(|c| c.form == s.form && !c.holdout)));
    summaries.push(summarize("CAM", "머리 카메라 흔들림", &cam));
    summaries.push(summarize("VX", "간헐 시각과 공간 기억", &vx));
    for line in ux_lines(&summaries) {
        println!("{line}");
    }
    let vx_sum = summaries.iter().find(|s| s.area == "VX").unwrap();
    assert_eq!(vx_sum.record_only, 1, "V3 compares the robot with itself: recorded, not scored");
    assert!(summaries.iter().all(|s| s.n_ref > 0), "every area has a reference cell");
    let sp = spread(&summaries).expect("a spread");
    assert_eq!(sp.areas, summaries.len());
}
