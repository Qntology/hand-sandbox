use noto_core::bench::{core_areas, forms, summarize, ux_lines, BenchScenario, Summary};

#[test]
fn cross_area_deviation_report() {
    let forms = forms();
    let mut summaries: Vec<Summary> = Vec::new();
    for area in core_areas() {
        let scenarios: Vec<BenchScenario> = (area.run)(&forms);
        assert!(!scenarios.is_empty(), "{}", area.id);
        assert!(scenarios.iter().all(|s| s.area == area.id), "{}", area.id);
        for s in &scenarios {
            for c in &s.cells {
                assert_eq!(c.form, s.form);
                let v = noto_core::bench::judge(c.value, c.rule);
                assert_eq!((v.z, v.score), (c.z, c.score), "{} {} {}:{}", area.id, s.id, c.layer, c.axis);
                assert!(c.value.map_or(true, f32::is_finite), "{} {} {}", area.id, s.id, c.axis);
            }
            if let Some(score) = s.score {
                assert!((0.0..=100.0).contains(&score), "{} {} {score}", area.id, s.id);
            }
        }
        let sum = summarize(area.id, area.name_ko, &scenarios);
        assert!(sum.n_ref > 0, "{} has reference cells", area.id);
        assert!(sum.median_abs_z.is_some_and(f32::is_finite) && sum.rms_z.is_some_and(f32::is_finite));
        summaries.push(sum);
    }
    for line in ux_lines(&summaries) {
        println!("{line}");
    }
    assert!(noto_core::bench::spread(&summaries).is_some());
}

#[test]
fn the_matrix_covers_every_base_once_per_plan() {
    let f = forms();
    let mut ids: Vec<String> = f.iter().map(|x| x.id()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), f.len(), "form ids are unique");
    for base in noto_core::bench::BASES {
        for plan in noto_core::bench::PLANS {
            assert!(f.iter().any(|x| x.plan == plan && x.base == base), "{plan} {base:?}");
        }
    }
    assert!(f.iter().any(|x| x.base == noto_core::BaseKind::Mecanum), "mecanum runs too");
}
