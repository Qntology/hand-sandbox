#![allow(dead_code)]

pub mod cam;
pub mod vx;

use noto_core::bench::BenchScenario;

pub fn print_scenario(tag: &str, s: &BenchScenario) -> f32 {
    for c in &s.cells {
        let v = c.value.map_or("미구현".to_string(), |v| format!("{v:.3}"));
        let sc = c.score.map_or(String::new(), |x| format!(" · 점수 {x:.0}"));
        let g = c.gate.as_ref().map_or(String::new(), |g| format!(" · 게이트 {g}"));
        println!("[{tag}] {}/{} {v} ({:?}){sc}{g} — {}", s.id, c.axis, c.rule, c.cite);
    }
    let scored: Vec<f32> = s.cells.iter().filter_map(|c| c.score).collect();
    let gated = !s.gates.is_empty();
    if scored.is_empty() && !gated {
        println!("[{tag}-SUM] {} {} 기록만 (비교할 기준 없음)", s.id, s.name_ko);
        return f32::NAN;
    }
    let score = if gated { 0.0 } else { scored.iter().sum::<f32>() / scored.len() as f32 };
    println!("[{tag}-SUM] {} {} 점수 {score:.1}{}", s.id, s.name_ko, if gated { " (게이트)" } else { "" });
    score
}
