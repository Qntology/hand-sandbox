use noto_core::bench::{form_id, BenchCell, BenchScenario, Rule};
use noto_core::{BaseKind, Body, BodyPlan, FingerCount};
use noto_runtime::{NoiseCfg, Runtime, RuntimeConfig};
use world_sim::world::DropItem;
use world_sim::{Action, ItemKind};

const LEX: &str = include_str!("../../../../data/world_lexicon.json");

const BLIND_ABS_FRAC: f32 = 0.115;
const BLIND_SPREAD_FRAC: f32 = 0.08;
const CITE_BLIND: &str = "눈 감고 걷기 4–12 m: 절대 오차 92 cm / 평균 8 m = 거리의 11.5% (Loomis 1992 실험 1, n = 10), 가변 오차 거리의 약 8% (Rieser 1990)";
const CITE_BLIND_SIGNED: &str = "부호 있는 오차 +55 cm / 8 m = +7% (Loomis 1992 실험 1)";
const CITE_DELAY: &str = "출발 전 기다림: 30 s 안에는 영향 없음 (Steenhuis & Goodale 1988), 없음 (Rieser 1990) — Thomson 1983의 8 s 효과는 재현되지 않음";
const VISION_SHARE_MAX: f32 = 0.5;
const CITE_SHARE: &str = "스스로 고른 시각 표집이 걷는 시간의 50% 미만이어도 안전하게 걸음 (Patla 1996)";
const FLOOR_PICK: (f32, f32) = (0.964, 0.019);
const CITE_FLOOR_PICK: &str = "로봇의 바닥 물건 집기 성공률 96.2% (TidyBot, Wu 2023) · 98.4% (Dusty, King 2012) · 94.7% (Xu 2009) — 평균 ± SD";
const CITE_MEMORY_REACH: &str = "기억으로 가리키기: 2 s 지연 뒤 오차가 약 2배 (Elliott & Madalena 1987, 초록만 — SD 없음, 기록만)";

fn mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
}

fn form(plan: &str, base: BaseKind) -> String {
    form_id(&Body::new(&BodyPlan::named(plan).expect("a body plan"), base, FingerCount::Five))
}

fn runtime(seed: u64, base: BaseKind, plan: &str, parcels: Option<usize>, intermittent: bool, noise: NoiseCfg) -> Runtime {
    let mut cfg = RuntimeConfig { seed, noise, parcels, ..RuntimeConfig::default() };
    cfg.noto.base = base;
    cfg.body = BodyPlan::named(plan).expect("a body plan");
    cfg.vision.intermittent = intermittent;
    Runtime::new(cfg, LEX, None).expect("runtime")
}

fn blind_walk(seed: u64, base: BaseKind, plan: &str, d: f32, delay_s: f32) -> Option<(f32, f32)> {
    let mut rt = runtime(seed, base, plan, Some(0), true, NoiseCfg::none());
    let (x0, z0, yaw) = {
        let w = rt.world();
        (w.agent.x, w.agent.z, w.agent.yaw)
    };
    let (tx, tz) = (x0 + d * yaw.cos(), z0 + d * yaw.sin());
    rt.world_mut().drops.push(DropItem::whole(9001, ItemKind::Parcel, tx, tz));
    for _ in 0..2 {
        rt.scan_now("bench").ok()?;
    }
    let lm = rt.gates().landmarks().iter().find(|l| l.link == "parcel" && l.confirmed)?.clone();
    for _ in 0..((delay_s / world_sim::TICK_S).round() as u32) {
        rt.world_mut().tick();
    }
    let go = rt.to_world(Action::MoveTo { x: lm.x, z: lm.z, reach_m: 0.6 });
    rt.world_mut().command(go).ok()?;
    let mut n = 0;
    while rt.world().busy() && n < 3000 {
        rt.world_mut().tick();
        n += 1;
    }
    let (px, pz) = rt.world().true_of(lm.x, lm.z);
    let err = (px - tx).hypot(pz - tz);
    let along = (px - tx) * yaw.cos() + (pz - tz) * yaw.sin();
    Some((err, along))
}

pub fn v1(plan: &str, base: BaseKind) -> BenchScenario {
    let id = format!("V1-{plan}-{base:?}");
    let f = form(plan, base);
    let cell = |axis: &str, value: Option<f32>, rule: Rule, cite: &str| BenchCell::new("VX", &id, "", axis, None, &f, value, rule, cite, false);
    let mut abs = Vec::new();
    let mut signed = Vec::new();
    for d in [4.0f32, 6.0, 8.0, 10.0] {
        for seed in 1..=4u64 {
            if let Some((e, along)) = blind_walk(seed, base, plan, d, 0.0) {
                abs.push(e / d);
                signed.push(-along / d);
            }
        }
    }
    let delayed: Vec<f32> = (1..=4u64).filter_map(|s| blind_walk(s, base, plan, 8.0, 10.0)).map(|(e, _)| e).collect();
    let now: Vec<f32> = (1..=4u64).filter_map(|s| blind_walk(s, base, plan, 8.0, 0.0)).map(|(e, _)| e).collect();
    let ratio = match (mean(&delayed), mean(&now)) {
        (Some(a), Some(b)) if b > 1e-4 => Some(a / b),
        _ => None,
    };
    let cells = vec![
        cell("abs_error_frac", mean(&abs), Rule::NoWorse { reference: BLIND_ABS_FRAC, tolerance: BLIND_SPREAD_FRAC }, CITE_BLIND),
        cell("signed_error_frac", mean(&signed), Rule::Raw, CITE_BLIND_SIGNED),
        cell("delay_10s_over_none", ratio, Rule::Raw, CITE_DELAY),
        cell("walks", Some(abs.len() as f32), Rule::Raw, ""),
    ];
    BenchScenario::from_cells("VX", &id, "본 물건까지 눈 감고 걷기", &f, false, cells)
}

pub fn v2(plan: &str) -> BenchScenario {
    let id = format!("V2-{plan}");
    let f = form(plan, BaseKind::HumanFoot);
    let cell = |axis: &str, value: Option<f32>, rule: Rule, cite: &str| BenchCell::new("VX", &id, "", axis, None, &f, value, rule, cite, false);
    let mut ok = 0u32;
    let mut runs = 0u32;
    let (mut side, mut picks, mut share_s, mut share_l) = (0u32, 0u32, 0.0f64, 0u32);
    let (mut mem_aim, mut seen_aim) = (Vec::new(), Vec::new());
    for seed in 1..=6u64 {
        let mut rt = runtime(seed, BaseKind::HumanFoot, plan, None, true, NoiseCfg::demo());
        let s = rt.submit("택배 상자 1개를 창고로 가져와");
        assert!(s.ok, "{:?}", s.error);
        rt.run_until_idle(9000);
        runs += 1;
        let q = rt.quests().last().cloned().expect("a quest");
        if q.state == noto_quest::State::Done {
            ok += 1;
        }
        let v = rt.vision_stats();
        side += v.side_picks;
        picks += v.picks;
        share_s += v.approach_s;
        share_l += v.approach_looks;
        mem_aim.extend(v.memory_aim_m.iter().copied());
        seen_aim.extend(v.seen_aim_m.iter().copied());
    }
    let rate = ok as f32 / runs.max(1) as f32;
    let share = (share_s > 0.0).then(|| (share_l as f64 * world_sim::TICK_S as f64 / share_s) as f32);
    let ratio = match (mean(&mem_aim), mean(&seen_aim)) {
        (Some(a), Some(b)) if b > 1e-5 => Some(a / b),
        _ => None,
    };
    let cells = vec![
        cell("success_rate", Some(rate), Rule::Like { reference: FLOOR_PICK.0, tolerance: FLOOR_PICK.1 }, CITE_FLOOR_PICK),
        cell("side_pick_share", (picks > 0).then(|| side as f32 / picks as f32), Rule::Raw, ""),
        cell("approach_vision_share", share, Rule::AtMost { limit: VISION_SHARE_MAX }, CITE_SHARE),
        cell("memory_over_seen_aim", ratio, Rule::Raw, CITE_MEMORY_REACH),
        cell("memory_aim_mm", mean(&mem_aim).map(|e| e * 1000.0), Rule::Raw, ""),
    ];
    BenchScenario::from_cells("VX", &id, "베이스캠프 택배 상자를 지나가며 오른쪽으로 집기", &f, false, cells)
}

pub fn v3(plan: &str) -> BenchScenario {
    let id = format!("V3-{plan}");
    let f = form(plan, BaseKind::HumanFoot);
    let cell = |axis: &str, value: Option<f32>, cite: &str| BenchCell::new("VX", &id, "", axis, None, &f, value, Rule::Raw, cite, false);
    let run = |intermittent: bool| -> (f32, f32, f32) {
        let (mut ok, mut scans, mut time) = (0u32, 0u64, 0.0f64);
        for seed in [3u64, 7] {
            let mut rt = runtime(seed, BaseKind::HumanFoot, plan, None, intermittent, NoiseCfg::demo());
            let s = rt.submit("원목 1개를 창고로 가져와");
            assert!(s.ok);
            rt.run_until_idle(12000);
            let q = rt.quests().last().cloned().expect("a quest");
            if q.state == noto_quest::State::Done {
                ok += 1;
                time += q.result.as_ref().map_or(0.0, |r| r.time_s);
            }
            scans += rt.eval().scans;
        }
        (ok as f32 / 2.0, scans as f32 / 2.0, if ok > 0 { (time / ok as f64) as f32 } else { f32::NAN })
    };
    let (on_ok, on_scans, on_t) = run(false);
    let (pat_ok, pat_scans, pat_t) = run(true);
    let cells = vec![
        cell("success_pattern", Some(pat_ok), "패턴 시각의 성공률"),
        cell("success_always", Some(on_ok), "늘 켠 시각의 성공률"),
        cell("success_kept", Some(pat_ok - on_ok), "패턴 − 늘 켬 (설계 목표: 정밀도 유지)"),
        cell("scans_pattern", Some(pat_scans), "퀘스트당 스캔 (계산 비용)"),
        cell("scans_always", Some(on_scans), "퀘스트당 스캔"),
        cell("time_pattern_s", Some(pat_t), ""),
        cell("time_always_s", Some(on_t), ""),
    ];
    BenchScenario::from_cells("VX", &id, "패턴 시각과 늘 켠 시각 (원목 1개)", &f, false, cells)
}

pub fn scenarios() -> Vec<BenchScenario> {
    vec![v1("mid", BaseKind::HumanFoot), v1("mid", BaseKind::Wheel), v2("mid"), v3("pro")]
}
