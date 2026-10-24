//! End-to-end scenarios: a typed command, a quest, a world, the gates, the logs.
//!
//! The recogniser here is the stand-in (the renderer's ground truth plus
//! injected errors). These tests show that the pieces fit and that the gates
//! behave consistently under a made-up error model; they say nothing about how
//! a real SigLIP2 would perform.

use hand_core::{BaseKind, HandConfig};
use hand_quest::{State, StepKind};
use hand_runtime::{FrameView, GateCfg, NoiseCfg, Runtime, RuntimeConfig};
use world_sim::{ItemKind, NodeKind};

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

/// The robot of these scenarios walks, so its hands reach the ground: with
/// the arm and shoulder height of the presets a wheeled body cannot bend down
/// to an item and pushes it instead (`a_wheeled_body_pushes_what_it_cannot_reach`).
fn legged(cfg: RuntimeConfig) -> RuntimeConfig {
    RuntimeConfig { hand: HandConfig { base: BaseKind::HumanFoot, ..cfg.hand }, ..cfg }
}

fn runtime_with(cfg: RuntimeConfig) -> Runtime {
    Runtime::new(legged(cfg), LEXICON, None).expect("runtime")
}

fn runtime(seed: u64, noise: NoiseCfg) -> Runtime {
    runtime_with(RuntimeConfig { seed, noise, ..RuntimeConfig::default() })
}

fn stock(rt: &Runtime, place: &str, item: ItemKind) -> u32 {
    rt.world().waypoint_by_key(place).map(|w| (w.stock_of(item) + 1e-3).floor() as u32).unwrap_or(0)
}

fn quest_log(rt: &Runtime) -> String {
    rt.log_tail(0, 5000, None).0.iter().filter(|l| matches!(l.cat.as_str(), "quest" | "search")).map(|l| format!("[{}] {}", l.cat, l.msg)).collect::<Vec<_>>().join("\n")
}

/// Submits, runs to the end, and returns the finished quest.
fn run_command(rt: &mut Runtime, text: &str, max_ticks: u64) -> hand_quest::Quest {
    let s = rt.submit(text);
    assert!(s.ok, "{text}: {:?} {:?}", s.error_code, s.error);
    let id = s.quest_id.clone().expect("a quest was made");
    rt.run_until_idle(max_ticks);
    rt.quest_book().get(&id).unwrap().clone()
}

fn assert_done(rt: &Runtime, q: &hand_quest::Quest) {
    assert_eq!(
        q.state,
        State::Done,
        "failure {:?}\nsteps {:#?}\nlog:\n{}",
        q.failure,
        q.steps.iter().map(|s| (&s.label_ko, s.state, &s.note)).collect::<Vec<_>>(),
        quest_log(rt)
    );
}

#[test]
fn a_typed_korean_command_is_carried_out_to_the_warehouse() {
    let mut rt = runtime(7, NoiseCfg::none());
    let q = run_command(&mut rt, "원목 2개를 캐서 창고로 가져와", 30_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Log), 2);
    let r = q.result.unwrap();
    assert!(r.time_s > 5.0 && r.energy_j > 0.0, "{r:?}");
    assert_eq!(r.precision, Some(1.0));
    assert_eq!(r.wasted_actions, 0);
    assert!(r.summary.contains("원목 2개를 창고에 인도"), "{}", r.summary);
    assert!(q.steps.iter().all(|s| s.state == State::Done && s.progress == 1.0));
    let kinds: Vec<StepKind> = q.steps.iter().map(|s| s.kind).collect();
    assert_eq!(kinds.last(), Some(&StepKind::Deliver));
    // The quest itself produced composite N:N states (carrying while braking at
    // the end of each move), scored with the research weights as they happened.
    let nn = rt.api_call("nn_status", &serde_json::Value::Null).unwrap();
    let scen = nn["scenarios"].as_array().unwrap();
    assert!(scen.iter().any(|s| s["scenario"] == "S4"), "carry + brake was scored: {nn}");
    for s in scen {
        println!("[NN-SUM] quest 원목 2개 · {} {} 점수 {} · 위상 {} · 사건 {} · 게이트 {} · 최소 여유 {:.2} · 최대 피크 {:.0}%", s["scenario"], s["name_ko"], s["score"], s["phases_seen"], s["events"], s["gated"], s["min_margin"].as_f64().unwrap_or(f64::NAN), s["peak_ratio"].as_f64().unwrap_or(0.0) * 100.0);
    }
}

#[test]
fn english_commands_work_through_the_same_dictionary() {
    let mut rt = runtime(7, NoiseCfg::none());
    let q = run_command(&mut rt, "bring 1 stone to the warehouse", 30_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Stone), 1);
}

#[test]
fn a_shortfall_sends_the_robot_back_to_search_for_more() {
    // A boulder drops two stones; three are asked for.
    let mut rt = runtime(7, NoiseCfg::none());
    let q = run_command(&mut rt, "돌 3개를 베이스캠프로 가져와", 40_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "base_camp", ItemKind::Stone), 3);
    assert!(q.steps[0].retries >= 1, "the search step was reopened: {:?}", q.steps[0]);
}

#[test]
fn several_resources_in_one_command_share_one_delivery() {
    let mut rt = runtime(11, NoiseCfg::none());
    let q = run_command(&mut rt, "원목 1개와 돌 1개를 창고로 가져와", 40_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Log), 1);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Stone), 1);
    let r = q.result.unwrap();
    assert!(r.items.contains(&("log".to_string(), 1)) && r.items.contains(&("stone".to_string(), 1)), "{:?}", r.items);
}

#[test]
fn collecting_without_a_destination_keeps_the_items_in_hand() {
    let mut rt = runtime(7, NoiseCfg::none());
    let q = run_command(&mut rt, "원목 1개 캐줘", 30_000);
    assert_done(&rt, &q);
    assert_eq!(rt.world().carried_count(ItemKind::Log), 1);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Log), 0);
}

#[test]
fn a_wheeled_body_pushes_what_it_cannot_reach() {
    // Shoulder 1.00 m, arm 0.42 m and no waist pitch: the palm stops about
    // 0.65 m above the ground, so the wheeled robot moves the log with its
    // bumper, and a collection with nowhere to push to cannot be done.
    let wheeled = || Runtime::new(RuntimeConfig { seed: 7, noise: NoiseCfg::none(), ..RuntimeConfig::default() }, LEXICON, None).expect("runtime");
    let mut rt = wheeled();
    assert_eq!(rt.world().hand.base.kind, BaseKind::Wheel);
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 40_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Log), 1);
    assert!(quest_log(&rt).contains("범퍼로 밀어"), "{}", quest_log(&rt));
    assert_eq!(rt.world().carried_count(ItemKind::Log), 0, "it never held the log");
    let mut rt = wheeled();
    let q = run_command(&mut rt, "원목 1개 캐줘", 30_000);
    assert_eq!(q.state, State::Failed, "{}", quest_log(&rt));
    let f = q.failure.as_ref().unwrap();
    assert_eq!(f.code, "reach_ground", "{f:?}");
    assert!(f.message_ko.contains("손바닥"), "{}", f.message_ko);
}

#[test]
fn going_to_a_place_ends_within_reach_of_it() {
    let mut rt = runtime(7, NoiseCfg::none());
    let q = run_command(&mut rt, "창고로 가", 30_000);
    assert_done(&rt, &q);
    let w = rt.world().waypoint_by_key("warehouse").unwrap();
    let d = ((w.x - rt.world().agent.x).powi(2) + (w.z - rt.world().agent.z).powi(2)).sqrt();
    assert!(d <= 1.6, "{d} m from the warehouse");
}

#[test]
fn commands_that_cannot_become_a_quest_say_why() {
    let mut rt = runtime(7, NoiseCfg::none());
    for (text, code) in [("다이아몬드 3개를 창고로 가져와", "unknown_item"), ("", "empty"), ("원목 0개를 창고로 가져와", "zero_quantity")] {
        let s = rt.submit(text);
        assert!(!s.ok, "{text}");
        assert_eq!(s.error_code.as_deref(), Some(code), "{text}: {:?}", s.error);
        assert!(s.quest_id.is_none() && s.error.as_deref().map_or(false, |m| !m.is_empty()));
    }
    assert!(rt.quests().is_empty(), "no quest was created");
}

#[test]
fn quests_run_one_after_another_in_the_order_given() {
    let mut rt = runtime(7, NoiseCfg::none());
    let a = rt.submit("원목 1개를 창고로 가져와").quest_id.unwrap();
    let b = rt.submit("돌 1개를 창고로 가져와").quest_id.unwrap();
    rt.step(20);
    assert_eq!(rt.quest_book().get(&a).unwrap().state, State::Active);
    assert_eq!(rt.quest_book().get(&b).unwrap().state, State::Pending);
    rt.run_until_idle(60_000);
    for id in [&a, &b] {
        assert_eq!(rt.quest_book().get(id).unwrap().state, State::Done, "{}", quest_log(&rt));
    }
    let (ta, tb) = (rt.quest_book().get(&a).unwrap().started_tick.unwrap(), rt.quest_book().get(&b).unwrap().started_tick.unwrap());
    assert!(ta < tb);
}

#[test]
fn stop_cancels_the_running_quest_and_the_queue() {
    let mut rt = runtime(7, NoiseCfg::none());
    let a = rt.submit("원목 2개를 창고로 가져와").quest_id.unwrap();
    let b = rt.submit("돌 1개를 창고로 가져와").quest_id.unwrap();
    rt.step(60);
    let s = rt.submit("멈춰");
    assert!(s.ok && s.stopped && s.quest_id.is_none(), "{:?}", s.error);
    for id in [&a, &b] {
        let q = rt.quest_book().get(id).unwrap();
        assert_eq!(q.state, State::Failed);
        assert_eq!(q.failure.as_ref().unwrap().code, "cancelled");
    }
    assert!(!rt.world().busy() && !rt.busy());
    // and it can be given work again
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 30_000);
    assert_done(&rt, &q);
}

#[test]
fn a_resource_that_does_not_exist_ends_in_a_search_timeout() {
    let cfg = RuntimeConfig { seed: 7, noise: NoiseCfg::none(), locate_timeout_s: 45.0, ..RuntimeConfig::default() };
    let mut rt = runtime_with(cfg);
    rt.world_mut().nodes.retain(|n| n.kind != NodeKind::Tree);
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 20_000);
    assert_eq!(q.state, State::Failed, "{}", quest_log(&rt));
    let f = q.failure.as_ref().unwrap();
    assert!(f.code == "locate_timeout", "{f:?}");
    assert!(f.message_ko.contains("원목을"), "{}", f.message_ko);
    assert_eq!(q.steps[0].state, State::Failed, "the failure is shown on the step that failed");
    assert!(q.result.is_some(), "a failed quest still reports time and energy");
    assert!(!rt.world().busy());
}

#[test]
fn a_load_that_does_not_fit_is_carried_in_round_trips() {
    // Before v0.5 this quest failed: the bay held iron ore up to about one
    // stone below the limit, so the second stone did not fit. Now the cargo
    // decision carries the first stone to the warehouse and comes back.
    let mut rt = runtime(7, NoiseCfg::none());
    let cap = rt.world().capacity_kg();
    let n = ((cap - 1.0) / ItemKind::IronOre.spec().mass_kg).floor();
    assert!(cap - n * ItemKind::IronOre.spec().mass_kg < 2.0 * ItemKind::Stone.spec().mass_kg, "two stones do not fit beside the ore");
    rt.world_mut().agent.carried.push((ItemKind::IronOre, n));
    let q = run_command(&mut rt, "돌 2개를 창고로 가져와", 40_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Stone), 2);
    let r = q.result.as_ref().unwrap();
    assert!(r.trips >= 2, "a round trip was made: {r:?}");
    assert!(r.summary.contains("왕복"), "{}", r.summary);
    // The ore is not this quest's cargo: it stays aboard.
    assert!((rt.world().carried_amount(ItemKind::IronOre) - n).abs() < 1e-4);
    assert!(quest_log(&rt).contains("왕복 1회차"));
}

#[test]
fn a_bay_full_of_other_cargo_is_reported_with_a_hint_to_set_it_down() {
    let mut rt = runtime(7, NoiseCfg::none());
    let cap = rt.world().capacity_kg();
    // Ore and coal up to less than a coal lump below the limit. A stone does
    // not fit, a trip would not help (the cargo is not the quest's), and
    // breaking the stone to squeeze it beside someone else's cargo is not done.
    let (iron, coal) = (ItemKind::IronOre.spec().mass_kg, ItemKind::Coal.spec().mass_kg);
    let n = (cap / iron).floor();
    let m = ((cap - n * iron) / coal).floor();
    rt.world_mut().agent.carried.extend([(ItemKind::IronOre, n), (ItemKind::Coal, m)]);
    let room = cap - n * iron - m * coal;
    assert!(room < ItemKind::Stone.spec().mass_kg, "{room} kg of room");
    let q = run_command(&mut rt, "돌 2개를 창고로 가져와", 30_000);
    assert_eq!(q.state, State::Failed, "{}", quest_log(&rt));
    let f = q.failure.as_ref().unwrap();
    assert_eq!(f.code, "capacity_exceeded", "{f:?}");
    assert!(f.message_ko.contains("적재 한도") && f.message_ko.contains("내려놓"), "{}", f.message_ko);
}

#[test]
fn the_same_seed_gives_the_same_run() {
    let run = || {
        let mut rt = runtime(5, NoiseCfg::demo());
        let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 40_000);
        (rt.world().tick, q.state, q.result.map(|r| (r.ticks, r.wasted_actions, r.summary)))
    };
    assert_eq!(run(), run());
}

/// What a batch of commands cost with the gates on or off.
struct Batch {
    done: usize,
    runs: usize,
    wasted: u32,
    ticks: u64,
    raw_precision: f64,
    gated_precision: f64,
    raw_recall: f64,
    gated_recall: f64,
}

fn batch(gates: bool, seeds: &[u64]) -> Batch {
    let (mut done, mut wasted, mut ticks) = (0, 0, 0u64);
    let (mut rp, mut gp, mut rr, mut gr) = (0.0, 0.0, 0.0, 0.0);
    for &seed in seeds {
        let cfg = RuntimeConfig { seed, noise: NoiseCfg { seed: seed + 100, ..NoiseCfg::demo() }, gates: GateCfg { enabled: gates, ..GateCfg::default() }, ..RuntimeConfig::default() };
        let mut rt = runtime_with(cfg);
        let q = run_command(&mut rt, "돌 2개를 창고로 가져와", 40_000);
        if q.state == State::Done {
            done += 1;
        }
        wasted += q.result.as_ref().map_or(0, |r| r.wasted_actions);
        ticks += rt.world().tick;
        let e = rt.eval().summary();
        rp += e.raw_precision.unwrap_or(0.0);
        gp += e.gated_precision.unwrap_or(0.0);
        rr += e.raw_recall.unwrap_or(0.0);
        gr += e.gated_recall.unwrap_or(0.0);
    }
    let n = seeds.len() as f64;
    Batch { done, runs: seeds.len(), wasted, ticks, raw_precision: rp / n, gated_precision: gp / n, raw_recall: rr / n, gated_recall: gr / n }
}

#[test]
fn under_injected_errors_the_gates_cut_wasted_work() {
    let seeds: Vec<u64> = (1..=12).map(|i| i * 2 + 1).collect();
    let on = batch(true, &seeds);
    let off = batch(false, &seeds);
    // `cargo run --release -p hand-runtime --example gate_ab` prints the same comparison.
    eprintln!(
        "gates on : {}/{} done, wasted {}, ticks {}, precision raw {:.2} -> gated {:.2}, recall raw {:.2} -> gated {:.2}",
        on.done, on.runs, on.wasted, on.ticks, on.raw_precision, on.gated_precision, on.raw_recall, on.gated_recall
    );
    eprintln!(
        "gates off: {}/{} done, wasted {}, ticks {}, precision raw {:.2} -> accepted {:.2}, recall raw {:.2} -> accepted {:.2}",
        off.done, off.runs, off.wasted, off.ticks, off.raw_precision, off.gated_precision, off.raw_recall, off.gated_recall
    );
    assert_eq!(on.done, on.runs, "every run finishes with the gates on");
    assert!(off.wasted >= 4, "the injected errors must cost something without gates, got {}", off.wasted);
    assert!(on.wasted * 2 <= off.wasted, "gates on wasted {} vs off {}", on.wasted, off.wasted);
    assert!(on.gated_precision > on.raw_precision + 0.05, "gated {:.2} vs raw {:.2}", on.gated_precision, on.raw_precision);
    assert!(on.gated_recall <= on.raw_recall + 1e-9, "gating can only lose detections, never add them");
    assert!((off.gated_precision - off.raw_precision).abs() < 1e-9, "with the gates off everything raw is accepted");
}

#[test]
fn touching_things_teaches_the_sds_what_to_trust() {
    // With heavy confusion and no gates the robot is wrong often; the SDS records each touch.
    let cfg = RuntimeConfig {
        seed: 7,
        noise: NoiseCfg { seed: 4, score_sd: 0.05, confusion_rate: 0.5, false_positive_rate: 0.3, miss_rate: 0.0 },
        gates: GateCfg { enabled: false, ..GateCfg::default() },
        ..RuntimeConfig::default()
    };
    let mut rt = runtime_with(cfg);
    for text in ["돌 2개를 창고로", "석탄 2개를 창고로", "돌 2개를 베이스캠프로"] {
        let q = run_command(&mut rt, text, 60_000);
        assert!(q.is_finished(), "{text}: {}", quest_log(&rt));
    }
    let axes = rt.sds_status().axes;
    assert!(axes.iter().any(|a| a.axis == "score_all"), "every detection's score is recorded");
    let st = rt.sds_status();
    assert!(!st.confusions.is_empty(), "wrong guesses are recorded as confusions: {:?}", st.confusions);
    let log = rt.log_tail(0, 5000, Some("vision")).0;
    assert!(log.iter().any(|l| l.msg.contains("접촉 불일치") || l.msg.contains("헛감지")), "mistakes show up in the log");
}

#[test]
fn logs_are_chained_on_disk_and_the_ledgers_are_written() {
    let dir = std::env::temp_dir().join(format!("hand-runtime-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let path;
    {
        let mut rt = Runtime::new(RuntimeConfig { seed: 7, noise: NoiseCfg::demo(), ..RuntimeConfig::default() }, LEXICON, Some(&dir)).unwrap();
        let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 40_000);
        assert_done(&rt, &q);
        rt.sds_mut().flush();
        path = rt.audit().path().expect("a file log").to_path_buf();
        let cats: std::collections::BTreeSet<String> = rt.log_tail(0, 5000, None).0.iter().map(|l| l.cat.clone()).collect();
        for c in ["world", "quest", "search", "vision", "device"] {
            assert!(cats.contains(c), "category {c} missing in {cats:?}");
        }
    }
    let ok = hand_sds::verify_file(&path).expect("the hash chain verifies");
    assert!(ok.lines > 20, "{} lines", ok.lines);
    let ledgers: Vec<_> = std::fs::read_dir(dir.join("sds")).map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect()).unwrap_or_default();
    assert!(ledgers.iter().any(|f| f.ends_with(".json")), "SDS files: {ledgers:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_manual_scan_gives_a_frame_with_boxes_a_body_status_and_a_forecast_series() {
    let mut rt = runtime(7, NoiseCfg::demo());
    for _ in 0..12 {
        rt.scan_now("manual").unwrap();
    }
    let r = rt.last_scan().unwrap().clone();
    assert!(!r.judged.is_empty(), "something is in view at the start");
    let f = rt.frame(FrameView::Agent, 70.0).unwrap();
    assert!(f.webp.starts_with(b"RIFF") && &f.webp[8..12] == b"WEBP");
    assert_eq!(f.detections.len(), r.judged.len(), "the boxes of the latest scan are attached");
    assert!(f.detections.iter().all(|d| d.bbox.iter().all(|v| (0.0..=1.0).contains(v)) && d.bbox[0] < d.bbox[2] && d.bbox[1] < d.bbox[3]));
    let top = rt.frame(FrameView::Top, 70.0).unwrap();
    assert!(top.detections.is_empty() && top.webp.len() > 100);
    let (w, h, rgb) = world_sim::decode_webp(&f.webp).unwrap();
    assert_eq!((w as usize, h as usize), (rt.camera().width, rt.camera().height));
    assert_eq!(rgb.len(), (w * h * 3) as usize);
    let body = rt.body_status();
    let head = body.parts.iter().find(|p| p.id == "head").expect("the head part shows vision");
    assert!(head.detail.contains("인식 신뢰도") && head.detail.contains("프레임"), "{}", head.detail);
    let ts = rt.ts_view(None, 3);
    assert!(!ts.history.is_empty() && ts.forecast.is_some(), "{ts:?}");
    assert_eq!(ts.forecast.as_ref().unwrap().point.len(), 3);
}
