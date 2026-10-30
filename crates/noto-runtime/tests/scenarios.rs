use noto_core::{BaseKind, NotoConfig};
use noto_quest::{State, StepKind};
use noto_runtime::{FrameView, GateCfg, NoiseCfg, Runtime, RuntimeConfig};
use world_sim::{ItemKind, NodeKind};

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn legged(cfg: RuntimeConfig) -> RuntimeConfig {
    RuntimeConfig { noto: NotoConfig { base: BaseKind::HumanFoot, ..cfg.noto }, ..cfg }
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

fn run_command(rt: &mut Runtime, text: &str, max_ticks: u64) -> noto_quest::Quest {
    let s = rt.submit(text);
    assert!(s.ok, "{text}: {:?} {:?}", s.error_code, s.error);
    let id = s.quest_id.clone().expect("a quest was made");
    rt.run_until_idle(max_ticks);
    rt.quest_book().get(&id).unwrap().clone()
}

fn assert_done(rt: &Runtime, q: &noto_quest::Quest) {
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
fn a_wheeled_body_lifts_the_log_on_its_forks_and_keeps_it_there_without_a_destination() {
    let wheeled = || Runtime::new(RuntimeConfig { seed: 7, noise: NoiseCfg::none(), ..RuntimeConfig::default() }, LEXICON, None).expect("runtime");
    let mut rt = wheeled();
    assert_eq!(rt.world().hand.base.kind, BaseKind::Wheel);
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 40_000);
    assert_done(&rt, &q);
    assert_eq!(stock(&rt, "warehouse", ItemKind::Log), 1);
    let log = quest_log(&rt);
    assert!(!log.contains("범퍼로 밀어"), "{log}");
    let world_log = rt.log_tail(0, 5000, None).0.iter().filter(|l| l.cat == "world").map(|l| l.msg.clone()).collect::<Vec<_>>().join("
");
    assert!(world_log.contains("포크") && world_log.contains("들어올림"), "{world_log}");
    let mut rt = wheeled();
    let q = run_command(&mut rt, "원목 1개 캐줘", 30_000);
    assert_done(&rt, &q);
    assert_eq!(rt.world().carried_count(ItemKind::Log), 1, "the log stays on the tines");
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
    assert!((rt.world().carried_amount(ItemKind::IronOre) - n).abs() < 1e-4);
    assert!(quest_log(&rt).contains("왕복 1회차"));
}

#[test]
fn a_bay_full_of_other_cargo_is_reported_with_a_hint_to_set_it_down() {
    let mut rt = runtime(7, NoiseCfg::none());
    let cap = rt.world().capacity_kg();
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
    eprintln!(
        "gates on : {}/{} done, wasted {}, ticks {}, precision raw {:.2} -> gated {:.2}, recall raw {:.2} -> gated {:.2}",
        on.done, on.runs, on.wasted, on.ticks, on.raw_precision, on.gated_precision, on.raw_recall, on.gated_recall
    );
    eprintln!(
        "gates off: {}/{} done, wasted {}, ticks {}, precision raw {:.2} -> accepted {:.2}, recall raw {:.2} -> accepted {:.2}",
        off.done, off.runs, off.wasted, off.ticks, off.raw_precision, off.gated_precision, off.raw_recall, off.gated_recall
    );
    assert_eq!(on.done, on.runs, "every run finishes with the gates on");
    assert!(off.wasted >= 2, "the injected errors must cost something without gates, got {}", off.wasted);
    assert!(on.wasted * 2 <= off.wasted, "gates on wasted {} vs off {}", on.wasted, off.wasted);
    assert!(on.gated_precision > on.raw_precision + 0.05, "gated {:.2} vs raw {:.2}", on.gated_precision, on.raw_precision);
    assert!(on.gated_recall <= on.raw_recall + 1e-9, "gating can only lose detections, never add them");
    assert!((off.gated_precision - off.raw_precision).abs() < 1e-9, "with the gates off everything raw is accepted");
}

#[test]
fn touching_things_teaches_the_sds_what_to_trust() {
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
    let dir = std::env::temp_dir().join(format!("noto-runtime-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
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
    let ok = noto_sds::verify_file(&path).expect("the hash chain verifies");
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

#[test]
fn a_parcel_seen_faintly_from_afar_is_walked_to_and_looked_down_at() {
    let cfg = RuntimeConfig {
        seed: 7,
        noise: NoiseCfg::none(),
        parcels: Some(0),
        terrain: world_sim::TerrainKind::Open,
        gates: GateCfg { min_score: 0.80, adaptive: false, ..GateCfg::default() },
        ..RuntimeConfig::default()
    };
    let mut rt = runtime_with(cfg);
    let (x, z, yaw) = (rt.world().agent.x, rt.world().agent.z, rt.world().agent.yaw);
    let (px, pz) = (x + 5.0 * yaw.cos(), z + 5.0 * yaw.sin());
    rt.world_mut().drops.push(world_sim::world::DropItem::whole(9001, ItemKind::Parcel, px, pz));
    let q = run_command(&mut rt, "택배 1개를 창고로 가져와", 20_000);
    let log = quest_log(&rt);
    assert!(log.contains("가까이 가서 내려다보며 확인"), "no walk-up look:\n{log}");
    assert!(log.contains("내려다보고 확인"), "the close look did not confirm it:\n{log}");
    assert_done(&rt, &q);
}

fn vision_lines(rt: &Runtime) -> String {
    rt.log_tail(0, 5000, Some("vision")).0.iter().map(|l| l.msg.clone()).collect::<Vec<_>>().join("\n")
}

/// What a log pick-up showed on the way: the deepest the neck bent, the deepest the head camera dipped below where it
/// looks standing still, the lowest the fork tines went, and the vision and quest logs.
struct PickLook {
    neck_deg: f32,
    dip_deg: f32,
    fork_low_m: f32,
    carried: bool,
    log: String,
}

fn pick_with_look(base: BaseKind, body: noto_core::BodyPlan) -> PickLook {
    let cfg = RuntimeConfig { seed: 7, noise: NoiseCfg::none(), noto: NotoConfig { base, ..RuntimeConfig::default().noto }, body, ..RuntimeConfig::default() };
    let mut rt = Runtime::new(cfg, LEXICON, None).expect("runtime");
    let rest = rt.head_view_json()["pitch_deg"].as_f64().unwrap_or(0.0) as f32;
    let s = rt.submit("원목 1개를 캐서 창고로 가져와");
    assert!(s.ok, "{:?}", s.error);
    let (mut neck_deg, mut dip_deg, mut fork_low_m, mut carried) = (0.0f32, 0.0f32, f32::INFINITY, false);
    let mut n = 0;
    while n < 30_000 && rt.busy() {
        rt.tick();
        neck_deg = neck_deg.max(rt.world().current_kin_pose().neck_deg);
        let v = rt.head_view_json();
        dip_deg = dip_deg.max(rest - v["pitch_deg"].as_f64().unwrap_or(0.0) as f32);
        fork_low_m = fork_low_m.min(rt.world().fork_m_now());
        carried |= rt.world().carried_count(ItemKind::Log) > 0;
        n += 1;
    }
    let log = format!("{}
-- quest --
{}", vision_lines(&rt), quest_log(&rt));
    PickLook { neck_deg, dip_deg, fork_low_m, carried, log }
}

#[test]
fn a_walker_bends_its_neck_to_look_at_the_item_before_the_hand_and_the_vision_check_runs() {
    let p = pick_with_look(BaseKind::HumanFoot, noto_core::BodyPlan::standard());
    assert!(p.neck_deg > 20.0, "the neck bent only {:.0}°
{}", p.neck_deg, p.log);
    assert!(p.dip_deg > 20.0, "the camera dipped only {:.0}°
{}", p.dip_deg, p.log);
    assert!(p.log.contains("비전 확인 ✓") && p.log.contains("고개"), "no head-down vision check:
{}", p.log);
}

#[test]
fn wheels_and_tracks_lift_with_forks_and_check_the_item_with_the_rider_and_the_fork_camera() {
    let lift = noto_core::BodyPlan { build: noto_core::dims::Build::Standard, ..noto_core::BodyPlan::named("wheel_lift").unwrap() };
    let noneck = noto_core::BodyPlan { morph: noto_core::morph::Morphology::named("wheel_noneck").unwrap(), ..noto_core::BodyPlan::standard() };
    for (base, body, tag, neck) in [
        (BaseKind::Wheel, noto_core::BodyPlan::standard(), "wheel", true),
        (BaseKind::Track, noto_core::BodyPlan::standard(), "track", true),
        (BaseKind::Mecanum, noto_core::BodyPlan::standard(), "mecanum", true),
        (BaseKind::Wheel, lift, "wheel+lift", true),
        (BaseKind::Wheel, noneck, "wheel, no neck", false),
    ] {
        let p = pick_with_look(base, body);
        if neck {
            // An operator glances down at the item before the tines go down, and again once it blocks the fork camera.
            assert!(p.neck_deg > 20.0 && p.dip_deg > 20.0, "{tag}: the rider bends the head to the item ({:.0}°, camera {:.0}° down)
{}", p.neck_deg, p.dip_deg, p.log);
            assert!(p.log.contains("교차 검증 ✓"), "{tag}: the rider's and the fork camera's looks agree:
{}", p.log);
            assert!(p.log.contains("재확인 ✓") && p.log.contains("가려져"), "{tag}: the rider looks again once the item blocks the fork camera:
{}", p.log);
        } else {
            // A fixed camera still tips with the chassis on uneven ground; there is no neck to bend.
            assert!(p.neck_deg < 1.0, "{tag}: no neck to bend ({:.0}°, camera {:.0}° down)
{}", p.neck_deg, p.dip_deg, p.log);
            assert!(!p.log.contains("탑승자 시선 ✓") && p.log.contains("포크 카메라만"), "{tag}: the fork camera alone checks:
{}", p.log);
        }
        assert!(p.fork_low_m < 0.01, "{tag}: the tines went down to the floor ({:.3} m)
{}", p.fork_low_m, p.log);
        assert!(p.carried, "{tag}: the log rode on the tines
{}", p.log);
        assert!(p.log.contains("비전 확인 ✓") && p.log.contains("포크 카메라"), "{tag}: no fork-camera vision check:
{}", p.log);
        assert!(!p.log.contains("범퍼로 밀어"), "{tag}: it lifted rather than pushed
{}", p.log);
    }
}

/// The map and quest of the run that froze on 2026-10-29: seed 1 on the rugged map (the window's default), wheels,
/// "원목 1개를 창고로 가져와". There the felled tree's logs, remembered from the real models a few centimetres off,
/// each counted as lying in front of the other and the pick-up swapped them without end inside one tick (the window
/// froze, the session log grew to 450 MB); the driver's unit tests pin that. Here the whole fetch must run through:
/// at most the one reorder the felled tree calls for, a log lifted and delivered. The quest runs on its own thread so
/// that a tick that spins fails the test instead of hanging it.
#[test]
fn a_lift_truck_lifts_a_felled_trees_logs_without_swapping_them_forever() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let noto = NotoConfig { base: BaseKind::Wheel, ..RuntimeConfig::default().noto };
        let cfg = RuntimeConfig { seed: 1, noto, terrain: world_sim::TerrainKind::Rugged, ..RuntimeConfig::default() };
        let mut rt = Runtime::new(cfg, LEXICON, None).expect("runtime");
        let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 30_000);
        let reorders = rt.log_tail(0, 5000, Some("quest")).0.iter().filter(|l| l.msg.contains("포크 길 앞에")).count();
        let _ = tx.send((q.state, q.failure.clone(), reorders, stock(&rt, "warehouse", ItemKind::Log), quest_log(&rt)));
    });
    let (state, failure, reorders, stocked, log) = rx.recv_timeout(std::time::Duration::from_secs(900)).expect("the quest spins inside one tick");
    assert!(reorders <= 1, "{reorders} reorders:\n{log}");
    assert_eq!(state, State::Done, "{failure:?}\n{log}");
    assert_eq!(stocked, 1, "{log}");
}
