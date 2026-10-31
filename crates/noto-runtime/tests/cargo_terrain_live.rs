use noto_core::{BaseKind, BodyPlan, NotoConfig, Tier};
use noto_quest::State;
use noto_sds::{Scope, Track};
use noto_runtime::{NoiseCfg, Runtime, RuntimeConfig, TestPhase};
use serde_json::{json, Value};
use world_sim::{ItemKind, TerrainKind};

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn runtime_with(cfg: RuntimeConfig) -> Runtime {
    let cfg = RuntimeConfig { noto: NotoConfig { base: BaseKind::HumanFoot, ..cfg.noto }, ..cfg };
    Runtime::new(cfg, LEXICON, None).expect("runtime")
}

fn runtime(seed: u64) -> Runtime {
    runtime_with(RuntimeConfig { seed, noise: NoiseCfg::none(), ..RuntimeConfig::default() })
}

fn quest_log(rt: &Runtime) -> String {
    rt.log_tail(0, 6000, None).0.iter().filter(|l| matches!(l.cat.as_str(), "quest" | "search" | "world")).map(|l| format!("[{}] {}", l.cat, l.msg)).collect::<Vec<_>>().join("\n")
}

fn run_command(rt: &mut Runtime, text: &str, max_ticks: u64) -> noto_quest::Quest {
    let s = rt.submit(text);
    assert!(s.ok, "{text}: {:?} {:?}", s.error_code, s.error);
    let id = s.quest_id.clone().expect("a quest was made");
    rt.run_until_idle(max_ticks);
    rt.quest_book().get(&id).unwrap().clone()
}

fn assert_done(rt: &Runtime, q: &noto_quest::Quest) {
    assert_eq!(q.state, State::Done, "failure {:?}\nsteps {:#?}\nlog:\n{}", q.failure, q.steps.iter().map(|s| (&s.label_ko, s.state, &s.note)).collect::<Vec<_>>(), quest_log(rt));
}

#[test]
fn carried_items_are_set_down_by_command_here_or_at_a_place() {
    let mut rt = runtime_with(RuntimeConfig { seed: 7, noise: NoiseCfg::none(), body: BodyPlan { build: noto_core::dims::Build::Standard, ..BodyPlan::named("pro").unwrap() }, ..RuntimeConfig::default() });
    let q = run_command(&mut rt, "원목 2개 캐줘", 30_000);
    assert_done(&rt, &q);
    assert_eq!(rt.world().carried_count(ItemKind::Log), 2);
    let drops_before = rt.world().drops.len();
    let q = run_command(&mut rt, "원목 1개 내려놔", 5_000);
    assert_done(&rt, &q);
    assert_eq!(rt.world().carried_count(ItemKind::Log), 1);
    assert_eq!(rt.world().drops.len(), drops_before + 1, "the log lies on the ground");
    assert!(q.result.as_ref().unwrap().summary.contains("내려놓았습니다"));
    let q = run_command(&mut rt, "베이스캠프에 다 내려놔", 30_000);
    assert_done(&rt, &q);
    assert!(rt.world().agent.carried.is_empty());
    assert_eq!(rt.world().waypoint_by_key("base_camp").unwrap().stock_of(ItemKind::Log), 1.0);
    let q = run_command(&mut rt, "다 내려놔", 2_000);
    assert_eq!(q.state, State::Failed);
    assert_eq!(q.failure.as_ref().unwrap().code, "nothing_to_drop");
}

#[test]
fn the_ground_reaches_the_physics_status_and_the_sds() {
    let mut rt = runtime(7);
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 40_000);
    assert_done(&rt, &q);
    let v = rt.api_call("physics_status", &Value::Null).unwrap();
    assert_eq!(v["surfaces"].as_array().unwrap().len(), 8);
    assert_eq!(v["ground"]["id"], "gravel", "the robot ends on the warehouse pad");
    assert!(v["joints"].as_array().unwrap().iter().any(|j| j["life_used"].as_f64().unwrap() > 0.0), "{}", v["joints"]);
    assert!(v["load"]["capacity_here_kg"].as_f64().unwrap() <= v["load"]["capacity_kg"].as_f64().unwrap() + 1e-3);
    let sds = rt.sds();
    let n = |track: Track, scope: &str, axis: &str| sds.exact_baseline(&Scope::new(track, scope, ""), axis).map_or(0, |b| b.2);
    assert!(n(Track::Balance, "fx:slip", "utilisation") > 0, "the friction picture of every move");
    assert!(sds.exact_reference(&Scope::new(Track::Balance, "fx:stability", ""), "margin").is_some(), "the margin against its reference");
    assert!(n(Track::Effort, "wear", "shoulder_r:life_used") > 0, "the reducers' wear");
}

#[test]
fn a_drop_command_reports_the_item_and_its_count_apart() {
    let mut rt = runtime(7);
    let parsed = |rt: &mut Runtime, text: &str| rt.api_call("command_submit", &json!({"text": text})).unwrap()["parsed"].clone();
    let p = parsed(&mut rt, "원목 1개 내려놔");
    assert_eq!((p["action"].as_str(), p["item_label_ko"].as_str(), p["count"].as_u64()), (Some("drop"), Some("원목"), Some(1)));
    let p = parsed(&mut rt, "원목 내려놔");
    assert_eq!((p["item_label_ko"].as_str(), p["count"].as_u64()), (Some("원목 전부"), None));
    let p = parsed(&mut rt, "다 내려놔");
    assert_eq!((p["item_label_ko"].as_str(), p["count"].as_u64()), (Some("적재함 전부"), None));
}

#[test]
fn the_inventory_button_and_the_snapshot_speak_the_same_amounts() {
    let mut rt = runtime(7);
    rt.world_mut().agent.carried.push((ItemKind::Stone, 1.5));
    let snap = noto_runtime::api::snapshot_json(&rt.snapshot());
    let c = &snap["agent"]["carrying"][0];
    assert_eq!((c["item_id"].as_str(), c["count"].as_u64(), c["amount"].as_f64()), (Some("stone"), Some(1), Some(1.5)));
    assert!((c["total_kg"].as_f64().unwrap() - 3.9).abs() < 1e-3 && (c["mass_kg"].as_f64().unwrap() - 2.6).abs() < 1e-3);
    let v = rt.api_call("inventory_drop", &json!({"item": "stone", "amount": 1})).unwrap();
    assert_eq!(v["ok"], true);
    rt.run_until_idle(3_000);
    assert!((rt.world().carried_amount(ItemKind::Stone) - 0.5).abs() < 1e-4, "one item set down, the half stays");
    assert!(rt.api_call("inventory_drop", &json!({"item": "log"})).is_err(), "no log aboard");
}

#[test]
fn a_stone_too_heavy_for_budget_arms_is_broken_and_carried_in_parts() {
    // The compact body: scaled up to 1.65 m even budget arms lift a whole stone.
    let cfg = RuntimeConfig { seed: 7, noise: NoiseCfg::none(), body: BodyPlan { arm: Tier::Budget, ..BodyPlan::default() }, ..RuntimeConfig::default() };
    let mut rt = runtime_with(cfg);
    let q = run_command(&mut rt, "돌 1개를 베이스캠프로 가져와", 40_000);
    assert_done(&rt, &q);
    let log = quest_log(&rt);
    assert!(log.contains("쪼갬"), "the stone was broken:\n{log}");
    assert!(log.contains("쪼개서"), "the cargo decision said why:\n{log}");
    let stock = rt.world().waypoint_by_key("base_camp").unwrap().stock_of(ItemKind::Stone);
    assert!((stock - 1.0).abs() < 1e-3, "one whole stone's worth arrived in pieces: {stock}");
    let mut rt = runtime_with(RuntimeConfig { seed: 7, noise: NoiseCfg::none(), body: BodyPlan { arm: Tier::Budget, ..BodyPlan::standard() }, ..RuntimeConfig::default() });
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 30_000);
    assert_eq!(q.state, State::Failed);
    assert_eq!(q.failure.as_ref().unwrap().code, "lift_overload");
}

#[test]
fn the_world_and_the_body_reset_apart() {
    let mut rt = runtime(7);
    let nodes = rt.world().nodes.len();
    let q = run_command(&mut rt, "원목 1개를 창고로 가져와", 30_000);
    assert_done(&rt, &q);
    let energy = rt.world().agent.energy_used_j;
    assert!(rt.world().nodes.len() < nodes && energy > 0.0);
    let v = rt.api_call("world_reset", &Value::Null).unwrap();
    assert_eq!(v["snapshot"]["nodes"].as_array().unwrap().len(), nodes, "the tree grew back");
    assert!(rt.world().waypoint_by_key("warehouse").unwrap().stock.is_empty(), "the pile is gone");
    assert!(rt.quests().is_empty());
    assert_eq!(rt.world().agent.energy_used_j, energy, "the body keeps its wear");
    let b = rt.api_call("body_reset", &Value::Null).unwrap();
    assert_eq!(rt.world().agent.energy_used_j, 0.0);
    let thorax = b["body"]["parts"].as_array().unwrap().iter().find(|p| p["id"] == "thorax").unwrap().clone();
    assert!((thorax["ratio"].as_f64().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn on_the_rugged_map_targets_hidden_behind_terrain_are_found_from_other_viewpoints() {
    let mut found = 0;
    let mut relocated = 0;
    for seed in [1u64, 3, 11] {
        let mut rt = runtime_with(RuntimeConfig { seed, noise: NoiseCfg::none(), terrain: TerrainKind::Rugged, ..RuntimeConfig::default() });
        let q = run_command(&mut rt, "철광석 1개를 창고로 가져와", 60_000);
        let log = quest_log(&rt);
        if log.contains("시점 (") {
            relocated += 1;
        }
        if q.state == State::Done {
            found += 1;
        } else {
            println!("[RUGGED] seed {seed}: {:?}\n{log}", q.failure);
        }
    }
    println!("[RUGGED] 철광석 퀘스트 완료 {found}/3, 다른 시점으로 이동한 실행 {relocated}/3");
    assert!(found >= 2, "{found}/3 rugged runs finished");
    assert!(relocated >= 1, "some run had to look from another viewpoint");
}

#[test]
fn a_typed_command_records_the_three_rows_in_the_live_world() {
    let mut rt = runtime(7);
    let v = rt.api_command_submit_rec("원목 1개를 창고로 가져와", true).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["recording"], true, "{v}");
    let q2 = rt.api_command_submit_rec("돌 1개를 베이스캠프로 가져와", true).unwrap();
    assert_eq!(q2["queued"], 1, "{q2}");
    let mut phases = vec![];
    let mut guard = 0;
    loop {
        let s = rt.api_sim_step(200).unwrap();
        let p = s["test"]["phase"].as_str().unwrap_or("").to_string();
        if phases.last() != Some(&p) {
            phases.push(p.clone());
        }
        if s["test"]["id"].as_str().is_some_and(|id| !id.ends_with("-t1")) {
            break;
        }
        guard += 1;
        assert!(guard < 3000, "the recording does not end: {phases:?}");
    }
    assert!(phases.starts_with(&["game".to_string(), "perceive".to_string(), "decide".to_string()]), "{phases:?}");
    let wh = rt.world().waypoint_by_key("warehouse").unwrap().stock_of(ItemKind::Log);
    assert_eq!(wh, 1.0, "{}", quest_log(&rt));
    let first = rt.quests().iter().find(|q| q.title.contains("원목")).expect("the game quest is in the live book");
    assert_eq!(first.state, State::Done);
    assert!(rt.live_running(), "the queued command is being recorded now");
    let st = rt.api_command_submit_rec("멈춰", true).unwrap();
    assert_eq!(st["stopped"], true);
    assert!(!rt.live_running());
    let status = rt.api_test_status().unwrap();
    assert!(status["active"] == false || status["phase"] != "game", "{status}");
}

#[test]
fn the_old_seeded_integrated_test_still_runs_beside_the_live_one() {
    let mut rt = runtime(7);
    rt.test_start("원목 1개를 창고로 가져와", 7, BodyPlan::standard(), Some(BaseKind::Wheel)).unwrap();
    let mut guard = 0;
    while rt.test_session().map(|t| t.phase) != Some(TestPhase::Done) {
        rt.test_advance(400).unwrap();
        guard += 1;
        assert!(guard < 2000);
    }
    let st = rt.api_test_status().unwrap();
    assert_eq!(st["live"], false);
    assert_eq!(st["game_quest"]["state"], "done");
}
