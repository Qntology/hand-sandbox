use noto_core::{BaseKind, BodyPlan, Tier};
use noto_runtime::{NoiseCfg, Runtime, RuntimeConfig, TestPhase};
use noto_sds::{Scope, Track};
use serde_json::Value;

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn runtime() -> Runtime {
    Runtime::new(RuntimeConfig { seed: 7, noise: NoiseCfg::none(), ..RuntimeConfig::default() }, LEXICON, None).expect("runtime")
}

fn run_test(rt: &mut Runtime, cmd: &str, body: BodyPlan, base: BaseKind) -> Value {
    rt.test_start(cmd, 7, body, Some(base)).expect("test starts");
    let mut guard = 0;
    while rt.test_session().map(|t| t.phase) != Some(TestPhase::Done) {
        rt.test_advance(400).expect("test advances");
        guard += 1;
        assert!(guard < 2000, "the test does not end");
    }
    rt.api_test_status().unwrap()
}

fn moments(rows: &[Value]) -> Vec<&str> {
    rows.iter().filter_map(|k| k["event"]["kind"].as_str()).collect()
}

#[test]
fn three_passes_record_keyframes_ml_tracks_decisions_and_success_rates() {
    let mut rt = runtime();
    let st = run_test(&mut rt, "원목 1개를 창고로 가져와", BodyPlan::standard(), BaseKind::Wheel);
    assert_eq!(st["game_quest"]["state"], "done", "{}", st["game_quest"]);
    let game = st["game"]["keyframes"].as_array().unwrap();
    let fixed: Vec<&Value> = game.iter().filter(|k| k["event"].is_null()).collect();
    assert!(fixed.len() > 10, "{} keyframes", fixed.len());
    let span = fixed.last().unwrap()["tick"].as_u64().unwrap() - fixed[0]["tick"].as_u64().unwrap();
    assert!((span / 10) as usize + 2 >= fixed.len(), "fixed 10-tick interval");
    assert!(st["game"]["video"].as_u64().unwrap() as usize > fixed.len(), "the video has more frames than keyframes");
    let m = moments(game);
    assert!(m.contains(&"contact") && m.contains(&"arrive"), "the push's moments are keyframes: {m:?}");
    assert!(game.iter().all(|k| !k["ml"].is_null()), "every keyframe analysed");
    let tracked = game.iter().filter(|k| !k["ml"]["target"].is_null()).count();
    assert!(tracked > 0, "the target was tracked on some keyframes");
    let dec = st["decide"]["keyframes"].as_array().unwrap();
    assert!(!dec.is_empty());
    for pass in ["game", "decide"] {
        let m = &st[pass]["memory"];
        let n = st[pass]["keyframes"].as_array().map_or(0, Vec::len);
        println!("[MEM] {pass}: {n} keyframes · truth pixels {} B (kept, they were {} B) · images {} B · video {} B", m["truth_pixels"], n * 192 * 128 * 8, m["images"], m["video"]);
        assert_eq!(m["truth_pixels"], 0, "{pass}: an analysed keyframe keeps no truth pixels: {m}");
    }
    let kinds: Vec<&str> = dec.iter().filter_map(|k| k["decision"]["kind"].as_str()).collect();
    let first_tracked = kinds.iter().position(|k| *k != "search").expect("a target was found");
    assert_eq!(kinds[first_tracked], "triangulate", "{kinds:?}");
    assert!(kinds.contains(&"approach") || kinds.contains(&"arrive"), "{kinds:?}");
    assert!(dec.iter().any(|k| !k["ml"]["target"]["tri_range"].is_null()), "a triangulated range");
    let kf = rt.api_test_keyframe("ml", first_tracked + 1).unwrap();
    assert!(kf["frame_webp"].as_str().unwrap().len() > 100);
    assert!(kf["ml_webp"].as_str().is_some());
    let stats = rt.api_test_stats().unwrap();
    let rows = stats["stats"].as_array().unwrap();
    let whole = |pass: &str| rows.iter().find(|r| r["pass"] == pass && r["step"] == "quest").cloned();
    let g = whole("game").expect("game pass recorded");
    assert_eq!((g["attempts"].as_u64(), g["successes"].as_u64()), (Some(1), Some(1)));
    assert!(whole("ml").is_some(), "ML pass recorded: {}", st["ml_quest"]);
    assert!(rows.iter().any(|r| r["step"] == "locate"));
    let sds = rt.sds_status();
    for l in ["ml", "body", "task"] {
        assert!(sds.ledgers.iter().any(|x| x.ledger == l && (x.observations > 0 || x.run_observations > 0)), "{l}: {:?}", sds.ledgers);
    }
    let nn = stats["nn"].as_array().unwrap();
    assert!(nn.iter().any(|c| !c["reference"].is_null()), "N:N cells with references");
    let live = rt.api_call("nn_status", &Value::Null).unwrap();
    assert!(!live["phases"].as_array().unwrap().is_empty(), "composite phases scored during the test");
    for s in live["scenarios"].as_array().unwrap() {
        println!("[NN-SUM] integrated · {} {} 점수 {} · 사건 {} · 게이트 {}", s["scenario"], s["name_ko"], s["score"], s["events"], s["gated"]);
    }
    assert!(sds.ledgers.iter().any(|x| x.ledger == "body"), "dynamics scopes live in the body ledger");
    // The target is a tree: its visible centre wanders by a share of its width between looks, so a short baseline
    // cannot fix its range to the 15 % a depth scale needs, and whether this run makes a long enough one depends on
    // the path. Triangulation runs; what scale it does set agrees with the truth.
    let keyframes: Vec<&Value> = dec.iter().chain(game.iter()).collect();
    assert!(keyframes.iter().any(|k| !k["ml"]["target"]["tri_range"].is_null()), "triangulated ranges");
    for k in keyframes {
        let t = &k["ml"]["target"];
        if let (Some(s), Some(d), Some(r)) = (t["scale"].as_f64(), t["depth_range"].as_f64(), t["true_range"].as_f64()) {
            assert!((s * d / r - 1.0).abs() < 0.25, "depth scale {s:.3} where the truth is {:.3}: {t}", r / d);
        }
    }
}

#[test]
fn a_legged_pick_is_recorded_moment_by_moment_with_its_focus_views_and_the_act_sds() {
    let mut rt = runtime();
    let st = run_test(&mut rt, "원목 1개를 창고로 가져와", BodyPlan::standard(), BaseKind::HumanFoot);
    assert_eq!(st["game_quest"]["state"], "done", "{}", st["game_quest"]);
    let game = st["game"]["keyframes"].as_array().unwrap();
    let m = moments(game);
    assert!(m.contains(&"bend_bottom") && m.contains(&"lift_off") && m.contains(&"stowed"), "the grasp's moments are keyframes: {m:?}");
    assert!(game.iter().all(|k| !k["focus"].is_null()), "focus digest on every keyframe");
    let at = game.iter().position(|k| k["event"]["kind"] == "lift_off").unwrap();
    let kf = rt.api_test_keyframe("game", at).unwrap();
    assert!(kf["focus"]["hand"]["grip"].is_object(), "the lift-off shows the grip: {}", kf["focus"]["hand"]);
    assert_eq!(kf["focus"]["foot"].as_array().map(|f| f.len()), Some(2), "both soles");
    assert!(kf["focus"]["parts"].as_array().is_some_and(|p| p.iter().any(|x| x == "hand")), "the lift-off is about the hand");
    for cam in ["hand", "side", "low"] {
        assert!(kf["focus_webp"][cam].as_str().is_some_and(|s| s.len() > 100), "{cam} close-up");
    }
    let sds = rt.sds();
    let ts = Scope::new(Track::Act, "grasp:ts", "log");
    assert!(sds.exact_reference(&ts, "preload_s").is_some(), "the lift's phases against the human lift");
    assert!(sds.exact_baseline(&ts, "lift_s").is_some(), "the rollout's own numbers as baselines");
    let feet = Scope::new(Track::Act, "nn:grasp", "foot");
    assert!(sds.exact_reference(&feet, "margin").is_some() || sds.exact_baseline(&feet, "margin").is_some(), "the feet while the hands lift");
    assert!(rt.sds_status().ledgers.iter().any(|l| l.ledger == "act" && (l.observations > 0 || l.run_observations > 0)), "the act ledger");
    let all = sds.status(100_000);
    assert!(all.axes.iter().any(|a| a.scope.starts_with("act|walk|") && a.scope.ends_with(":humanfoot:f5") && a.axis == "ok"), "the walks by form");
    assert!(all.axes.iter().any(|a| a.scope.starts_with("act|walk:ts|") && a.axis == "step_m"), "the gait on each ground");
    assert!(all.axes.iter().any(|a| a.scope.starts_with("act|walk|") && a.axis == "no_fall" && a.n > 0 && a.mean > 0.0), "walks that ended standing");
    let stats = rt.api_test_stats().unwrap();
    assert!(stats["nn"].as_array().unwrap().iter().any(|c| c["source"] == "arm_lift" && !c["reference"].is_null()), "the arm lift against its reference");
    let dec = st["decide"]["keyframes"].as_array().unwrap();
    let scales: Vec<String> = dec.iter().chain(game.iter()).filter_map(|k| k["ml"]["target"]["scale"].as_f64().map(|s| format!("{s:.2}@{}", k["event"]["kind"].as_str().unwrap_or("fixed")))).collect();
    println!("[INTEGRATED] legs: depth scales {scales:?}");
}

#[test]
fn a_budget_arm_fails_where_a_mid_arm_succeeds_and_both_are_recorded() {
    let mut rt = runtime();
    let budget = BodyPlan { arm: Tier::Budget, ..BodyPlan::standard() };
    let st = run_test(&mut rt, "원목 1개를 창고로 가져와", budget, BaseKind::HumanFoot);
    assert_eq!(st["game_quest"]["state"], "failed");
    assert_eq!(st["game_quest"]["failure"]["code"], "lift_overload", "{}", st["game_quest"]);
    let stats = rt.api_test_stats().unwrap();
    let rows = stats["stats"].as_array().unwrap();
    let r = rows.iter().find(|r| r["pass"] == "game" && r["step"] == "quest" && r["preset"] == budget.id()).expect("budget run recorded");
    assert_eq!(r["successes"], 0);
    let collect = rows.iter().find(|r| r["pass"] == "game" && r["step"] == "collect" && r["preset"] == budget.id()).expect("the failing step");
    assert_eq!(collect["successes"], 0);
}
