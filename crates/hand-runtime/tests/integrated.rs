//! The integrated test end to end: game pass with recording, ML perception on
//! its keyframes, ML decision pass with triangulation, records and SDS.
//!
//! The ML models here are the stand-ins (no weights in the repository) and the
//! region-growing tracker; the test shows the passes fit together and record
//! what they should, not how TIPSv2, SigLIP2 or SAM 3.1 would perform.

use hand_core::{BaseKind, BodyPlan, Tier};
use hand_runtime::{NoiseCfg, Runtime, RuntimeConfig, TestPhase};
use serde_json::Value;

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn runtime() -> Runtime {
    Runtime::new(RuntimeConfig { seed: 7, noise: NoiseCfg::none(), ..RuntimeConfig::default() }, LEXICON, None).expect("runtime")
}

fn run_test(rt: &mut Runtime, cmd: &str, body: BodyPlan) -> Value {
    rt.test_start(cmd, 7, body, Some(BaseKind::Wheel)).expect("test starts");
    let mut guard = 0;
    while rt.test_session().map(|t| t.phase) != Some(TestPhase::Done) {
        rt.test_advance(400).expect("test advances");
        guard += 1;
        assert!(guard < 2000, "the test does not end");
    }
    rt.api_test_status().unwrap()
}

#[test]
fn three_passes_record_keyframes_ml_tracks_decisions_and_success_rates() {
    let mut rt = runtime();
    let st = run_test(&mut rt, "원목 1개를 창고로 가져와", BodyPlan::default());
    assert_eq!(st["game_quest"]["state"], "done", "{}", st["game_quest"]);
    // Row 1: fixed keyframes, about one per second of the game pass.
    let game = st["game"]["keyframes"].as_array().unwrap();
    assert!(game.len() > 10, "{} keyframes", game.len());
    let span = game.last().unwrap()["tick"].as_u64().unwrap() - game[0]["tick"].as_u64().unwrap();
    assert!((span / 10) as usize + 2 >= game.len(), "fixed 10-tick interval");
    assert!(st["game"]["video"].as_u64().unwrap() as usize > game.len(), "the video has more frames than keyframes");
    // Row 2: every game keyframe went through the ML tracks.
    assert!(game.iter().all(|k| !k["ml"].is_null()), "every keyframe analysed");
    let tracked = game.iter().filter(|k| !k["ml"]["target"].is_null()).count();
    assert!(tracked > 0, "the target was tracked on some keyframes");
    // Row 3: the ML pass decided; the first decision on a tracked target is a triangulation step.
    let dec = st["decide"]["keyframes"].as_array().unwrap();
    assert!(!dec.is_empty());
    let kinds: Vec<&str> = dec.iter().filter_map(|k| k["decision"]["kind"].as_str()).collect();
    let first_tracked = kinds.iter().position(|k| *k != "search").expect("a target was found");
    assert_eq!(kinds[first_tracked], "triangulate", "{kinds:?}");
    assert!(kinds.contains(&"approach") || kinds.contains(&"arrive"), "{kinds:?}");
    assert!(dec.iter().any(|k| !k["ml"]["target"]["tri_range"].is_null()), "a triangulated range");
    // A keyframe has its pictures.
    let kf = rt.api_test_keyframe("ml", first_tracked + 1).unwrap();
    assert!(kf["frame_webp"].as_str().unwrap().len() > 100);
    assert!(kf["ml_webp"].as_str().is_some());
    // Success rate per pass and step is in the records.
    let stats = rt.api_test_stats().unwrap();
    let rows = stats["stats"].as_array().unwrap();
    let whole = |pass: &str| rows.iter().find(|r| r["pass"] == pass && r["step"] == "quest").cloned();
    let g = whole("game").expect("game pass recorded");
    assert_eq!((g["attempts"].as_u64(), g["successes"].as_u64()), (Some(1), Some(1)));
    assert!(whole("ml").is_some(), "ML pass recorded: {}", st["ml_quest"]);
    assert!(rows.iter().any(|r| r["step"] == "locate"));
    // The SDS has the ML tracks and the balance N:N cells.
    let sds = rt.sds_status();
    for l in ["ml", "body", "task"] {
        assert!(sds.ledgers.iter().any(|x| x.ledger == l && (x.observations > 0 || x.run_observations > 0)), "{l}: {:?}", sds.ledgers);
    }
    let nn = stats["nn"].as_array().unwrap();
    assert!(nn.iter().any(|c| c["source"] == "arm_lift" && !c["reference"].is_null()), "N:N cells with references");
    // The depth scale was learned from triangulation (stand-in scale is 0.8, so about 1/0.8 = 1.25).
    let scales: Vec<f64> = dec.iter().chain(game.iter()).filter_map(|k| k["ml"]["target"]["scale"].as_f64()).collect();
    assert!(!scales.is_empty());
    let mean = scales.iter().sum::<f64>() / scales.len() as f64;
    assert!(mean > 1.0 && mean < 1.6, "depth scale {mean}: {scales:?}");
}

#[test]
fn a_budget_arm_fails_where_a_mid_arm_succeeds_and_both_are_recorded() {
    let mut rt = runtime();
    let budget = BodyPlan { arm: Tier::Budget, ..BodyPlan::default() };
    let st = run_test(&mut rt, "원목 1개를 창고로 가져와", budget);
    assert_eq!(st["game_quest"]["state"], "failed");
    assert_eq!(st["game_quest"]["failure"]["code"], "lift_overload", "{}", st["game_quest"]);
    let stats = rt.api_test_stats().unwrap();
    let rows = stats["stats"].as_array().unwrap();
    let r = rows.iter().find(|r| r["pass"] == "game" && r["step"] == "quest" && r["preset"] == budget.id()).expect("budget run recorded");
    assert_eq!(r["successes"], 0);
    let collect = rows.iter().find(|r| r["pass"] == "game" && r["step"] == "collect" && r["preset"] == budget.id()).expect("the failing step");
    assert_eq!(collect["successes"], 0);
}
