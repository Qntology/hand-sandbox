//! The integrated test's records go to Lance and come back with the same counts.

use noto_data::test_tables::{write_mirror, TABLE_BALANCE_EFFORT, TABLE_ML_KEYFRAMES, TABLE_QUEST_RUNS, TABLE_QUEST_STATS, TABLE_QUEST_STEPS};
use noto_data::LanceStore;
use noto_sds::{BalanceRow, KeyframeRow, QuestRun, StepRecord, TestDb};

fn run(pass: &str, ok: bool) -> QuestRun {
    QuestRun {
        run_id: format!("t1-{pass}"),
        session: "s".into(),
        pass: pass.into(),
        quest_id: "q1".into(),
        kind: "fetch".into(),
        target: "log".into(),
        destination: Some("warehouse".into()),
        preset: "mid".into(),
        seed: 7,
        ok,
        failure: (!ok).then(|| "lift_overload".into()),
        ticks: 300,
        time_s: 30.0,
        energy_j: 4000.0,
        steps: vec![
            StepRecord { step: "locate".into(), attempts: 2, ok: true, fail_code: None, time_s: 0.0 },
            StepRecord { step: "collect".into(), attempts: 1, ok, fail_code: (!ok).then(|| "lift_overload".into()), time_s: 0.0 },
        ],
        created_ms: 1,
    }
}

#[tokio::test]
async fn test_records_round_trip_through_lance() {
    let dir = std::env::temp_dir().join(format!("noto-data-testtables-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = LanceStore::open(dir.to_str().unwrap()).await.unwrap();

    let mut db = TestDb::memory();
    db.record_run(run("game", true));
    db.record_run(run("ml", false));
    db.record_keyframe(KeyframeRow {
        run_id: "t1-ml".into(),
        pass: "ml".into(),
        index: 0,
        tick: 0,
        time_s: 0.0,
        target: "tree".into(),
        depth_mode: "stand-in".into(),
        depth_absrel: Some(0.2),
        depth_delta1: None,
        locate_mode: "stand-in".into(),
        locate_score: Some(0.8),
        locate_err_px: None,
        track_mode: "region-grow".into(),
        track_id: Some(3),
        track_iou_prev: None,
        track_iou_truth: Some(0.5),
        tri_range_m: Some(6.1),
        tri_parallax_deg: Some(8.0),
        depth_scale: Some(1.25),
        range_used_m: Some(6.1),
        range_true_m: Some(6.0),
        decision: "tri_step(angled, 0.9 m)".into(),
        ts_method: "last".into(),
        ts_residual: None,
        fusion_flags: String::new(),
        balance_margin: Some(0.7),
        effort_max: Some(0.3),
    });
    db.record_balance(BalanceRow {
        run_id: "t1-game".into(),
        pass: "game".into(),
        tick: 10,
        source: "arm_lift".into(),
        source_seg: "arm".into(),
        target: "foot".into(),
        state: "flat|pick|com_shift_frac".into(),
        preset: "mid".into(),
        comp: 0.3,
        comp_unit: String::new(),
        reference: None,
        tolerance: None,
        torque_nm: 12.0,
        effort: 0.4,
        effort_noff: 0.5,
        margin: 0.6,
        margin_noff: 0.5,
        ff_gain: 1.0,
    });
    let batch = db.take_mirror();
    let c = write_mirror(&store, &batch, &db.stats()).await.unwrap();
    assert_eq!((c.quest_runs, c.quest_steps, c.ml_keyframes, c.balance_effort), (2, 4, 1, 1));
    assert_eq!(store.row_count(TABLE_QUEST_RUNS).await.unwrap(), 2);
    assert_eq!(store.row_count(TABLE_QUEST_STEPS).await.unwrap(), 4);
    assert_eq!(store.row_count(TABLE_ML_KEYFRAMES).await.unwrap(), 1);
    assert_eq!(store.row_count(TABLE_BALANCE_EFFORT).await.unwrap(), 1);
    let stats = store.row_count(TABLE_QUEST_STATS).await.unwrap();
    assert_eq!(stats, db.stats().len());
    // The success table is replaced, not appended to.
    write_mirror(&store, &db.take_mirror(), &db.stats()).await.unwrap();
    assert_eq!(store.row_count(TABLE_QUEST_STATS).await.unwrap(), stats);
    let _ = std::fs::remove_dir_all(&dir);
}
