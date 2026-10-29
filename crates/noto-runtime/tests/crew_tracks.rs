// Sub-workers around the first worker, and a command run over them on each track: 개별 (the count split, a quest
// per worker) and 협업 (humanoids fell and stage at a loading bay, lift trucks take the batch onto their tines and
// haul it), with the first worker as a humanoid and as a lift truck (its own fork procedure on the staged logs, on a
// stereo eye: SGBM).
//   cargo test -p noto-runtime --test crew_tracks -- --nocapture
use noto_core::{BaseKind, NotoConfig};
use noto_quest::{State, StepKind};
use noto_runtime::crew::TrackMode;
use noto_runtime::{NoiseCfg, Runtime, RuntimeConfig};
use world_sim::ItemKind;

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn runtime(seed: u64, base: BaseKind) -> Runtime {
    let cfg = RuntimeConfig { seed, noise: NoiseCfg::none(), noto: NotoConfig { base, ..RuntimeConfig::default().noto }, ..RuntimeConfig::default() };
    Runtime::new(cfg, LEXICON, None).expect("runtime")
}

fn stock(rt: &Runtime, place: &str, item: ItemKind) -> u32 {
    rt.world().waypoint_by_key(place).map(|w| (w.stock_of(item) + 1e-3).floor() as u32).unwrap_or(0)
}

fn crew_log(rt: &Runtime) -> String {
    rt.log_tail(0, 20_000, None).0.iter().filter(|l| matches!(l.cat.as_str(), "quest" | "crew" | "vision")).map(|l| format!("[{}] {}", l.cat, l.msg)).collect::<Vec<_>>().join("\n")
}

fn run(rt: &mut Runtime, text: &str, max_ticks: u64) -> String {
    let s = rt.submit(text);
    assert!(s.ok, "{text}: {:?} {:?}", s.error_code, s.error);
    let id = s.quest_id.clone().expect("a quest");
    let n = rt.run_until_idle(max_ticks);
    assert!(n < max_ticks, "still busy after {max_ticks} ticks\n{}", crew_log(rt));
    id
}

/// Runs on a thread of its own, so that a tick that spins fails the test instead of hanging it.
fn guarded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(1200)).expect("the crew spins inside one tick or never ends")
}

#[test]
fn sub_workers_are_added_around_the_first_worker() {
    let mut rt = runtime(7, BaseKind::HumanFoot);
    let ids = rt.crew_add(3, "mixed", None).unwrap();
    assert_eq!(ids, vec![2, 3, 4]);
    let w = rt.world();
    let forms: Vec<_> = w.crew.iter().map(|c| c.form).collect();
    assert_eq!(forms, vec![world_sim::Form::Forklift, world_sim::Form::Humanoid, world_sim::Form::Humanoid], "mixed: one lift truck, the rest humanoids");
    for c in &w.crew {
        let d = (c.x - w.agent.x).hypot(c.z - w.agent.z);
        assert!(d < 4.5, "worker {} is {d:.1} m from the first worker", c.id);
    }
    let v = rt.crew_json();
    assert_eq!(v["workers"].as_array().unwrap().len(), 4, "the first worker and three sub-workers");
    assert_eq!(v["mode"], "solo");
    let snap = noto_runtime::api::snapshot_json(&rt.snapshot());
    assert_eq!(snap["crew"].as_array().unwrap().len(), 3);
    assert!(rt.crew_add(9, "humanoid", None).is_ok(), "fills up to the cap");
    assert_eq!(rt.world().crew.len(), world_sim::crew::CREW_MAX);
    assert!(rt.crew_add(1, "humanoid", None).is_err(), "no room left");
    assert_eq!(rt.crew_clear(), world_sim::crew::CREW_MAX);
    assert!(rt.world().crew.is_empty());
}

#[test]
fn the_solo_track_splits_the_count_and_every_worker_delivers_its_share() {
    let (log, stocked, quests) = guarded(|| {
        let mut rt = runtime(7, BaseKind::HumanFoot);
        rt.crew_add(2, "humanoid", None).unwrap();
        rt.set_track(TrackMode::Solo);
        run(&mut rt, "원목 3개를 창고로 가져와", 40_000);
        let quests: Vec<(u32, State, String)> = rt.quest_book().all().iter().map(|q| (q.owner, q.state, q.title.clone())).collect();
        (crew_log(&rt), stock(&rt, "warehouse", ItemKind::Log), quests)
    });
    assert_eq!(quests.len(), 3, "one quest per worker: {quests:?}");
    let owners: Vec<u32> = quests.iter().map(|q| q.0).collect();
    assert_eq!(owners, vec![1, 2, 3]);
    assert!(quests.iter().all(|q| q.1 == State::Done && q.2.starts_with("[개별 · 작업자")), "{quests:#?}\n{log}");
    assert_eq!(stocked, 3, "{log}");
    assert!(log.contains("개별 트랙: 원목 3개를 3명이 나눔"), "{log}");
}

#[test]
fn the_team_track_with_a_humanoid_first_worker_stages_and_a_crew_lift_truck_hauls() {
    let (log, stocked, team, steps, first) = guarded(|| {
        let mut rt = runtime(7, BaseKind::HumanFoot);
        rt.crew_add(2, "mixed", None).unwrap();
        rt.set_track(TrackMode::Team);
        let id = run(&mut rt, "원목 4개를 창고로 가져와", 60_000);
        let q = rt.quest_book().get(&id).unwrap().clone();
        let first: Vec<(State, String)> = rt.quest_book().all().iter().filter(|q| q.owner == 1).map(|q| (q.state, q.title.clone())).collect();
        let steps: Vec<(StepKind, State)> = q.steps.iter().map(|s| (s.kind, s.state)).collect();
        let staged = rt.crew_json()["team"]["counts"]["staged"].as_f64().unwrap_or(f64::NAN);
        (crew_log(&rt), stock(&rt, "warehouse", ItemKind::Log), (q.state, q.owner, q.title.clone(), q.result.map(|r| r.summary), staged), steps, first)
    });
    assert_eq!(team.0, State::Done, "{team:?}\n{log}");
    assert_eq!(team.1, 0, "the team quest belongs to the team");
    assert!(team.2.starts_with("[협업]"), "{}", team.2);
    assert_eq!(steps.len(), 7, "① 편성 … ⑦ 인도");
    assert_eq!(steps.iter().map(|s| s.0).collect::<Vec<_>>(), vec![StepKind::Assign, StepKind::Goto, StepKind::Mine, StepKind::Stage, StepKind::Load, StepKind::Carry, StepKind::Deliver]);
    assert!(steps.iter().all(|s| s.1 == State::Done));
    assert!(stocked >= 4, "{stocked} logs in the warehouse\n{log}");
    assert!(!first.is_empty() && first.iter().all(|(s, t)| *s == State::Done && t.starts_with("[협업 · 작업자 1 · 채취·적치]")), "the first worker harvested with its own pipeline: {first:?}\n{log}");
    for stage in ["① 편성", "② 상차장", "③ 채취", "④ 적치", "⑤ 상차", "⑥ 운반", "⑦ 인도"] {
        assert!(log.contains(stage), "the flow passes {stage}\n{log}");
    }
    assert!(team.3.as_deref().is_some_and(|s| s.contains("협업 완료")), "{:?}", team.3);
    // Harvesters stop picking up once the count is made sure of: at most what each still held then on top.
    assert!(team.4 <= 4.0 + 2.0, "staged {} for a count of 4 with two humanoids
{log}", team.4);
}

#[test]
fn the_team_track_with_a_lift_truck_first_worker_takes_the_staged_logs_with_its_fork_procedure_on_stereo() {
    let (log, stocked, team, first, depth) = guarded(|| {
        let mut rt = runtime(7, BaseKind::Wheel);
        rt.set_cameras(2).unwrap();
        rt.crew_add(2, "humanoid", None).unwrap();
        rt.set_track(TrackMode::Team);
        let id = run(&mut rt, "원목 3개를 창고로 가져와", 60_000);
        let q = rt.quest_book().get(&id).unwrap().clone();
        let first: Vec<(State, String)> = rt.quest_book().all().iter().filter(|q| q.owner == 1).map(|q| (q.state, q.title.clone())).collect();
        (crew_log(&rt), stock(&rt, "warehouse", ItemKind::Log), (q.state, q.failure), first, rt.ml_pipeline().depth_mode())
    });
    assert_eq!(depth, noto_vision::stereo::MODE, "two cameras: SGBM in the depth slot, no TIPSv2");
    assert_eq!(team.0, State::Done, "{:?}\n{log}", team.1);
    assert!(stocked >= 3, "{stocked}\n{log}");
    assert!(!first.is_empty() && first.iter().all(|(s, t)| *s == State::Done && t.starts_with("[협업 · 작업자 1 · 상차·운반]")), "{first:?}\n{log}");
    assert!(log.contains("상차장 적치분"), "the lift truck's task starts at the staged batch\n{log}");
    assert!(log.contains("SGBM 스테레오"), "its looks range with the stereo pair\n{log}");
    assert!(log.contains("교차 검증"), "rider and carriage camera are checked against each other\n{log}");
}

#[test]
fn a_team_without_a_lift_truck_runs_the_solo_track_and_says_so() {
    let (log, stocked) = guarded(|| {
        let mut rt = runtime(7, BaseKind::HumanFoot);
        rt.crew_add(1, "humanoid", None).unwrap();
        rt.set_track(TrackMode::Team);
        run(&mut rt, "원목 2개를 창고로 가져와", 40_000);
        (crew_log(&rt), stock(&rt, "warehouse", ItemKind::Log))
    });
    assert!(log.contains("개별 트랙으로 진행"), "{log}");
    assert_eq!(stocked, 2, "{log}");
}
