//! A unit's body changed by a sentence, and a load lost on the way: the two paths through the runtime that the world's own
//! tests cannot walk, because they end in a quest and a saved preset.

use noto_core::{BaseKind, NotoConfig};
use noto_runtime::{Runtime, RuntimeConfig};
use serde_json::{json, Value};
use world_sim::{ItemKind, WorldEvent};

fn runtime(base: BaseKind) -> Runtime {
    let cfg = RuntimeConfig::default();
    Runtime::new(RuntimeConfig { noto: NotoConfig { base, ..cfg.noto }, ..cfg }, include_str!("../../../data/world_lexicon.json"), None).unwrap()
}

fn call(rt: &mut Runtime, cmd: &str, args: Value) -> Value {
    rt.api_call(cmd, &args).unwrap_or_else(|e| panic!("{cmd}: {e}"))
}

#[test]
fn a_sentence_lengthens_the_arms_of_a_unit_where_it_stands() {
    let mut rt = runtime(BaseKind::HumanFoot);
    let at = (rt.world().agent.x, rt.world().agent.z);
    let nodes = rt.world().nodes.len();
    let before = call(&mut rt, "rig_params", json!({"unit": 1}));
    let arm = |v: &Value| v["params"].as_array().unwrap().iter().find(|p| p["param"] == "arm_len_m").unwrap()["value_m"].as_f64().unwrap();
    let plan = call(&mut rt, "rig_prompt", json!({"unit": 1, "text": "팔 길이를 15cm 늘려줘"}));
    assert_eq!(plan["ok"], true, "{plan}");
    assert!(plan["design"]["components"].as_array().unwrap().len() > 8);
    assert_eq!(plan["calls"][0]["tool"], "rig_edit");
    let done = call(&mut rt, "rig_commit", json!({"unit": 1}));
    assert_eq!(done["ok"], true, "{done}");
    assert!(done["sds_key"].as_str().unwrap().starts_with("scraft:"), "{done}");
    assert!(done["checks"].as_array().unwrap().len() >= 4);
    let after = call(&mut rt, "rig_params", json!({"unit": 1}));
    assert!((arm(&after) - arm(&before) - 0.15).abs() < 0.002, "{after}");
    assert_eq!((rt.world().agent.x, rt.world().agent.z), at, "it stays where it stands");
    assert_eq!(rt.world().nodes.len(), nodes, "the world is not made again");
    assert!(rt.cfg.body.id().contains("scraft:"), "its statistics go under their own key: {}", rt.cfg.body.id());
    let u = call(&mut rt, "units_list", json!({}));
    assert_eq!(u["units"][0]["rig_id"], done["preset_id"], "{u}");
}

#[test]
fn a_lift_trucks_tines_change_and_a_sub_worker_keeps_its_own_body() {
    let mut rt = runtime(BaseKind::Wheel);
    let added = call(&mut rt, "crew_add", json!({"count": 1, "form": "forklift"}));
    let id = rt.world().crew[0].id;
    let (x, z) = (rt.world().crew[0].x, rt.world().crew[0].z);
    let plan = call(&mut rt, "rig_prompt", json!({"unit": id, "text": "포크 길이를 1.4m로 늘려줘"}));
    assert_eq!(plan["ok"], true, "{plan} {added}");
    assert_eq!(call(&mut rt, "rig_commit", json!({"unit": id}))["ok"], true);
    let w = &rt.world().crew[0];
    assert!((w.body.dims.tine_len_m - 1.4).abs() < 0.002, "{}", w.body.dims.tine_len_m);
    assert_eq!((w.x, w.z), (x, z));
    assert!((w.footprint().front_m - rt.world().footprint().front_m) > 0.2, "the first worker's tines did not grow with it");
}

#[test]
fn a_foot_changes_in_place_and_a_sentence_it_does_not_understand_changes_nothing() {
    let mut rt = runtime(BaseKind::Wheel);
    let done = call(&mut rt, "unit_apply", json!({"unit": 1, "base": "track"}));
    assert_eq!(done["ok"], true, "{done}");
    assert_eq!(rt.world().hand.base.kind, BaseKind::Track);
    let r = call(&mut rt, "rig_prompt", json!({"unit": 1, "text": "오늘 날씨 어때"}));
    assert_eq!(r["ok"], false);
    assert!(r["error_ko"].as_str().unwrap().contains("알아듣지"));
    assert!(rt.api_call("rig_commit", &json!({"unit": 1})).is_err(), "nothing to commit");
    let r = call(&mut rt, "rig_prompt", json!({"unit": 1, "text": "포크 길이를 30m로"}));
    assert_eq!(r["ok"], false, "{r}");
    assert!(r["error_ko"].as_str().unwrap().contains("범위"), "{r}");
}

#[test]
fn a_robot_left_standing_at_the_place_does_not_keep_the_next_one_from_delivering() {
    // The rugged map of the window's first run: the first worker ends up on the warehouse's flag.
    let cfg = RuntimeConfig::default();
    let mut rt = Runtime::new(RuntimeConfig { noto: NotoConfig { base: BaseKind::Wheel, ..cfg.noto }, seed: 1, terrain: world_sim::TerrainKind::Rugged, ..cfg }, include_str!("../../../data/world_lexicon.json"), None).unwrap();
    call(&mut rt, "crew_add", json!({"count": 1, "form": "forklift"}));
    let id = rt.world().crew[0].id;
    assert_eq!(call(&mut rt, "rig_prompt", json!({"unit": id, "text": "포크 길이를 1.4m로 늘려줘"}))["ok"], true);
    assert_eq!(call(&mut rt, "rig_commit", json!({"unit": id}))["ok"], true);
    assert!(rt.submit("원목 2개를 창고로 가져와").ok);
    let settled = |rt: &Runtime| rt.quests().iter().all(|q| !matches!(q.state, noto_quest::State::Active | noto_quest::State::Pending));
    let mut n = 0;
    while n < 12_000 && !settled(&rt) {
        rt.tick();
        n += 1;
    }
    assert!(n < 12_000, "they all came to an end: {:?}", rt.quests().iter().map(|q| (&q.title, &q.state)).collect::<Vec<_>>());
    assert!(rt.quests().iter().all(|q| q.state == noto_quest::State::Done), "{:?}", rt.quests().iter().map(|q| (&q.title, &q.state, &q.failure)).collect::<Vec<_>>());
    assert_eq!(rt.world().waypoint_by_key("warehouse").unwrap().stock_of(ItemKind::Log), 2.0);
}

#[test]
fn a_load_knocked_off_on_the_way_is_fetched_again_and_the_quest_still_ends() {
    for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
        let mut rt = runtime(base);
        let sub = rt.submit("원목 1개를 창고로 가져와");
        assert!(sub.ok, "{:?}", sub.error);
        let (mut knocked, mut lost) = (false, false);
        for _ in 0..12000 {
            let carrying = rt.world().carried_amount(ItemKind::Log) >= 1.0;
            if carrying && !knocked && rt.world().agent.speed_mps > 0.2 {
                knocked = true;
                // The load already past the end of its support, sliding on: the next tick takes it off.
                let ride = &mut rt.world_mut().ride;
                ride.state.shift = [3.0, 0.0];
                ride.state.vel = [1.0, 0.0];
            }
            let ev = rt.tick();
            lost |= ev.iter().any(|e| matches!(e.event, WorldEvent::CargoLost { .. }));
            if !rt.busy() {
                break;
            }
        }
        assert!(lost, "{base:?}: it got as far as carrying the log");
        let q = rt.quests().last().unwrap();
        assert_eq!(rt.world().waypoint_by_key("warehouse").unwrap().stock_of(ItemKind::Log), 1.0, "{base:?} delivered after going back: {:?} {:?}", q.state, q.failure);
    }
}

#[test]
fn a_walker_reads_a_curb_off_its_depth_and_bends_its_knees_for_it() {
    let mut rt = runtime(BaseKind::HumanFoot);
    let (cx, cz) = rt.world().terrain.cell_of(rt.world().agent.x, rt.world().agent.z).unwrap();
    {
        let w = rt.world_mut();
        w.nodes.clear();
        w.drops.clear();
        for z in 0..w.terrain.d {
            for x in 0..w.terrain.w {
                w.terrain.set_height_cm(x, z, 0);
                w.terrain.set_curb_cm(x, z, if x > cx + 8 { 15 } else { 0 });
            }
        }
        w.drops.push(world_sim::world::DropItem::whole(800, ItemKind::Parcel, w.agent.x + 0.5, w.agent.z));
        w.agent.yaw = 0.0;
    }
    let goal = world_sim::terrain::cell_center(cx + 14, cz);
    rt.world_mut().command(world_sim::Action::pick(800)).unwrap();
    while rt.world().busy() {
        rt.tick();
    }
    assert_eq!(rt.world().carried_count(ItemKind::Parcel), 1);
    let mut taken = None;
    rt.world_mut().command(world_sim::Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).ok();
    for _ in 0..800 {
        rt.step_look_now();
        for e in rt.tick() {
            if let WorldEvent::StepTaken { crouch_m, a_z, .. } = e.event {
                taken = taken.or(Some((crouch_m, a_z)));
            }
        }
        if !rt.world().busy() {
            break;
        }
    }
    let log = rt.log_tail(0, 500, Some("physics")).0;
    let reads: Vec<&str> = log.iter().filter(|l| l.msg.contains("(깊이 대역)")).map(|l| l.msg.as_str()).collect();
    println!("[STEP] {reads:?} {taken:?}");
    assert!(!reads.is_empty(), "it read the step off the depth: {:?}", log.iter().map(|l| &l.msg).collect::<Vec<_>>());
    let (crouch, a_z) = taken.expect("it took the step");
    assert!(crouch > 0.06 && a_z < 12.0, "knees bent {crouch}, jolt {a_z}");
    assert_eq!(rt.world().carried_count(ItemKind::Parcel), 1, "the parcel stayed in its hands");
}
