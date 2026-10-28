use noto_core::parts::{BodyPlan, Tier};
use noto_sds::{Scope, Track};
use noto_runtime::{Runtime, RuntimeConfig};
use serde_json::json;
use world_sim::Camera;

fn runtime() -> Runtime {
    let lexicon = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).unwrap();
    Runtime::new(RuntimeConfig::default(), &lexicon, None).unwrap()
}

#[test]
fn a_preset_from_the_ui_reaches_the_joint_dynamics_and_the_overlay() {
    let mut rt = runtime();
    for (preset, tier) in [("budget", Tier::Budget), ("pro", Tier::Pro), ("mid", Tier::Mid)] {
        rt.api_call("body_plan_set", &json!({"preset": preset})).unwrap();
        assert_eq!(rt.world().body.arm.tier, tier, "{preset}: the world was rebuilt with the plan");
        let pose = rt.world().body_pose().expect("a pose");
        let shoulder = pose.joints.iter().find(|j| j.name == "shoulder_r").unwrap();
        assert_eq!(shoulder.peak_nm, BodyPlan::uniform(tier).arm_preset().shoulder.peak_nm, "{preset}");
        let cam = Camera { width: 160, height: 108, fov_deg: 64.0, max_dist_m: 30.0, ..Camera::default() };
        let (_, overlay) = noto_runtime::chase::render(rt.world(), &cam, -30.0);
        let o = overlay.unwrap();
        assert_eq!(o["plan"], preset);
        let neck = o["joints"].as_array().unwrap().iter().find(|j| j["name"] == "neck").unwrap();
        assert_eq!(neck["peak_nm"].as_f64().unwrap() as f32, BodyPlan::uniform(tier).head_preset().neck.peak_nm, "{preset}");
        println!("[PRESET] api {preset}: shoulder {} N·m, neck {} N·m in the overlay", shoulder.peak_nm, neck["peak_nm"]);
    }
    rt.api_call("world_new", &json!({"seed": 3, "size": 48, "body": {"head": "mid", "chest": "mid", "belly": "pro", "arm": "budget", "leg": "mid"}})).unwrap();
    assert_eq!(rt.world().body.belly.tier, Tier::Pro);
    assert!(rt.world().body_pose().unwrap().joints.iter().any(|j| j.name == "waist_pitch" && j.kind == "sim"), "a pro belly has a driven waist pitch");
}

#[test]
fn a_quest_records_the_joint_dynamics_in_the_sds() {
    let mut rt = runtime();
    rt.api_world_new(7, 64, None, Some("human_foot")).unwrap();
    rt.api_command_submit("원목 1개를 창고로 가져와").unwrap();
    for _ in 0..60 {
        rt.api_sim_step(40).unwrap();
    }
    let n = rt.sds().exact_baseline(&Scope::new(Track::Effort, "dynamics", ""), "min_margin").map_or(0, |b| b.2);
    assert!(n > 0, "dynamics scopes in the SDS");
    let m = rt.world().motion.as_ref().expect("the world simulated a posture change");
    assert!(!m.rollout.samples.is_empty());
}
