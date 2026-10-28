use std::sync::Arc;

use anyhow::Result;
use noto_core::{grasp, Actuation, BaseKind, ContactSource, GraspResult, GraspType, Hand, JointAngles, JointKind};
use lancedb::arrow::arrow_array::{ArrayRef, BooleanArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray};

use crate::scene::SceneSpec;
use crate::schema::{contacts_schema, joint_states_schema, samples_schema};

pub const RECORDED_LAYERS: &str = "joint_states,contacts,object_pose,friction,mass_properties";

pub struct SampleRows {
    pub samples: RecordBatch,
    pub contacts: RecordBatch,
    pub joint_states: RecordBatch,
    pub stable: bool,
}

fn kind_name(kind: JointKind) -> &'static str {
    match kind {
        JointKind::Hinge => "hinge",
        JointKind::Universal => "universal",
        JointKind::Ball => "ball",
    }
}

fn base_name(base: BaseKind) -> &'static str {
    match base {
        BaseKind::Wheel => "wheel",
        BaseKind::Mecanum => "mecanum",
        BaseKind::Track => "track",
        BaseKind::HumanFoot => "human_foot",
    }
}

fn actuation_name(actuation: Actuation) -> String {
    match actuation {
        Actuation::Direct => "direct".to_string(),
        Actuation::Coupled { ratio } => format!("coupled_{ratio:.2}"),
    }
}

fn grasp_type_name(t: GraspType) -> &'static str {
    match t {
        GraspType::None => "none",
        GraspType::Pinch => "pinch",
        GraspType::Power => "power",
    }
}

pub fn simulate_scene(scene: &SceneSpec) -> Result<SampleRows> {
    let hand = scene.hand.build()?;
    let outcome = grasp(&hand, &scene.payload, &scene.setup);
    Ok(SampleRows {
        samples: sample_batch(scene, &hand, &outcome.result)?,
        contacts: contact_batch(scene, &outcome.result)?,
        joint_states: joint_batch(scene, &hand, &outcome.angles)?,
        stable: outcome.result.stable,
    })
}

fn sample_batch(scene: &SceneSpec, hand: &Hand, r: &GraspResult) -> Result<RecordBatch> {
    let p = &scene.payload;
    let [w, h, d] = p.size_m;
    let m = p.mass_kg;
    let inertia = [
        m / 12.0 * (h * h + d * d),
        m / 12.0 * (w * w + d * d),
        m / 12.0 * (w * w + h * h),
    ];
    let object_y = h * 0.5;
    let finger_contacts = r
        .contacts
        .iter()
        .filter(|c| matches!(c.source, ContactSource::Finger { .. }))
        .count() as i32;

    let joints = hand.config.joints;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![scene.sample_id.as_str()])),
        Arc::new(StringArray::from(vec![scene.label.as_str()])),
        Arc::new(StringArray::from(vec![scene.material.as_str()])),
        Arc::new(Int64Array::from(vec![scene.seed as i64])),
        Arc::new(Int32Array::from(vec![hand.config.fingers.count() as i32])),
        Arc::new(Int32Array::from(vec![joints.segments as i32])),
        Arc::new(StringArray::from(vec![base_name(hand.config.base)])),
        Arc::new(StringArray::from(vec![format!(
            "{}+{}",
            kind_name(joints.base_kind),
            kind_name(joints.distal_kind)
        )
        .as_str()])),
        Arc::new(StringArray::from(vec![actuation_name(joints.actuation).as_str()])),
        Arc::new(Float32Array::from(vec![w])),
        Arc::new(Float32Array::from(vec![h])),
        Arc::new(Float32Array::from(vec![d])),
        Arc::new(Float32Array::from(vec![m])),
        Arc::new(Float32Array::from(vec![p.friction])),
        Arc::new(Float32Array::from(vec![inertia[0]])),
        Arc::new(Float32Array::from(vec![inertia[1]])),
        Arc::new(Float32Array::from(vec![inertia[2]])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(Float32Array::from(vec![object_y])),
        Arc::new(Float32Array::from(vec![scene.setup.z_offset])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(Float32Array::from(vec![0.0f32])),
        Arc::new(StringArray::from(vec![grasp_type_name(r.grasp_type)])),
        Arc::new(Int32Array::from(vec![finger_contacts])),
        Arc::new(Int32Array::from(vec![r.contacts.len() as i32])),
        Arc::new(BooleanArray::from(vec![r.force_closed])),
        Arc::new(BooleanArray::from(vec![r.stable])),
        Arc::new(Float32Array::from(vec![r.friction_capacity_n])),
        Arc::new(Float32Array::from(vec![r.required_n])),
        Arc::new(Float32Array::from(vec![r.max_payload_kg])),
        Arc::new(StringArray::from(vec![RECORDED_LAYERS])),
    ];
    Ok(RecordBatch::try_new(samples_schema(), columns)?)
}

fn contact_batch(scene: &SceneSpec, r: &GraspResult) -> Result<RecordBatch> {
    let n = r.contacts.len();
    let sample_ids: Vec<&str> = vec![scene.sample_id.as_str(); n];
    let indices: Vec<i32> = (0..n as i32).collect();
    let sources: Vec<i32> = r
        .contacts
        .iter()
        .map(|c| match c.source {
            ContactSource::Finger { index } => index as i32,
            ContactSource::Palm => -1,
        })
        .collect();
    let col = |f: fn(&noto_core::Contact) -> f32| -> Float32Array {
        Float32Array::from(r.contacts.iter().map(f).collect::<Vec<f32>>())
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(sample_ids)),
        Arc::new(Int32Array::from(indices)),
        Arc::new(Int32Array::from(sources)),
        Arc::new(col(|c| c.point.x)),
        Arc::new(col(|c| c.point.y)),
        Arc::new(col(|c| c.point.z)),
        Arc::new(col(|c| c.normal.x)),
        Arc::new(col(|c| c.normal.y)),
        Arc::new(col(|c| c.normal.z)),
        Arc::new(col(|c| c.force_n)),
        Arc::new(Float32Array::from(vec![scene.payload.friction; n])),
    ];
    Ok(RecordBatch::try_new(contacts_schema(), columns)?)
}

fn joint_batch(scene: &SceneSpec, hand: &Hand, angles: &[Vec<JointAngles>]) -> Result<RecordBatch> {
    let mut sample_ids: Vec<&str> = Vec::new();
    let mut finger_index = Vec::new();
    let mut finger_name: Vec<&str> = Vec::new();
    let mut is_thumb = Vec::new();
    let mut joint_index = Vec::new();
    let mut joint_kind: Vec<&str> = Vec::new();
    let mut flex = Vec::new();
    let mut abd = Vec::new();
    let mut roll = Vec::new();
    let mut flex_min = Vec::new();
    let mut flex_max = Vec::new();
    let mut max_torque = Vec::new();

    for (fi, (finger, finger_angles)) in hand.fingers.iter().zip(angles).enumerate() {
        for (ji, (joint, a)) in finger.joints.iter().zip(finger_angles).enumerate() {
            sample_ids.push(scene.sample_id.as_str());
            finger_index.push(fi as i32);
            finger_name.push(finger.name.as_str());
            is_thumb.push(finger.is_thumb);
            joint_index.push(ji as i32);
            joint_kind.push(kind_name(joint.kind));
            flex.push(a.flex);
            abd.push(a.abd);
            roll.push(a.roll);
            flex_min.push(joint.flex_min);
            flex_max.push(joint.flex_max);
            max_torque.push(joint.max_torque);
        }
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(sample_ids)),
        Arc::new(Int32Array::from(finger_index)),
        Arc::new(StringArray::from(finger_name)),
        Arc::new(BooleanArray::from(is_thumb)),
        Arc::new(Int32Array::from(joint_index)),
        Arc::new(StringArray::from(joint_kind)),
        Arc::new(Float32Array::from(flex)),
        Arc::new(Float32Array::from(abd)),
        Arc::new(Float32Array::from(roll)),
        Arc::new(Float32Array::from(flex_min)),
        Arc::new(Float32Array::from(flex_max)),
        Arc::new(Float32Array::from(max_torque)),
    ];
    Ok(RecordBatch::try_new(joint_states_schema(), columns)?)
}
