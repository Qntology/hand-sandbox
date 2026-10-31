//! Changing a robot without a CAD window. A design is the body's dimensions drawn out (`design_of`), so an edit is a
//! change of those dimensions: longer arms, a longer tine, a taller mast. The edited design goes out as the same URDF
//! and meshes a CAD export would give and comes back through the same measuring (`RigSpec::from_urdf`), so a robot
//! edited here and one drawn in an editor are the same kind of thing.

use std::collections::HashMap;

use serde::Serialize;

use crate::balance::Body;

use super::{add, design_of, cross, norm, scale, sub, DesignSolid, RigDesign, RigForm, V3};

/// What an edit can change. The ranges keep a body the moves still work for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Param {
    ArmLen,
    LegLen,
    TorsoLen,
    ShoulderWidth,
    TineLen,
    TineSpread,
    MastTop,
    Wheelbase,
    TruckWidth,
    WheelRadius,
}

impl Param {
    pub const ALL: [Param; 10] = [Param::ArmLen, Param::LegLen, Param::TorsoLen, Param::ShoulderWidth, Param::TineLen, Param::TineSpread, Param::MastTop, Param::Wheelbase, Param::TruckWidth, Param::WheelRadius];

    pub fn id(self) -> &'static str {
        match self {
            Param::ArmLen => "arm_len_m",
            Param::LegLen => "leg_len_m",
            Param::TorsoLen => "torso_len_m",
            Param::ShoulderWidth => "shoulder_width_m",
            Param::TineLen => "tine_len_m",
            Param::TineSpread => "tine_spread_m",
            Param::MastTop => "mast_top_m",
            Param::Wheelbase => "wheelbase_m",
            Param::TruckWidth => "truck_width_m",
            Param::WheelRadius => "wheel_radius_m",
        }
    }

    pub fn from_id(id: &str) -> Option<Param> {
        Param::ALL.into_iter().find(|p| p.id() == id)
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Param::ArmLen => "팔 길이",
            Param::LegLen => "다리 길이",
            Param::TorsoLen => "몸통 길이",
            Param::ShoulderWidth => "어깨 폭",
            Param::TineLen => "포크 길이",
            Param::TineSpread => "포크 간격",
            Param::MastTop => "마스트 높이",
            Param::Wheelbase => "축거",
            Param::TruckWidth => "차체 폭",
            Param::WheelRadius => "바퀴 반지름",
        }
    }

    pub fn form(self) -> RigForm {
        match self {
            Param::ArmLen | Param::LegLen | Param::TorsoLen | Param::ShoulderWidth => RigForm::Humanoid,
            _ => RigForm::Forklift,
        }
    }

    /// The values this can take (m).
    pub fn range(self) -> [f32; 2] {
        match self {
            Param::ArmLen => [0.30, 1.20],
            Param::LegLen => [0.40, 1.20],
            Param::TorsoLen => [0.25, 0.90],
            Param::ShoulderWidth => [0.25, 0.70],
            Param::TineLen => [0.60, 2.00],
            Param::TineSpread => [0.12, 0.90],
            Param::MastTop => [1.40, 4.50],
            Param::Wheelbase => [0.70, 2.20],
            Param::TruckWidth => [0.70, 1.60],
            Param::WheelRadius => [0.12, 0.40],
        }
    }

    pub fn value(self, b: &Body) -> f32 {
        let d = &b.dims;
        match self {
            Param::ArmLen => d.upper_arm_m + d.forearm_m,
            Param::LegLen => d.hip_m - d.ankle_m,
            Param::TorsoLen => d.shoulder_m - d.hip_m,
            Param::ShoulderWidth => 2.0 * d.shoulder_half_m,
            Param::TineLen => d.tine_len_m,
            Param::TineSpread => d.tine_spread_m,
            Param::MastTop => d.mast_top_m,
            Param::Wheelbase => b.support_len_m,
            Param::TruckWidth => b.support_width_m,
            Param::WheelRadius => d.wheel_r_m,
        }
    }

    fn set(self, b: &mut Body, v: f32) {
        let d = &mut b.dims;
        match self {
            Param::ArmLen => {
                let k = v / (d.upper_arm_m + d.forearm_m).max(1e-3);
                d.upper_arm_m *= k;
                d.forearm_m *= k;
            }
            Param::LegLen => {
                let up = v - (d.hip_m - d.ankle_m);
                d.hip_m += up;
                d.waist_m += up;
                d.shoulder_m += up;
            }
            Param::TorsoLen => {
                let up = v - (d.shoulder_m - d.hip_m);
                d.waist_m += 0.4 * up;
                d.shoulder_m += up;
            }
            Param::ShoulderWidth => d.shoulder_half_m = 0.5 * v,
            Param::TineLen => d.tine_len_m = v,
            Param::TineSpread => d.tine_spread_m = v,
            Param::MastTop => d.mast_top_m = v,
            Param::Wheelbase => {
                d.wheelbase_m = Some(v);
                b.support_len_m = v;
            }
            Param::TruckWidth => {
                d.chassis_w_m = v;
                b.support_width_m = v;
            }
            Param::WheelRadius => d.wheel_r_m = v,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RigEdit {
    pub param: Param,
    /// The value it takes (m).
    pub value_m: f32,
}

/// The parameters of a body of its form, with what they are now.
pub fn params_of(body: &Body) -> Vec<(Param, f32)> {
    let form = if body.legged() { RigForm::Humanoid } else { RigForm::Forklift };
    Param::ALL.into_iter().filter(|p| p.form() == form).map(|p| (p, p.value(body))).collect()
}

/// The design of `body` with `edits` made.
pub fn edited_design(body: &Body, edits: &[RigEdit]) -> Result<RigDesign, String> {
    let form = if body.legged() { RigForm::Humanoid } else { RigForm::Forklift };
    let mut b = body.clone();
    for e in edits {
        if e.param.form() != form {
            return Err(format!("{}은(는) {} 몸에 없는 치수입니다", e.param.label_ko(), form.label_ko()));
        }
        let [lo, hi] = e.param.range();
        if !(lo..=hi).contains(&e.value_m) {
            return Err(format!("{} {:.2} m는 범위({lo:.2} ~ {hi:.2} m) 밖입니다", e.param.label_ko(), e.value_m));
        }
        e.param.set(&mut b, e.value_m);
    }
    Ok(design_of(&b))
}

fn solid_tris(s: &DesignSolid) -> Vec<[V3; 3]> {
    match *s {
        DesignSolid::Box { center, size } => {
            let (lo, hi) = (sub(center, scale(size, 0.5)), add(center, scale(size, 0.5)));
            let c = |x: usize, y: usize, z: usize| [if x == 0 { lo[0] } else { hi[0] }, if y == 0 { lo[1] } else { hi[1] }, if z == 0 { lo[2] } else { hi[2] }];
            let quads = [[c(0, 0, 0), c(0, 1, 0), c(1, 1, 0), c(1, 0, 0)], [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)], [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)], [c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), c(1, 1, 0)], [c(0, 0, 0), c(0, 0, 1), c(0, 1, 1), c(0, 1, 0)], [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)]];
            quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect()
        }
        DesignSolid::Cylinder { base, axis, height, radius } => {
            const SIDES: usize = 18;
            let up = scale(axis, 1.0 / norm(axis).max(1e-6));
            let other = if up[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let u = cross(up, other);
            let u = scale(u, 1.0 / norm(u).max(1e-6));
            let v = cross(up, u);
            let top = add(base, scale(up, height));
            let rim = |c: V3, k: usize| {
                let a = k as f32 * std::f32::consts::TAU / SIDES as f32;
                add(add(c, scale(u, radius * a.cos())), scale(v, radius * a.sin()))
            };
            (0..SIDES).flat_map(|k| [[base, rim(base, k + 1), rim(base, k)], [top, rim(top, k), rim(top, k + 1)], [rim(base, k), rim(base, k + 1), rim(top, k + 1)], [rim(base, k), rim(top, k + 1), rim(top, k)]]).collect()
        }
    }
}

fn binary_stl(tris: &[[V3; 3]]) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out.extend((tris.len() as u32).to_le_bytes());
    for t in tris {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let n = scale(n, 1.0 / norm(n).max(1e-9));
        for v in std::iter::once(n).chain(t.iter().copied()) {
            for c in v {
                out.extend(c.to_le_bytes());
            }
        }
        out.extend([0u8; 2]);
    }
    out
}

fn solid_volume(s: &DesignSolid) -> f32 {
    match *s {
        DesignSolid::Box { size, .. } => size[0] * size[1] * size[2],
        DesignSolid::Cylinder { height, radius, .. } => std::f32::consts::PI * radius * radius * height,
    }
}

fn solid_centre(s: &DesignSolid) -> V3 {
    match *s {
        DesignSolid::Box { center, .. } => center,
        DesignSolid::Cylinder { base, axis, height, .. } => add(base, scale(axis, 0.5 * height / norm(axis).max(1e-6))),
    }
}

/// A design as a CAD package would export it: a URDF, with each link's frame at the joint that carries it, and an STL
/// of each link in that frame.
pub fn export(design: &RigDesign) -> (String, HashMap<String, Vec<u8>>) {
    let mut at: HashMap<&str, V3> = HashMap::new();
    for j in &design.joints {
        at.insert(j.b.as_str(), j.at);
    }
    let frame = |name: &str| at.get(name).copied().unwrap_or([0.0; 3]);
    let mut files = HashMap::new();
    let mut xml = format!("<robot name=\"{}\"><link name=\"base_link\"/>", design.name);
    for c in &design.components {
        let o = frame(&c.name);
        let tris: Vec<[V3; 3]> = c.solids.iter().flat_map(solid_tris).map(|t| t.map(|p| sub(p, o))).collect();
        let file = format!("{}.stl", c.name);
        files.insert(file.clone(), binary_stl(&tris));
        let vol: f32 = c.solids.iter().map(solid_volume).sum::<f32>().max(1e-9);
        let com = sub(c.solids.iter().fold([0.0; 3], |a, s| add(a, scale(solid_centre(s), solid_volume(s) / vol))), o);
        let [r, g, b] = c.color.map(|v| v as f32 / 255.0);
        xml += &format!(
            "<link name=\"{}\"><inertial><origin xyz=\"{} {} {}\"/><mass value=\"{}\"/></inertial><visual><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry><mesh filename=\"{file}\"/></geometry><material name=\"{}\"><color rgba=\"{r} {g} {b} 1\"/></material></visual></link>",
            c.name, com[0], com[1], com[2], c.mass_kg, c.name
        );
    }
    let root = &design.components[0].name;
    xml += &format!("<joint name=\"{root}_fixed\" type=\"fixed\"><parent link=\"base_link\"/><child link=\"{root}\"/></joint>");
    for j in &design.joints {
        let kind = match (j.kind, j.limits) {
            ("rigid", _) => "fixed",
            ("slider", _) => "prismatic",
            (_, None) => "continuous",
            _ => "revolute",
        };
        let o = sub(j.at, frame(&j.a));
        let limit = j.limits.map_or(String::new(), |[lo, hi]| {
            let (lo, hi) = if j.kind == "slider" { (lo, hi) } else { (lo.to_radians(), hi.to_radians()) };
            format!("<limit lower=\"{lo}\" upper=\"{hi}\" effort=\"1000\" velocity=\"10\"/>")
        });
        xml += &format!("<joint name=\"{}\" type=\"{kind}\"><parent link=\"{}\"/><child link=\"{}\"/><origin xyz=\"{} {} {}\" rpy=\"0 0 0\"/><axis xyz=\"{} {} {}\"/>{limit}</joint>", j.name, j.a, j.b, o[0], o[1], o[2], j.axis[0], j.axis[1], j.axis[2]);
    }
    xml += "</robot>";
    (xml, files)
}

#[cfg(test)]
mod tests {
    use super::super::{register, urdf, RigSpec};
    use super::*;
    use crate::base::BaseKind;
    use crate::dims::Build;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    fn body(base: BaseKind) -> Body {
        Body::new(&BodyPlan::standard(), base, FingerCount::Five)
    }

    fn measured(design: &RigDesign) -> RigSpec {
        let (text, files) = export(design);
        RigSpec::from_urdf(&urdf::parse(&text).unwrap(), "t", &files).unwrap()
    }

    #[test]
    fn an_unedited_design_comes_back_as_the_body_it_was_made_from() {
        let b = body(BaseKind::HumanFoot);
        let r = measured(&edited_design(&b, &[]).unwrap());
        assert_eq!(r.form, RigForm::Humanoid);
        assert!((r.dims.upper_arm_m - b.dims.upper_arm_m).abs() < 1e-3 && (r.dims.hip_m - b.dims.hip_m).abs() < 1e-3, "{:?}", r.dims);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let t = body(BaseKind::Wheel);
        let r = measured(&edited_design(&t, &[]).unwrap());
        assert_eq!(r.form, RigForm::Forklift);
        assert!((r.dims.tine_len_m - t.dims.tine_len_m).abs() < 1e-3 && (r.dims.wheelbase_m.unwrap() - t.support_len_m).abs() < 1e-3, "{:?}", r.dims);
    }

    #[test]
    fn a_longer_arm_measures_longer_and_a_body_built_from_it_reaches_further() {
        let b = body(BaseKind::HumanFoot);
        let want = Param::ArmLen.value(&b) + 0.15;
        let r = measured(&edited_design(&b, &[RigEdit { param: Param::ArmLen, value_m: want }]).unwrap());
        assert!((r.dims.upper_arm_m + r.dims.forearm_m - want).abs() < 1e-3);
        let id = register(r);
        let new = Body::new(&BodyPlan { build: Build::Rig(id), ..BodyPlan::standard() }, BaseKind::HumanFoot, FingerCount::Five);
        assert!((Param::ArmLen.value(&new) - want).abs() < 1e-3);
        assert!(!new.rig().unwrap().visuals.is_empty(), "the meshes are drawn");
    }

    #[test]
    fn a_tine_edit_on_a_truck_measures_and_an_arm_edit_on_it_is_refused() {
        let t = body(BaseKind::Wheel);
        let r = measured(&edited_design(&t, &[RigEdit { param: Param::TineLen, value_m: 1.5 }]).unwrap());
        assert!((r.dims.tine_len_m - 1.5).abs() < 1e-3);
        assert!(edited_design(&t, &[RigEdit { param: Param::ArmLen, value_m: 0.8 }]).unwrap_err().contains("몸에 없는"));
        assert!(edited_design(&t, &[RigEdit { param: Param::TineLen, value_m: 9.0 }]).unwrap_err().contains("범위"));
    }
}
