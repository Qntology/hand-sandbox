//! Reads what the CAD editor exports: a URDF robot description (links with mass, centre of mass and inertia, joints
//! with origin, axis and limits, a mesh per link) and the binary STL meshes beside it. Lengths are metres.
//!
//! The URDF is a tree hanging from `base_link`, written at the design's current joint values, so walking it gives
//! every link's place in the design, which is how a robot's lengths are measured (`RigSpec::from_urdf`).

use std::collections::HashMap;

type V3 = [f32; 3];
type M3 = [[f32; 3]; 3];

/// A rigid transform: rotation then translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tf {
    pub r: M3,
    pub t: V3,
}

impl Tf {
    pub const ID: Tf = Tf { r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t: [0.0; 3] };

    /// URDF's `xyz` and `rpy`: R = Rz(yaw) Ry(pitch) Rx(roll).
    pub fn of(xyz: V3, rpy: V3) -> Tf {
        let (sr, cr) = rpy[0].sin_cos();
        let (sp, cp) = rpy[1].sin_cos();
        let (sy, cy) = rpy[2].sin_cos();
        Tf { r: [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]], t: xyz }
    }

    pub fn dir(&self, v: V3) -> V3 {
        [0, 1, 2].map(|i| self.r[i][0] * v[0] + self.r[i][1] * v[1] + self.r[i][2] * v[2])
    }

    pub fn at(&self, p: V3) -> V3 {
        let d = self.dir(p);
        [d[0] + self.t[0], d[1] + self.t[1], d[2] + self.t[2]]
    }

    pub fn then(&self, b: &Tf) -> Tf {
        let col = |k: usize| self.dir([b.r[0][k], b.r[1][k], b.r[2][k]]);
        let (c0, c1, c2) = (col(0), col(1), col(2));
        Tf { r: [[c0[0], c1[0], c2[0]], [c0[1], c1[1], c2[1]], [c0[2], c1[2], c2[2]]], t: self.at(b.t) }
    }

    /// Takes a point in this frame's parent back into this frame.
    pub fn inv_at(&self, p: V3) -> V3 {
        let d = [p[0] - self.t[0], p[1] - self.t[1], p[2] - self.t[2]];
        [0, 1, 2].map(|i| self.r[0][i] * d[0] + self.r[1][i] * d[1] + self.r[2][i] * d[2])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UrdfVisual {
    pub file: String,
    pub origin: Tf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UrdfLink {
    pub name: String,
    pub mass_kg: f32,
    /// Centre of mass in the link frame.
    pub com: V3,
    pub visuals: Vec<UrdfVisual>,
    /// The first body's colour.
    pub color: Option<[u8; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UrdfJoint {
    pub name: String,
    /// `revolute`, `continuous`, `prismatic` or `fixed`.
    pub kind: String,
    pub parent: String,
    pub child: String,
    pub origin: Tf,
    pub axis: V3,
    /// Lower and upper, rad or m.
    pub limit: Option<[f32; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Urdf {
    pub name: String,
    pub links: Vec<UrdfLink>,
    pub joints: Vec<UrdfJoint>,
}

/// One tag's name, attributes and whether it closes itself.
struct Tag<'a> {
    name: &'a str,
    attrs: Vec<(&'a str, &'a str)>,
    closes: bool,
}

impl Tag<'_> {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }
}

fn tags(xml: &str) -> Vec<Tag<'_>> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find('<') {
        rest = &rest[i + 1..];
        if let Some(body) = rest.strip_prefix("!--") {
            rest = body.split_once("-->").map_or("", |(_, r)| r);
            continue;
        }
        if rest.starts_with('?') {
            rest = rest.split_once("?>").map_or("", |(_, r)| r);
            continue;
        }
        let Some(end) = rest.find('>') else { break };
        let (inner, after) = (&rest[..end], &rest[end + 1..]);
        rest = after;
        if inner.starts_with('/') {
            out.push(Tag { name: inner[1..].trim(), attrs: Vec::new(), closes: true });
            continue;
        }
        let selfclose = inner.ends_with('/');
        let inner = inner.trim_end_matches('/');
        let (name, mut tail) = inner.split_once(char::is_whitespace).unwrap_or((inner, ""));
        let mut attrs = Vec::new();
        while let Some(eq) = tail.find('=') {
            let key = tail[..eq].trim();
            let val = tail[eq + 1..].trim_start();
            let Some(q) = val.chars().next().filter(|c| *c == '"' || *c == '\'') else { break };
            let Some(close) = val[1..].find(q) else { break };
            attrs.push((key, &val[1..1 + close]));
            tail = &val[close + 2..];
        }
        out.push(Tag { name, attrs, closes: false });
        if selfclose {
            out.push(Tag { name, attrs: Vec::new(), closes: true });
        }
    }
    out
}

fn v3(s: Option<&str>) -> V3 {
    let mut v = [0.0; 3];
    for (k, part) in s.unwrap_or("").split_whitespace().take(3).enumerate() {
        v[k] = part.parse().unwrap_or(0.0);
    }
    v
}

fn origin_of(t: &Tag) -> Tf {
    Tf::of(v3(t.attr("xyz")), v3(t.attr("rpy")))
}

pub fn parse(xml: &str) -> Result<Urdf, String> {
    let all = tags(xml);
    let mut u = Urdf::default();
    let mut link: Option<UrdfLink> = None;
    let mut joint: Option<UrdfJoint> = None;
    let mut part = ""; // inertial, visual, collision
    let mut visual_origin = Tf::ID;
    for t in &all {
        if t.closes {
            match t.name {
                "link" => u.links.extend(link.take()),
                "joint" => u.joints.extend(joint.take()),
                "inertial" | "visual" | "collision" => part = "",
                _ => {}
            }
            continue;
        }
        match t.name {
            "robot" => u.name = t.attr("name").unwrap_or("robot").to_string(),
            "link" => link = Some(UrdfLink { name: t.attr("name").unwrap_or("").to_string(), mass_kg: 0.0, com: [0.0; 3], visuals: Vec::new(), color: None }),
            "joint" if link.is_none() => joint = Some(UrdfJoint { name: t.attr("name").unwrap_or("").to_string(), kind: t.attr("type").unwrap_or("fixed").to_string(), parent: String::new(), child: String::new(), origin: Tf::ID, axis: [0.0, 0.0, 1.0], limit: None }),
            "inertial" | "visual" | "collision" => {
                part = t.name;
                visual_origin = Tf::ID;
            }
            "origin" => {
                if let Some(j) = joint.as_mut() {
                    j.origin = origin_of(t);
                } else if let Some(l) = link.as_mut() {
                    match part {
                        "inertial" => l.com = v3(t.attr("xyz")),
                        "visual" => visual_origin = origin_of(t),
                        _ => {}
                    }
                }
            }
            "mass" => {
                if let (Some(l), "inertial") = (link.as_mut(), part) {
                    l.mass_kg = t.attr("value").and_then(|v| v.parse().ok()).unwrap_or(0.0);
                }
            }
            "mesh" => {
                if let (Some(l), "visual", Some(file)) = (link.as_mut(), part, t.attr("filename")) {
                    l.visuals.push(UrdfVisual { file: file.to_string(), origin: visual_origin });
                }
            }
            "color" => {
                if let (Some(l), "visual") = (link.as_mut(), part) {
                    l.color = t.attr("rgba").map(|c| v3(Some(c)).map(|x| (x * 255.0).round().clamp(0.0, 255.0) as u8));
                }
            }
            "parent" => {
                if let (Some(j), Some(n)) = (joint.as_mut(), t.attr("link")) {
                    j.parent = n.to_string();
                }
            }
            "child" => {
                if let (Some(j), Some(n)) = (joint.as_mut(), t.attr("link")) {
                    j.child = n.to_string();
                }
            }
            "axis" => {
                if let Some(j) = joint.as_mut() {
                    j.axis = v3(t.attr("xyz"));
                }
            }
            "limit" => {
                if let Some(j) = joint.as_mut() {
                    if let (Some(lo), Some(hi)) = (t.attr("lower").and_then(|v| v.parse().ok()), t.attr("upper").and_then(|v| v.parse().ok())) {
                        j.limit = Some([lo, hi]);
                    }
                }
            }
            _ => {}
        }
    }
    if u.links.is_empty() {
        return Err("URDF에 링크가 없습니다".into());
    }
    Ok(u)
}

impl Urdf {
    /// Each link's frame in the design, walking the tree from its root.
    pub fn world_frames(&self) -> HashMap<String, Tf> {
        let mut frames: HashMap<String, Tf> = HashMap::new();
        let children: Vec<&str> = self.joints.iter().map(|j| j.child.as_str()).collect();
        for l in self.links.iter().filter(|l| !children.contains(&l.name.as_str())) {
            frames.insert(l.name.clone(), Tf::ID);
        }
        // At most the depth of the tree in passes.
        for _ in 0..self.joints.len() + 1 {
            let mut grew = false;
            for j in &self.joints {
                if frames.contains_key(&j.child) {
                    continue;
                }
                if let Some(p) = frames.get(&j.parent).copied() {
                    frames.insert(j.child.clone(), p.then(&j.origin));
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        frames
    }
}

/// Triangles of a binary STL (metres, as the exporter writes).
pub fn parse_stl(bytes: &[u8]) -> Result<Vec<[V3; 3]>, String> {
    if bytes.len() < 84 {
        return Err("STL가 너무 짧습니다".into());
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    if bytes.len() < 84 + 50 * n {
        return Err(format!("STL 길이가 삼각형 {n}개에 모자랍니다"));
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let rec = &bytes[84 + 50 * k..84 + 50 * (k + 1)];
        let f = |i: usize| f32::from_le_bytes([rec[i], rec[i + 1], rec[i + 2], rec[i + 3]]);
        out.push([[f(12), f(16), f(20)], [f(24), f(28), f(32)], [f(36), f(40), f(44)]]);
    }
    Ok(out)
}

/// Standard base64, as the page sends the mesh files.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let val = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a') as u32 + 26),
            b'0'..=b'9' => Some((c - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("base64 글자가 아닙니다: {}", c as char))?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<!-- Exported by SolidCraft; units: m, kg, rad -->
<robot name="noto_robot">
  <link name="base_link">
  </link>
  <link name="chassis">
    <inertial>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <mass value="206.4"/>
      <inertia ixx="11.7" ixy="0" ixz="0" iyy="17.9" iyz="0" izz="28.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="noto_robot-chassis.stl"/></geometry>
    </visual>
  </link>
  <link name="mast">
    <inertial>
      <origin xyz="0 0 0.6" rpy="0 0 0"/>
      <mass value="58.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0.6 -0.4" rpy="0 0 -1.5707963267948966"/>
      <geometry><mesh filename="noto_robot-mast.stl"/></geometry>
    </visual>
  </link>
  <joint name="chassis_fixed" type="fixed">
    <parent link="base_link"/>
    <child link="chassis"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="mast_lift" type="prismatic">
    <parent link="chassis"/>
    <child link="mast"/>
    <origin xyz="0.6 0 0.4" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="450" velocity="0.1"/>
  </joint>
</robot>
"#;

    #[test]
    fn the_editor_s_export_reads_back_as_a_tree_of_links() {
        let u = parse(SAMPLE).unwrap();
        assert_eq!(u.name, "noto_robot");
        assert_eq!(u.links.len(), 3);
        assert_eq!(u.joints.len(), 2);
        let mast = u.links.iter().find(|l| l.name == "mast").unwrap();
        assert!((mast.mass_kg - 58.05).abs() < 1e-3);
        assert_eq!(mast.visuals[0].file, "noto_robot-mast.stl");
        let lift = &u.joints[1];
        assert_eq!((lift.kind.as_str(), lift.parent.as_str(), lift.child.as_str()), ("prismatic", "chassis", "mast"));
        assert_eq!(lift.limit, Some([0.0, 1.0]));
        let f = u.world_frames();
        let m = f["mast"];
        assert!((m.t[0] - 0.6).abs() < 1e-6 && (m.t[2] - 0.4).abs() < 1e-6, "{m:?}");
        // The mesh's own frame comes back to the design's.
        let v = m.then(&mast.visuals[0].origin);
        for p in [[0.1, 0.2, 0.3], [1.0, -2.0, 0.5]] {
            let q = v.at(p);
            assert!((q[0] - p[0]).abs() < 1e-5 && (q[1] - p[1]).abs() < 1e-5 && (q[2] - p[2]).abs() < 1e-5, "{p:?} -> {q:?}");
        }
    }

    #[test]
    fn frames_compose_and_invert() {
        let a = Tf::of([1.0, 2.0, 3.0], [0.3, -0.2, 1.1]);
        let p = [0.4, -0.7, 0.2];
        let q = a.at(p);
        let back = a.inv_at(q);
        assert!((0..3).all(|i| (back[i] - p[i]).abs() < 1e-5), "{back:?}");
    }

    #[test]
    fn binary_stl_and_base64_round_trip() {
        let mut bytes = vec![0u8; 80];
        bytes.extend(1u32.to_le_bytes());
        for v in [0.0f32, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
            bytes.extend(v.to_le_bytes());
        }
        bytes.extend([0u8, 0]);
        let t = parse_stl(&bytes).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0][2], [0.7, 0.8, 0.9]);
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8h").unwrap(), b"hello!");
        assert!(base64_decode("a$b").is_err());
    }
}
