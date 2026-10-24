//! Contact materials of the actions: friction of the finger pads on each
//! object and of each object on each ground (with the scatter of repeated
//! measurements), restitution of an object landing, and the elasticity of the
//! pads and the grounds. Values, sources and the arithmetic:
//! `docs/research/actions.md` §1. These are model inputs; the benchmark
//! scores with other sources.

use serde::Serialize;

use crate::surface::Surface;

/// What an object is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemMaterial {
    Wood,
    Stone,
    Coal,
    IronOre,
}

impl ItemMaterial {
    pub const ALL: [ItemMaterial; 4] = [ItemMaterial::Wood, ItemMaterial::Stone, ItemMaterial::Coal, ItemMaterial::IronOre];

    pub fn id(self) -> &'static str {
        match self {
            ItemMaterial::Wood => "wood",
            ItemMaterial::Stone => "stone",
            ItemMaterial::Coal => "coal",
            ItemMaterial::IronOre => "iron_ore",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            ItemMaterial::Wood => "나무",
            ItemMaterial::Stone => "돌",
            ItemMaterial::Coal => "석탄",
            ItemMaterial::IronOre => "철광석",
        }
    }
}

/// A friction coefficient and the scatter of its measurements (coefficient of variation).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Friction {
    pub mean: f32,
    pub cov: f32,
}

/// Object-to-object scatter of a pad's friction (est, research §1.1: repeat
/// tests on one pair scatter 1–7%; natural objects differ more than that).
pub const PAD_MU_COV: f32 = 0.10;

/// Elastomer finger pad on the object, dry (research §1.1).
pub fn pad_on(item: ItemMaterial) -> Friction {
    let mean = match item {
        // Rubber on other materials, the cargo-securing design value [direct, CTU Code 2014].
        ItemMaterial::Wood => 0.60,
        // Rough granite: shoe rubber slides at 0.57–0.67 (midpoint 0.62); static = kinetic / 0.75 [derived].
        ItemMaterial::Stone => 0.83,
        // Coal on steel holds 0.31–0.38; rubber grips more, coal dust less than clean rubber's 0.6 [est].
        ItemMaterial::Coal => 0.45,
        // A rough mineral lump like stone [substituted].
        ItemMaterial::IronOre => 0.83,
    };
    Friction { mean, cov: PAD_MU_COV }
}

/// Scatter of a block's sliding friction on soil from drag to drag
/// (coefficient of variation; logs on clay 6–14%, research §1.2).
pub const GROUND_SLIDE_COV: f32 = 0.10;
/// Kinetic over static friction of a block on a ground (est: grounds without both numbers).
pub const KINETIC_SHARE: f32 = 0.9;

/// The object sliding on the ground (pushing): static and kinetic friction (research §1.2).
pub fn item_on(item: ItemMaterial, ground: Surface) -> (Friction, Friction) {
    let wood = item == ItemMaterial::Wood;
    let (s, k) = match ground {
        // Douglas fir on smooth concrete 0.58; stone on a hard face like hard rock on hard rock 0.55.
        Surface::Firm => (if wood { 0.58 } else { 0.55 }, None),
        // Dressed rock on rock 0.55–0.70: the middle.
        Surface::Rock => (0.62, None),
        // Fir on packed soil 0.58; a block on silt or clay 0.30–0.50 (middle of the clays 0.40); both near 0.5.
        Surface::Dirt => (if wood { 0.58 } else { 0.50 }, None),
        // A sled on natural turf slides at 0.45.
        Surface::Grass => (0.50, Some(0.45)),
        // Concrete on clean gravel 0.55–0.60.
        Surface::Gravel => (0.57, None),
        // A sled on dry sand at ≤2.5 kPa: about 0.5 sliding.
        Surface::Sand => (0.55, Some(0.50)),
        // A little water lowers it 10–40% (26% the middle case).
        Surface::WetSand => (0.41, Some(0.37)),
        Surface::Water => (0.0, Some(0.0)),
    };
    let k = k.unwrap_or(KINETIC_SHARE * s);
    (Friction { mean: s, cov: GROUND_SLIDE_COV }, Friction { mean: k, cov: GROUND_SLIDE_COV })
}

/// Normal coefficient of restitution of a small block landing on the ground (research §1.3).
pub fn restitution(item: ItemMaterial, ground: Surface) -> f32 {
    let _ = item;
    match ground {
        // An 8 cm rock block on a rock slab rebounds 0.21 [direct, laboratory drop].
        Surface::Firm | Surface::Rock => 0.21,
        // On medium-dense sand 0.10 [direct, same tests].
        Surface::Sand | Surface::WetSand => 0.10,
        // Between the slab and the sand [est].
        Surface::Gravel => 0.15,
        Surface::Dirt => 0.12,
        Surface::Grass => 0.10,
        Surface::Water => 0.0,
    }
}

/// Contact stiffness of the ground under a small block, N/m (impact force and
/// duration): the ground's deformation modulus times the 0.1 m face (flat
/// punch), capped by the block's own contact (research §1.3).
pub fn ground_stiffness(ground: Surface) -> f32 {
    match ground {
        // A hard face: the block's own corner contact governs (est).
        Surface::Firm | Surface::Rock => 2.0e6,
        // Medium-dense sand Evd 4.5 MPa × 0.1 m [derived]; loose dry sand softer (est).
        Surface::WetSand => 4.5e5,
        Surface::Sand => 2.0e5,
        Surface::Gravel => 6.0e5,
        Surface::Dirt => 3.0e5,
        Surface::Grass => 1.5e5,
        Surface::Water => 1.0e4,
    }
}

/// Young's modulus of the finger pad, Pa: Shore A30 platinum silicone,
/// median of uniaxial tests 0.80 MPa (research §1.1).
pub const PAD_E_PA: f32 = 0.80e6;
/// Poisson's ratio of the pad (rubber-like, nearly incompressible).
pub const PAD_NU: f32 = 0.48;
/// Radius of curvature of a fingertip pad, metres (est: a 20 mm wide rounded pad).
pub const PAD_RADIUS_M: f32 = 0.010;
/// Effective moving mass of a finger at contact, kg (est: a 25 g distal chain).
pub const FINGER_MASS_KG: f32 = 0.025;

/// Radius of a soft fingertip's contact patch under normal force `n`, metres
/// (Hertz, sphere on a flat rigid face: `a = (3·N·R / (4·E*))^(1/3)`; the
/// measured power law `a = c·N^γ` has γ ≤ 1/3, Hertz its upper end).
pub fn contact_radius_m(n: f32) -> f32 {
    let e_star = PAD_E_PA / (1.0 - PAD_NU * PAD_NU);
    (3.0 * n.max(0.0) * PAD_RADIUS_M / (4.0 * e_star)).cbrt()
}

/// Largest twisting moment a soft contact resists before it spins, N·m
/// (Hertzian pressure: `3π/16 · μ · N · a`).
pub fn torsion_capacity_nm(mu: f32, n: f32) -> f32 {
    3.0 * std::f32::consts::PI / 16.0 * mu * n * contact_radius_m(n)
}

/// Peak force of a contact closing at `v` between a mass `m` and a stiffness
/// `k` (undamped spring contact `v·√(k·m)`), and its duration `π·√(m/k)`.
pub fn impact(v: f32, m: f32, k: f32) -> (f32, f32) {
    (v.abs() * (k * m).sqrt(), std::f32::consts::PI * (m / k).sqrt())
}

/// Contact stiffness of a soft pad pressing a rigid face at force `n`, N/m (`dN/dδ` of Hertz: `2·E*·a`).
pub fn pad_stiffness(n: f32) -> f32 {
    let e_star = PAD_E_PA / (1.0 - PAD_NU * PAD_NU);
    2.0 * e_star * contact_radius_m(n.max(0.5))
}

/// 1σ of the depth a stereo camera measures at range `z`, metres
/// (`Δz = z²·Δd / (f·B)`, research §1.4): focal length in pixels, baseline, subpixel error.
pub fn stereo_depth_sigma(z_m: f32, focal_px: f32, baseline_m: f32, subpixel: f32) -> f32 {
    z_m * z_m * subpixel / (focal_px * baseline_m).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_contacts_grow_with_force_and_twist_more() {
        let (a1, a2) = (contact_radius_m(10.0), contact_radius_m(80.0));
        assert!(a2 > a1 && (a2 / a1 - 2.0).abs() < 0.01, "a ∝ N^(1/3): {a1} {a2}");
        assert!(torsion_capacity_nm(0.6, 80.0) > 8.0 * torsion_capacity_nm(0.6, 10.0));
        let (f, dt) = impact(0.4, 0.025, pad_stiffness(5.0));
        assert!(f > 1.0 && f < 50.0 && dt > 1e-4 && dt < 0.05, "{f} N over {dt} s");
        assert!(restitution(ItemMaterial::Stone, Surface::Firm) > restitution(ItemMaterial::Stone, Surface::Sand));
        let (s, k) = item_on(ItemMaterial::Wood, Surface::Gravel);
        assert!(k.mean < s.mean);
        // The D435 law at 1 m: f = 424 px, B = 50 mm, 0.08 px → 3.8 mm.
        assert!((stereo_depth_sigma(1.0, 424.0, 0.05, 0.08) - 0.0038).abs() < 1e-4);
    }
}
