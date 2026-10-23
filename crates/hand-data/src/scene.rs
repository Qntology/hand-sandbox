//! Domain randomization: draws a synthetic scene (hand configuration, object
//! size, mass, material friction and placement) from a seed.
//!
//! The random generator is a small SplitMix64, so the same seed always gives
//! the same scene on every platform without an extra dependency.

use hand_core::{Actuation, BaseKind, FingerCount, GraspSetup, HandConfig, JointKind, JointOptions, Payload};

/// Deterministic 64-bit generator (SplitMix64).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform float in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform index in `[0, n)`.
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Material with a friction range. The ranges are modelling assumptions to
/// be checked against the sources in `data/README.md`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub name: &'static str,
    pub friction: (f32, f32),
}

pub const MATERIALS: [Material; 4] = [
    Material {
        name: "cardboard",
        friction: (0.40, 0.65),
    },
    Material {
        name: "plastic",
        friction: (0.30, 0.45),
    },
    Material {
        name: "rubber",
        friction: (0.70, 1.00),
    },
    Material {
        name: "polybag",
        friction: (0.35, 0.55),
    },
];

/// Everything needed to simulate one synthetic episode.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneSpec {
    pub sample_id: String,
    pub seed: u64,
    pub label: String,
    pub material: String,
    pub hand: HandConfig,
    pub payload: Payload,
    pub setup: GraspSetup,
}

/// Draws a scene from `seed`. `label` names the object class.
pub fn random_scene(seed: u64, label: &str) -> SceneSpec {
    let mut rng = SplitMix64::new(seed);

    let fingers = FingerCount::ALL[rng.index(FingerCount::ALL.len())];
    let knuckle_kinds = [JointKind::Hinge, JointKind::Universal, JointKind::Ball];
    let distal_kinds = [JointKind::Hinge, JointKind::Universal];
    let actuation = if rng.unit() < 0.5 {
        Actuation::Direct
    } else {
        Actuation::Coupled {
            ratio: rng.range(0.5, 0.8),
        }
    };
    let hand = HandConfig {
        fingers,
        joints: JointOptions {
            segments: 2 + rng.index(3) as u8,
            base_kind: knuckle_kinds[rng.index(knuckle_kinds.len())],
            distal_kind: distal_kinds[rng.index(distal_kinds.len())],
            actuation,
        },
        base: BaseKind::ALL[rng.index(BaseKind::ALL.len())],
    };

    let material = MATERIALS[rng.index(MATERIALS.len())];
    let size_m = [
        rng.range(0.05, 0.14),
        rng.range(0.05, 0.14),
        rng.range(0.05, 0.14),
    ];
    let payload = Payload {
        name: label.to_string(),
        size_m,
        mass_kg: rng.range(0.2, 3.0),
        friction: rng.range(material.friction.0, material.friction.1),
    };
    let setup = GraspSetup {
        z_offset: rng.range(-0.02, 0.02),
        ..GraspSetup::default()
    };

    SceneSpec {
        sample_id: format!("syn-{seed:016x}"),
        seed,
        label: label.to_string(),
        material: material.name.to_string(),
        hand,
        payload,
        setup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_scene() {
        assert_eq!(random_scene(7, "carton"), random_scene(7, "carton"));
        assert_ne!(random_scene(7, "carton"), random_scene(8, "carton"));
    }

    #[test]
    fn drawn_scenes_build_valid_hands_and_ranges() {
        for seed in 0..200u64 {
            let s = random_scene(seed, "carton");
            let hand = s.hand.build().expect("valid hand");
            assert_eq!(hand.fingers.len(), s.hand.fingers.count() as usize);
            assert!((0.2..3.0).contains(&s.payload.mass_kg));
            assert!(s.payload.friction > 0.0 && s.payload.friction <= 1.0);
        }
    }

    #[test]
    fn rng_is_uniform_enough_for_index_draws() {
        let mut rng = SplitMix64::new(42);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[rng.index(4)] += 1;
        }
        assert!(counts.iter().all(|&c| c > 800));
    }
}
