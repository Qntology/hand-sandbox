#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }

    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

pub fn hash2(seed: u64, x: i32, y: i32) -> u64 {
    let mut z = seed
        ^ (x as i64 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (y as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    z ^ (z >> 33)
}

fn lattice(seed: u64, x: i32, y: i32) -> f32 {
    (hash2(seed, x, y) >> 40) as f32 / (1u64 << 24) as f32
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

pub fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (smooth(x - x0), smooth(y - y0));
    let (xi, yi) = (x0 as i32, y0 as i32);
    let a = lattice(seed, xi, yi);
    let b = lattice(seed, xi + 1, yi);
    let c = lattice(seed, xi, yi + 1);
    let d = lattice(seed, xi + 1, yi + 1);
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    top + (bottom - top) * fy
}

pub fn fbm(seed: u64, x: f32, y: f32, octaves: u32) -> f32 {
    let (mut amp, mut freq, mut sum, mut norm) = (1.0f32, 1.0f32, 0.0f32, 0.0f32);
    for o in 0..octaves {
        sum += amp * value_noise(seed.wrapping_add(o as u64 * 7919), x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x));
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn below_stays_below() {
        let mut r = Rng::new(7);
        assert!((0..1000).all(|_| r.below(5) < 5));
    }

    #[test]
    fn noise_is_bounded_and_continuous() {
        let mut prev = value_noise(3, 0.0, 0.0);
        for i in 1..200 {
            let v = value_noise(3, i as f32 * 0.01, 0.0);
            assert!((0.0..=1.0).contains(&v));
            assert!((v - prev).abs() < 0.1, "noise jumped between close samples");
            prev = v;
        }
        let f = fbm(9, 1.3, 2.7, 3);
        assert!((0.0..=1.0).contains(&f));
    }

    #[test]
    fn different_seeds_give_different_fields() {
        let a: Vec<f32> = (0..20).map(|i| value_noise(1, i as f32 * 0.7, 0.3)).collect();
        let b: Vec<f32> = (0..20).map(|i| value_noise(2, i as f32 * 0.7, 0.3)).collect();
        assert_ne!(a, b);
    }
}
