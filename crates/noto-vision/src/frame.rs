use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Rgb {
        assert_eq!(data.len(), (width * height * 3) as usize, "RGB buffer size");
        Rgb { width, height, data }
    }

    #[inline]
    pub fn px(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    pub fn to_dynamic(&self) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(self.width, self.height, self.data.clone()).expect("size checked in new"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthKind {
    Metric,
    Relative,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub kind: DepthKind,
    pub data: Vec<f32>,
}

impl DepthMap {
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.data[(y * self.width + x) as usize]
    }

    pub fn at_scaled(&self, x: f32, y: f32, src_w: u32, src_h: u32) -> f32 {
        let sx = ((x / src_w.max(1) as f32) * self.width as f32).clamp(0.0, self.width as f32 - 1.0) as u32;
        let sy = ((y / src_h.max(1) as f32) * self.height as f32).clamp(0.0, self.height as f32 - 1.0) as u32;
        self.at(sx, sy)
    }

    pub fn median_in(&self, mask: &Mask) -> Option<f32> {
        let mut v: Vec<f32> = Vec::new();
        for y in 0..mask.height {
            for x in 0..mask.width {
                if mask.get(x, y) {
                    let d = self.at_scaled(x as f32 + 0.5, y as f32 + 0.5, mask.width, mask.height);
                    if d.is_finite() && d > 0.0 {
                        v.push(d);
                    }
                }
            }
        }
        if v.is_empty() {
            return None;
        }
        v.sort_by(f32::total_cmp);
        Some(v[v.len() / 2])
    }

    pub fn colorize(&self) -> Rgb {
        let (lo, hi) = self.range();
        self.colorize_in(lo, hi)
    }

    pub fn range(&self) -> (f32, f32) {
        let finite: Vec<f32> = self.data.iter().copied().filter(|d| d.is_finite() && *d > 0.0).collect();
        if finite.is_empty() {
            return (0.0, 1.0);
        }
        let mut s = finite;
        s.sort_by(f32::total_cmp);
        (s[s.len() / 50], s[s.len() - 1 - s.len() / 50].max(s[s.len() / 50] + 1e-3))
    }

    pub fn colorize_in(&self, lo: f32, hi: f32) -> Rgb {
        let mut out = Vec::with_capacity((self.width * self.height * 3) as usize);
        for &d in &self.data {
            if !d.is_finite() || d <= 0.0 {
                out.extend_from_slice(&[12, 12, 16]);
                continue;
            }
            let t = ((1.0 / d - 1.0 / hi) / (1.0 / lo.max(1e-3) - 1.0 / hi)).clamp(0.0, 1.0);
            out.extend_from_slice(&turbo(t));
        }
        Rgb::new(self.width, self.height, out)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalMap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 3]>,
}

impl NormalMap {
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> [f32; 3] {
        self.data[(y * self.width + x) as usize]
    }

    pub fn at_scaled(&self, x: f32, y: f32, src_w: u32, src_h: u32) -> [f32; 3] {
        let sx = ((x / src_w.max(1) as f32) * self.width as f32).clamp(0.0, self.width as f32 - 1.0) as u32;
        let sy = ((y / src_h.max(1) as f32) * self.height as f32).clamp(0.0, self.height as f32 - 1.0) as u32;
        self.at(sx, sy)
    }

    pub fn colorize(&self) -> Rgb {
        let mut out = Vec::with_capacity((self.width * self.height * 3) as usize);
        for n in &self.data {
            out.extend(n.iter().map(|c| ((c.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8));
        }
        Rgb::new(self.width, self.height, out)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SegMap {
    pub width: u32,
    pub height: u32,
    pub labels: Vec<u8>,
    pub sky: Vec<f32>,
}

impl SegMap {
    #[inline]
    pub fn label_at(&self, x: u32, y: u32) -> u8 {
        self.labels[(y * self.width + x) as usize]
    }

    pub fn label_scaled(&self, x: f32, y: f32, src_w: u32, src_h: u32) -> u8 {
        let sx = ((x / src_w.max(1) as f32) * self.width as f32).clamp(0.0, self.width as f32 - 1.0) as u32;
        let sy = ((y / src_h.max(1) as f32) * self.height as f32).clamp(0.0, self.height as f32 - 1.0) as u32;
        self.label_at(sx, sy)
    }

    pub fn is_sky(&self, i: usize) -> bool {
        self.labels[i] as usize == crate::tips::ADE_SKY || self.sky[i] >= 0.5
    }

    pub fn sky_fraction(&self) -> f32 {
        (0..self.labels.len()).filter(|&i| self.is_sky(i)).count() as f32 / self.labels.len().max(1) as f32
    }

    pub fn colorize(&self) -> Rgb {
        let mut out = Vec::with_capacity((self.width * self.height * 3) as usize);
        for &l in &self.labels {
            out.extend_from_slice(&crate::tips::ade_colour(l as usize));
        }
        Rgb::new(self.width, self.height, out)
    }
}

pub fn turbo(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) as f64;
    let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    [(r.clamp(0.0, 1.0) * 255.0) as u8, (g.clamp(0.0, 1.0) * 255.0) as u8, (b.clamp(0.0, 1.0) * 255.0) as u8]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub bits: Vec<bool>,
}

impl Mask {
    pub fn empty(width: u32, height: u32) -> Mask {
        Mask { width, height, bits: vec![false; (width * height) as usize] }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> bool {
        self.bits[(y * self.width + x) as usize]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32) {
        let i = (y * self.width + x) as usize;
        self.bits[i] = true;
    }

    pub fn area(&self) -> u32 {
        self.bits.iter().filter(|b| **b).count() as u32
    }

    pub fn centroid(&self) -> Option<(f32, f32)> {
        let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u64);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    sx += x as f64 + 0.5;
                    sy += y as f64 + 0.5;
                    n += 1;
                }
            }
        }
        (n > 0).then(|| ((sx / n as f64) as f32, (sy / n as f64) as f32))
    }

    pub fn bbox(&self) -> Option<[u32; 4]> {
        let mut b: Option<[u32; 4]> = None;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    b = Some(match b {
                        None => [x, y, x, y],
                        Some(o) => [o[0].min(x), o[1].min(y), o[2].max(x), o[3].max(y)],
                    });
                }
            }
        }
        b
    }

    pub fn iou(&self, other: &Mask) -> f32 {
        if self.width != other.width || self.height != other.height {
            return 0.0;
        }
        let (mut inter, mut uni) = (0u32, 0u32);
        for (a, b) in self.bits.iter().zip(&other.bits) {
            inter += (*a && *b) as u32;
            uni += (*a || *b) as u32;
        }
        if uni == 0 {
            0.0
        } else {
            inter as f32 / uni as f32
        }
    }

    pub fn rle(&self) -> Vec<Vec<u32>> {
        (0..self.height)
            .map(|y| {
                let mut runs = Vec::new();
                let mut x = 0;
                while x < self.width {
                    if self.get(x, y) {
                        let s = x;
                        while x < self.width && self.get(x, y) {
                            x += 1;
                        }
                        runs.push(s);
                        runs.push(x - s);
                    } else {
                        x += 1;
                    }
                }
                runs
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_measures() {
        let mut m = Mask::empty(4, 3);
        m.set(1, 1);
        m.set(2, 1);
        assert_eq!(m.area(), 2);
        assert_eq!(m.bbox(), Some([1, 1, 2, 1]));
        assert_eq!(m.centroid(), Some((2.0, 1.5)));
        assert_eq!(m.rle()[1], vec![1, 2]);
        let mut n = Mask::empty(4, 3);
        n.set(2, 1);
        assert!((m.iou(&n) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn depth_median_and_colours() {
        let d = DepthMap { width: 2, height: 2, kind: DepthKind::Metric, data: vec![1.0, 2.0, 3.0, f32::INFINITY] };
        let mut m = Mask::empty(2, 2);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            m.set(x, y);
        }
        assert_eq!(d.median_in(&m), Some(2.0));
        let c = d.colorize();
        assert_eq!(c.data.len(), 12);
        assert_eq!(c.px(1, 1), [12, 12, 16], "sky stays dark");
        assert_ne!(c.px(0, 0), c.px(0, 1), "near and far differ");
    }
}
