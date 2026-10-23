//! Images the ML logic works on: an RGB keyframe, a depth map, a mask.

use serde::{Deserialize, Serialize};

/// An 8-bit RGB image, row-major.
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

/// What a depth map's numbers mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthKind {
    /// Forward distance in metres (as the renderer writes it).
    Metric,
    /// Metric up to one unknown scale factor.
    Relative,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub kind: DepthKind,
    /// Forward depth per pixel; not finite where the model has no answer (sky).
    pub data: Vec<f32>,
}

impl DepthMap {
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.data[(y * self.width + x) as usize]
    }

    /// Value at a pixel of an image of another size (nearest neighbour).
    pub fn at_scaled(&self, x: f32, y: f32, src_w: u32, src_h: u32) -> f32 {
        let sx = ((x / src_w.max(1) as f32) * self.width as f32).clamp(0.0, self.width as f32 - 1.0) as u32;
        let sy = ((y / src_h.max(1) as f32) * self.height as f32).clamp(0.0, self.height as f32 - 1.0) as u32;
        self.at(sx, sy)
    }

    /// Median of the finite values under a mask of another size.
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

    /// 8-bit colour image of the map (near is warm, far is cool), for the UI.
    pub fn colorize(&self) -> Rgb {
        let finite: Vec<f32> = self.data.iter().copied().filter(|d| d.is_finite() && *d > 0.0).collect();
        let (lo, hi) = if finite.is_empty() {
            (0.0, 1.0)
        } else {
            let mut s = finite.clone();
            s.sort_by(f32::total_cmp);
            (s[s.len() / 50], s[s.len() - 1 - s.len() / 50].max(s[s.len() / 50] + 1e-3))
        };
        let mut out = Vec::with_capacity((self.width * self.height * 3) as usize);
        for &d in &self.data {
            if !d.is_finite() || d <= 0.0 {
                out.extend_from_slice(&[12, 12, 16]);
                continue;
            }
            // Inverse depth spreads the near range, where decisions are made.
            let t = ((1.0 / d - 1.0 / hi) / (1.0 / lo.max(1e-3) - 1.0 / hi)).clamp(0.0, 1.0);
            out.extend_from_slice(&turbo(t));
        }
        Rgb::new(self.width, self.height, out)
    }
}

/// Polynomial fit of the "turbo" colour map (Mikhailov 2019), t in 0..1.
pub fn turbo(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) as f64;
    let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    [(r.clamp(0.0, 1.0) * 255.0) as u8, (g.clamp(0.0, 1.0) * 255.0) as u8, (b.clamp(0.0, 1.0) * 255.0) as u8]
}

/// A binary mask, row-major.
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

    /// Centroid in pixels, `None` when empty.
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

    /// Inclusive box `[x0, y0, x1, y1]`, `None` when empty.
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

    /// Run-length encoding of the rows, `[start, length, ...]` per row, for the UI.
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
