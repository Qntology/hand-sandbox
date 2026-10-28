use image::{imageops::FilterType, DynamicImage};

use crate::config::Siglip2Config;

#[derive(Clone, Debug, PartialEq)]
pub struct NaflexInput {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_dim: usize,
    pub width: u32,
    pub height: u32,
}

pub fn naflex_grid(width: u32, height: u32, patch: usize, max_patches: usize) -> (usize, usize) {
    let scaled_dim = |scale: f64, size: u32| -> usize {
        let scaled = scale * size as f64;
        let divisible = (scaled / patch as f64).ceil() as usize * patch;
        divisible.max(patch)
    };

    let eps = 1e-5;
    let (mut lo, mut hi) = (eps, 100.0f64);
    while hi - lo >= eps {
        let mid = (lo + hi) / 2.0;
        let th = scaled_dim(mid, height);
        let tw = scaled_dim(mid, width);
        let n = (th / patch) * (tw / patch);
        if n <= max_patches {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let th = scaled_dim(lo, height);
    let tw = scaled_dim(lo, width);
    ((th / patch).max(1), (tw / patch).max(1))
}

pub fn preprocess(image: &DynamicImage, cfg: &Siglip2Config) -> NaflexInput {
    let p = cfg.patch_size;
    let (grid_h, grid_w) = naflex_grid(image.width(), image.height(), p, cfg.max_patches);
    let (rw, rh) = ((grid_w * p) as u32, (grid_h * p) as u32);
    let rgb = image.resize_exact(rw, rh, FilterType::Triangle).to_rgb8();

    let patch_dim = p * p * 3;
    let mut patches = vec![0.0f32; grid_h * grid_w * patch_dim];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let base = (gy * grid_w + gx) * patch_dim;
            for py in 0..p {
                for px in 0..p {
                    let pixel = rgb.get_pixel((gx * p + px) as u32, (gy * p + py) as u32);
                    let at = base + (py * p + px) * 3;
                    for c in 0..3 {
                        patches[at + c] = pixel[c] as f32 / 127.5 - 1.0;
                    }
                }
            }
        }
    }

    NaflexInput {
        patches,
        grid_h,
        grid_w,
        patch_dim,
        width: image.width(),
        height: image.height(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn grid_respects_patch_budget_and_aspect() {
        let (h, w) = naflex_grid(640, 480, 16, 256);
        assert!(h * w <= 256);
        assert!(h >= 1 && w >= 1);
        assert!(w > h);
        let (h2, w2) = naflex_grid(480, 640, 16, 256);
        assert!(h2 > w2);
    }

    #[test]
    fn square_image_fills_the_budget() {
        let (h, w) = naflex_grid(1000, 1000, 16, 256);
        assert_eq!((h, w), (16, 16));
    }

    #[test]
    fn preprocess_shapes_and_range() {
        let cfg = Siglip2Config {
            patch_size: 4,
            max_patches: 16,
            ..Siglip2Config::default()
        };
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 20, image::Rgb([255, 0, 128])));
        let input = preprocess(&img, &cfg);
        assert_eq!(input.patch_dim, 4 * 4 * 3);
        assert_eq!(input.patches.len(), input.grid_h * input.grid_w * input.patch_dim);
        assert!(input.patches.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert!((input.patches[0] - 1.0).abs() < 1e-6);
        assert!((input.patches[1] + 1.0).abs() < 1e-6);
    }
}
