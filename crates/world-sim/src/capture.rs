use std::io::Cursor;
use std::time::Instant;

use image::codecs::webp::WebPEncoder;
use image::{ExtendedColorType, ImageFormat, ImageReader};

#[derive(Clone, Debug)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub lossless: bool,
    pub quality: f32,
    pub encode_ms: f32,
}

impl Encoded {
    pub fn ratio(&self) -> f32 {
        (self.width as f32 * self.height as f32 * 3.0) / self.bytes.len().max(1) as f32
    }
}

pub fn encode_webp(rgb: &[u8], width: u32, height: u32, quality: f32) -> Result<Encoded, String> {
    if rgb.len() != (width as usize) * (height as usize) * 3 {
        return Err(format!("expected {} bytes for {width}x{height} RGB, got {}", width * height * 3, rgb.len()));
    }
    if width == 0 || height == 0 {
        return Err("empty frame".into());
    }
    let start = Instant::now();
    let lossless = quality >= 100.0;
    let bytes = if lossless {
        let mut out = Vec::new();
        WebPEncoder::new_lossless(&mut out)
            .encode(rgb, width, height, ExtendedColorType::Rgb8)
            .map_err(|e| format!("lossless webp: {e}"))?;
        out
    } else {
        let mem = webp::Encoder::from_rgb(rgb, width, height).encode(quality.clamp(0.0, 99.0));
        mem.to_vec()
    };
    Ok(Encoded {
        bytes,
        width,
        height,
        lossless,
        quality: if lossless { 100.0 } else { quality },
        encode_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

pub fn decode_webp(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let img = ImageReader::with_format(Cursor::new(bytes), ImageFormat::WebP)
        .decode()
        .map_err(|e| format!("decode webp: {e}"))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok((w, h, img.into_raw()))
}

pub fn psnr(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffers must have the same size");
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64;
    if mse == 0.0 {
        f32::INFINITY
    } else {
        (10.0 * (255.0f64 * 255.0 / mse).log10()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&[(x * 255 / w) as u8, (y * 255 / h) as u8, ((x + y) * 255 / (w + h)) as u8]);
            }
        }
        v
    }

    #[test]
    fn lossless_round_trip_is_exact() {
        let rgb = gradient(64, 48);
        let e = encode_webp(&rgb, 64, 48, 100.0).unwrap();
        assert!(e.lossless);
        assert_eq!(&e.bytes[0..4], b"RIFF");
        assert_eq!(&e.bytes[8..12], b"WEBP");
        let (w, h, back) = decode_webp(&e.bytes).unwrap();
        assert_eq!((w, h), (64, 48));
        assert_eq!(back, rgb);
    }

    #[test]
    fn lossy_is_smaller_and_close_to_the_original() {
        let rgb = gradient(128, 96);
        let lossless = encode_webp(&rgb, 128, 96, 100.0).unwrap();
        let lossy = encode_webp(&rgb, 128, 96, 70.0).unwrap();
        assert!(!lossy.lossless);
        assert!(lossy.bytes.len() < lossless.bytes.len(), "{} vs {}", lossy.bytes.len(), lossless.bytes.len());
        assert!(lossy.ratio() > 3.0);
        let (_, _, back) = decode_webp(&lossy.bytes).unwrap();
        assert_eq!(back.len(), rgb.len());
        let p = psnr(&rgb, &back);
        assert!(p > 28.0, "psnr {p} dB");
    }

    #[test]
    fn size_mismatch_is_an_error() {
        assert!(encode_webp(&[0u8; 10], 4, 4, 80.0).is_err());
        assert!(encode_webp(&[], 0, 0, 80.0).is_err());
        assert!(decode_webp(b"not a webp").is_err());
    }

    #[test]
    fn psnr_of_identical_buffers_is_infinite() {
        let a = vec![5u8; 30];
        assert!(psnr(&a, &a).is_infinite());
        let b = vec![15u8; 30];
        assert!((psnr(&a, &b) - 28.13).abs() < 0.05);
    }
}
