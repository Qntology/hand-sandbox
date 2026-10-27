//! Daylight (v1.0): where the sun stands, how much light it and the sky
//! give, and in what colour, for the renderer (`render`) and the head
//! camera's exposure (`world::eye`). Numbers and sources:
//! `docs/research/lighting.md` (§5 lists the engine inputs).
//!
//! * [`Lighting::Flat`] is the renderer as it was before v1.0: every face lit
//!   by a fixed factor of its axis (top 1.0, x faces 0.82, z faces 0.66), the
//!   constant sky gradient and fog, and 10 000 lx for the exposure. It is the
//!   default, so seeded runs and every benchmark keep their pixels byte for
//!   byte.
//! * [`Lighting::Sun`] puts the sun at an elevation and a compass azimuth.
//!   [`solar_position`] finds it for a latitude, a day and a clock with the
//!   NOAA solar position series (NOAA GMD, "General Solar Position
//!   Calculations"; the azimuth in the quadrant-safe form of the research
//!   pack) and the banded refraction of NOAA's solar calculator.
//! * The light on a horizontal face is the IESNA daylight availability model
//!   (IESNA Lighting Handbook, coefficients as reproduced by Golmohammadi,
//!   Shekari & Mahjub 2009): a direct beam `E_dn = E_xt·exp(−c·m)` through the
//!   Kasten & Young (1989) relative air mass `m`, and a sky term
//!   `A + B·sin^C h`, for a clear, a partly cloudy and an overcast sky.
//! * A face receives the beam by the cosine of its angle to the sun when a
//!   ray toward the sun is free (otherwise it lies in a shadow), the sky by
//!   the share of the dome it sees, `(1 + n_y)/2` (an isotropic sky), and the
//!   light the ground reflects by the rest, `ρ_g·E_h·(1 − n_y)/2`. The
//!   camera's auto exposure meters the global horizontal illuminance `E_h`
//!   (the scene lux), so a sunlit top face renders as bright as the flat
//!   mode's top face; faces turned to a low sun get brighter, shadows darker.
//!   Light adds in linear units; the pixels are sRGB, so the gain is applied
//!   through the display gamma (IEC 61966-2-1, ≈ 2.2).
//! * Colour: the direct beam reddens with the air mass (Rayleigh plus
//!   Ångström aerosol optical depth at a representative wavelength of each
//!   channel); the sky's light is the scattered share, proportional to the
//!   same optical depth (bluish); an overcast sky is grey. The camera's white
//!   balance is fixed to the sun at the zenith.
//! * When the scene is too dark for the longest exposure at the highest gain
//!   the frame darkens by what is missing.
//!
//! Directions: the world has x and z on the ground and y up; the UI's map
//! (and `noto-runtime::api`) shows +x to the east and −z to the north, so a
//! compass azimuth `A` (0 = north, 90 = east) points along `(sin A, −cos A)`
//! in `(x, z)`.

use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::world::{World, EXPOSURE_MAX_S, SCENE_LUX};

/// How the scene is lit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Lighting {
    /// The pre-v1.0 renderer: fixed face factors, constant sky, 10 000 lx.
    #[default]
    Flat,
    /// A sun at `elevation_deg` above the horizon (apparent, refraction
    /// included) and `azimuth_deg` on the compass (0 north, 90 east), under a
    /// sky from clear (`overcast` 0) through partly cloudy (0.5) to overcast (1).
    Sun {
        elevation_deg: f32,
        azimuth_deg: f32,
        #[serde(default)]
        overcast: f32,
    },
}

// ----- the sun's position (research lighting.md §1) -----

/// Where the sun is for a place and an instant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SolarPosition {
    /// Elevation above the horizon with refraction, degrees.
    pub elevation_deg: f32,
    /// The geometric elevation (no refraction), degrees.
    pub geometric_elevation_deg: f32,
    /// Compass azimuth, degrees clockwise from north.
    pub azimuth_deg: f32,
    pub declination_deg: f32,
    /// Hour angle, degrees (negative in the morning).
    pub hour_angle_deg: f32,
    /// Equation of time, minutes.
    pub eqtime_min: f32,
}

/// The sun's position at `hour` (local clock time, fractional hours) of
/// `day_of_year` (1 = 1 January) at latitude `lat_deg` (north positive) and
/// longitude `lon_deg` (east positive), in a time zone `tz_h` hours from UTC.
/// NOAA GMD "General Solar Position Calculations": the fractional year from
/// the UTC hour, the equation of time and declination series, the true solar
/// time, the hour angle and the zenith; the azimuth as
/// `arccos[(sin δ − sin φ·cos Z)/(cos φ·sin Z)]`, mirrored after noon; the
/// apparent elevation with the banded refraction of NOAA's calculator.
pub fn solar_position(lat_deg: f32, lon_deg: f32, tz_h: f32, day_of_year: u32, hour: f32) -> SolarPosition {
    let (lat, lon, tz, hour) = (lat_deg as f64, lon_deg as f64, tz_h as f64, hour as f64);
    let utc_h = hour - tz;
    let g = 2.0 * std::f64::consts::PI / 365.0 * (day_of_year.max(1) as f64 - 1.0 + (utc_h - 12.0) / 24.0);
    let eqtime = 229.18 * (0.000075 + 0.001868 * g.cos() - 0.032077 * g.sin() - 0.014615 * (2.0 * g).cos() - 0.040849 * (2.0 * g).sin());
    let decl = 0.006918 - 0.399912 * g.cos() + 0.070257 * g.sin() - 0.006758 * (2.0 * g).cos() + 0.000907 * (2.0 * g).sin() - 0.002697 * (3.0 * g).cos()
        + 0.00148 * (3.0 * g).sin();
    let time_offset = eqtime + 4.0 * lon - 60.0 * tz;
    let tst = hour * 60.0 + time_offset;
    let mut ha = tst / 4.0 - 180.0;
    ha = (ha + 180.0).rem_euclid(360.0) - 180.0;
    let phi = lat.to_radians();
    let cos_z = (phi.sin() * decl.sin() + phi.cos() * decl.cos() * ha.to_radians().cos()).clamp(-1.0, 1.0);
    let zen = cos_z.acos();
    let geo = 90.0 - zen.to_degrees();
    let denom = phi.cos() * zen.sin();
    let a0 = if denom.abs() < 1e-9 { 180.0 } else { ((decl.sin() - phi.sin() * cos_z) / denom).clamp(-1.0, 1.0).acos().to_degrees() };
    let azimuth = if ha < 0.0 { a0 } else { 360.0 - a0 };
    SolarPosition {
        elevation_deg: (geo + refraction_deg(geo)) as f32,
        geometric_elevation_deg: geo as f32,
        azimuth_deg: azimuth.rem_euclid(360.0) as f32,
        declination_deg: decl.to_degrees() as f32,
        hour_angle_deg: ha as f32,
        eqtime_min: eqtime as f32,
    }
}

/// Atmospheric refraction of a geometric elevation `h`, degrees, added to it
/// (NOAA solar calculator, after Meeus; four bands).
pub fn refraction_deg(h: f64) -> f64 {
    let t = h.to_radians().tan();
    let arcsec = if h > 85.0 {
        0.0
    } else if h > 5.0 {
        58.1 / t - 0.07 / t.powi(3) + 0.000086 / t.powi(5)
    } else if h > -0.575 {
        1735.0 + h * (-518.2 + h * (103.4 + h * (-12.79 + h * 0.711)))
    } else {
        -20.774 / t
    };
    arcsec / 3600.0
}

// ----- how much light (research lighting.md §2) -----

/// The luminous solar constant, lux: the sun's illuminance outside the
/// atmosphere at the mean Sun–Earth distance (IESNA, 127.5 klx). The year's
/// ±3.4 % from the orbit's eccentricity is left out.
pub const SOLAR_CONSTANT_LX: f32 = 127_500.0;

/// One sky of the IESNA daylight availability model: the direct beam's
/// extinction coefficient `c` (none: no beam gets through) and the sky's
/// horizontal illuminance `A + B·sin^C h`, lux.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyType {
    pub extinction: Option<f32>,
    pub a_lx: f32,
    pub b_lx: f32,
    pub c: f32,
}

/// IESNA clear sky: c = 0.21; A = 0.8, B = 15.5 klx, C = 0.5.
pub const CLEAR_SKY: SkyType = SkyType { extinction: Some(0.21), a_lx: 800.0, b_lx: 15_500.0, c: 0.5 };
/// IESNA partly cloudy sky: c = 0.80; A = 0.3, B = 45.0 klx, C = 1.0.
pub const PARTLY_CLOUDY_SKY: SkyType = SkyType { extinction: Some(0.80), a_lx: 300.0, b_lx: 45_000.0, c: 1.0 };
/// IESNA overcast sky: no beam; A = 0.3, B = 21.0 klx, C = 1.0.
pub const OVERCAST_SKY: SkyType = SkyType { extinction: None, a_lx: 300.0, b_lx: 21_000.0, c: 1.0 };

/// Elevation at which civil twilight ends, degrees (the definition of civil
/// twilight). Below the horizon the sky term fades linearly from its
/// horizon value to nothing there (est: the IESNA model stops at h = 0).
pub const CIVIL_TWILIGHT_DEG: f32 = -6.0;

/// Relative optical air mass at an elevation (Kasten & Young 1989:
/// `1/[cos z + 0.50572·(96.07995 − z)^−1.6364]`, z the zenith angle in
/// degrees; about 38 at the horizon).
pub fn air_mass(elevation_deg: f32) -> f32 {
    let z = (90.0 - elevation_deg).clamp(0.0, 90.0);
    1.0 / (z.to_radians().cos() + 0.50572 * (96.07995 - z).powf(-1.6364))
}

impl SkyType {
    /// Direct normal and sky (diffuse horizontal) illuminance at an elevation, lux.
    pub fn components(&self, elevation_deg: f32) -> (f32, f32) {
        if elevation_deg <= 0.0 {
            let fade = (1.0 - elevation_deg / CIVIL_TWILIGHT_DEG).clamp(0.0, 1.0);
            return (0.0, self.a_lx * fade);
        }
        let s = elevation_deg.to_radians().sin();
        let beam = self.extinction.map_or(0.0, |c| SOLAR_CONSTANT_LX * (-c * air_mass(elevation_deg)).exp());
        (beam, self.a_lx + self.b_lx * s.powf(self.c))
    }
}

/// Direct normal and sky illuminance under a cloud cover from 0 (clear)
/// through 0.5 (partly cloudy) to 1 (overcast): linear between the model's
/// three skies (est: the model gives only the three).
pub fn sky_components(elevation_deg: f32, overcast: f32) -> (f32, f32) {
    let w = overcast.clamp(0.0, 1.0);
    let lerp = |a: (f32, f32), b: (f32, f32), t: f32| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
    let (c, p, o) = (CLEAR_SKY.components(elevation_deg), PARTLY_CLOUDY_SKY.components(elevation_deg), OVERCAST_SKY.components(elevation_deg));
    if w <= 0.5 {
        lerp(c, p, 2.0 * w)
    } else {
        lerp(p, o, 2.0 * w - 1.0)
    }
}

// ----- colour (research lighting.md §3) -----

/// Representative wavelengths of the red, green and blue channels, µm (est:
/// near the dominant wavelengths of the sRGB primaries).
pub const CHANNEL_UM: [f32; 3] = [0.61, 0.55, 0.465];
/// Ångström turbidity of a clear sky and its wavelength exponent (research
/// pack: β about 0.05–0.1 for a clear sky, α = 1.3).
pub const ANGSTROM_BETA: f32 = 0.05;
pub const ANGSTROM_ALPHA: f32 = 1.3;
/// Relative luminance of the sRGB (BT.709) primaries.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// Display gamma of the frame's sRGB values (IEC 61966-2-1, ≈ 2.2).
pub const DISPLAY_GAMMA: f32 = 2.2;
/// Share of the global light the ground reflects onto faces that see it (est: grass and soil).
pub const GROUND_ALBEDO: f32 = 0.2;
/// Highest sensor gain over ISO 100: ISO 1600, the top of the OAK-D Lite
/// RGB control range (research camera.md §4, [29]).
pub const SENSOR_GAIN_MAX: f32 = 16.0;

/// Rayleigh optical depth of the atmosphere at sea level at a wavelength in
/// µm, `0.008569·λ⁻⁴·(1 + 0.0113·λ⁻² + 0.00013·λ⁻⁴)` (the research pack's
/// form; the original was not re-read).
pub fn rayleigh_depth(um: f32) -> f32 {
    0.008569 * um.powi(-4) * (1.0 + 0.0113 * um.powi(-2) + 0.00013 * um.powi(-4))
}

/// Total vertical optical depth (Rayleigh plus aerosol) of each channel.
pub fn channel_depth() -> [f32; 3] {
    CHANNEL_UM.map(|um| rayleigh_depth(um) + ANGSTROM_BETA * um.powf(-ANGSTROM_ALPHA))
}

fn luma(c: [f32; 3]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}

fn unit_luma(c: [f32; 3]) -> [f32; 3] {
    let l = luma(c).max(1e-12);
    c.map(|v| v / l)
}

/// Colour of the direct beam at an elevation, relative luminance 1: its
/// transmission through `m − 1` more air masses than at the zenith, per channel.
pub fn sun_colour(elevation_deg: f32) -> [f32; 3] {
    let m = air_mass(elevation_deg.max(0.0));
    let tau = channel_depth();
    unit_luma(tau.map(|t| (-(m - 1.0) * t).exp()))
}

/// Colour of the clear sky's light, relative luminance 1: the share of the
/// beam the atmosphere scatters, proportional to each channel's optical
/// depth (thin-atmosphere single scattering, est), going grey as clouds close.
pub fn sky_colour(overcast: f32) -> [f32; 3] {
    let blue = unit_luma(channel_depth());
    let w = overcast.clamp(0.0, 1.0);
    blue.map(|b| b + (1.0 - b) * w)
}

/// Share of the auto exposure's target the camera reaches at an
/// illuminance: 1 while the longest exposure (33 ms) at the highest gain
/// covers the meter's time `t = 0.5/L` (`world::eye::exposure_s`), less below.
pub fn exposure_reach(lux: f32) -> f32 {
    let luminance = 0.18 * lux.max(0.0) / PI;
    if luminance <= 0.0 {
        return 0.0;
    }
    (EXPOSURE_MAX_S * SENSOR_GAIN_MAX * luminance / 0.5).min(1.0)
}

/// Unit vector toward the sun (world, y up) from an elevation and a compass azimuth.
pub fn sun_direction(elevation_deg: f32, azimuth_deg: f32) -> [f32; 3] {
    let (se, ce) = elevation_deg.to_radians().sin_cos();
    let (sa, ca) = azimuth_deg.to_radians().sin_cos();
    [ce * sa, se, -ce * ca]
}

/// The sun and sky of a [`Lighting::Sun`], worked out once per frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Daylight {
    pub elevation_deg: f32,
    pub azimuth_deg: f32,
    pub overcast: f32,
    /// Toward the sun (world, y up).
    pub sun_dir: [f32; 3],
    /// Direct normal, direct horizontal, sky (diffuse horizontal) and global horizontal illuminance, lux.
    pub direct_normal_lx: f32,
    pub direct_lx: f32,
    pub sky_lx: f32,
    pub global_lx: f32,
    /// Colours of the beam and of the sky light, relative luminance 1.
    pub sun_rgb: [f32; 3],
    pub sky_rgb: [f32; 3],
    /// Share of the exposure target the sensor reaches (1 in daylight).
    pub reach: f32,
}

/// The sky's glow around a low sun: angular scale, degrees, and its share
/// of the sky colour at the horizon (est, for the picture only).
const GLOW_DEG: f32 = 12.0;
const GLOW_SHARE: f32 = 0.55;
/// Elevation over which the glow fades as the sun climbs, degrees (est).
const GLOW_FADE_DEG: f32 = 15.0;
/// Angular radius of the sun's disc, degrees (0.53° across, the standard value).
const SUN_DISC_DEG: f32 = 0.265;

impl Daylight {
    pub fn new(elevation_deg: f32, azimuth_deg: f32, overcast: f32) -> Daylight {
        let overcast = overcast.clamp(0.0, 1.0);
        let (dn, sky) = sky_components(elevation_deg, overcast);
        let direct = dn * elevation_deg.to_radians().sin().max(0.0);
        let global = direct + sky;
        Daylight {
            elevation_deg,
            azimuth_deg,
            overcast,
            sun_dir: sun_direction(elevation_deg, azimuth_deg),
            direct_normal_lx: dn,
            direct_lx: direct,
            sky_lx: sky,
            global_lx: global,
            sun_rgb: sun_colour(elevation_deg),
            sky_rgb: sky_colour(overcast),
            reach: exposure_reach(global),
        }
    }

    /// Light on a face with unit normal `n` (world), per channel, over the
    /// metered horizontal illuminance (1 = a sunlit top face), linear units.
    /// `lit`: the ray toward the sun is free.
    pub fn face_light(&self, n: [f32; 3], lit: bool) -> [f32; 3] {
        if self.global_lx <= 0.0 {
            return [0.0; 3];
        }
        let cos = n[0] * self.sun_dir[0] + n[1] * self.sun_dir[1] + n[2] * self.sun_dir[2];
        let beam = if lit { self.direct_normal_lx * cos.max(0.0) } else { 0.0 };
        let sky = self.sky_lx * 0.5 * (1.0 + n[1]);
        let ground = GROUND_ALBEDO * self.global_lx * 0.5 * (1.0 - n[1]);
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            out[c] = (beam * self.sun_rgb[c] + sky * self.sky_rgb[c] + ground) / self.global_lx;
        }
        out
    }

    /// The factor a face's sRGB colour is multiplied by: [`Daylight::face_light`]
    /// times the exposure's reach, through the display gamma.
    pub fn face_gain(&self, n: [f32; 3], lit: bool) -> [f32; 3] {
        self.face_light(n, lit).map(|k| (k * self.reach).max(0.0).powf(1.0 / DISPLAY_GAMMA))
    }

    /// Whether the sun stands high enough for a face to be lit directly.
    pub fn has_beam(&self) -> bool {
        self.direct_normal_lx > 0.0 && self.elevation_deg > 0.0
    }

    /// The sky seen along `d` (unit): the flat sky's gradient `base` (sRGB),
    /// greyed by clouds, warmed into a glow around a low sun, the sun's disc,
    /// darkened when the exposure falls short.
    pub fn sky_px(&self, d: [f32; 3], base: [f32; 3]) -> [f32; 3] {
        let grey = luma(base);
        let mut c = [0.0f32; 3];
        for k in 0..3 {
            c[k] = base[k] + (grey - base[k]) * self.overcast;
        }
        let cos = (d[0] * self.sun_dir[0] + d[1] * self.sun_dir[1] + d[2] * self.sun_dir[2]).clamp(-1.0, 1.0);
        let angle = cos.acos().to_degrees();
        let peak = self.sun_rgb.iter().fold(0.0f32, |a, &b| a.max(b)).max(1e-6);
        let warm = self.sun_rgb.map(|v| v / peak);
        if self.has_beam() {
            let w = GLOW_SHARE * (-angle / GLOW_DEG).exp() * (-self.elevation_deg.max(0.0) / GLOW_FADE_DEG).exp() * (1.0 - self.overcast);
            for k in 0..3 {
                c[k] += (255.0 * warm[k] - c[k]) * w;
            }
            if angle < SUN_DISC_DEG {
                c = warm.map(|v| 255.0 * v);
            }
        }
        let dim = self.reach.powf(1.0 / DISPLAY_GAMMA);
        c.map(|v| v * dim)
    }
}

impl Lighting {
    /// A sun at a place's clock: [`solar_position`] with the clock read as
    /// mean solar time of the time zone's own meridian (longitude `15·tz`).
    pub fn from_clock(lat_deg: f32, day_of_year: u32, hour: f32) -> Lighting {
        let p = solar_position(lat_deg, 0.0, 0.0, day_of_year, hour);
        Lighting::Sun { elevation_deg: p.elevation_deg, azimuth_deg: p.azimuth_deg, overcast: 0.0 }
    }

    /// One of the named times of day (`dawn`, `morning`, `noon`,
    /// `afternoon`, `dusk`, or their Korean names): the sun over Seoul at the
    /// March equinox at that clock.
    pub fn preset(id: &str) -> Option<Lighting> {
        PRESETS.iter().find(|p| p.id == id || p.label_ko == id).map(|p| Lighting::from_clock(PRESET_LAT_DEG, PRESET_DAY, p.hour))
    }

    /// The sun and sky of this lighting (`None` for the flat mode).
    pub fn daylight(&self) -> Option<Daylight> {
        match *self {
            Lighting::Flat => None,
            Lighting::Sun { elevation_deg, azimuth_deg, overcast } => Some(Daylight::new(elevation_deg, azimuth_deg, overcast)),
        }
    }

    /// Illuminance the camera's exposure meters, lux: 10 000 for the flat
    /// mode (`world::SCENE_LUX`), the global horizontal illuminance otherwise.
    pub fn scene_lux(&self) -> f32 {
        self.daylight().map_or(SCENE_LUX, |d| d.global_lx)
    }

    /// What the UI shows of the lighting.
    pub fn summary(&self) -> LightSummary {
        let lux = self.scene_lux();
        let base = LightSummary {
            mode: match self {
                Lighting::Flat => "flat",
                Lighting::Sun { .. } => "sun",
            },
            label_ko: match self {
                Lighting::Flat => "고정 조명".to_string(),
                Lighting::Sun { elevation_deg, azimuth_deg, .. } => format!("해 고도 {elevation_deg:.1}° · 방위 {azimuth_deg:.0}°"),
            },
            lux,
            exposure_ms: crate::world::exposure_s(lux) * 1000.0,
            reach: exposure_reach(lux),
            sun: None,
        };
        match self.daylight() {
            None => base,
            Some(d) => LightSummary {
                reach: d.reach,
                sun: Some(SunSummary {
                    elevation_deg: d.elevation_deg,
                    azimuth_deg: d.azimuth_deg,
                    overcast: d.overcast,
                    direct_normal_lux: d.direct_normal_lx,
                    direct_lux: d.direct_lx,
                    sky_lux: d.sky_lx,
                    sun_rgb: d.sun_rgb,
                    sky_rgb: d.sky_rgb,
                    shadow_per_m: if d.has_beam() { Some(1.0 / d.elevation_deg.to_radians().tan()) } else { None },
                }),
                ..base
            },
        }
    }
}

/// The lighting as the UI reads it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LightSummary {
    pub mode: &'static str,
    pub label_ko: String,
    /// Illuminance the exposure meters, lux, and the exposure it sets, ms.
    pub lux: f32,
    pub exposure_ms: f32,
    /// Share of the exposure target the sensor reaches.
    pub reach: f32,
    pub sun: Option<SunSummary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SunSummary {
    pub elevation_deg: f32,
    pub azimuth_deg: f32,
    pub overcast: f32,
    pub direct_normal_lux: f32,
    pub direct_lux: f32,
    pub sky_lux: f32,
    pub sun_rgb: [f32; 3],
    pub sky_rgb: [f32; 3],
    /// Shadow length per metre of height (`1/tan h`; none without a beam).
    pub shadow_per_m: Option<f32>,
}

/// A named time of day.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Preset {
    pub id: &'static str,
    pub label_ko: &'static str,
    /// Clock time, hours (mean solar time of the zone's meridian).
    pub hour: f32,
}

/// Latitude of the presets, degrees north (Seoul, 37.57° N).
pub const PRESET_LAT_DEG: f32 = 37.57;
/// Day of the year of the presets (21 March, about the equinox: sunrise and sunset near 6 and 18 h).
pub const PRESET_DAY: u32 = 80;

/// The times of day the UI offers: just after sunrise, mid-morning, noon,
/// mid-afternoon and just before sunset.
pub const PRESETS: [Preset; 5] = [
    Preset { id: "dawn", label_ko: "새벽", hour: 6.25 },
    Preset { id: "morning", label_ko: "아침", hour: 9.0 },
    Preset { id: "noon", label_ko: "정오", hour: 12.0 },
    Preset { id: "afternoon", label_ko: "오후", hour: 15.0 },
    Preset { id: "dusk", label_ko: "해질녘", hour: 17.75 },
];

impl World {
    /// The lighting of the world (`WorldConfig::lighting`).
    pub fn lighting(&self) -> Lighting {
        self.cfg.lighting
    }

    /// Changes the lighting; nothing else in the world changes with it.
    pub fn set_lighting(&mut self, lighting: Lighting) {
        self.cfg.lighting = lighting;
    }

    /// Illuminance the head camera's exposure meters now, lux.
    pub fn scene_lux(&self) -> f32 {
        self.cfg.lighting.scene_lux()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Camera;
    use crate::world::{exposure_s, WorldConfig};

    #[test]
    fn flat_lighting_meters_the_old_ten_thousand_lux() {
        assert_eq!(Lighting::default(), Lighting::Flat);
        assert_eq!(Lighting::Flat.scene_lux(), SCENE_LUX);
        assert!(Lighting::Flat.daylight().is_none());
        let w = World::new(WorldConfig::default()).unwrap();
        assert_eq!(w.scene_lux(), 10_000.0);
        // The serialised default reads back, and an old config without the field is flat.
        let v = serde_json::to_value(Lighting::Sun { elevation_deg: 30.0, azimuth_deg: 200.0, overcast: 0.0 }).unwrap();
        assert_eq!(v["mode"], "sun");
        assert_eq!(serde_json::from_value::<Lighting>(v).unwrap(), Lighting::Sun { elevation_deg: 30.0, azimuth_deg: 200.0, overcast: 0.0 });
        assert_eq!(serde_json::from_value::<Lighting>(serde_json::json!({"mode": "sun", "elevation_deg": 10.0, "azimuth_deg": 90.0})).unwrap(), Lighting::Sun { elevation_deg: 10.0, azimuth_deg: 90.0, overcast: 0.0 });
    }

    #[test]
    fn the_sun_follows_the_noaa_series() {
        // Equinox noon: the sun stands at 90° − latitude in the south.
        let noon = solar_position(37.57, 0.0, 0.0, 80, 12.0);
        assert!((noon.elevation_deg - (90.0 - 37.57)).abs() < 1.5, "{noon:?}");
        assert!((noon.azimuth_deg - 180.0).abs() < 4.0, "{noon:?}");
        // Morning in the east, afternoon in the west, mirrored about solar noon
        // (the clock's noon shifted by the equation of time).
        let solar_noon = 12.0 - noon.eqtime_min / 60.0;
        let am = solar_position(37.57, 0.0, 0.0, 80, solar_noon - 3.0);
        let pm = solar_position(37.57, 0.0, 0.0, 80, solar_noon + 3.0);
        assert!(am.azimuth_deg > 90.0 && am.azimuth_deg < 180.0 && pm.azimuth_deg > 180.0 && pm.azimuth_deg < 270.0, "{am:?} {pm:?}");
        assert!((am.elevation_deg - pm.elevation_deg).abs() < 0.2 && (am.azimuth_deg + pm.azimuth_deg - 360.0).abs() < 0.5, "{am:?} {pm:?}");
        assert!(am.hour_angle_deg < 0.0 && pm.hour_angle_deg > 0.0);
        // The June solstice's declination is the tilt of the axis; December's its negative.
        assert!((solar_position(0.0, 0.0, 0.0, 172, 12.0).declination_deg - 23.44).abs() < 0.3);
        assert!((solar_position(0.0, 0.0, 0.0, 355, 12.0).declination_deg + 23.44).abs() < 0.3);
        // Night: below the horizon.
        assert!(solar_position(37.57, 0.0, 0.0, 80, 0.0).elevation_deg < -40.0);
        // A longitude east of the zone's meridian brings noon earlier by 4 min per degree.
        let east = solar_position(37.57, 7.5, 0.0, 80, 11.5);
        assert!((east.hour_angle_deg - noon.hour_angle_deg).abs() < 0.2, "{east:?}");
        // Refraction lifts the sun most at the horizon (about half a degree).
        assert!(refraction_deg(0.0) > 0.45 && refraction_deg(0.0) < 0.5 && refraction_deg(45.0) < 0.02 && refraction_deg(89.0) == 0.0);
    }

    #[test]
    fn a_higher_sun_gives_more_light_and_clouds_take_the_beam() {
        assert!((air_mass(90.0) - 1.0).abs() < 1e-3 && (air_mass(0.0) - 37.9).abs() < 0.5 && (air_mass(30.0) - 2.0).abs() < 0.02);
        let lux = |h: f32| Daylight::new(h, 180.0, 0.0).global_lx;
        let mut last = 0.0;
        for h in [-5.0, 0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0] {
            assert!(lux(h) > last, "{h}: {} after {last}", lux(h));
            last = lux(h);
        }
        // Computed from the model's own equations (research lighting.md §2): h = 30° about 53.6 klx, 60° about 102 klx.
        assert!((lux(30.0) - 53_600.0).abs() < 1_500.0 && (lux(60.0) - 101_900.0).abs() < 2_000.0, "{} {}", lux(30.0), lux(60.0));
        // Low sun: the sky gives most of the light.
        let low = Daylight::new(5.0, 90.0, 0.0);
        assert!(low.sky_lx / low.global_lx > 0.7);
        // Overcast: no beam, much less light.
        let grey = Daylight::new(45.0, 180.0, 1.0);
        assert_eq!(grey.direct_normal_lx, 0.0);
        assert!(grey.global_lx < 0.3 * lux(45.0));
        // Past civil twilight it is dark: the frame goes black.
        assert_eq!(Daylight::new(-7.0, 270.0, 0.0).global_lx, 0.0);
        assert_eq!(Daylight::new(-7.0, 270.0, 0.0).face_gain([0.0, 1.0, 0.0], true), [0.0; 3]);
        assert!(exposure_reach(100_000.0) == 1.0 && exposure_reach(5.0) < 0.5);
    }

    #[test]
    fn a_low_sun_is_red_and_the_sky_is_blue() {
        let (high, low) = (sun_colour(80.0), sun_colour(5.0));
        assert!(low[0] / low[2] > 2.0 * high[0] / high[2], "{low:?} {high:?}");
        assert!((sun_colour(90.0)[0] - 1.0).abs() < 1e-3 && (sun_colour(90.0)[2] - 1.0).abs() < 1e-3);
        let sky = sky_colour(0.0);
        assert!(sky[2] > sky[1] && sky[1] > sky[0]);
        assert!(sky_colour(1.0).iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // A sunlit top face renders as the flat mode's top face; one in shadow much darker.
        let d = Daylight::new(50.0, 180.0, 0.0);
        let lit = d.face_light([0.0, 1.0, 0.0], true);
        assert!((luma(lit) - 1.0).abs() < 1e-3);
        assert!(luma(d.face_light([0.0, 1.0, 0.0], false)) < 0.25);
    }

    #[test]
    fn azimuths_are_compass_bearings_on_the_map() {
        let east = sun_direction(0.0, 90.0);
        assert!((east[0] - 1.0).abs() < 1e-6 && east[2].abs() < 1e-6);
        let north = sun_direction(0.0, 0.0);
        assert!((north[2] + 1.0).abs() < 1e-6);
        let south_high = sun_direction(60.0, 180.0);
        assert!(south_high[2] > 0.0 && (south_high[1] - 60f32.to_radians().sin()).abs() < 1e-6);
        // A wall facing the sun gets more light than the one facing away.
        let d = Daylight::new(30.0, 90.0, 0.0);
        assert!(luma(d.face_light([1.0, 0.0, 0.0], true)) > 2.0 * luma(d.face_light([-1.0, 0.0, 0.0], true)));
    }

    #[test]
    fn presets_run_from_dawn_to_dusk() {
        let elev: Vec<f32> = PRESETS.iter().map(|p| match Lighting::preset(p.id).unwrap() {
            Lighting::Sun { elevation_deg, .. } => elevation_deg,
            Lighting::Flat => unreachable!(),
        }).collect();
        assert!(elev.iter().all(|&e| e > 0.0 && e < 60.0), "{elev:?}");
        assert!(elev[0] < elev[1] && elev[1] < elev[2] && elev[2] > elev[3] && elev[3] > elev[4], "{elev:?}");
        assert!(elev[0] < 6.0 && elev[4] < 8.0);
        assert_eq!(Lighting::preset("정오"), Lighting::preset("noon"));
        assert!(Lighting::preset("midnight").is_none());
    }

    #[test]
    fn a_low_sun_lengthens_the_exposure_and_the_blur() {
        let mut w = World::new(WorldConfig::default()).unwrap();
        let cam = Camera::default();
        let flat = w.head_view(&cam).exposure.exposure_s;
        assert!((flat - exposure_s(SCENE_LUX)).abs() < 1e-9);
        w.set_lighting(Lighting::preset("noon").unwrap());
        let noon = w.head_view(&cam).exposure.exposure_s;
        w.set_lighting(Lighting::Sun { elevation_deg: 1.0, azimuth_deg: 270.0, overcast: 1.0 });
        let dusk = w.head_view(&cam).exposure.exposure_s;
        assert!(noon < flat && dusk > 5.0 * noon, "flat {flat} noon {noon} dusk {dusk}");
        assert!(w.scene_lux() < 1_000.0);
    }
}
