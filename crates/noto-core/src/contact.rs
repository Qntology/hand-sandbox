//! Contact physics of the soft pads the robot touches the world with: the
//! finger pads on an object and the sole patches of a foot on the ground
//! (v1.0, `docs/research/tactile.md`).
//!
//! **Finger pads.** A pad is a soft Hertzian body:
//!
//! * the contact patch grows as `a = (3·N·R / 4E*)^(1/3)` (`material::contact_radius_m`;
//!   the measured soft-finger power law `a = c·N^γ` has γ ≤ 1/3, Hertz its
//!   upper end — Xydas & Kao 1999);
//! * the pad's friction falls with the normal force as finger pads' does
//!   (`μ ∝ N^−0.21`: Armstrong & Seo 2009 measured a 46% drop from 1 N to
//!   20 N), never below half the table value (Buchholz, Frederick & Armstrong
//!   1988: at most a 52.7% drop over 9–55 N), and changes with a wet or a
//!   dusty object ([`pad_mu`]);
//! * a pad resists a tangential force and a twist together inside the
//!   elliptic limit surface `(f_t/μN)² + (m_n/(e_n·μN))² ≤ 1`, `e_n = 3π/16·a`
//!   (Xydas & Kao 1999; Ciocarlie, Lackner & Allen 2007; Popov 2010), and slides
//!   along the surface normal there (maximum dissipation);
//! * below the limit the edge of the patch already slips (Cattaneo–Mindlin):
//!   the sticking share of the patch area is `(1 − q/q_max)^(2/3)`
//!   (Johnson 1985), the simulated tactile slip signal ([`stick_ratio`]);
//! * a load on an object held by several pads shares out by the pads'
//!   tangential stiffness (Mindlin: `8·G*·a`), each capped at its limit
//!   surface, the rest carried by the others; a twist the same way by the
//!   pads' torsional stiffness ([`share_load`]); an opposed pinch (a digit
//!   on each face) holds by the friction of both faces ([`PINCH_FACES`]),
//!   the robot's grasp from above by its fingers' face alone
//!   (`act::grasp::SQUEEZE_SURFACES`);
//! * before it slides a patch creeps along the face by the
//!   Cattaneo–Mindlin displacement ([`preslip_displacement_m`]);
//! * the grip loop feels the shear with a fingertip sensor at its frame
//!   rate and a two-frame confirmation, and grips for the load it feels
//!   ([`ramp_hold`]; `act::grasp` leads a planned load the same way).
//!
//! **Feet.** A sole is four compliant patches (heel, midfoot, metatarsal
//! heads, toes) on a rigid rear foot and a toe segment hinged to it at the
//! metatarsal line by a passive spring–damper toe joint (HRP-2TJ's
//! `Kp`/`Kd`, Sellaouti et al. 2006, carried to the robot by its weight and
//! CoM height). While the centre of pressure stays behind the hinge the foot
//! is flat and its pressure is that of a rigid plate on an elastic
//! foundation (the soft toe takes only what its spring lets it); once the
//! centre of pressure passes the hinge the rear foot rolls up about it (heel
//! off), the toes carry the load and the toe joint dorsiflexes against its
//! spring ([`sole_contact`]). Shear spreads over the patches by their
//! tangential stiffness until a lightly loaded patch slips (partial slip at
//! the sole), each patch with its own friction cone.
//!
//! These are engine inputs (Group A); the tactile benchmark scores the
//! results against other sources (`docs/research/tactile.md` §5).

use serde::Serialize;

use crate::act::material;
use crate::balance::Body;

// ----- finger pads -----

/// Exponent of the pad friction's fall with normal force, `μ ∝ N^n`: a 46%
/// drop of the finger pad's mean friction from 1 N to 20 N gives
/// `n = ln 0.54 / ln 20` (Armstrong & Seo 2009, Ergonomics 52:609) [derived].
pub const PAD_MU_FORCE_EXP: f32 = -0.21;
/// Normal force at which the material table (`material::pad_on`) holds, N
/// (est: the middle of the 1/10/20 N finger-pad tests of Armstrong & Seo 2009).
pub const PAD_MU_REF_N: f32 = 10.0;
/// The force law is applied between these forces and held outside, N: the
/// tested range from 1 N (Armstrong & Seo 2009) to 55 N (Buchholz et al. 1988).
pub const PAD_MU_N_RANGE: [f32; 2] = [1.0, 55.0];
/// The pad's friction never falls below this share of the table value: the
/// largest drop over the pinch range was 52.7% (Buchholz, Frederick &
/// Armstrong 1988, Ergonomics 31:317, Table 3).
pub const PAD_MU_FLOOR: f32 = 0.5;
/// Finger-to-finger scatter of the pad friction on one object (coefficient
/// of variation): repeat tests on one pair scatter 1–7% (`material`'s
/// `PAD_MU_COV` note, Elkington 2024) — the middle [derived].
pub const FINGER_MU_COV: f32 = 0.04;
/// A wet object: smooth vinyl rubber against moist skin keeps 0.49/0.56 of its
/// dry friction (Buchholz et al. 1988 Table 2) — the closest non-porous,
/// rubber-like pair to a silicone pad [substituted].
pub const WET_PAD_FACTOR: f32 = 0.875;
/// A dusty object: the material table lowers clean rubber's 0.60 to 0.45 for
/// coal dust (`material::pad_on`), the same share for any dust [derived from
/// that estimate].
pub const DUSTY_PAD_FACTOR: f32 = 0.75;
/// The stick ratio below which the robot's tactile sensing calls an
/// incipient slip (est: the pack's example threshold 0.5 on the
/// Cattaneo–Mindlin stick ratio).
pub const STICK_ALARM: f32 = 0.5;
/// Frame rate of the fingertip tactile sensor, Hz: a DIGIT-class vision
/// tactile sensor, 640×480 at 60 fps (Lambeta et al. 2020, IEEE RA-L, Table I).
pub const TACTILE_FPS: f32 = 60.0;
/// Frames a contact change must show before the controller acts on it (est:
/// two consecutive frames, so one noisy frame does not trigger).
pub const DETECT_FRAMES: f32 = 2.0;
/// Light seating force each finger presses with before the load comes on, N
/// (est: of the order of the fingertip's closing impact, `act::grasp`).
pub const PRELOAD_N: f32 = 1.0;

/// How long the tactile loop takes to see a contact change, seconds.
pub fn detect_delay_s() -> f32 {
    DETECT_FRAMES / TACTILE_FPS
}

/// The surface of the object under the pads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PadSurface {
    #[default]
    Dry,
    Wet,
    Dusty,
}

/// A pad's friction at normal force `n` (N) on an object whose table value
/// (at [`PAD_MU_REF_N`]) is `mu_table`.
pub fn pad_mu(mu_table: f32, n: f32, surface: PadSurface) -> f32 {
    let n = n.clamp(PAD_MU_N_RANGE[0], PAD_MU_N_RANGE[1]);
    let law = (n / PAD_MU_REF_N).powf(PAD_MU_FORCE_EXP).max(PAD_MU_FLOOR);
    let s = match surface {
        PadSurface::Dry => 1.0,
        PadSurface::Wet => WET_PAD_FACTOR,
        PadSurface::Dusty => DUSTY_PAD_FACTOR,
    };
    mu_table * law * s
}

/// Radius of a pad's contact patch at normal force `n`, metres (Hertz).
pub fn patch_radius_m(n: f32) -> f32 {
    material::contact_radius_m(n)
}

/// The torsion radius of a Hertzian patch: the twist it resists is
/// `e_n · μ · N`, `e_n = 3π/16 · a` (Popov 2010; Ciocarlie et al. 2007 eq. 10).
pub fn torsion_radius_m(n: f32) -> f32 {
    3.0 * std::f32::consts::PI / 16.0 * patch_radius_m(n)
}

/// Area of a pad's contact patch at normal force `n`, m² (`π·a²`).
pub fn patch_area_m2(n: f32) -> f32 {
    std::f32::consts::PI * patch_radius_m(n).powi(2)
}

/// The tangential contact modulus of a soft pad on a rigid face, Pa
/// (Mindlin: `G* = [(2 − ν₁)/G₁ + (2 − ν₂)/G₂]⁻¹`, the rigid face adding
/// nothing; `G = E / (2·(1 + ν))`, so `G* = E / (2·(2 − ν)(1 + ν))`; Johnson 1985 §7.2).
pub fn tangential_modulus_pa() -> f32 {
    material::PAD_E_PA / (2.0 * (2.0 - material::PAD_NU) * (1.0 + material::PAD_NU))
}

/// Tangential (Mindlin) stiffness of a pad patch before any slip, N/m: `8·G*·a`.
pub fn tangential_stiffness(n: f32) -> f32 {
    8.0 * tangential_modulus_pa() * patch_radius_m(n.max(0.05))
}

/// How far a pad's patch has moved along the face under a tangential load
/// `q` before it slides, metres (Cattaneo–Mindlin with the slip annulus
/// growing in from the edge: `δ = 3μN/(16·G*·a)·[1 − (1 − q/μN)^(2/3)]`,
/// Johnson 1985 §7.2); at `q = μN` the whole patch slides.
pub fn preslip_displacement_m(q: f32, mu: f32, n: f32) -> f32 {
    let f = (mu * n).max(1e-9);
    let a = patch_radius_m(n).max(1e-6);
    3.0 * f / (16.0 * tangential_modulus_pa() * a) * (1.0 - (1.0 - (q.abs() / f).min(1.0)).powf(2.0 / 3.0))
}

/// How much of a pad's elliptic limit surface a tangential force `ft` and a
/// twist `m` use together (1 on the surface: the pad slides).
pub fn limit_use(ft: f32, m: f32, mu: f32, n: f32) -> f32 {
    let f_max = (mu * n).max(1e-9);
    let m_max = (torsion_radius_m(n) * mu * n).max(1e-12);
    ((ft / f_max).powi(2) + (m / m_max).powi(2)).sqrt()
}

/// The tangential force a pad still holds while it resists a twist `m`
/// (the limit surface's tangential extent at that twist), N.
pub fn tangential_capacity(m: f32, mu: f32, n: f32) -> f32 {
    let f_max = (mu * n).max(0.0);
    let m_max = (torsion_radius_m(n) * mu * n).max(1e-12);
    f_max * (1.0 - (m / m_max).powi(2)).max(0.0).sqrt()
}

/// The direction a pad slides in on its limit surface (maximum dissipation:
/// the slip velocity `(v, ω)` lies along the surface's normal,
/// `∝ (f_t/f_max², m/m_max²)`), as the unit vector of `(v, e_n·ω)` — the
/// sliding speed and the speed the twist turns the patch at its torsion
/// radius `e_n`, both velocities, so the shares do not depend on the units.
/// Since `m_max = e_n·f_max` that is the normal in the scaled space,
/// `∝ (f_t/f_max, m/m_max)`.
pub fn slip_direction(ft: f32, m: f32, mu: f32, n: f32) -> [f32; 2] {
    let f_max = (mu * n).max(1e-9);
    let m_max = (torsion_radius_m(n) * mu * n).max(1e-12);
    let g = [ft / f_max, m / m_max];
    let l = (g[0] * g[0] + g[1] * g[1]).sqrt();
    if l > 1e-12 {
        [g[0] / l, g[1] / l]
    } else {
        [1.0, 0.0]
    }
}

/// The share of a pad's patch that still sticks under a tangential load `q`
/// when it slides at `q_max` (Cattaneo–Mindlin: the stick radius is
/// `a·(1 − q/q_max)^(1/3)`, its area the square of that).
pub fn stick_ratio(q: f32, q_max: f32) -> f32 {
    if q_max <= 1e-9 {
        return 0.0;
    }
    (1.0 - (q.abs() / q_max).min(1.0)).powf(2.0 / 3.0)
}

/// What the tactile signal tells the controller about a pad's friction: the
/// friction that puts this load at this stick ratio (the inverse of
/// [`stick_ratio`], `μ = q / (N·(1 − φ^(3/2)))`).
pub fn mu_from_stick(q: f32, n: f32, stick: f32) -> f32 {
    let used = 1.0 - stick.clamp(0.0, 1.0).powf(1.5);
    if used <= 1e-4 || n <= 1e-6 {
        f32::INFINITY
    } else {
        q.abs() / (n * used)
    }
}

/// Where a pad is between holding and sliding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SlipState {
    #[default]
    Stick,
    /// The patch's edge slips (stick ratio below [`STICK_ALARM`]); the pad still holds.
    Incipient,
    /// The pad slides (on its limit surface).
    Slip,
}

/// One pad's load from a shared load (`share_load`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PadLoad {
    /// Normal force, N.
    pub n: f32,
    /// Friction at this force.
    pub mu: f32,
    /// Tangential force the pad carries, N, and the twist, N·m.
    pub ft: f32,
    pub torsion_nm: f32,
    /// Limit-surface use (1: sliding), stick ratio, state.
    pub cone_use: f32,
    pub stick: f32,
    pub state: SlipState,
    /// Patch radius, metres.
    pub radius_m: f32,
    /// Which way it slides once it slides, as the unit vector of (the
    /// sliding speed along its tangential force, the twist's speed at its
    /// torsion radius about the face normal) ([`slip_direction`]).
    pub slip_dir: [f32; 2],
}

/// The pads' answer to a load: each pad's share, and whether the hold fails.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Shared {
    pub pads: Vec<PadLoad>,
    /// The tangential load the pads together can hold with this twist, N.
    pub capacity_n: f32,
    /// The pads together cannot hold the load, or the twist: the object
    /// slides out, or turns.
    pub gross_slip: bool,
    /// The smallest stick ratio of a loaded pad.
    pub stick_min: f32,
}

/// Shares a tangential load `load_n` and a twist `twist_nm` (about the
/// grasp axis) over pads pressed with `normals` at frictions `mus`: each
/// pad takes the twist by its torsional stiffness (`∝ a³`) and the load by
/// its tangential stiffness (`∝ a`) while it sticks; a pad at its limit
/// surface carries its capacity and the rest goes to the others. A twist
/// beyond every pad's torsion limit together, like a load beyond the pads'
/// capacity, is a gross slip. Each pad's stick ratio follows its use of the
/// limit surface (the load and the twist together).
pub fn share_load(normals: &[f32], mus: &[f32], load_n: f32, twist_nm: f32) -> Shared {
    let k = normals.len();
    let mut pads: Vec<PadLoad> = (0..k).map(|i| PadLoad { n: normals[i].max(0.0), mu: mus[i], radius_m: patch_radius_m(normals[i].max(0.0)), ..PadLoad::default() }).collect();
    let on: Vec<usize> = (0..k).filter(|&i| pads[i].n > 1e-6).collect();
    if on.is_empty() {
        return Shared { pads, capacity_n: 0.0, gross_slip: load_n > 1e-6 || twist_nm.abs() > 1e-9, stick_min: 0.0 };
    }
    // The twist by torsional stiffness (∝ a³), each pad no more than its own
    // torsion limit; what a pad at its limit cannot take goes round to the others.
    let m_max: Vec<f32> = (0..k).map(|i| torsion_radius_m(pads[i].n) * pads[i].mu * pads[i].n).collect();
    let mut t_free: Vec<bool> = (0..k).map(|i| pads[i].n > 1e-6).collect();
    let mut t_left = twist_nm.abs();
    for _ in 0..k + 1 {
        let a3: f32 = (0..k).filter(|&i| t_free[i]).map(|i| pads[i].radius_m.powi(3)).sum();
        if a3 <= 1e-15 || t_left <= 1e-12 {
            break;
        }
        let share: Vec<f32> = (0..k).map(|i| if t_free[i] { t_left * pads[i].radius_m.powi(3) / a3 } else { 0.0 }).collect();
        let capped: Vec<usize> = (0..k).filter(|&i| t_free[i] && pads[i].torsion_nm + share[i] >= m_max[i]).collect();
        if capped.is_empty() {
            for i in 0..k {
                pads[i].torsion_nm += share[i];
            }
            t_left = 0.0;
            break;
        }
        for &i in &capped {
            pads[i].torsion_nm = m_max[i];
            t_free[i] = false;
        }
        t_left = (twist_nm.abs() - pads.iter().map(|p| p.torsion_nm).sum::<f32>()).max(0.0);
    }
    let twist_slip = t_left > 1e-9;
    for p in pads.iter_mut() {
        p.torsion_nm = p.torsion_nm.copysign(twist_nm);
    }
    // What each pad holds tangentially beside its twist.
    let cap: Vec<f32> = (0..k).map(|i| if pads[i].n > 1e-6 { tangential_capacity(pads[i].torsion_nm, pads[i].mu, pads[i].n) } else { 0.0 }).collect();
    let capacity_n: f32 = cap.iter().sum();
    // Stick by stiffness, capped; the overflow goes round again.
    let stiff: Vec<f32> = (0..k).map(|i| if pads[i].n > 1e-6 { tangential_stiffness(pads[i].n) } else { 0.0 }).collect();
    let mut free: Vec<bool> = (0..k).map(|i| pads[i].n > 1e-6).collect();
    let mut left = load_n.max(0.0);
    for _ in 0..k + 1 {
        let ks: f32 = (0..k).filter(|&i| free[i]).map(|i| stiff[i]).sum();
        if ks <= 1e-12 || left <= 1e-9 {
            break;
        }
        let mut capped = false;
        for i in 0..k {
            if free[i] && left * stiff[i] / ks >= cap[i] {
                capped = true;
            }
        }
        if !capped {
            for i in 0..k {
                if free[i] {
                    pads[i].ft += left * stiff[i] / ks;
                }
            }
            left = 0.0;
            break;
        }
        for i in 0..k {
            if free[i] && left * stiff[i] / ks >= cap[i] {
                pads[i].ft = cap[i];
                free[i] = false;
            }
        }
        left = (load_n.max(0.0) - pads.iter().map(|p| p.ft).sum::<f32>()).max(0.0);
    }
    let gross_slip = (left > 1e-6 && load_n > capacity_n + 1e-6) || twist_slip;
    let mut stick_min = 1.0f32;
    for i in 0..k {
        let p = &mut pads[i];
        if p.n <= 1e-6 {
            continue;
        }
        p.cone_use = limit_use(p.ft, p.torsion_nm, p.mu, p.n);
        p.slip_dir = slip_direction(p.ft, p.torsion_nm, p.mu, p.n);
        // The share that sticks under the load and the twist together: the Cattaneo–Mindlin law on
        // the limit surface's use. Exact for a tangential load alone (the use is then `ft/(μN)`); with
        // a twist the patch's edge slips by the twist too (est: the twist's slip annulus taken on the
        // same law), so a pad at its torsion limit sticks nowhere, as it slides.
        p.stick = stick_ratio(p.cone_use, 1.0);
        p.state = if gross_slip || p.cone_use >= 0.999 {
            SlipState::Slip
        } else if p.stick < STICK_ALARM {
            SlipState::Incipient
        } else {
            SlipState::Stick
        };
        if gross_slip {
            p.stick = 0.0;
        }
        stick_min = stick_min.min(p.stick);
    }
    Shared { pads, capacity_n, gross_slip, stick_min }
}

/// One finger's contact on the held object at one instant, in the hand frame
/// (`sim`: z along the fingers, y toward the palm side), for the views.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct FingerContact {
    pub finger: u8,
    /// The phalanx that touches (0 the proximal one).
    pub phalanx: u8,
    /// The contact point on the object's surface, and the surface normal there (out of the object).
    pub point: [f32; 3],
    pub normal: [f32; 3],
    /// Normal force the pad presses with, N.
    pub fn_n: f32,
    /// Tangential force the pad holds the object with (on the object), N, and its size.
    pub ft: [f32; 3],
    pub ft_n: f32,
    pub torsion_nm: f32,
    /// The pad's friction at this force (drawn), its limit-surface use and stick ratio.
    pub mu: f32,
    pub cone_use: f32,
    pub stick: f32,
    pub state: SlipState,
    /// Contact patch radius, metres.
    pub radius_m: f32,
    /// Which way the pad slides once it slides: the unit vector of (the
    /// sliding speed along `ft`, the twist's speed about `normal` at the
    /// torsion radius), its limit surface's normal (maximum dissipation).
    pub slip_dir: [f32; 2],
}

/// Up to five finger contacts, serialised as a list of the ones in use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FingerContacts {
    pub n: u8,
    pub c: [FingerContact; 5],
}

impl FingerContacts {
    pub fn push(&mut self, c: FingerContact) {
        if (self.n as usize) < self.c.len() {
            self.c[self.n as usize] = c;
            self.n += 1;
        }
    }

    pub fn as_slice(&self) -> &[FingerContact] {
        &self.c[..self.n as usize]
    }
}

impl Serialize for FingerContacts {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(self.n as usize))?;
        for c in self.as_slice() {
            seq.serialize_element(c)?;
        }
        seq.end()
    }
}

// ----- the grip loop under an outside load -----

/// Change of shear the fingertip sensor must show before the grip loop acts
/// on it, N (est: a tenth of a newton; the force resolutions quoted for
/// three-axis taxel skins and fluid-filled fingertips could not be
/// confirmed, `docs/research/tactile.md` §1).
pub const SHEAR_DETECT_N: f32 = 0.1;

/// A pinch holding an object still while an outside load pulls it along the
/// squeezed faces ([`ramp_hold`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RampHold {
    /// From the load's onset to the grip's first rise, seconds (`None`: the grip never moved).
    pub latency_s: Option<f32>,
    /// At the end (the plateau held, or the slide): the grip, the load, and
    /// the safety margin `1 − LF / capacity` (the share of the grip above the
    /// grip that would just slip).
    pub grip_n: f32,
    pub load_n: f32,
    pub safety_margin: f32,
    /// The smallest stick ratio a pad reached, and whether the object slid out.
    pub stick_min: f32,
    pub slipped: bool,
}

/// The longest ramp and the longest hold [`ramp_hold`] plays, seconds: a
/// slower ramp is cut off there (est: far beyond any pull the bench asks
/// about, so a rate near zero cannot run for ever).
pub const RAMP_MAX_S: f32 = 60.0;

/// The tactile grip loop of an opposed pinch — `pads` pads on each of the
/// two squeezed faces ([`PINCH_FACES`]), all pressed alike — holding an
/// object (pad table friction `mu_table` on `surface`) with the grip
/// `grip0_n` (the normal force on each face), when an outside load ramps up
/// at `rate_n_s` to `plateau_n` and stays for `hold_s`:
///
/// * the fingertip sensor reports the shear of the frame it last took
///   ([`TACTILE_FPS`]), [`detect_delay_s`] late;
/// * once the reported load has risen [`SHEAR_DETECT_N`] the controller
///   commands `GF = grip0 + S·LF/(2·μ_est)` from every frame on (the load per
///   squeezed face, `safety` = S);
/// * the grip follows the command as a first-order lag whose 10–90% rise
///   takes `rise_s` (2.2 time constants);
/// * every millisecond the pads answer the load (`share_load`, `pad_mu`):
///   the stick ratio, and a slide once the load passes what they hold.
///
/// The ramp and the hold are each played for at most [`RAMP_MAX_S`].
pub fn ramp_hold(mu_table: f32, surface: PadSurface, mu_est: f32, safety: f32, grip0_n: f32, pads: usize, rate_n_s: f32, plateau_n: f32, hold_s: f32, rise_s: f32) -> RampHold {
    let dt = 1e-3f32;
    let frame = 1.0 / TACTILE_FPS;
    let delay = detect_delay_s();
    let tau = (rise_s / 2.2).max(1e-3);
    let rate = rate_n_s.max(1e-6);
    let load_at = |t: f32| (rate * t).clamp(0.0, plateau_n.max(0.0));
    let k = pads.max(1);
    let steps = (((plateau_n.max(0.0) / rate).min(RAMP_MAX_S) + hold_s.clamp(0.0, RAMP_MAX_S)) / dt).ceil() as usize;
    let mut grip = grip0_n.max(0.0);
    let mut latency_s = None;
    let mut stick_min = 1.0f32;
    let mut out = RampHold { latency_s: None, grip_n: grip, load_n: 0.0, safety_margin: 1.0, stick_min: 1.0, slipped: false };
    let mut acting = false;
    for i in 1..=steps {
        let t = i as f32 * dt;
        // What the sensor reports now: the load of its last frame before `t − delay`.
        let seen = if t >= delay { load_at(((t - delay) / frame).floor() * frame) } else { 0.0 };
        acting = acting || seen >= SHEAR_DETECT_N;
        let cmd = if acting { grip0_n + safety * seen / (PINCH_FACES * mu_est.max(0.05)) } else { grip0_n };
        grip += (cmd - grip) * (1.0 - (-dt / tau).exp());
        if latency_s.is_none() && grip > grip0_n + 1e-4 {
            latency_s = Some(t);
        }
        let load = load_at(t);
        let n = grip / k as f32;
        let normals = vec![n; 2 * k];
        let mus: Vec<f32> = normals.iter().map(|&ni| pad_mu(mu_table, ni, surface)).collect();
        let st = share_load(&normals, &mus, load, 0.0);
        if load > 1e-6 {
            stick_min = stick_min.min(st.stick_min);
        }
        out = RampHold { latency_s, grip_n: grip, load_n: load, safety_margin: 1.0 - load / st.capacity_n.max(1e-6), stick_min, slipped: st.gross_slip };
        if st.gross_slip {
            break;
        }
    }
    out
}

/// The faces an opposed pinch holds an object by: a digit presses each
/// face with the grip force (each the other's reaction, Newton's third
/// law), so both faces' friction carries the load — Amontons' law with both
/// surfaces counted, the convention of the pad-friction measurements the
/// friction law rests on (the load per surface over the pinch force;
/// Buchholz, Frederick & Armstrong 1988). Only for a grip that has a pad on
/// each face ([`ramp_hold`]): the robot's hand closing on an object from
/// above presses one face with its fingers and has no opposing digit
/// (`sim::top_grasp`), so it counts one face (`act::grasp::SQUEEZE_SURFACES`).
pub const PINCH_FACES: f32 = 2.0;

// ----- feet -----

/// Where the sole's patches end, as shares of the foot length from the
/// heel's back edge: heel, midfoot, metatarsal heads, toes (est: the usual
/// pedobarographic masks — heel the rear 30%, the toes in front of the
/// metatarsal line).
pub const SOLE_BOUNDS_FL: [f32; 5] = [0.0, 0.30, 0.55, 0.75, 1.0];
/// The toe joint (the metatarsophalangeal line) lies at the front of the
/// metatarsal patch (est, as the masks).
pub const TOE_HINGE_FL: f32 = 0.75;
/// The ankle lies this share of the sole in front of its back edge (as `kin`).
pub const ANKLE_FL: f32 = 0.25;
/// HRP-2TJ's passive toe joint (54 kg, CoM 0.814 m): stiffness 10 N·m/rad
/// and damping 5 N·m·s/rad, the set tuned for a smooth heel rise
/// (Sellaouti, Stasse, Kajita, Yokoi & Kheddar 2006, IROS, pp. 4909–4914).
pub const HRP2TJ_TOE_KP: f32 = 10.0;
pub const HRP2TJ_TOE_KD: f32 = 5.0;
pub const HRP2TJ_MASS_KG: f32 = 54.0;
pub const HRP2TJ_COM_Z_M: f32 = 0.814;
/// The toe joint's dorsiflexion stop, degrees (est: a little under a
/// person's toe-off range; no robot value was found).
pub const TOE_DORSI_MAX_DEG: f32 = 60.0;
/// Torsion a uniformly loaded square patch of side `s` resists is
/// `0.383·s·μ·N` (`act::push`'s footprint constant, `s(√2 + ln(1+√2))/6`).
pub const SQUARE_TORSION: f32 = 0.383;
/// How far the sole's rubber compresses at most, metres (est: half of an
/// 8 mm rubber sole over the foot plate). A rigid plate whose resultant
/// sits at its very edge would press a line of infinite pressure; the
/// rubber's reach bounds the edge pressure at `k·δ_max` instead, so the
/// loaded band at an edge is never shorter than `2F/(k·δ_max)` and the
/// centre of pressure stays a third of that inside it.
pub const SOLE_COMPRESS_M: f32 = 0.004;
/// Names of the sole patches, heel first.
pub const SOLE_PATCHES: [&str; 4] = ["heel", "midfoot", "metatarsal", "toe"];
pub const SOLE_PATCHES_KO: [&str; 4] = ["뒤꿈치", "중족", "앞발 볼", "발가락"];

/// A robot's sole: its size, the foundation stiffness under it and its toe joint.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SoleModel {
    pub foot_len_m: f32,
    pub foot_width_m: f32,
    /// Foundation stiffness per metre of sole length, N/m per m: the sole's
    /// heel stiffness (`act::gait::SOLE_STIFFNESS`) spread over the heel.
    pub k_per_m: f32,
    /// Toe joint spring and damper.
    pub toe_kp: f32,
    pub toe_kd: f32,
    /// How much of a rigid toe's flat-foot load the sprung toe takes (`Kp / (Kp + K_ground)`).
    pub toe_share: f32,
}

impl SoleModel {
    /// The sole of a legged body (sizes from its leg preset). The toe joint
    /// is HRP-2TJ's, scaled by the pendulum's `m·g·z_CoM` for the stiffness
    /// and by `√(z_CoM)` for the time constant `Kd/Kp` (dynamic similarity).
    pub fn of(body: &Body) -> SoleModel {
        let fl = body.leg.foot_len_m;
        let fw = body.leg.foot_width_m;
        let k_per_m = crate::act::gait::SOLE_STIFFNESS / (SOLE_BOUNDS_FL[1] * fl).max(1e-3);
        let z = body.com_height_m().max(0.1);
        let kp = HRP2TJ_TOE_KP * (body.mass_kg() * z) / (HRP2TJ_MASS_KG * HRP2TJ_COM_Z_M);
        let tau = HRP2TJ_TOE_KD / HRP2TJ_TOE_KP * (z / HRP2TJ_COM_Z_M).sqrt();
        let l_toe = (1.0 - TOE_HINGE_FL) * fl;
        let k_rot = k_per_m * l_toe.powi(3) / 3.0;
        SoleModel { foot_len_m: fl, foot_width_m: fw, k_per_m, toe_kp: kp, toe_kd: kp * tau, toe_share: kp / (kp + k_rot) }
    }

    /// The patches' ends along the sole from the heel's back edge, metres.
    fn bounds(&self) -> [f32; 5] {
        SOLE_BOUNDS_FL.map(|f| f * self.foot_len_m)
    }

    /// The least distance of the centre of pressure from an edge the sole
    /// carries a normal force `f` (N) at: the third of the shortest band its
    /// rubber holds `f` over ([`SOLE_COMPRESS_M`]), metres.
    pub fn edge_inset_m(&self, f: f32) -> f32 {
        2.0 * f.max(0.0) / (3.0 * self.k_per_m.max(1.0) * SOLE_COMPRESS_M)
    }

    /// The toe hinge from the heel's back edge, metres.
    pub fn hinge_m(&self) -> f32 {
        TOE_HINGE_FL * self.foot_len_m
    }

    /// The toe angle the spring settles at under a moment `m` about the hinge, degrees.
    pub fn static_toe_deg(&self, m: f32) -> f32 {
        (m.max(0.0) / self.toe_kp.max(1e-3)).to_degrees().min(TOE_DORSI_MAX_DEG)
    }

    /// One step of the toe joint's spring–damper under the moment `m`: the angle after `dt`, degrees.
    pub fn toe_step_deg(&self, deg: f32, m: f32, dt: f32) -> f32 {
        if m <= 0.0 {
            // The centre of pressure is back behind the hinge: the heel is down again.
            return 0.0;
        }
        let th = deg.to_radians();
        let tau = (self.toe_kd / self.toe_kp.max(1e-3)).max(1e-3);
        let target = m / self.toe_kp.max(1e-3);
        // Exact step of `Kd·θ̇ + Kp·θ = M` with M held over dt.
        let next = target + (th - target) * (-dt / tau).exp();
        next.to_degrees().clamp(0.0, TOE_DORSI_MAX_DEG)
    }
}

/// One sole patch at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SolePatch {
    pub name: &'static str,
    pub label_ko: &'static str,
    /// The patch along the foot in the base frame (x), metres.
    pub x0: f32,
    pub x1: f32,
    /// Normal force, shear (x, y) and twist it carries.
    pub normal_n: f32,
    pub shear: [f32; 2],
    pub torsion_nm: f32,
    /// Highest pressure on it, kPa.
    pub peak_kpa: f32,
    /// Its friction cone (with the twist) used, and whether it slides.
    pub cone_use: f32,
    pub slipping: bool,
}

/// A foot's sole at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SoleContact {
    pub patches: [SolePatch; 4],
    /// The rear foot has rolled up about the toe hinge (heel off).
    pub rocker: bool,
    /// The foot is held flat with its centre of pressure ahead of the toe
    /// hinge: the toe joint holds the moment at zero angle (a passive toe
    /// would let the heel rise here; a pose that does not track the toe).
    pub toe_held: bool,
    /// Toe dorsiflexion (the rear foot's pitch over the toes in a rocker), degrees, and the hinge's moment, N·m.
    pub toe_deg: f32,
    pub toe_torque_nm: f32,
    /// Centre of pressure along x, and the loaded stretch of the sole `[x0, x1]`, base frame.
    pub cop_x: f32,
    pub loaded: [f32; 2],
    /// Friction under the foot, the whole foot's cone use and the largest patch's.
    pub mu: f32,
    pub cone_use: f32,
    pub cone_use_max: f32,
    /// Share of the loaded sole that still sticks.
    pub stick_share: f32,
}

/// `∫ max(0, a + b·u) du` and `∫ u·max(0, a + b·u) du` over `[u0, u1]`.
fn clipped_line(a: f32, b: f32, u0: f32, u1: f32) -> (f32, f32) {
    let (v0, v1) = if b.abs() < 1e-12 {
        if a > 0.0 {
            (u0, u1)
        } else {
            return (0.0, 0.0);
        }
    } else {
        let r = -a / b;
        if b > 0.0 {
            (u0.max(r), u1)
        } else {
            (u0, u1.min(r))
        }
    };
    if v1 <= v0 {
        return (0.0, 0.0);
    }
    let f = a * (v1 - v0) + 0.5 * b * (v1 * v1 - v0 * v0);
    let m = 0.5 * a * (v1 * v1 - v0 * v0) + b * (v1 * v1 * v1 - v0 * v0 * v0) / 3.0;
    (f, m)
}

/// A rigid plate on an elastic foundation of relative stiffness `w[i]` over
/// the stretches `[bounds[i], bounds[i+1]]`: the pressure line whose
/// resultant lies at `cop`. The family of lines `cos ψ + sin ψ·ξ` (ξ the
/// position scaled to `[−1, 1]`, clipped at zero where the plate lifts off)
/// moves its resultant from the back edge (ψ → −3π/4) to the front edge
/// (ψ → 3π/4); the angle is found by bisection. Returns `(a, b)` of the
/// line `a + b·u` (u in the bounds' coordinate) scaled so that the
/// stiffness-weighted integral is 1.
fn plate_line(bounds: &[f32], w: &[f32], cop: f32) -> (f32, f32) {
    let (u0, u1) = (bounds[0], bounds[bounds.len() - 1]);
    let h = 0.5 * (u1 - u0).max(1e-6);
    let mid_u = 0.5 * (u0 + u1);
    let at = |psi: f32| {
        let (s, c) = psi.sin_cos();
        // cos ψ + sin ψ · (u − mid) / h
        let (a, b) = (c - s * mid_u / h, s / h);
        let (mut f, mut m) = (0.0, 0.0);
        for i in 0..w.len() {
            let (fi, mi) = clipped_line(a, b, bounds[i], bounds[i + 1]);
            f += w[i] * fi;
            m += w[i] * mi;
        }
        (a, b, f, m)
    };
    let target = cop.clamp(u0 + 1e-4 * h, u1 - 1e-4 * h);
    let lim = 0.75 * std::f32::consts::PI - 1e-4;
    let (mut lo, mut hi) = (-lim, lim);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let (_, _, f, m) = at(mid);
        let c = if f > 1e-12 { m / f } else if mid < 0.0 { u0 } else { u1 };
        if c < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (a, b, f, _) = at(0.5 * (lo + hi));
    let f = f.max(1e-12);
    (a / f, b / f)
}

/// The sole of a stance foot under its share of the ground reaction: the
/// normal force `normal_n`, the shear `shear` (x, y) and the twist
/// `torsion_nm` about the vertical, its centre of pressure `cop_x` (base
/// frame, along the foot) and the ankle at `ankle_x`; `toe_deg` the toe's
/// dorsiflexion the pose has (0: the foot is flat; above 0 the rear foot has
/// rolled up about the toe hinge), `mu` the ground's friction under it (0 or
/// less: not known, the shear is shared by stiffness without a friction cap).
pub fn sole_contact(sole: &SoleModel, ankle_x: f32, normal_n: f32, shear: [f32; 2], torsion_nm: f32, cop_x: f32, toe_deg: f32, mu: f32) -> SoleContact {
    let b = sole.bounds();
    let x_back = ankle_x - ANKLE_FL * sole.foot_len_m;
    let mut out = SoleContact { mu, ..SoleContact::default() };
    for i in 0..4 {
        out.patches[i] = SolePatch { name: SOLE_PATCHES[i], label_ko: SOLE_PATCHES_KO[i], x0: x_back + b[i], x1: x_back + b[i + 1], ..SolePatch::default() };
    }
    let f = normal_n.max(0.0);
    if f <= 1e-6 {
        out.cop_x = cop_x;
        return out;
    }
    let hinge = sole.hinge_m();
    let rocker = toe_deg > 1e-3;
    let l_toe = b[4] - b[3];
    // On the rocker the load is ahead of the metatarsal band only. At an
    // edge the rubber's reach keeps the centre of pressure a little inside.
    let inset = sole.edge_inset_m(f);
    let u_c = if rocker {
        (cop_x - x_back).clamp(hinge, sole.foot_len_m - inset.min(l_toe / 3.0))
    } else {
        let e = inset.min(sole.foot_len_m / 3.0);
        (cop_x - x_back).clamp(e, sole.foot_len_m - e)
    };
    let width = sole.foot_width_m.max(1e-3);
    let mut force = [0.0f32; 4];
    let mut peak = [0.0f32; 4];
    let mut loaded_len = [0.0f32; 4];
    let mut loaded = [b[4], b[0]];
    if !rocker {
        // Flat: a rigid plate on the foundation, the sprung toe softer by its
        // share; ahead of the hinge the toe is held at zero (the whole sole one plate).
        out.toe_held = u_c > hinge;
        let w = [1.0, 1.0, 1.0, if out.toe_held { 1.0 } else { sole.toe_share }];
        let (a, bb) = plate_line(&b, &w, u_c);
        for i in 0..4 {
            let (fi, _) = clipped_line(a, bb, b[i], b[i + 1]);
            force[i] = f * w[i] * fi;
            let d0 = (a + bb * b[i]).max(0.0);
            let d1 = (a + bb * b[i + 1]).max(0.0);
            // Peak pressure: the line's highest point over the patch (force per unit length over the width).
            peak[i] = f * w[i] * d0.max(d1) / width / 1000.0;
            let (v0, v1) = if bb.abs() < 1e-12 { (b[i], b[i + 1]) } else if bb > 0.0 { (b[i].max(-a / bb), b[i + 1]) } else { (b[i], b[i + 1].min(-a / bb)) };
            loaded_len[i] = (v1 - v0).max(0.0);
            if force[i] > 1e-6 {
                loaded[0] = loaded[0].min(v0);
                loaded[1] = loaded[1].max(v1);
            }
        }
        // The toes' pressure about the hinge: what the toe spring (or the held toe) holds.
        let (fi, mi) = clipped_line(a, bb, b[3], b[4]);
        out.toe_torque_nm = f * w[3] * (mi - hinge * fi);
        out.toe_deg = if out.toe_held { 0.0 } else { sole.static_toe_deg(out.toe_torque_nm) };
    } else {
        // Rocker: the rear foot rolls up about the hinge; the metatarsal heads at the hinge and the toes carry the load.
        out.rocker = true;
        let x_tc = hinge + 0.5 * l_toe;
        let f_toe = f * ((u_c - hinge) / (x_tc - hinge)).clamp(0.0, 1.0);
        let f_meta = f - f_toe;
        out.toe_torque_nm = f * (u_c - hinge);
        out.toe_deg = toe_deg.min(TOE_DORSI_MAX_DEG);
        // The toes: uniform while their resultant is behind their middle, then a plate of their own.
        force[3] = f_toe;
        if u_c <= x_tc {
            peak[3] = f_toe / (l_toe * width) / 1000.0;
            loaded_len[3] = l_toe;
        } else {
            let tb = [b[3], b[4]];
            let (a, bb) = plate_line(&tb, &[1.0], u_c);
            let d0 = (a + bb * b[3]).max(0.0);
            let d1 = (a + bb * b[4]).max(0.0);
            peak[3] = f_toe * d0.max(d1) / width / 1000.0;
            let v0 = if bb > 1e-12 { b[3].max(-a / bb) } else { b[3] };
            loaded_len[3] = (b[4] - v0).max(0.0);
        }
        // The metatarsal heads: a triangle of foundation pressure behind the hinge,
        // as long as the tilted rear foot stays pressed in (Winkler: b = √(2F/(k·tan θ))).
        if f_meta > 1e-6 {
            let tan = out.toe_deg.to_radians().tan().max(1e-4);
            let band = (2.0 * f_meta / (sole.k_per_m * tan)).sqrt().max(2.0 * f_meta / (sole.k_per_m * SOLE_COMPRESS_M)).min(b[3] - b[2]);
            force[2] = f_meta;
            peak[2] = 2.0 * f_meta / (band * width) / 1000.0;
            loaded_len[2] = band;
        }
        loaded = [if f_meta > 1e-6 { hinge - loaded_len[2] } else { b[4] - loaded_len[3] }, b[4]];
    }
    // Shear by the patches' tangential stiffness (∝ loaded length), each capped at μ·F (partial slip at the sole).
    let s_mag = (shear[0] * shear[0] + shear[1] * shear[1]).sqrt();
    let dir = if s_mag > 1e-9 { [shear[0] / s_mag, shear[1] / s_mag] } else { [1.0, 0.0] };
    let known = mu.is_finite() && mu > 0.0;
    let cap: Vec<f32> = (0..4).map(|i| if known { mu * force[i] } else { f32::INFINITY }).collect();
    let total_cap: f32 = if known { cap.iter().sum() } else { f32::INFINITY };
    let mut q = [0.0f32; 4];
    let mut slipping = [false; 4];
    if !known {
        let lsum: f32 = loaded_len.iter().sum::<f32>().max(1e-9);
        for i in 0..4 {
            q[i] = s_mag * loaded_len[i] / lsum;
        }
    } else if s_mag >= total_cap - 1e-9 {
        for i in 0..4 {
            q[i] = cap[i];
            slipping[i] = force[i] > 1e-6;
        }
    } else {
        let lsum: f32 = loaded_len.iter().sum::<f32>().max(1e-9);
        let (mut lo, mut hi) = (0.0f32, s_mag / lsum * 1e3 + 1.0);
        for _ in 0..40 {
            let u = 0.5 * (lo + hi);
            let s: f32 = (0..4).map(|i| (u * loaded_len[i]).min(cap[i])).sum();
            if s < s_mag {
                lo = u;
            } else {
                hi = u;
            }
        }
        let u = 0.5 * (lo + hi);
        for i in 0..4 {
            q[i] = (u * loaded_len[i]).min(cap[i]);
            slipping[i] = force[i] > 1e-6 && u * loaded_len[i] >= cap[i] - 1e-9;
        }
    }
    let mut stuck_len = 0.0;
    let mut use_max = 0.0f32;
    for i in 0..4 {
        let p = &mut out.patches[i];
        p.normal_n = force[i];
        p.shear = [q[i] * dir[0], q[i] * dir[1]];
        p.torsion_nm = torsion_nm * force[i] / f;
        p.peak_kpa = peak[i];
        p.slipping = slipping[i];
        if force[i] > 1e-6 {
            if known {
                let e = SQUARE_TORSION * (loaded_len[i].max(1e-3) * width).sqrt();
                let fmax = mu * force[i];
                p.cone_use = ((q[i] / fmax).powi(2) + (p.torsion_nm / (e * fmax)).powi(2)).sqrt();
                use_max = use_max.max(p.cone_use);
            }
            if !slipping[i] {
                stuck_len += loaded_len[i];
            }
        }
    }
    let loaded_total: f32 = loaded_len.iter().sum::<f32>().max(1e-9);
    out.stick_share = (stuck_len / loaded_total).clamp(0.0, 1.0);
    out.cop_x = x_back + u_c;
    out.loaded = [x_back + loaded[0].min(loaded[1]), x_back + loaded[1].max(loaded[0])];
    out.cone_use = if known { s_mag / (mu * f) } else { 0.0 };
    out.cone_use_max = use_max;
    out
}

/// Where the ankle sits when the rear foot has rolled up `toe_deg` about the
/// toe hinge of a sole of `foot_len_m` (the toes flat on the ground): forward
/// and up from where it stands with the foot flat, metres. The rear foot is a
/// rigid body turning about the hinge `(TOE_HINGE_FL − ANKLE_FL)·L` ahead of
/// the ankle, the ankle `ankle_h_m` above the sole.
pub fn rocker_ankle_shift(foot_len_m: f32, ankle_h_m: f32, toe_deg: f32) -> [f32; 2] {
    let th = toe_deg.clamp(0.0, TOE_DORSI_MAX_DEG).to_radians();
    let d = (TOE_HINGE_FL - ANKLE_FL) * foot_len_m;
    let (s, c) = th.sin_cos();
    [d * (1.0 - c) + ankle_h_m * s, d * s + ankle_h_m * (c - 1.0)]
}

/// What a standing robot's base can resist at its hands before its contacts
/// slide or it tips (`balance::Traction` builds on it): the friction left
/// over the shear it already carries, forward and backward, N.
pub fn friction_room_n(mu: f32, normal_n: f32, shear_x_n: f32) -> [f32; 2] {
    let cap = mu.max(0.0) * normal_n.max(0.0);
    [(cap - shear_x_n).max(0.0), (cap + shear_x_n).max(0.0)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    #[test]
    fn pad_friction_falls_with_force_and_holds_its_floor() {
        let mu = pad_mu(0.6, PAD_MU_REF_N, PadSurface::Dry);
        assert!((mu - 0.6).abs() < 1e-6);
        // 1 N → 20 N: the 46% drop of the finger-pad tests.
        let drop = pad_mu(0.6, 20.0, PadSurface::Dry) / pad_mu(0.6, 1.0, PadSurface::Dry);
        assert!((drop - 0.54).abs() < 0.01, "{drop}");
        assert!(pad_mu(0.6, 1e4, PadSurface::Dry) >= 0.6 * PAD_MU_FLOOR - 1e-6);
        assert!(pad_mu(0.6, 10.0, PadSurface::Wet) < mu && pad_mu(0.6, 10.0, PadSurface::Dusty) < mu);
    }

    #[test]
    fn the_limit_surface_couples_force_and_twist() {
        let (mu, n) = (0.6, 10.0);
        assert!((limit_use(mu * n, 0.0, mu, n) - 1.0).abs() < 1e-5);
        let m_max = torsion_radius_m(n) * mu * n;
        assert!((limit_use(0.0, m_max, mu, n) - 1.0).abs() < 1e-5);
        // Half the twist leaves √(3/4) of the tangential capacity.
        assert!((tangential_capacity(0.5 * m_max, mu, n) / (mu * n) - 0.75f32.sqrt()).abs() < 1e-4);
        // Sliding under pure force goes along the force; under pure twist, in twist.
        assert!((slip_direction(1.0, 0.0, mu, n)[0] - 1.0).abs() < 1e-6);
        assert!((slip_direction(0.0, 0.01, mu, n)[1] - 1.0).abs() < 1e-6);
        // On the surface at equal shares of force and twist the slide and the rim speed of the turn are equal:
        // the direction is the surface's normal in its own scale, so the units do not tilt it.
        let s = 0.5f32.sqrt();
        let d = slip_direction(s * mu * n, s * m_max, mu, n);
        assert!((d[0] - s).abs() < 1e-5 && (d[1] - s).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn a_twist_past_one_pad_moves_to_the_others_and_past_all_of_them_turns_the_object() {
        // A light pad saturates in twist first; what it cannot take goes to the heavy one.
        let (light, heavy) = (1.0f32, 20.0f32);
        let m_light = torsion_radius_m(light) * 0.6 * light;
        let m_heavy = torsion_radius_m(heavy) * 0.6 * heavy;
        let tw = 0.5 * (m_light + m_heavy);
        let s = share_load(&[light, heavy], &[0.6, 0.6], 0.0, tw);
        assert!(!s.gross_slip, "{s:?}");
        assert!((s.pads[0].torsion_nm + s.pads[1].torsion_nm - tw).abs() < 1e-6, "the whole twist is held: {s:?}");
        assert!(s.pads.iter().zip([m_light, m_heavy]).all(|(p, m)| p.torsion_nm <= m + 1e-7));
        // The sign of the twist is kept.
        let neg = share_load(&[light, heavy], &[0.6, 0.6], 0.0, -tw);
        assert!(neg.pads.iter().all(|p| p.torsion_nm <= 0.0) && !neg.gross_slip);
        // More twist than every pad resists together turns the object, even with no load on it.
        let over = share_load(&[light, heavy], &[0.6, 0.6], 0.0, 1.1 * (m_light + m_heavy));
        assert!(over.gross_slip && over.stick_min == 0.0, "{over:?}");
    }

    #[test]
    fn a_pad_at_its_torsion_limit_sticks_nowhere_and_the_sensors_feel_it() {
        // The finger scatter: equal forces, unequal friction. The twist passes the weaker pad's limit
        // (each takes half by stiffness) but not both limits together.
        let n = 10.0f32;
        let mus = [0.45f32, 0.75];
        let m = |mu: f32| torsion_radius_m(n) * mu * n;
        let tw = 0.5 * (2.0 * m(mus[0]) + m(mus[0]) + m(mus[1]));
        for load in [0.0f32, 1.0] {
            let s = share_load(&[n, n], &mus, load, tw);
            assert!(!s.gross_slip && (s.pads[0].torsion_nm - m(mus[0])).abs() < 1e-6, "{load}: {s:?}");
            // The capped pad slides in twist: no stick left, and the hand's alarm sees it.
            assert!(s.pads[0].state == SlipState::Slip && s.pads[0].stick < 0.05, "{load}: {:?}", s.pads[0]);
            assert!(s.stick_min < STICK_ALARM, "{load}: {}", s.stick_min);
            // Stick ratio and state agree on every pad.
            for p in &s.pads {
                assert!((p.state == SlipState::Stick) == (p.stick >= STICK_ALARM && p.cone_use < 0.999), "{p:?}");
            }
        }
        // A tangential load alone keeps the Cattaneo–Mindlin ratio on the pad's force.
        let s = share_load(&[10.0], &[0.6], 3.0, 0.0);
        assert!((s.pads[0].stick - stick_ratio(3.0, 6.0)).abs() < 1e-5, "{:?}", s.pads[0]);
    }

    #[test]
    fn the_stick_zone_shrinks_to_nothing_at_slip_and_tells_the_friction() {
        assert_eq!(stick_ratio(0.0, 5.0), 1.0);
        assert!(stick_ratio(5.0, 5.0) < 1e-6);
        let phi = stick_ratio(3.0, 6.0);
        assert!((phi - 0.5f32.powf(2.0 / 3.0)).abs() < 1e-5);
        // The controller reads μ back from the stick ratio.
        assert!((mu_from_stick(3.0, 10.0, phi) - 0.6).abs() < 1e-4);
    }

    #[test]
    fn a_load_shares_by_stiffness_and_the_overflow_moves_on() {
        // Two equal pads share equally and stick.
        let s = share_load(&[10.0, 10.0], &[0.6, 0.6], 6.0, 0.0);
        assert!(!s.gross_slip && (s.pads[0].ft - 3.0).abs() < 1e-4 && s.pads[0].state == SlipState::Stick);
        // A light pad with low friction caps and passes the rest on.
        let s = share_load(&[2.0, 20.0], &[0.3, 0.6], 8.0, 0.0);
        assert!((s.pads[0].ft - 0.6).abs() < 1e-4 && s.pads[0].state == SlipState::Slip && !s.gross_slip, "{s:?}");
        assert!((s.pads[0].ft + s.pads[1].ft - 8.0).abs() < 1e-3);
        // Without a twist the sliding pad slides along its force.
        assert!((s.pads[0].slip_dir[0] - 1.0).abs() < 1e-5 && s.pads[0].slip_dir[1].abs() < 1e-5, "{:?}", s.pads[0].slip_dir);
        // More than both hold: the object slides out.
        let s = share_load(&[5.0, 5.0], &[0.5, 0.5], 6.0, 0.0);
        assert!(s.gross_slip && s.stick_min == 0.0);
        // A twist takes tangential room away.
        let a = share_load(&[10.0, 10.0], &[0.6, 0.6], 0.0, 0.0).capacity_n;
        let b = share_load(&[10.0, 10.0], &[0.6, 0.6], 0.0, 0.02).capacity_n;
        assert!(b < a, "{b} {a}");
    }

    #[test]
    fn a_patch_creeps_before_it_slides() {
        let (mu, n) = (0.6, 2.0);
        assert_eq!(preslip_displacement_m(0.0, mu, n), 0.0);
        let full = preslip_displacement_m(mu * n, mu, n);
        // The whole-slip creep is 3μN/(16·G*·a), and the first bit of load moves the patch by the no-slip stiffness.
        assert!((full - 3.0 * mu * n / (16.0 * tangential_modulus_pa() * patch_radius_m(n))).abs() < 1e-7);
        let small = preslip_displacement_m(0.01 * mu * n, mu, n);
        assert!((small * tangential_stiffness(n) / (0.01 * mu * n) - 1.0).abs() < 0.02, "{small}");
        assert!(preslip_displacement_m(0.5 * mu * n, mu, n) < 0.5 * full, "the creep speeds up toward the slide");
        // A heavier press spreads a bigger patch.
        assert!(patch_area_m2(2.0) > patch_area_m2(1.0) && (patch_area_m2(8.0) / patch_area_m2(1.0) - 4.0).abs() < 0.01);
    }

    #[test]
    fn the_grip_loop_answers_a_pull_after_its_sensor_sees_it() {
        let hold = |rate: f32, plateau: f32| ramp_hold(0.6, PadSurface::Dry, 0.6, 2.0, PRELOAD_N, 1, rate, plateau, 0.5, 0.0875);
        let lat: Vec<f32> = [2.0, 4.0, 8.0, 32.0].iter().map(|&r| hold(r, 2.0).latency_s.unwrap()).collect();
        // The response starts no sooner than the sensor's delay, and sooner for a faster pull.
        assert!(lat.iter().all(|l| *l >= detect_delay_s()), "{lat:?}");
        assert!(lat.windows(2).all(|w| w[1] <= w[0]) && lat[0] > lat[3], "{lat:?}");
        // On the plateau the grip holds the load with the commanded margin, and nothing slid.
        for p in [1.0, 2.0, 4.0] {
            let h = hold(4.0, p);
            assert!(!h.slipped && h.safety_margin > 0.3 && h.safety_margin < 0.9, "{h:?}");
            assert!(h.grip_n > PRELOAD_N + 2.0 * p / (PINCH_FACES * 0.6) - 0.05, "{h:?}");
        }
        // A grip that never answers (a sensor that never reports) lets a heavy pull slide out.
        let blind = ramp_hold(0.6, PadSurface::Dry, 0.6, 2.0, PRELOAD_N, 1, 4.0, 4.0, 0.5, 1e9);
        assert!(blind.slipped, "{blind:?}");
        // A pull that hardly rises is cut off at the longest ramp instead of running for ever.
        let crawl = ramp_hold(0.6, PadSurface::Dry, 0.6, 2.0, PRELOAD_N, 1, 1e-9, 4.0, 0.5, 0.0875);
        assert!(!crawl.slipped && crawl.load_n < 1e-3 && crawl.latency_s.is_none(), "{crawl:?}");
    }

    #[test]
    fn finger_contacts_serialise_as_a_list() {
        let mut c = FingerContacts::default();
        c.push(FingerContact { finger: 1, fn_n: 3.0, ..FingerContact::default() });
        let j = serde_json::to_value(c).unwrap();
        assert_eq!(j.as_array().unwrap().len(), 1);
        assert_eq!(j[0]["finger"], 1);
    }

    fn legs() -> Body {
        Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five)
    }

    #[test]
    fn a_flat_foot_loads_the_heel_behind_and_the_forefoot_ahead() {
        let b = legs();
        let s = SoleModel::of(&b);
        assert!(s.toe_kp > 1.0 && s.toe_kp < 10.0 && s.toe_share > 0.0 && s.toe_share < 0.5, "{s:?}");
        let fl = s.foot_len_m;
        let w = 300.0;
        // The centre of pressure over the ankle: the heel carries most.
        let back = sole_contact(&s, 0.0, w, [0.0; 2], 0.0, 0.0, 0.0, 0.7);
        let sum: f32 = back.patches.iter().map(|p| p.normal_n).sum();
        assert!((sum - w).abs() < 0.5, "{sum}");
        assert!(!back.rocker && back.patches[0].normal_n > 0.5 * w, "{:?}", back.patches);
        // The resultant is where it was asked.
        let m: f32 = back.patches.iter().map(|p| p.normal_n * 0.5 * (p.x0 + p.x1)).sum::<f32>() / w;
        assert!(m.abs() < 0.25 * fl, "{m}");
        // Further forward the heel unloads before the hinge (0.40 fl ahead of the ankle);
        // past the hinge (0.50 fl) the rear foot rolls up.
        let fore = sole_contact(&s, 0.0, w, [0.0; 2], 0.0, 0.40 * fl, 0.0, 0.7);
        assert!(!fore.rocker && !fore.toe_held && fore.patches[0].normal_n < 1e-3 && fore.patches[2].normal_n > 0.5 * w, "{:?}", fore.patches);
        // Past the hinge with the heel up (the gait's toe angle): the forefoot carries it.
        let push = sole_contact(&s, 0.0, w, [30.0, 0.0], 0.0, 0.60 * fl, 20.0, 0.7);
        assert!(push.rocker && push.patches[0].normal_n == 0.0 && push.patches[3].normal_n > 0.5 * w, "{push:?}");
        assert!(push.toe_deg > 0.0 && push.toe_torque_nm > 0.0);
        assert!(push.loaded[0] >= 0.30 * fl - 1e-4, "the support is the forefoot: {:?}", push.loaded);
        // The same load on a foot held flat: the toe holds it, the heel is not up.
        let held = sole_contact(&s, 0.0, w, [30.0, 0.0], 0.0, 0.60 * fl, 0.0, 0.7);
        assert!(!held.rocker && held.toe_held && held.toe_deg == 0.0 && held.patches[3].normal_n > 0.0, "{held:?}");
        // Shear within friction sticks; past it the patches slide.
        assert!(push.cone_use < 1.0 && push.patches.iter().all(|p| !p.slipping));
        let slide = sole_contact(&s, 0.0, w, [250.0, 0.0], 0.0, 0.0, 0.0, 0.7);
        assert!(slide.cone_use > 1.0 && slide.patches.iter().filter(|p| p.normal_n > 0.0).all(|p| p.slipping));
        // Friction not known (the statics alone): the shear spreads by stiffness, nothing slips.
        let blind = sole_contact(&s, 0.0, w, [250.0, 20.0], 0.0, 0.0, 0.0, 0.0);
        let sx: f32 = blind.patches.iter().map(|p| p.shear[0]).sum();
        assert!((sx - 250.0).abs() < 0.5 && blind.patches.iter().all(|p| !p.slipping) && blind.cone_use == 0.0, "{blind:?}");
    }

    #[test]
    fn the_rear_foot_turns_about_the_toe_hinge() {
        assert_eq!(rocker_ankle_shift(0.2, 0.08, 0.0), [0.0, 0.0]);
        let s = rocker_ankle_shift(0.2, 0.08, 30.0);
        // The ankle 0.10 m behind the hinge rises about 0.10·sin 30° (less the ankle height's turn) and moves forward.
        assert!(s[1] > 0.035 && s[1] < 0.05 && s[0] > 0.0, "{s:?}");
    }

    #[test]
    #[ignore]
    fn print_soles() {
        for tier in [Tier::Budget, Tier::Mid, Tier::Pro] {
            let b = Body::new(&BodyPlan::uniform(tier), BaseKind::HumanFoot, FingerCount::Five);
            let s = SoleModel::of(&b);
            println!("[SOLE] {tier:?} {:.1} kg, CoM {:.3} m: Kp {:.2} N·m/rad, Kd/Kp {:.3} s, toe share {:.3}, k {:.3e} N/m/m", b.mass_kg(), b.com_height_m(), s.toe_kp, s.toe_kd / s.toe_kp, s.toe_share, s.k_per_m);
        }
    }

    #[test]
    fn the_toe_spring_settles_with_its_time_constant() {
        let s = SoleModel::of(&legs());
        let m = 2.0;
        let mut d = 0.0;
        for _ in 0..200 {
            d = s.toe_step_deg(d, m, 0.02);
        }
        assert!((d - s.static_toe_deg(m)).abs() < 0.05, "{d} vs {}", s.static_toe_deg(m));
        assert_eq!(s.toe_step_deg(10.0, -1.0, 0.02), 0.0, "the heel comes down");
    }
}
