//! Compute backend selection: CPU, CUDA, Metal, Vulkan or ROCm.
//!
//! Which backends exist is decided at compile time by cargo features. Each
//! feature only switches on its candle backend, so a build without a GPU SDK
//! still runs on the CPU.
//!
//! The rules follow the logis-center app (`utils/device_utils.rs`) and the
//! endmill sandbox (`ml/mod.rs`):
//!
//! * At start-up the first GPU backend of this build that opens is used
//!   ([`preferred`]: CUDA, ROCm, Vulkan, Metal), unless `NOTO_DEVICE` says
//!   otherwise (`cpu`, `cuda`, `rocm`, `vulkan`, `metal`, `gpu`/`auto`, with an
//!   optional `:ordinal`).
//! * Every model gets its device from one cache ([`Backend::device`]), so the
//!   SigLIP2, TIPS, SAM and TTM tensors of one backend live on the same device
//!   handle instead of a new context per load (logis-center `DEVICE_CACHE`).
//! * After a batched call the idle memory of the device pool is given back
//!   ([`trim_idle_pool`], logis-center `trim_idle_gpu_pool`).
//! * How many rows go into one forward pass depends on the backend
//!   ([`BatchPolicy`], logis-center `BatchOpts::for_device`).
//! * A Windows process that used ROCm ends through [`rocm_exit_now_if_used`],
//!   which skips the rocBLAS/HIP DLL unload where the logis-center tests saw
//!   processes hang.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};

/// Picks the backend at start-up (see the module docs).
pub const DEVICE_ENV: &str = "NOTO_DEVICE";
/// Overrides the rows of one image forward pass ([`BatchPolicy::image_rows`]).
pub const BATCH_ROWS_ENV: &str = "NOTO_BATCH_ROWS";
/// `1` or `0` forces the fixed-shape text batches on or off ([`BatchPolicy::fixed_shape`]).
pub const FIXED_SHAPE_ENV: &str = "NOTO_FIXED_SHAPE";
/// Pool memory a CUDA or ROCm device keeps after [`trim_idle_pool`], bytes.
pub const POOL_KEEP_BYTES: usize = 256 << 20;
/// A Vulkan pool is only trimmed above this many idle bytes (logis-center: 64 MB).
pub const VULKAN_TRIM_MIN_BYTES: usize = 64 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

/// GPU backends in the order [`preferred`] tries them.
pub const GPU_ORDER: [Backend; 4] = [Backend::Cuda, Backend::Rocm, Backend::Vulkan, Backend::Metal];

impl Backend {
    pub const ALL: [Backend; 5] = [
        Backend::Cpu,
        Backend::Cuda,
        Backend::Metal,
        Backend::Vulkan,
        Backend::Rocm,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
        }
    }

    pub fn parse(s: &str) -> Option<Backend> {
        Backend::ALL
            .into_iter()
            .find(|b| b.name().eq_ignore_ascii_case(s.trim()))
    }

    /// The backend a device belongs to.
    pub fn of(device: &Device) -> Backend {
        match device {
            Device::Cuda(_) => Backend::Cuda,
            Device::Metal(_) => Backend::Metal,
            Device::Vulkan(_) => Backend::Vulkan,
            Device::Rocm(_) => Backend::Rocm,
            _ => Backend::Cpu,
        }
    }

    /// Whether this build includes the backend.
    pub fn compiled(self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Cuda => cfg!(feature = "cuda"),
            Backend::Metal => cfg!(feature = "metal"),
            Backend::Vulkan => cfg!(feature = "vulkan"),
            Backend::Rocm => cfg!(feature = "rocm"),
        }
    }

    /// Device `ordinal` of this backend, opened once per process and shared
    /// by every caller afterwards.
    pub fn device(self, ordinal: usize) -> Result<Device> {
        static CACHE: OnceLock<Mutex<HashMap<(Backend, usize), Device>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(d) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&(self, ordinal)) {
            return Ok(d.clone());
        }
        let d = self.open(ordinal)?;
        let mut g = cache.lock().unwrap_or_else(|p| p.into_inner());
        Ok(g.entry((self, ordinal)).or_insert(d).clone())
    }

    /// Opens a new handle to device `ordinal` of this backend. Prefer
    /// [`Backend::device`], which reuses the handle.
    pub fn open(self, ordinal: usize) -> Result<Device> {
        match self {
            Backend::Cpu => Ok(Device::Cpu),
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    return Ok(Device::new_cuda(ordinal)?);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    bail!("this build has no CUDA support (enable the `cuda` feature)")
                }
            }
            Backend::Metal => {
                #[cfg(feature = "metal")]
                {
                    return Ok(Device::new_metal(ordinal)?);
                }
                #[cfg(not(feature = "metal"))]
                {
                    let _ = ordinal;
                    bail!("this build has no Metal support (enable the `metal` feature)")
                }
            }
            Backend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    return Ok(Device::new_vulkan(ordinal)?);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    let _ = ordinal;
                    bail!("this build has no Vulkan support (enable the `vulkan` feature)")
                }
            }
            Backend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    let d = Device::new_rocm(ordinal)?;
                    ROCM_USED.store(true, Ordering::SeqCst);
                    return Ok(d);
                }
                #[cfg(not(feature = "rocm"))]
                {
                    let _ = ordinal;
                    bail!("this build has no ROCm support (enable the `rocm` feature)")
                }
            }
        }
    }
}

/// The device a process should start on, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub backend: Backend,
    pub ordinal: usize,
    /// One line on how the choice was made (the `NOTO_DEVICE` value, a fallback).
    pub reason: String,
}

/// Parses a `NOTO_DEVICE` value: `None` means "the first GPU that opens".
fn parse_request(raw: &str) -> Result<(Option<Backend>, usize)> {
    let raw = raw.trim().to_ascii_lowercase();
    let (name, ordinal) = match raw.split_once(':') {
        Some((n, o)) => (n.to_string(), o.trim().parse::<usize>().map_err(|_| anyhow::anyhow!("{DEVICE_ENV}={raw}: ordinal is not a number"))?),
        None => (raw.clone(), 0),
    };
    match name.as_str() {
        "" | "auto" | "gpu" => Ok((None, ordinal)),
        n => match Backend::parse(n) {
            Some(b) => Ok((Some(b), ordinal)),
            None => bail!("{DEVICE_ENV}={raw}: unknown backend (cpu, cuda, rocm, vulkan, metal, gpu)"),
        },
    }
}

/// Chooses the start-up device from `request` (a `NOTO_DEVICE` value).
/// Without a request, or when the requested backend cannot be opened, the
/// first GPU of [`GPU_ORDER`] that this build has and that opens wins; the CPU
/// is the last resort. Opening goes through [`Backend::device`], so the probe
/// leaves a usable device in the cache.
pub fn select(request: Option<&str>) -> Selection {
    let mut notes = Vec::new();
    let (wanted, ordinal) = match request.map(parse_request) {
        Some(Ok(r)) => r,
        Some(Err(e)) => {
            notes.push(e.to_string());
            (None, 0)
        }
        None => (None, 0),
    };
    if let Some(b) = wanted {
        if b == Backend::Cpu {
            return Selection { backend: b, ordinal: 0, reason: format!("{DEVICE_ENV}=cpu") };
        }
        match b.device(ordinal) {
            Ok(_) => return Selection { backend: b, ordinal, reason: format!("{DEVICE_ENV}={}:{ordinal}", b.name()) },
            Err(e) => notes.push(format!("{} 장치 {ordinal}을 열 수 없음: {e}", b.name())),
        }
    }
    for b in GPU_ORDER {
        if !b.compiled() {
            continue;
        }
        match b.device(ordinal) {
            Ok(_) => {
                notes.push(format!("이 빌드의 첫 GPU 백엔드 {}:{ordinal}", b.name()));
                return Selection { backend: b, ordinal, reason: notes.join("; ") };
            }
            Err(e) => notes.push(format!("{}: {e}", b.name())),
        }
    }
    if notes.is_empty() {
        notes.push("GPU 기능 없이 빌드됨".into());
    }
    Selection { backend: Backend::Cpu, ordinal: 0, reason: format!("CPU ({})", notes.join("; ")) }
}

/// [`select`] with the `NOTO_DEVICE` environment variable.
pub fn preferred() -> Selection {
    select(std::env::var(DEVICE_ENV).ok().as_deref())
}

/// Short name of a device for logs and the UI: `CUDA:0`, `ROCm:0 (RX 6600)`,
/// `Vulkan:0 (name, discrete)`, `Metal`, `CPU`.
pub fn device_label(device: &Device) -> String {
    match device {
        Device::Cuda(_) => "CUDA:0".into(),
        Device::Rocm(r) => match r.name() {
            Ok(n) if !n.is_empty() => format!("ROCm:0 ({n})"),
            _ => "ROCm:0".into(),
        },
        Device::Vulkan(v) => format!("Vulkan:0 ({}, {})", v.name(), v.device_type()),
        Device::Metal(_) => "Metal".into(),
        _ => "CPU".into(),
    }
}

/// Device family for cache keys: vectors computed on different backends are
/// not bit-identical, so cached vectors must not cross them.
pub fn device_family(device: &Device) -> &'static str {
    Backend::of(device).name()
}

/// `(free, total)` device memory in bytes, `None` on the CPU.
pub fn mem_info(device: &Device) -> Option<(u64, u64)> {
    device.mem_info().ok().flatten().map(|(f, t)| (f as u64, t as u64))
}

/// Gives the idle part of the device's memory pool back to the driver after
/// a batched call. CUDA and ROCm keep [`POOL_KEEP_BYTES`]; Vulkan is trimmed
/// only when more than [`VULKAN_TRIM_MIN_BYTES`] sit idle. Errors are ignored:
/// trimming is housekeeping, never a reason to fail an inference.
pub fn trim_idle_pool(device: &Device) {
    match device {
        Device::Cuda(_) | Device::Rocm(_) => {
            let _ = device.trim_memory_pool(POOL_KEEP_BYTES);
        }
        Device::Vulkan(_) => {
            if device.pool_idle_bytes().unwrap_or(0) >= VULKAN_TRIM_MIN_BYTES {
                let _ = device.trim_memory_pool(0);
            }
        }
        _ => {}
    }
}

/// How many rows go into one forward pass on a device.
///
/// * Images: `image_rows(tokens)` = `token_budget / tokens`, at most
///   `max_rows`. Image outputs are not cached across calls, so batches are
///   never padded.
/// * Texts: on CUDA and ROCm a text batch is padded to its full row count
///   with copies of its first row (`fixed_shape`), so the GEMM kernels the
///   driver picks, and with them the vector of a phrase, do not depend on
///   which other phrases share the call. The phrase bank caches those vectors
///   on disk. On the CPU, Vulkan and Metal padding only costs time and the
///   rows do not change with the batch (logis-center ⑥), so they are dynamic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPolicy {
    pub max_rows: usize,
    pub token_budget: usize,
    pub fixed_shape: bool,
}

impl BatchPolicy {
    pub fn for_device(device: &Device) -> BatchPolicy {
        let gpu = !device.is_cpu();
        let mut p = BatchPolicy {
            max_rows: if gpu { 16 } else { 8 },
            // About 8 NaFlex images of 256 patches, or 2 TIPS keyframes of 864 patches, per pass on a GPU.
            token_budget: if gpu { 2048 } else { 1024 },
            fixed_shape: matches!(device, Device::Cuda(_) | Device::Rocm(_)),
        };
        if let Some(rows) = std::env::var(BATCH_ROWS_ENV).ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|r| *r > 0) {
            p.max_rows = rows;
            p.token_budget = p.token_budget.max(rows);
        }
        if let Ok(v) = std::env::var(FIXED_SHAPE_ENV) {
            p.fixed_shape = v.trim() != "0";
        }
        p
    }

    /// Images of `tokens` patches per forward pass.
    pub fn image_rows(&self, tokens: usize) -> usize {
        (self.token_budget / tokens.max(1)).clamp(1, self.max_rows.max(1))
    }
}

/// Whether this process opened a ROCm device.
static ROCM_USED: AtomicBool = AtomicBool::new(false);

/// On Windows, a process that used ROCm finishes its GPU work and ends here
/// with `TerminateProcess`, skipping the DLL unload: logis-center saw rocBLAS's
/// exit handler call into an already stopped HIP runtime and spin forever (2
/// of 68 test processes on an RX 6600 with HIP SDK 7.2). Rust destructors do
/// not run on `std::process::exit` either, so only C `atexit` work is lost.
/// `NOTO_ROCM_FAST_EXIT=0` keeps the normal exit. Everywhere else this
/// returns and does nothing.
pub fn rocm_exit_now_if_used(code: u32) {
    #[cfg(all(windows, feature = "rocm"))]
    {
        use std::io::Write;
        if !ROCM_USED.load(Ordering::SeqCst) {
            return;
        }
        if std::env::var("NOTO_ROCM_FAST_EXIT").map(|v| v.trim() == "0").unwrap_or(false) {
            return;
        }
        if let Ok(d) = Backend::Rocm.device(0) {
            let _ = d.synchronize();
            let _ = d.trim_memory_pool(0);
        }
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: ends this process; nothing runs after the call.
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
            TerminateProcess(GetCurrentProcess(), code);
        }
    }
    let _ = code;
    let _ = ROCM_USED.load(Ordering::Relaxed);
}

/// One backend as the UI shows it: whether it is compiled in, and whether a
/// device of that backend can actually be opened on this machine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub backend: Backend,
    pub compiled: bool,
    pub available: bool,
    pub detail: String,
}

/// Probes every backend. Opening a device can be slow (driver load), so call
/// this once and cache the result. The devices it opens stay in the cache.
pub fn probe_backends() -> Vec<BackendStatus> {
    Backend::ALL
        .into_iter()
        .map(|backend| {
            let compiled = backend.compiled();
            if !compiled {
                return BackendStatus {
                    backend,
                    compiled,
                    available: false,
                    detail: "not compiled into this build".into(),
                };
            }
            match backend.device(0) {
                Ok(dev) => {
                    let mem = mem_info(&dev).map(|(f, t)| format!(", {:.1} / {:.1} GB free", f as f64 / 1e9, t as f64 / 1e9)).unwrap_or_default();
                    BackendStatus {
                        backend,
                        compiled,
                        available: true,
                        detail: format!("{}{mem}", device_label(&dev)),
                    }
                }
                Err(e) => BackendStatus {
                    backend,
                    compiled,
                    available: false,
                    detail: e.to_string(),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for b in Backend::ALL {
            assert_eq!(Backend::parse(b.name()), Some(b));
        }
        assert_eq!(Backend::parse("CUDA"), Some(Backend::Cuda));
        assert_eq!(Backend::parse("tpu"), None);
    }

    #[test]
    fn cpu_is_always_available() {
        assert!(Backend::Cpu.compiled());
        assert!(Backend::Cpu.open(0).is_ok());
        assert_eq!(Backend::of(&Backend::Cpu.device(0).unwrap()), Backend::Cpu);
    }

    #[test]
    fn uncompiled_backend_reports_missing_feature() {
        if !Backend::Rocm.compiled() {
            let err = Backend::Rocm.open(0).unwrap_err().to_string();
            assert!(err.contains("rocm"));
        }
    }

    #[test]
    fn device_requests_parse_and_fall_back() {
        assert_eq!(parse_request("cuda:1").unwrap(), (Some(Backend::Cuda), 1));
        assert_eq!(parse_request(" GPU ").unwrap(), (None, 0));
        assert!(parse_request("tpu").is_err());
        assert!(parse_request("cuda:x").is_err());
        assert_eq!(select(Some("cpu")).backend, Backend::Cpu);
        // Whatever is compiled in, the choice opens.
        let s = select(None);
        assert!(s.backend.device(s.ordinal).is_ok(), "{s:?}");
        if !Backend::Rocm.compiled() {
            let s = select(Some("rocm"));
            assert_ne!(s.backend, Backend::Rocm);
            assert!(s.reason.contains("rocm"), "{}", s.reason);
        }
    }

    #[test]
    fn the_cache_hands_out_one_device_per_backend() {
        let a = Backend::Cpu.device(0).unwrap();
        let b = Backend::Cpu.device(0).unwrap();
        assert!(a.same_device(&b));
        assert_eq!(device_label(&a), "CPU");
        assert_eq!(device_family(&a), "cpu");
        assert!(mem_info(&a).is_none());
        trim_idle_pool(&a);
    }

    #[test]
    fn batch_policy_is_dynamic_on_the_cpu() {
        let p = BatchPolicy::for_device(&Device::Cpu);
        if std::env::var(FIXED_SHAPE_ENV).is_err() {
            assert!(!p.fixed_shape);
        }
        assert!(p.image_rows(256) >= 1 && p.image_rows(256) <= p.max_rows);
        assert_eq!(p.image_rows(100_000), 1);
    }
}
