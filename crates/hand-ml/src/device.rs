//! Compute backend selection: CPU, CUDA, Metal, Vulkan or ROCm.
//!
//! Which backends exist is decided at compile time by cargo features. Each
//! feature only switches on its candle backend, so a build without a GPU SDK
//! still runs on the CPU.

use anyhow::{bail, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

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

    /// Opens device `ordinal` of this backend.
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
                    return Ok(Device::new_rocm(ordinal)?);
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
/// this once and cache the result.
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
            match backend.open(0) {
                Ok(dev) => BackendStatus {
                    backend,
                    compiled,
                    available: true,
                    detail: format!("{dev:?}"),
                },
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
    }

    #[test]
    fn uncompiled_backend_reports_missing_feature() {
        if !Backend::Rocm.compiled() {
            let err = Backend::Rocm.open(0).unwrap_err().to_string();
            assert!(err.contains("rocm"));
        }
    }
}
