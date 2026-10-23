//! SigLIP2 vision-language embeddings on candle, with CUDA, Metal, Vulkan and
//! ROCm device selection.
//!
//! * [`device`] picks the compute backend (features decide what is compiled in).
//! * [`model::Siglip2`] loads `google/siglip2-base-patch16-naflex` and embeds
//!   images and texts. [`preprocess`] implements the NaFlex patching.
//! * [`parallel`] embeds phrases and frame tiles in deduplicated, cached,
//!   batched passes.
//! * [`classify`] runs zero-shot classification against the logistics word
//!   dictionary.

pub mod classify;
pub mod config;
pub mod device;
pub mod encoder;
pub mod model;
pub mod parallel;
pub mod preprocess;
pub mod text;
pub mod tokenizer;
pub mod vision;

pub use classify::{classify, Anchors, DictEntry, Dictionary, Prediction};
pub use config::Siglip2Config;
pub use device::{probe_backends, Backend, BackendStatus};
pub use model::{DenseEmbedding, Siglip2};
pub use parallel::{BatchOpts, ImageEmbedder, PhraseBank, TextEmbedder, Tile};
