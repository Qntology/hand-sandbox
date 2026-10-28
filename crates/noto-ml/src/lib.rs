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
pub use device::{device_label, preferred, probe_backends, select, trim_idle_pool, Backend, BackendStatus, BatchPolicy, Selection};
pub use model::{DenseEmbedding, Siglip2};
pub use parallel::{BatchOpts, ImageEmbedder, PhraseBank, TextEmbedder, Tile};
