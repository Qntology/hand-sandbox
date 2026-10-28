pub mod lexical;
pub mod lexicon;
pub mod numbers;
pub mod parse;
pub mod semantic;
pub mod tails;
pub mod text;
pub mod vision;

use serde::Serialize;

pub use lexical::{Category, Hit, Via};
pub use lexicon::{EntityKind, Lexicon, Node, Phrase, StructureEntity, VisionEntity};
pub use parse::{DropTarget, ParsedCommand, Parser, PlaceRef, Plan, Problem, Target};
pub use semantic::{PhraseEmbedder, SemanticBank, SemanticOpts};
pub use vision::{EntityScore, VisionBank, VisionPrompting};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TraceLine {
    pub stage: String,
    pub detail: String,
}

impl TraceLine {
    pub fn new(stage: &str, detail: impl Into<String>) -> TraceLine {
        TraceLine { stage: stage.to_string(), detail: detail.into() }
    }
}
