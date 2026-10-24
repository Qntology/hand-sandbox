//! Command search for the world simulator.
//!
//! A typed Korean (or English) command such as "나무 두 개를 캐서 창고로
//! 가져와" is turned into a plan by the same method logis-center uses for its
//! search box: a word dictionary of `{semantic, bias, prejudice, exact_match}`
//! nodes, closed particle and ending tables with final-consonant agreement,
//! exact and prefix matching first, embedding span scoring for the rest.
//!
//! * [`lexicon`] loads the dictionary (`data/world_lexicon.json`);
//! * [`tails`], [`numbers`], [`lexical`] are the lexical stage;
//! * [`semantic`] is the embedding stage, generic over a [`PhraseEmbedder`];
//! * [`vision`] scores image vectors against the same dictionary's vision prompts;
//! * [`parse::Parser`] ties the stages together.
//!
//! The crate has no model dependency. Without an embedder the lexical stage
//! still works.

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
pub use lexicon::{EntityKind, Lexicon, Node, Phrase, VisionEntity};
pub use parse::{DropTarget, ParsedCommand, Parser, PlaceRef, Plan, Problem, Target};
pub use semantic::{PhraseEmbedder, SemanticBank, SemanticOpts};
pub use vision::{EntityScore, VisionBank};

/// One line of the search trace, for the log view.
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
