//! Embedding scoring, ported from logis-center's phrase scoring.
//!
//! A query vector is scored against a node by
//!
//! ```text
//! score = bias − prejudice_weight · prejudice − verb_penalty
//! bias        = weighted max-pool over the node's bias phrases
//! prejudice   = max-pool over the node's prejudice phrases
//! verb_penalty = beta · max-pool over every verb phrase
//! ```
//!
//! with `prejudice_weight` 0.3 and `beta` 0.05 for spans of up to two words,
//! 0.7 and 0.10 for longer ones (the constants of logis-center's Plinko
//! window). Spans are then thinned by non-maximum suppression (a better
//! overlapping span wins) and assigned greedily so each node is used once.
//!
//! Nothing in this file knows a model: callers pass a [`PhraseEmbedder`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::lexical::{Category, Hit, Via};
use crate::lexicon::{Lexicon, Node, Phrase};
use crate::tails::{peel, TailKind};
use crate::TraceLine;

/// Anything that turns phrases into vectors (one per phrase, in order).
pub trait PhraseEmbedder {
    fn embed_phrases(&self, phrases: &[String]) -> Result<Vec<Arc<[f32]>>>;
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn is_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

/// Largest cosine against any phrase, floored at 0.
pub fn max_pool(q: &[f32], phrases: &[Arc<[f32]>]) -> f32 {
    phrases.iter().map(|p| cosine(q, p)).fold(0.0f32, f32::max)
}

/// Largest weight-scaled cosine, floored at 0; zero vectors are skipped.
pub fn weighted_max_pool(q: &[f32], phrases: &[(Arc<[f32]>, f32)]) -> f32 {
    let mut best = 0.0f32;
    for (p, w) in phrases {
        if is_zero(p) {
            continue;
        }
        best = best.max(cosine(q, p) * w);
    }
    best
}

/// Prejudice wins when it outweighs the bias by the cohesion allowance.
pub fn prejudice_dominates(own: f32, prej: f32, cohesion: f32) -> bool {
    if own <= 0.0 {
        return true;
    }
    let relief = cohesion.clamp(0.0, 0.5);
    prej > own * (1.0 + relief)
}

#[derive(Clone, Debug)]
pub struct NodeVectors {
    pub id: String,
    pub bias: Vec<(Arc<[f32]>, f32)>,
    pub prejudice: Vec<Arc<[f32]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeScore {
    pub id: String,
    pub bias: f32,
    pub prejudice: f32,
    pub verb: f32,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct SemanticOpts {
    /// A span must score at least this to be accepted.
    pub min_score: f32,
    /// ... and beat the runner-up node by this much.
    pub margin: f32,
    /// Longest span, in tokens.
    pub max_span: usize,
    /// Cohesion allowance for [`prejudice_dominates`].
    pub cohesion: f32,
}

impl Default for SemanticOpts {
    /// The thresholds are starting points chosen without the real embedding
    /// model (it cannot be downloaded in the build sandbox). They need tuning
    /// against real vectors; `hand-runtime` lets the SDS baseline move them.
    fn default() -> Self {
        SemanticOpts { min_score: 0.30, margin: 0.04, max_span: 3, cohesion: 0.0 }
    }
}

/// Vectors of every dictionary phrase that the command search scores against.
pub struct SemanticBank {
    pub resources: Vec<NodeVectors>,
    pub places: Vec<NodeVectors>,
    verb_phrases: Vec<Arc<[f32]>>,
}

fn pairs(
    phrases: &[Phrase],
    vectors: &HashMap<&str, Arc<[f32]>>,
) -> Vec<(Arc<[f32]>, f32)> {
    phrases.iter().filter_map(|p| vectors.get(p.text.as_str()).map(|v| (v.clone(), p.weight))).collect()
}

impl SemanticBank {
    /// Embeds every command-side phrase of the lexicon in one call.
    pub fn build(lex: &Lexicon, emb: &dyn PhraseEmbedder) -> Result<SemanticBank> {
        let mut texts: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for n in lex.resources.iter().chain(&lex.places).chain(&lex.verbs) {
            for p in n.bias.iter().chain(&n.prejudice) {
                if seen.insert(p.text.clone()) {
                    texts.push(p.text.clone());
                }
            }
        }
        let vecs = emb.embed_phrases(&texts)?;
        if vecs.len() != texts.len() {
            bail!("embedder returned {} vectors for {} phrases", vecs.len(), texts.len());
        }
        let by_text: HashMap<&str, Arc<[f32]>> = texts.iter().map(String::as_str).zip(vecs).collect();
        let node_vectors = |n: &Node| NodeVectors {
            id: n.id.clone(),
            bias: pairs(&n.bias, &by_text),
            prejudice: n.prejudice.iter().filter_map(|p| by_text.get(p.text.as_str()).cloned()).collect(),
        };
        let verb_phrases: Vec<Arc<[f32]>> = lex
            .verbs
            .iter()
            .flat_map(|n| n.bias.iter())
            .filter_map(|p| by_text.get(p.text.as_str()).cloned())
            .collect();
        Ok(SemanticBank {
            resources: lex.resources.iter().map(node_vectors).collect(),
            places: lex.places.iter().map(node_vectors).collect(),
            verb_phrases,
        })
    }

    fn nodes(&self, category: Category) -> &[NodeVectors] {
        match category {
            Category::Resource => &self.resources,
            _ => &self.places,
        }
    }

    /// Scores `q` (a span of `words` tokens) against every node of `category`, best first.
    pub fn score(&self, q: &[f32], words: usize, category: Category) -> Vec<NodeScore> {
        let (pw, beta) = if words <= 2 { (0.3, 0.05) } else { (0.7, 0.10) };
        let verb = max_pool(q, &self.verb_phrases);
        let mut out: Vec<NodeScore> = self
            .nodes(category)
            .iter()
            .map(|n| {
                let bias = weighted_max_pool(q, &n.bias);
                let prejudice = max_pool(q, &n.prejudice);
                NodeScore {
                    id: n.id.clone(),
                    bias,
                    prejudice,
                    verb,
                    score: bias - prejudice * pw - verb * beta,
                }
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out
    }
}

struct Span {
    start: usize,
    end: usize,
    /// Strings to embed for this span: as typed, and with the last token's tail removed.
    variants: Vec<String>,
}

fn spans(tokens: &[String], free: &[bool], max_span: usize) -> Vec<Span> {
    let mut out = Vec::new();
    for i in 0..tokens.len() {
        for n in 1..=max_span {
            if i + n > tokens.len() || (i..i + n).any(|k| !free.get(k).copied().unwrap_or(false)) {
                break;
            }
            let head = tokens[i..i + n - 1].join(" ");
            let last = &tokens[i + n - 1];
            let join = |tail: &str| if head.is_empty() { tail.to_string() } else { format!("{head} {tail}") };
            let mut variants = vec![join(last)];
            // The reading with the longest closed tail removed.
            if let Some(r) = peel(last).into_iter().filter(|r| !r.tail.is_empty()).max_by_key(|r| r.tail.chars().count()) {
                let v = join(&r.stem);
                if !variants.contains(&v) {
                    variants.push(v);
                }
            }
            if variants.iter().all(|v| v.chars().count() < 2) {
                continue;
            }
            out.push(Span { start: i, end: i + n, variants });
        }
    }
    out
}

/// Finds resource and place words that the lexical stage did not know, by
/// scoring free token spans against the bank. `free[i]` marks tokens no
/// earlier stage used.
pub fn search_spans(
    bank: &SemanticBank,
    emb: &dyn PhraseEmbedder,
    tokens: &[String],
    free: &[bool],
    opts: &SemanticOpts,
    trace: &mut Vec<TraceLine>,
) -> Result<Vec<Hit>> {
    let spans = spans(tokens, free, opts.max_span.max(1));
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let mut texts: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for s in &spans {
        for v in &s.variants {
            if seen.insert(v.clone()) {
                texts.push(v.clone());
            }
        }
    }
    let vecs = emb.embed_phrases(&texts)?;
    if vecs.len() != texts.len() {
        bail!("embedder returned {} vectors for {} spans", vecs.len(), texts.len());
    }
    let vec_of: HashMap<&str, Arc<[f32]>> = texts.iter().map(String::as_str).zip(vecs).collect();

    struct Cand {
        start: usize,
        end: usize,
        category: Category,
        node: String,
        score: f32,
        text: String,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for s in &spans {
        let words = s.end - s.start;
        // Best over variants, per node, across both categories.
        let mut best: Vec<(Category, NodeScore)> = Vec::new();
        for cat in [Category::Resource, Category::Place] {
            let mut per_node: HashMap<String, NodeScore> = HashMap::new();
            for v in &s.variants {
                let Some(q) = vec_of.get(v.as_str()) else { continue };
                if is_zero(q) {
                    continue;
                }
                for ns in bank.score(q, words, cat) {
                    let slot = per_node.entry(ns.id.clone()).or_insert_with(|| ns.clone());
                    if ns.score > slot.score {
                        *slot = ns;
                    }
                }
            }
            best.extend(per_node.into_values().map(|n| (cat, n)));
        }
        best.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        let Some((cat, top)) = best.first().cloned() else { continue };
        let second = best.get(1).map(|b| b.1.score).unwrap_or(0.0);
        let typed = tokens[s.start..s.end].join(" ");
        if top.score < opts.min_score {
            continue;
        }
        if prejudice_dominates(top.bias, top.prejudice, opts.cohesion) {
            trace.push(TraceLine::new(
                "semantic",
                format!("{typed:?} looks like {} but its prejudice phrases score higher; rejected", top.id),
            ));
            continue;
        }
        if top.score - second < opts.margin {
            trace.push(TraceLine::new(
                "semantic",
                format!(
                    "{typed:?}: {} {:.2} vs {} {:.2} is inside the {:.2} margin; rejected",
                    top.id,
                    top.score,
                    best.get(1).map(|b| b.1.id.as_str()).unwrap_or("-"),
                    second,
                    opts.margin
                ),
            ));
            continue;
        }
        cands.push(Cand { start: s.start, end: s.end, category: cat, node: top.id, score: top.score, text: typed });
    }

    // Non-maximum suppression, then one span per node.
    cands.sort_by(|a, b| b.score.total_cmp(&a.score).then((b.end - b.start).cmp(&(a.end - a.start))));
    let mut taken: HashSet<usize> = HashSet::new();
    let mut used: HashSet<(Category, String)> = HashSet::new();
    let mut hits = Vec::new();
    for c in cands {
        if (c.start..c.end).any(|k| taken.contains(&k)) {
            continue;
        }
        if !used.insert((c.category, c.node.clone())) {
            continue;
        }
        (c.start..c.end).for_each(|k| {
            taken.insert(k);
        });
        trace.push(TraceLine::new(
            "semantic",
            format!("{:?} -> {:?} {} by embedding (score {:.2})", c.text, c.category, c.node, c.score),
        ));
        // Report the tail the stem variant dropped, for the role logic.
        let last = &tokens[c.end - 1];
        let reading = peel(last).into_iter().filter(|r| !r.tail.is_empty()).max_by_key(|r| r.tail.chars().count());
        let (tail, tail_kind) = reading.map(|r| (r.tail, r.kind)).unwrap_or((String::new(), TailKind::Bare));
        hits.push(Hit {
            category: c.category,
            node: c.node,
            start: c.start,
            end: c.end,
            via: Via::Embedding,
            score: c.score,
            text: c.text,
            tail,
            tail_kind,
        });
    }
    hits.sort_by_key(|h| h.start);
    Ok(hits)
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A stand-in embedder: character unigrams and bigrams hashed into a
    /// fixed-width vector, so phrases that share characters are close. It
    /// models "related words are near" well enough to test the plumbing; it is
    /// not a language model and says nothing about real accuracy.
    pub struct BigramEmbedder;

    pub fn bigram_vector(text: &str) -> Vec<f32> {
        const DIM: usize = 2048;
        let mut v = vec![0.0f32; DIM];
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        let mut bump = |s: &str, w: f32| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in s.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            // finalize so nearby strings spread over the whole vector
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            v[(h % DIM as u64) as usize] += w;
        };
        for c in &chars {
            bump(&c.to_string(), 0.3);
        }
        for w in chars.windows(2) {
            bump(&w.iter().collect::<String>(), 1.0);
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    }

    impl PhraseEmbedder for BigramEmbedder {
        fn embed_phrases(&self, phrases: &[String]) -> Result<Vec<Arc<[f32]>>> {
            Ok(phrases.iter().map(|p| bigram_vector(p).into()).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::from_json(
            r#"{
          "version": 1,
          "resource": {
            "log": {"bias": "원목, 장작, 통나무토막, firewood", "prejudice": "돌멩이, 바위", "exact_match": ["나무"]},
            "stone": {"bias": "돌멩이, 바위, 자갈", "prejudice": "장작, 원목", "exact_match": ["돌"]}
          },
          "place": {
            "warehouse": {"bias": "창고, 보관소, 물류창고", "prejudice": "베이스캠프", "exact_match": ["창고"]},
            "base_camp": {"bias": "베이스캠프, 기지, 본진", "prejudice": "창고", "exact_match": ["캠프"]}
          },
          "verb": {
            "harvest": {"bias": "채굴하다, 베어내다", "exact_match": ["채굴"]},
            "pickup": {"bias": "줍다, 집어올리다", "exact_match": ["줍"]},
            "deliver": {"bias": "가져오다, 운반하다, 옮기다", "exact_match": ["운반"]},
            "go": {"bias": "이동하다", "exact_match": ["이동"]},
            "stop": {"bias": "멈추다", "exact_match": ["멈춰"]}
          },
          "filler": {"exact_match": ["좀"]}
        }"#,
        )
        .unwrap()
    }

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn run(s: &str, opts: &SemanticOpts) -> (Vec<Hit>, Vec<TraceLine>) {
        let l = lex();
        let bank = SemanticBank::build(&l, &BigramEmbedder).unwrap();
        let t = toks(s);
        let mut trace = Vec::new();
        let hits = search_spans(&bank, &BigramEmbedder, &t, &vec![true; t.len()], opts, &mut trace).unwrap();
        (hits, trace)
    }

    #[test]
    fn pooling_helpers_match_the_logis_definitions() {
        let a: Arc<[f32]> = vec![1.0f32, 0.0].into();
        let b: Arc<[f32]> = vec![0.0f32, 1.0].into();
        let z: Arc<[f32]> = vec![0.0f32, 0.0].into();
        assert_eq!(max_pool(&[1.0, 0.0], &[b.clone(), a.clone()]), 1.0);
        assert_eq!(max_pool(&[1.0, 0.0], &[b.clone()]), 0.0, "negative or zero cosine floors at 0");
        assert!((weighted_max_pool(&[1.0, 0.0], &[(a.clone(), 0.8), (z, 1.0)]) - 0.8).abs() < 1e-6);
        assert!(prejudice_dominates(0.4, 0.5, 0.0));
        assert!(!prejudice_dominates(0.4, 0.5, 0.5), "cohesion widens the allowance");
        assert!(prejudice_dominates(0.0, 0.0, 0.0));
    }

    #[test]
    fn a_word_that_shares_a_bias_phrase_is_found_by_embedding() {
        let (hits, _) = run("장작을", &SemanticOpts::default());
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].node.as_str(), hits[0].category, hits[0].via), ("log", Category::Resource, Via::Embedding));
        assert_eq!(hits[0].tail, "을");
    }

    #[test]
    fn places_are_found_too_and_the_verb_penalty_keeps_verbs_out() {
        let (hits, _) = run("물류창고로", &SemanticOpts::default());
        assert_eq!(hits[0].node, "warehouse");
        assert_eq!(hits[0].tail_kind, TailKind::Direction);
        // "운반하다" is a verb phrase: it must not turn into a place or resource
        let (hits, _) = run("운반하다", &SemanticOpts::default());
        assert!(hits.is_empty(), "{hits:?}");
    }

    #[test]
    fn unrelated_words_do_not_match() {
        let (hits, _) = run("도끼로 휘두르기", &SemanticOpts::default());
        assert!(hits.is_empty(), "{hits:?}");
    }

    #[test]
    fn prejudice_phrases_push_confusable_words_away() {
        // "돌멩이" is a bias phrase of stone and a prejudice phrase of log: stone must win.
        let (hits, _) = run("돌멩이", &SemanticOpts::default());
        assert_eq!(hits[0].node, "stone");
    }

    #[test]
    fn a_tight_margin_rejects_an_ambiguous_span_and_says_why() {
        // scores never differ by more than 1, so a margin of 2 rejects every span
        let wide = SemanticOpts { min_score: 0.05, margin: 2.0, ..SemanticOpts::default() };
        let (hits, trace) = run("장작을", &wide);
        assert!(hits.is_empty());
        assert!(trace.iter().any(|t| t.detail.contains("margin")));
    }

    #[test]
    fn overlapping_spans_are_thinned_and_each_node_is_used_once() {
        let (hits, _) = run("원목 장작 창고", &SemanticOpts::default());
        let logs = hits.iter().filter(|h| h.node == "log").count();
        assert!(logs <= 1, "{hits:?}");
        assert!(hits.iter().any(|h| h.node == "warehouse"));
        // no two hits overlap
        for (i, a) in hits.iter().enumerate() {
            for b in hits.iter().skip(i + 1) {
                assert!(a.end <= b.start || b.end <= a.start, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn occupied_tokens_are_not_scored() {
        let l = lex();
        let bank = SemanticBank::build(&l, &BigramEmbedder).unwrap();
        let t = toks("장작을 창고로");
        let hits = search_spans(&bank, &BigramEmbedder, &t, &[false, true], &SemanticOpts::default(), &mut Vec::new()).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node, "warehouse");
    }
}
