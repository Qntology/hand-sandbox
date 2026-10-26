//! From a typed command to a plan.
//!
//! Stages, each recorded in the trace:
//! 1. normalize and split into eojeol tokens;
//! 2. read quantities ("두 개", "3개");
//! 3. lexical matching (exact word, word behind a closed tail, unique prefix);
//! 4. embedding span search over the tokens nothing else used (optional);
//! 5. roles: which place is the destination (by its tail or preposition),
//!    which quantity belongs to which resource, what the verbs ask for;
//! 6. the plan, or a [`Problem`] saying what is missing.

use std::collections::HashSet;

use anyhow::Result;
use serde::Serialize;

use crate::lexical::{Category, Hit, LexMatcher, Via};
use crate::lexicon::Lexicon;
use crate::numbers::{Quantity, MAX_QUANTITY};
use crate::semantic::{search_spans, PhraseEmbedder, SemanticBank, SemanticOpts};
use crate::tails::TailKind;
use crate::text::{normalize, tokens as split_tokens};
use crate::TraceLine;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Target {
    pub resource: String,
    pub quantity: u32,
    pub via: Via,
    pub score: f32,
    /// The words as typed.
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaceRef {
    pub place: String,
    pub via: Via,
    pub score: f32,
    pub text: String,
}

/// Something to set down from the robot's own cargo.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DropTarget {
    pub resource: String,
    /// Items to set down; `None` (no number said) sets down all of that kind.
    pub quantity: Option<u32>,
    pub via: Via,
    pub score: f32,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Plan {
    /// Gather the targets and bring them to a place.
    Fetch { targets: Vec<Target>, destination: PlaceRef },
    /// Gather the targets and keep them in hand.
    Collect { targets: Vec<Target> },
    /// Walk to a place.
    Goto { place: PlaceRef },
    /// Set down what is already carried: the named items (everything when
    /// `items` is empty), here or, with `place`, after going there.
    Drop { items: Vec<DropTarget>, place: Option<PlaceRef> },
    /// Abort what is running.
    Stop,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Problem {
    pub code: String,
    /// Korean explanation shown to the user.
    pub message: String,
}

impl Problem {
    fn new(code: &str, message: impl Into<String>) -> Problem {
        Problem { code: code.to_string(), message: message.into() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParsedCommand {
    pub text: String,
    pub normalized: String,
    pub tokens: Vec<String>,
    pub hits: Vec<Hit>,
    pub quantities: Vec<(u32, usize, usize)>,
    /// Words ruled out by "말고", "except", ...
    pub excluded: Vec<String>,
    /// Content words nothing understood. They are ignored, and listed here.
    pub unmatched: Vec<String>,
    pub warnings: Vec<String>,
    pub plan: Option<Plan>,
    pub problem: Option<Problem>,
    pub trace: Vec<TraceLine>,
}

impl ParsedCommand {
    pub fn is_ready(&self) -> bool {
        self.plan.is_some()
    }
}

pub struct Parser {
    lex: Lexicon,
    matcher: LexMatcher,
    bank: Option<SemanticBank>,
    pub opts: SemanticOpts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Destination,
    Origin,
    Unspecified,
}

impl Parser {
    /// A parser that matches words only.
    pub fn new(lex: Lexicon) -> Parser {
        let matcher = LexMatcher::new(&lex);
        Parser { lex, matcher, bank: None, opts: SemanticOpts::default() }
    }

    /// Embeds the lexicon's command phrases so unknown words can be matched by meaning.
    pub fn attach_embedder(&mut self, emb: &dyn PhraseEmbedder) -> Result<()> {
        self.bank = Some(SemanticBank::build(&self.lex, emb)?);
        Ok(())
    }

    pub fn has_embedder(&self) -> bool {
        self.bank.is_some()
    }

    pub fn lexicon(&self) -> &Lexicon {
        &self.lex
    }

    /// Parses one command. `emb` is used only when [`Parser::attach_embedder`] ran.
    pub fn parse(&self, text: &str, emb: Option<&dyn PhraseEmbedder>) -> ParsedCommand {
        let mut trace: Vec<TraceLine> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        let normalized = normalize(text);
        let tokens = split_tokens(&normalized);
        trace.push(TraceLine::new("tokenize", format!("{text:?} -> {tokens:?}")));
        let mut out = ParsedCommand {
            text: text.to_string(),
            normalized,
            tokens: tokens.clone(),
            hits: Vec::new(),
            quantities: Vec::new(),
            excluded: Vec::new(),
            unmatched: Vec::new(),
            warnings: Vec::new(),
            plan: None,
            problem: None,
            trace: Vec::new(),
        };
        if tokens.is_empty() {
            out.problem = Some(Problem::new("empty", "명령이 비어 있습니다."));
            out.trace = trace;
            return out;
        }

        // 2. quantities
        let mut blocked = vec![false; tokens.len()];
        let mut quantities: Vec<Quantity> = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            match self.lex.quantity.read(&tokens, i) {
                Some(q) => {
                    trace.push(TraceLine::new(
                        "quantity",
                        format!("{:?} -> {}", tokens[q.start..q.end].join(" "), q.value),
                    ));
                    (q.start..q.end).for_each(|k| blocked[k] = true);
                    i = q.end;
                    quantities.push(q);
                }
                None => i += 1,
            }
        }

        // 3. lexical
        let mut hits = self.matcher.find(&tokens, &blocked, &mut trace);

        // 4. embedding over what is left
        if let (Some(bank), Some(emb)) = (&self.bank, emb) {
            let mut free: Vec<bool> = blocked.iter().map(|b| !b).collect();
            for h in &hits {
                (h.start..h.end).for_each(|k| free[k] = false);
            }
            // Filler, verbs and negation words never reach the embedding stage.
            match search_spans(bank, emb, &tokens, &free, &self.opts, &mut trace) {
                Ok(extra) => hits.extend(extra),
                Err(e) => {
                    warnings.push(format!("의미 검색을 건너뛰었습니다: {e}"));
                    trace.push(TraceLine::new("semantic", format!("skipped: {e}")));
                }
            }
            hits.sort_by_key(|h| h.start);
        }

        // 5. roles
        let mut excluded = vec![false; hits.len()];
        for (k, h) in hits.iter().enumerate() {
            if h.category != Category::Negation {
                continue;
            }
            let prev = (0..k).rev().find(|&j| matches!(hits[j].category, Category::Resource | Category::Place));
            let next = ((k + 1)..hits.len()).find(|&j| matches!(hits[j].category, Category::Resource | Category::Place));
            let adjacent_prev = prev.filter(|&j| hits[j].end == h.start);
            if let Some(j) = adjacent_prev.or(next).or(prev) {
                excluded[j] = true;
                out.excluded.push(hits[j].text.clone());
                trace.push(TraceLine::new("roles", format!("{:?} is ruled out by {:?}", hits[j].text, h.text)));
            }
        }

        let mut resources: Vec<usize> = (0..hits.len())
            .filter(|&k| hits[k].category == Category::Resource && !excluded[k])
            .collect();
        // a resource named twice keeps its first mention
        let mut seen = HashSet::new();
        resources.retain(|&k| seen.insert(hits[k].node.clone()));

        let mut qty: Vec<Option<u32>> = vec![None; hits.len()];
        for q in &quantities {
            let between_is_light = |from: usize, to: usize| {
                (from..to).all(|t| {
                    hits.iter().any(|h| h.start <= t && t < h.end && matches!(h.category, Category::Filler))
                })
            };
            let before = resources
                .iter()
                .rev()
                .copied()
                .find(|&k| hits[k].end <= q.start && qty[k].is_none() && between_is_light(hits[k].end, q.start));
            let after = resources
                .iter()
                .copied()
                .find(|&k| hits[k].start >= q.end && qty[k].is_none() && between_is_light(q.end, hits[k].start));
            let chosen = before.or(after).or_else(|| {
                // fall back to the nearest unassigned resource on either side
                resources
                    .iter()
                    .copied()
                    .filter(|&k| qty[k].is_none())
                    .min_by_key(|&k| {
                        if hits[k].end <= q.start { q.start - hits[k].end } else { hits[k].start.saturating_sub(q.end) }
                    })
            });
            match chosen {
                Some(k) => {
                    qty[k] = Some(q.value);
                    trace.push(TraceLine::new(
                        "roles",
                        format!("quantity {} belongs to {:?}", q.value, hits[k].text),
                    ));
                }
                None => warnings.push(format!("수량 {}을(를) 붙일 자원이 없어 무시했습니다.", q.value)),
            }
        }

        let mut targets: Vec<Target> = Vec::new();
        let mut problem: Option<Problem> = None;
        for &k in &resources {
            let mut n = qty[k].unwrap_or(1);
            if n == 0 {
                problem = Some(Problem::new("zero_quantity", format!("{:?}의 수량이 0입니다.", hits[k].text)));
            }
            if n > MAX_QUANTITY {
                warnings.push(format!("수량 {n}은(는) 최대 {MAX_QUANTITY}개로 줄였습니다."));
                n = MAX_QUANTITY;
            }
            targets.push(Target {
                resource: hits[k].node.clone(),
                quantity: n,
                via: hits[k].via,
                score: hits[k].score,
                text: hits[k].text.clone(),
            });
        }

        // places and their roles
        let mut places: Vec<(usize, Role)> = Vec::new();
        for (k, h) in hits.iter().enumerate() {
            if h.category != Category::Place || excluded[k] {
                continue;
            }
            let from_tail = match h.tail_kind {
                TailKind::Direction => Some(Role::Destination),
                TailKind::Source => Some(Role::Origin),
                _ => None,
            };
            let from_prep = || {
                // the nearest earlier hit, skipping "the"-like filler, if adjacent to this place
                (0..k).rev().find(|&j| hits[j].category != Category::Filler).and_then(|j| match hits[j].category {
                    Category::Direction => Some(Role::Destination),
                    Category::Source => Some(Role::Origin),
                    _ => None,
                })
            };
            places.push((k, from_tail.or_else(from_prep).unwrap_or(Role::Unspecified)));
        }
        let verbs: HashSet<&str> = hits
            .iter()
            .enumerate()
            .filter(|(k, h)| h.category == Category::Verb && !excluded[*k])
            .map(|(_, h)| h.node.as_str())
            .collect();

        let destinations: Vec<usize> = places.iter().filter(|p| p.1 == Role::Destination).map(|p| p.0).collect();
        let origins: Vec<usize> = places.iter().filter(|p| p.1 == Role::Origin).map(|p| p.0).collect();
        let loose: Vec<usize> = places.iter().filter(|p| p.1 == Role::Unspecified).map(|p| p.0).collect();
        let place_ref = |k: usize| PlaceRef {
            place: hits[k].node.clone(),
            via: hits[k].via,
            score: hits[k].score,
            text: hits[k].text.clone(),
        };

        let mut destination: Option<usize> = None;
        let distinct = |ks: &[usize]| ks.iter().map(|&k| hits[k].node.as_str()).collect::<HashSet<_>>().len();
        if distinct(&destinations) > 1 {
            problem = problem.or(Some(Problem::new(
                "conflicting_destinations",
                "도착 지점이 둘 이상입니다. 한 곳만 말해 주세요.",
            )));
        } else if let Some(&k) = destinations.first() {
            destination = Some(k);
        } else if distinct(&loose) == 1 {
            let k = loose[0];
            warnings.push(format!("{:?}에 방향 표지(로, 에, to)가 없어 도착 지점으로 보았습니다.", hits[k].text));
            destination = Some(k);
        } else if distinct(&loose) > 1 {
            problem = problem.or(Some(Problem::new(
                "conflicting_destinations",
                "장소가 둘 이상인데 어느 쪽이 도착 지점인지 알 수 없습니다. '창고로'처럼 말해 주세요.",
            )));
        }
        if !origins.is_empty() {
            let names: Vec<&str> = origins.iter().map(|&k| hits[k].text.as_str()).collect();
            if destination.is_none() && targets.is_empty() {
                problem = problem.or(Some(Problem::new(
                    "unsupported_origin",
                    format!("출발 지점({})을 지정하는 명령은 아직 지원하지 않습니다.", names.join(", ")),
                )));
            } else {
                warnings.push(format!("출발 지점({})은 사용하지 않고 무시했습니다.", names.join(", ")));
            }
        }

        // unmatched content words
        let mut covered = blocked.clone();
        for h in &hits {
            (h.start..h.end).for_each(|k| covered[k] = true);
        }
        out.unmatched = tokens.iter().enumerate().filter(|(k, _)| !covered[*k]).map(|(_, t)| t.clone()).collect();
        if !out.unmatched.is_empty() {
            warnings.push(format!("이해하지 못해 무시한 말: {}", out.unmatched.join(", ")));
        }

        // 6. plan
        let gathers = ["harvest", "pickup", "deliver"].iter().any(|v| verbs.contains(v));
        let plan = if problem.is_some() {
            None
        } else if verbs.contains("stop") {
            if !targets.is_empty() {
                warnings.push("정지 명령과 채취 명령이 함께 있어 정지만 실행합니다.".to_string());
            }
            Some(Plan::Stop)
        } else if verbs.contains("drop") && !gathers {
            // Set down what is carried; a number limits it, no number means all of it.
            let items: Vec<DropTarget> = resources
                .iter()
                .map(|&k| DropTarget { resource: hits[k].node.clone(), quantity: qty[k].map(|n| n.min(MAX_QUANTITY)), via: hits[k].via, score: hits[k].score, text: hits[k].text.clone() })
                .collect();
            trace.push(TraceLine::new("roles", format!("drop {} from the cargo{}", if items.is_empty() { "everything".to_string() } else { format!("{} kind(s)", items.len()) }, destination.map(|k| format!(" at {:?}", hits[k].text)).unwrap_or_default())));
            Some(Plan::Drop { items, place: destination.map(place_ref) })
        } else if !targets.is_empty() {
            match destination {
                Some(k) => Some(Plan::Fetch { targets: targets.clone(), destination: place_ref(k) }),
                None if verbs.contains("deliver") => {
                    problem = Some(Problem::new(
                        "missing_destination",
                        "어디로 가져올지 알 수 없습니다. 예: '나무 2개를 창고로 가져와'",
                    ));
                    None
                }
                None => Some(Plan::Collect { targets: targets.clone() }),
            }
        } else if destination.is_some() && !gathers {
            destination.map(|k| Plan::Goto { place: place_ref(k) })
        } else if gathers {
            // The verbs ask to gather or carry something, but no known resource was named.
            let unknown = if out.unmatched.is_empty() {
                String::new()
            } else {
                format!(" (이해하지 못한 말: {})", out.unmatched.join(", "))
            };
            problem = Some(Problem::new(
                "unknown_item",
                format!("무엇을 가져올지 알 수 없습니다{unknown}. 나무·돌·석탄·철광석 중에서 말해 주세요."),
            ));
            None
        } else {
            problem = Some(Problem::new(
                "no_target",
                "무엇을 할지 알 수 없습니다. 자원(나무·돌·석탄·철광석)이나 장소(창고·베이스캠프)를 말해 주세요.",
            ));
            None
        };
        trace.push(TraceLine::new(
            "plan",
            match (&plan, &problem) {
                (Some(p), _) => format!("{p:?}"),
                (None, Some(pr)) => format!("no plan: {} ({})", pr.code, pr.message),
                _ => "no plan".to_string(),
            },
        ));

        out.hits = hits;
        out.quantities = quantities.iter().map(|q| (q.value, q.start, q.end)).collect();
        out.warnings = warnings;
        out.plan = plan;
        out.problem = problem;
        out.trace = trace;
        out
    }
}
