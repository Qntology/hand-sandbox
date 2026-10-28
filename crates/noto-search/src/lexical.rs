use std::collections::{BTreeMap, HashMap, HashSet};

use serde::Serialize;

use crate::lexicon::{Lexicon, Node};
use crate::tails::{peel, verb_remainder, TailKind};
use crate::TraceLine;

pub const MAX_SPAN: usize = 3;
const PREFIX_COVERAGE: f32 = 0.66;
const PREFIX_MIN_CHARS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Resource,
    Place,
    Negation,
    Direction,
    Source,
    Verb,
    Filler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Via {
    Exact,
    Tail,
    Prefix,
    Embedding,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Hit {
    pub category: Category,
    pub node: String,
    pub start: usize,
    pub end: usize,
    pub via: Via,
    pub score: f32,
    pub text: String,
    pub tail: String,
    pub tail_kind: TailKind,
}

#[derive(Clone, Debug)]
struct Entry {
    category: Category,
    node: String,
}

pub struct LexMatcher {
    words: HashMap<String, Vec<Entry>>,
    verb_stems: Vec<(String, String)>,
    prefix_words: Vec<(String, Category, String)>,
}

fn key_of(words: &str) -> String {
    words.split_whitespace().collect()
}

fn add(words: &mut HashMap<String, Vec<Entry>>, w: &str, category: Category, node: &str) {
    let key = key_of(w);
    if key.is_empty() {
        return;
    }
    let e = Entry { category, node: node.to_string() };
    let slot = words.entry(key).or_default();
    if !slot.iter().any(|x| x.category == e.category && x.node == e.node) {
        slot.push(e);
    }
}

impl LexMatcher {
    pub fn new(lex: &Lexicon) -> LexMatcher {
        let mut words: HashMap<String, Vec<Entry>> = HashMap::new();
        let mut verb_stems = Vec::new();
        let mut prefix_words = Vec::new();

        let mut noun = |nodes: &[Node], category: Category, prefix_words: &mut Vec<(String, Category, String)>| {
            for n in nodes {
                for w in n.exact_match.iter().chain(&n.surface) {
                    add(&mut words, w, category, &n.id);
                    let key = key_of(w);
                    if key.chars().count() > PREFIX_MIN_CHARS {
                        prefix_words.push((key, category, n.id.clone()));
                    }
                }
            }
        };
        noun(&lex.resources, Category::Resource, &mut prefix_words);
        noun(&lex.places, Category::Place, &mut prefix_words);

        for n in &lex.verbs {
            for w in &n.surface {
                add(&mut words, w, Category::Verb, &n.id);
            }
            for w in &n.exact_match {
                if w.contains(' ') || w.chars().count() < 2 {
                    add(&mut words, w, Category::Verb, &n.id);
                } else {
                    verb_stems.push((w.clone(), n.id.clone()));
                }
            }
        }
        for w in lex.filler.exact_match.iter().chain(&lex.filler.surface).chain(&lex.located_words) {
            add(&mut words, w, Category::Filler, "");
        }
        for w in lex.negation.exact_match.iter().chain(&lex.negation.surface) {
            add(&mut words, w, Category::Negation, "");
        }
        for w in &lex.direction_words {
            add(&mut words, w, Category::Direction, "");
        }
        for w in &lex.source_words {
            add(&mut words, w, Category::Source, "");
        }
        LexMatcher { words, verb_stems, prefix_words }
    }

    fn verb_hit(&self, token: &str, start: usize) -> Option<Hit> {
        let mut best: Option<(usize, &str, &str)> = None;
        for (stem, node) in &self.verb_stems {
            if let Some(rest) = verb_remainder(token, stem) {
                let len = stem.chars().count();
                if best.map(|b| len > b.0).unwrap_or(true) {
                    best = Some((len, node.as_str(), rest));
                }
            }
        }
        best.map(|(_, node, rest)| Hit {
            category: Category::Verb,
            node: node.to_string(),
            start,
            end: start + 1,
            via: if rest.is_empty() { Via::Exact } else { Via::Tail },
            score: if rest.is_empty() { 1.0 } else { 0.95 },
            text: token.to_string(),
            tail: rest.to_string(),
            tail_kind: TailKind::Bare,
        })
    }

    pub fn find(&self, tokens: &[String], blocked: &[bool], trace: &mut Vec<TraceLine>) -> Vec<Hit> {
        let mut candidates: Vec<Hit> = Vec::new();
        for i in 0..tokens.len() {
            if blocked.get(i).copied().unwrap_or(false) {
                continue;
            }
            if let Some(h) = self.verb_hit(&tokens[i], i) {
                candidates.push(h);
            }
            for n in 1..=MAX_SPAN {
                if i + n > tokens.len() {
                    break;
                }
                if (i..i + n).any(|k| blocked.get(k).copied().unwrap_or(false)) {
                    break;
                }
                let head: String = tokens[i..i + n - 1].iter().map(String::as_str).collect();
                let last = &tokens[i + n - 1];
                let typed = tokens[i..i + n].join(" ");
                let mut exact_found = false;
                for reading in peel(last) {
                    let key = format!("{head}{}", reading.stem);
                    if let Some(owners) = self.words.get(&key) {
                        exact_found = true;
                        for o in owners {
                            candidates.push(Hit {
                                category: o.category,
                                node: o.node.clone(),
                                start: i,
                                end: i + n,
                                via: if reading.tail.is_empty() { Via::Exact } else { Via::Tail },
                                score: if reading.tail.is_empty() { 1.0 } else { 0.98 },
                                text: typed.clone(),
                                tail: reading.tail.clone(),
                                tail_kind: reading.kind,
                            });
                        }
                    }
                }
                if exact_found {
                    continue;
                }
                for reading in peel(last) {
                    let key = format!("{head}{}", reading.stem);
                    let len = key.chars().count();
                    if len < PREFIX_MIN_CHARS {
                        continue;
                    }
                    let mut owners: BTreeMap<(Category, &str), usize> = BTreeMap::new();
                    for (word, cat, node) in &self.prefix_words {
                        let wl = word.chars().count();
                        if wl > len && word.starts_with(&key) && len as f32 >= PREFIX_COVERAGE * wl as f32 {
                            owners.insert((*cat, node.as_str()), wl);
                        }
                    }
                    if owners.len() == 1 {
                        let ((cat, node), wl) = owners.into_iter().next().unwrap();
                        candidates.push(Hit {
                            category: cat,
                            node: node.to_string(),
                            start: i,
                            end: i + n,
                            via: Via::Prefix,
                            score: 0.7 * (len as f32 / wl as f32),
                            text: typed.clone(),
                            tail: reading.tail.clone(),
                            tail_kind: reading.kind,
                        });
                        break;
                    } else if owners.len() > 1 {
                        trace.push(TraceLine::new(
                            "lexical",
                            format!("{typed:?} is a prefix of several dictionary words; left to the embedding stage"),
                        ));
                    }
                }
            }
        }

        candidates.sort_by(|a, b| {
            (b.end - b.start)
                .cmp(&(a.end - a.start))
                .then(b.score.total_cmp(&a.score))
                .then(a.category.cmp(&b.category))
                .then(a.start.cmp(&b.start))
        });
        let mut taken: HashSet<usize> = HashSet::new();
        let mut chosen: Vec<Hit> = Vec::new();
        for h in candidates {
            if (h.start..h.end).any(|k| taken.contains(&k)) {
                continue;
            }
            (h.start..h.end).for_each(|k| {
                taken.insert(k);
            });
            chosen.push(h);
        }
        chosen.sort_by_key(|h| h.start);
        for h in &chosen {
            trace.push(TraceLine::new(
                "lexical",
                format!(
                    "{:?} -> {:?} {} via {:?} (score {:.2}{})",
                    h.text,
                    h.category,
                    if h.node.is_empty() { "-" } else { &h.node },
                    h.via,
                    h.score,
                    if h.tail.is_empty() { String::new() } else { format!(", tail {:?}", h.tail) }
                ),
            ));
        }
        chosen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> Lexicon {
        Lexicon::from_json(
            r#"{
          "version": 1,
          "resource": {
            "log": {"exact_match": ["나무", "통나무", "log", "logs"]},
            "stone": {"exact_match": ["돌", "돌멩이"]},
            "iron_ore": {"exact_match": ["철광석", "iron ore"]}
          },
          "place": {
            "base_camp": {"exact_match": ["베이스캠프", "캠프"]},
            "warehouse": {"exact_match": ["창고", "warehouse"]}
          },
          "verb": {
            "harvest": {"exact_match": ["채굴", "mine"], "surface": ["캐서", "캐"]},
            "pickup": {"exact_match": ["줍", "pick up"]},
            "deliver": {"exact_match": ["가져오", "bring"], "surface": ["가져와", "가져와줘"]},
            "go": {"exact_match": ["이동"]},
            "stop": {"exact_match": ["멈춰"]}
          },
          "filler": {"exact_match": ["좀", "please", "the"]},
          "negation": {"exact_match": ["말고", "except"]},
          "preposition": {"direction": ["to", "into"], "source": ["from"]}
        }"#,
        )
        .unwrap()
    }

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn find(s: &str) -> Vec<Hit> {
        let m = LexMatcher::new(&lex());
        let t = toks(s);
        m.find(&t, &vec![false; t.len()], &mut Vec::new())
    }

    fn pick<'a>(hits: &'a [Hit], cat: Category) -> Vec<&'a Hit> {
        hits.iter().filter(|h| h.category == cat).collect()
    }

    #[test]
    fn exact_words_and_tails() {
        let h = find("나무를 창고로 가져와");
        let r = pick(&h, Category::Resource);
        assert_eq!((r[0].node.as_str(), r[0].via, r[0].tail.as_str()), ("log", Via::Tail, "를"));
        let p = pick(&h, Category::Place);
        assert_eq!(p[0].node, "warehouse");
        assert_eq!(p[0].tail_kind, TailKind::Direction);
        let v = pick(&h, Category::Verb);
        assert_eq!((v[0].node.as_str(), v[0].via), ("deliver", Via::Exact));
    }

    #[test]
    fn tails_must_agree_with_the_stem_or_the_word_stays_unmatched() {
        let h = find("나무을");
        assert!(pick(&h, Category::Resource).is_empty());
    }

    #[test]
    fn multiword_phrases_match_with_and_without_spaces() {
        let h = find("bring iron ore");
        assert_eq!(pick(&h, Category::Resource)[0].node, "iron_ore");
        assert_eq!(pick(&h, Category::Resource)[0].end - pick(&h, Category::Resource)[0].start, 2);
        let h = find("베이스 캠프로");
        assert_eq!(pick(&h, Category::Place)[0].node, "base_camp");
        let h = find("베이스캠프에서");
        assert_eq!(pick(&h, Category::Place)[0].tail_kind, TailKind::Source);
    }

    #[test]
    fn verbs_take_closed_endings_only() {
        assert_eq!(pick(&find("채굴해서"), Category::Verb)[0].node, "harvest");
        assert_eq!(pick(&find("채굴해줘"), Category::Verb)[0].tail, "해줘");
        assert!(pick(&find("채굴장"), Category::Verb).is_empty(), "채굴장 is a place, not the verb");
        assert_eq!(pick(&find("가져오세요"), Category::Verb)[0].node, "deliver");
        assert_eq!(pick(&find("가져와줘"), Category::Verb)[0].node, "deliver");
        assert_eq!(pick(&find("캐서"), Category::Verb)[0].node, "harvest");
        assert!(pick(&find("캐릭터"), Category::Verb).is_empty());
    }

    #[test]
    fn a_unique_prefix_is_accepted_and_an_ambiguous_one_is_not() {
        let h = find("철광");
        let r = pick(&h, Category::Resource);
        assert_eq!((r[0].node.as_str(), r[0].via), ("iron_ore", Via::Prefix));
        assert!(r[0].score < 0.7 && r[0].score > 0.4);
        assert!(pick(&find("베이스"), Category::Place).is_empty());
    }

    #[test]
    fn longer_spans_beat_shorter_ones_inside_them() {
        let h = find("iron ore");
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].node, "iron_ore");
    }

    #[test]
    fn filler_negation_and_prepositions_are_recognised() {
        let h = find("please bring logs to the warehouse");
        assert_eq!(pick(&h, Category::Filler).len(), 2);
        assert_eq!(pick(&h, Category::Direction).len(), 1);
        let h = find("나무 말고 돌");
        assert_eq!(pick(&h, Category::Negation).len(), 1);
    }

    #[test]
    fn blocked_tokens_are_skipped() {
        let m = LexMatcher::new(&lex());
        let t = toks("돌 나무");
        let hits = m.find(&t, &[true, false], &mut Vec::new());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node, "log");
    }
}
