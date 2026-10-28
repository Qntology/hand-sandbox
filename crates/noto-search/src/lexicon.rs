use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use crate::text::normalize;

#[derive(Clone, Debug, PartialEq)]
pub struct Phrase {
    pub text: String,
    pub weight: f32,
}

pub fn split_phrase_list(raw: &str) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, c) in chars.iter().enumerate() {
        let sep = match *c {
            ',' | '\n' | '|' => true,
            '/' => !is_abbreviation_slash(&chars, i),
            _ => false,
        };
        if sep {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        } else {
            cur.push(*c);
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    let mut seen = HashSet::new();
    out.retain(|p| seen.insert(p.clone()));
    out
}

fn is_abbreviation_slash(chars: &[char], i: usize) -> bool {
    let mut left = 0usize;
    let mut left_alpha = false;
    let mut j = i;
    while j > 0 && chars[j - 1].is_ascii_alphanumeric() {
        left += 1;
        left_alpha |= chars[j - 1].is_ascii_alphabetic();
        j -= 1;
    }
    let mut right = 0usize;
    let mut right_alpha = false;
    let mut k = i + 1;
    while k < chars.len() && chars[k].is_ascii_alphanumeric() {
        right += 1;
        right_alpha |= chars[k].is_ascii_alphabetic();
        k += 1;
    }
    left_alpha && right_alpha && (1..=2).contains(&left) && (1..=2).contains(&right)
}

pub fn weighted_phrases(raw: &str) -> Vec<Phrase> {
    split_phrase_list(raw)
        .into_iter()
        .map(|text| {
            let compact: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
            let total = compact.len().max(1);
            let digits = compact.iter().filter(|c| c.is_ascii_digit()).count();
            let ratio = digits as f32 / total as f32;
            let weight = if ratio >= 0.25 {
                0.80
            } else if digits > 0 {
                0.95
            } else {
                1.0
            };
            Phrase { text, weight }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: String,
    pub semantic: String,
    pub bias: Vec<Phrase>,
    pub prejudice: Vec<Phrase>,
    pub exact_match: Vec<String>,
    pub surface: Vec<String>,
    pub link: Option<String>,
    pub placed: bool,
    pub home: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Source,
    Drop,
    Waypoint,
}

#[derive(Clone, Debug)]
pub struct StructureEntity {
    pub id: String,
    pub link: String,
    pub semantic: String,
    pub bias: Vec<Phrase>,
    pub prejudice: Vec<Phrase>,
}

#[derive(Clone, Debug)]
pub struct VisionEntity {
    pub id: String,
    pub kind: EntityKind,
    pub link: String,
    pub semantic: String,
    pub bias: Vec<Phrase>,
    pub prejudice: Vec<Phrase>,
}

#[derive(Clone, Debug, Default)]
pub struct QuantityTable {
    pub counters: Vec<String>,
    pub native_units: BTreeMap<String, u32>,
    pub native_tens: BTreeMap<String, u32>,
    pub sino_digits: BTreeMap<String, u32>,
    pub sino_tens: BTreeMap<String, u32>,
    pub words: BTreeMap<String, u32>,
}

#[derive(Clone, Debug)]
pub struct Lexicon {
    pub version: u32,
    pub resources: Vec<Node>,
    pub places: Vec<Node>,
    pub verbs: Vec<Node>,
    pub filler: Node,
    pub negation: Node,
    pub direction_words: Vec<String>,
    pub source_words: Vec<String>,
    pub located_words: Vec<String>,
    pub quantity: QuantityTable,
    pub vision: Vec<VisionEntity>,
    pub structures: Vec<StructureEntity>,
}

#[derive(Deserialize)]
struct RawNode {
    #[serde(default)]
    semantic: String,
    #[serde(default)]
    bias: String,
    #[serde(default)]
    prejudice: String,
    #[serde(default)]
    exact_match: Vec<String>,
    #[serde(default)]
    surface: Vec<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    placed: bool,
    #[serde(default)]
    home: bool,
}

#[derive(Deserialize)]
struct RawEntity {
    kind: String,
    link: String,
    #[serde(default)]
    semantic: String,
    #[serde(default)]
    bias: String,
    #[serde(default)]
    prejudice: String,
}

#[derive(Deserialize, Default)]
struct RawQuantity {
    #[serde(default)]
    counters: Vec<String>,
    #[serde(default)]
    native_units: BTreeMap<String, u32>,
    #[serde(default)]
    native_tens: BTreeMap<String, u32>,
    #[serde(default)]
    sino_digits: BTreeMap<String, u32>,
    #[serde(default)]
    sino_tens: BTreeMap<String, u32>,
    #[serde(default)]
    words: BTreeMap<String, u32>,
}

#[derive(Deserialize, Default)]
struct RawPreposition {
    #[serde(default)]
    direction: Vec<String>,
    #[serde(default)]
    source: Vec<String>,
    #[serde(default)]
    located: Vec<String>,
}

#[derive(Deserialize)]
struct RawLexicon {
    version: u32,
    resource: BTreeMap<String, RawNode>,
    place: BTreeMap<String, RawNode>,
    verb: BTreeMap<String, RawNode>,
    filler: RawNode,
    #[serde(default)]
    negation: Option<RawNode>,
    #[serde(default)]
    preposition: RawPreposition,
    #[serde(default)]
    quantity: RawQuantity,
    #[serde(default)]
    vision: BTreeMap<String, RawEntity>,
}

fn norm_list(words: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    words
        .iter()
        .map(|w| normalize(w))
        .filter(|w| !w.is_empty())
        .filter(|w| seen.insert(w.clone()))
        .collect()
}

fn node_of(id: &str, raw: RawNode) -> Node {
    Node {
        id: id.to_string(),
        semantic: raw.semantic,
        bias: weighted_phrases(&raw.bias),
        prejudice: weighted_phrases(&raw.prejudice),
        exact_match: norm_list(&raw.exact_match),
        surface: norm_list(&raw.surface),
        link: raw.source,
        placed: raw.placed,
        home: raw.home,
    }
}

pub const KNOWN_VERBS: [&str; 6] = ["harvest", "pickup", "deliver", "go", "stop", "drop"];
pub const REQUIRED_VERBS: [&str; 5] = ["harvest", "pickup", "deliver", "go", "stop"];

impl Lexicon {
    pub fn from_json(raw: &str) -> Result<Lexicon> {
        let r: RawLexicon = serde_json::from_str(raw).context("parsing the world lexicon")?;
        let resources: Vec<Node> = r.resource.into_iter().map(|(k, v)| node_of(&k, v)).collect();
        let places: Vec<Node> = r.place.into_iter().map(|(k, v)| node_of(&k, v)).collect();
        let verbs: Vec<Node> = r.verb.into_iter().map(|(k, v)| node_of(&k, v)).collect();
        let filler = node_of("filler", r.filler);
        let negation = node_of(
            "negation",
            r.negation.unwrap_or(RawNode {
                semantic: String::new(),
                bias: String::new(),
                prejudice: String::new(),
                exact_match: Vec::new(),
                surface: Vec::new(),
                source: None,
                placed: false,
                home: false,
            }),
        );
        let mut vision = Vec::new();
        let mut structures = Vec::new();
        for (id, e) in r.vision {
            let kind = match e.kind.as_str() {
                "source" => EntityKind::Source,
                "drop" => EntityKind::Drop,
                "waypoint" => EntityKind::Waypoint,
                "structure" => {
                    structures.push(StructureEntity { id, link: e.link, semantic: e.semantic, bias: weighted_phrases(&e.bias), prejudice: weighted_phrases(&e.prejudice) });
                    continue;
                }
                other => bail!("vision entity {id}: unknown kind {other:?} (source, drop, waypoint or structure)"),
            };
            vision.push(VisionEntity {
                id,
                kind,
                link: e.link,
                semantic: e.semantic,
                bias: weighted_phrases(&e.bias),
                prejudice: weighted_phrases(&e.prejudice),
            });
        }
        let q = r.quantity;
        let quantity = QuantityTable {
            counters: norm_list(&q.counters),
            native_units: q.native_units,
            native_tens: q.native_tens,
            sino_digits: q.sino_digits,
            sino_tens: q.sino_tens,
            words: q.words,
        };
        let lex = Lexicon {
            version: r.version,
            resources,
            places,
            verbs,
            filler,
            negation,
            direction_words: norm_list(&r.preposition.direction),
            source_words: norm_list(&r.preposition.source),
            located_words: norm_list(&r.preposition.located),
            quantity,
            vision,
            structures,
        };
        lex.validate()?;
        Ok(lex)
    }

    pub fn validate(&self) -> Result<()> {
        for group in [&self.resources, &self.places, &self.verbs] {
            let mut owner: HashMap<String, &str> = HashMap::new();
            let mut ids = HashSet::new();
            for n in group.iter() {
                if !ids.insert(n.id.clone()) {
                    bail!("duplicate node id {:?}", n.id);
                }
                if n.exact_match.is_empty() && n.surface.is_empty() && n.bias.is_empty() {
                    bail!("node {:?} has no exact_match, surface or bias words", n.id);
                }
                for w in n.exact_match.iter().chain(n.surface.iter()) {
                    let key: String = w.split_whitespace().collect();
                    if let Some(prev) = owner.insert(key, &n.id) {
                        if prev != n.id {
                            bail!("word {w:?} is claimed by both {prev:?} and {:?}", n.id);
                        }
                    }
                }
            }
        }
        for v in &self.verbs {
            if !KNOWN_VERBS.contains(&v.id.as_str()) {
                bail!("verb node {:?} is not one of {:?}", v.id, KNOWN_VERBS);
            }
        }
        for want in REQUIRED_VERBS {
            if !self.verbs.iter().any(|v| v.id == want) {
                bail!("the lexicon has no verb node {want:?}");
            }
        }
        for e in &self.vision {
            let ok = match e.kind {
                EntityKind::Source | EntityKind::Drop => self.resources.iter().any(|r| r.id == e.link),
                EntityKind::Waypoint => self.places.iter().any(|p| p.id == e.link),
            };
            if !ok {
                return Err(anyhow!("vision entity {:?} links to {:?}, which is not defined", e.id, e.link));
            }
            if e.bias.is_empty() {
                bail!("vision entity {:?} has no bias phrases", e.id);
            }
        }
        for e in &self.structures {
            if !self.places.iter().any(|p| p.id == e.link) {
                return Err(anyhow!("structure entity {:?} links to {:?}, which is not a place", e.id, e.link));
            }
            if e.bias.is_empty() {
                bail!("structure entity {:?} has no bias phrases", e.id);
            }
        }
        if self.places.iter().filter(|p| p.home).count() > 1 {
            bail!("more than one place is marked home");
        }
        for r in self.resources.iter().filter(|r| r.placed) {
            if !self.vision.iter().any(|e| e.kind == EntityKind::Drop && e.link == r.id) {
                bail!("placed resource {:?} has no drop vision entity", r.id);
            }
        }
        Ok(())
    }

    pub fn resource(&self, id: &str) -> Option<&Node> {
        self.resources.iter().find(|n| n.id == id)
    }

    pub fn place(&self, id: &str) -> Option<&Node> {
        self.places.iter().find(|n| n.id == id)
    }

    pub fn verb(&self, id: &str) -> Option<&Node> {
        self.verbs.iter().find(|n| n.id == id)
    }

    pub fn is_placed(&self, resource: &str) -> bool {
        self.resource(resource).is_some_and(|n| n.placed)
    }

    pub fn home(&self) -> Option<&Node> {
        self.places.iter().find(|p| p.home)
    }

    pub fn all_phrases(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let mut push = |p: &Phrase| {
            if seen.insert(p.text.clone()) {
                out.push(p.text.clone());
            }
        };
        for n in self.resources.iter().chain(&self.places).chain(&self.verbs) {
            n.bias.iter().chain(&n.prejudice).for_each(&mut push);
        }
        self.filler.bias.iter().for_each(&mut push);
        for e in &self.vision {
            e.bias.iter().chain(&e.prejudice).for_each(&mut push);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrase_lists_split_dedupe_and_keep_short_abbreviations() {
        assert_eq!(split_phrase_list("a, b | c\nd, a"), vec!["a", "b", "c", "d"]);
        assert_eq!(split_phrase_list("w/o lid, tree/rock"), vec!["w/o lid", "tree", "rock"]);
        assert!(split_phrase_list("  ,, ").is_empty());
    }

    #[test]
    fn digit_heavy_phrases_weigh_less() {
        let p = weighted_phrases("log, 2026-03-15, route 66 east");
        assert_eq!(p[0].weight, 1.0);
        assert_eq!(p[1].weight, 0.80);
        assert_eq!(p[2].weight, 0.95);
    }

    const MIN: &str = r#"{
      "version": 1,
      "resource": {
        "log": {"semantic": "wood", "bias": "log, wood", "prejudice": "rock", "exact_match": ["나무", "Log"]},
        "stone": {"semantic": "rock", "bias": "stone", "exact_match": ["돌"]}
      },
      "place": {"warehouse": {"semantic": "store", "bias": "warehouse", "exact_match": ["창고"]}},
      "verb": {
        "harvest": {"exact_match": ["채굴"]}, "pickup": {"exact_match": ["줍"]},
        "deliver": {"exact_match": ["운반"]}, "go": {"exact_match": ["이동"]}, "stop": {"exact_match": ["멈춰"]}
      },
      "filler": {"exact_match": ["좀"]},
      "vision": {
        "tree": {"kind": "source", "link": "log", "bias": "a tree"},
        "warehouse": {"kind": "waypoint", "link": "warehouse", "bias": "a warehouse"}
      }
    }"#;

    #[test]
    fn a_minimal_lexicon_loads_and_normalizes_words() {
        let l = Lexicon::from_json(MIN).unwrap();
        assert_eq!(l.resources.len(), 2);
        assert_eq!(l.resource("log").unwrap().exact_match, vec!["나무", "log"]);
        assert_eq!(l.vision.len(), 2);
        assert!(l.all_phrases().contains(&"a tree".to_string()));
    }

    #[test]
    fn a_word_claimed_twice_is_rejected() {
        let bad = MIN.replace(r#""exact_match": ["돌"]"#, r#""exact_match": ["돌", "나무"]"#);
        let err = Lexicon::from_json(&bad).unwrap_err().to_string();
        assert!(err.contains("claimed by both"), "{err}");
    }

    #[test]
    fn a_dangling_vision_link_is_rejected() {
        let bad = MIN.replace(r#""link": "log""#, r#""link": "gold""#);
        let err = Lexicon::from_json(&bad).unwrap_err().to_string();
        assert!(err.contains("not defined"), "{err}");
    }

    #[test]
    fn structures_are_kept_apart_from_the_vision_entities_and_need_a_place() {
        let with = MIN.replace(
            r#""warehouse": {"kind": "waypoint", "link": "warehouse", "bias": "a warehouse"}"#,
            r#""warehouse": {"kind": "waypoint", "link": "warehouse", "bias": "a warehouse"},
               "shelf": {"kind": "structure", "link": "warehouse", "bias": "a steel shelf"}"#,
        );
        let l = Lexicon::from_json(&with).unwrap();
        assert_eq!(l.vision.len(), 2, "the recognisers' label set stays as it was");
        assert_eq!(l.structures.len(), 1);
        assert!(!l.all_phrases().contains(&"a steel shelf".to_string()));
        let bad = with.replace(r#""kind": "structure", "link": "warehouse""#, r#""kind": "structure", "link": "log""#);
        assert!(Lexicon::from_json(&bad).unwrap_err().to_string().contains("not a place"));
    }

    #[test]
    fn a_placed_resource_needs_a_drop_entity() {
        let placed = MIN.replace(r#""exact_match": ["돌"]}"#, r#""exact_match": ["돌"], "placed": true}"#);
        let err = Lexicon::from_json(&placed).unwrap_err().to_string();
        assert!(err.contains("no drop vision entity"), "{err}");
        let ok = placed.replace(r#""tree": {"kind": "source""#, r#""stone": {"kind": "drop", "link": "stone", "bias": "a stone"}, "tree": {"kind": "source""#);
        let l = Lexicon::from_json(&ok).unwrap();
        assert!(l.is_placed("stone") && !l.is_placed("log") && !l.is_placed("gold"));
    }

    #[test]
    fn a_missing_verb_node_is_rejected() {
        let bad = MIN.replace(r#""stop": {"exact_match": ["멈춰"]}"#, r#""wait": {"exact_match": ["대기"]}"#);
        assert!(Lexicon::from_json(&bad).is_err());
    }
}
