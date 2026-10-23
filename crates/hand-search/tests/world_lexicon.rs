//! The shipped dictionary (`data/world_lexicon.json`) driven through the parser.
//!
//! These are consistency tests over the project's own vocabulary and rules.
//! They say nothing about how well a real embedding model understands
//! unseen Korean; the semantic test uses a character-bigram stand-in.

use std::sync::Arc;

use hand_search::{
    Category, Lexicon, ParsedCommand, Parser, PhraseEmbedder, Plan, Via, VisionBank,
};

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn parser() -> Parser {
    Parser::new(Lexicon::from_json(LEXICON).expect("the shipped lexicon loads"))
}

fn parse(text: &str) -> ParsedCommand {
    parser().parse(text, None)
}

fn fetch(p: &ParsedCommand) -> (Vec<(String, u32)>, String) {
    match p.plan.as_ref().unwrap_or_else(|| panic!("no plan for {:?}: {:?}", p.text, p.problem)) {
        Plan::Fetch { targets, destination } => (
            targets.iter().map(|t| (t.resource.clone(), t.quantity)).collect(),
            destination.place.clone(),
        ),
        other => panic!("expected Fetch for {:?}, got {other:?}", p.text),
    }
}

#[test]
fn the_shipped_lexicon_is_valid_and_complete() {
    let l = Lexicon::from_json(LEXICON).unwrap();
    for id in ["log", "stone", "coal", "iron_ore"] {
        assert!(l.resource(id).is_some(), "resource {id}");
        assert!(!l.vision.iter().filter(|e| e.link == id).collect::<Vec<_>>().is_empty(), "vision for {id}");
    }
    for id in ["base_camp", "warehouse"] {
        assert!(l.place(id).is_some());
        assert!(l.vision.iter().any(|e| e.link == id));
    }
    // the world defines these exact entity ids (world-sim kinds / waypoint keys)
    for id in ["tree", "boulder", "coal_ore", "iron_ore_node", "log", "stone", "coal", "iron_ore", "base_camp", "warehouse"] {
        assert!(l.vision.iter().any(|e| e.id == id), "vision entity {id}");
    }
    assert!(l.all_phrases().len() > 100);
}

#[test]
fn harvest_and_bring_with_a_quantity() {
    let p = parse("나무 두 개를 캐서 창고로 가져와");
    assert_eq!(fetch(&p), (vec![("log".to_string(), 2)], "warehouse".to_string()));
    assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    assert!(p.unmatched.is_empty());
}

#[test]
fn a_digit_quantity_and_a_polite_verb() {
    let p = parse("석탄 3개 베이스캠프로 옮겨줘");
    assert_eq!(fetch(&p), (vec![("coal".to_string(), 3)], "base_camp".to_string()));
}

#[test]
fn quantity_before_the_noun_with_a_genitive() {
    let p = parse("두 개의 돌을 창고에 가져다 놔");
    assert_eq!(fetch(&p), (vec![("stone".to_string(), 2)], "warehouse".to_string()));
}

#[test]
fn two_resources_each_with_its_own_count() {
    let p = parse("나무 2개 돌 3개 창고로 가져와");
    assert_eq!(
        fetch(&p),
        (vec![("log".to_string(), 2), ("stone".to_string(), 3)], "warehouse".to_string())
    );
}

#[test]
fn a_resource_ruled_out_is_not_fetched() {
    let p = parse("나무 말고 돌 창고로 가져와");
    assert_eq!(fetch(&p), (vec![("stone".to_string(), 1)], "warehouse".to_string()));
    assert_eq!(p.excluded, vec!["나무".to_string()]);
}

#[test]
fn english_commands_work_too() {
    let p = parse("bring 2 logs to the warehouse");
    assert_eq!(fetch(&p), (vec![("log".to_string(), 2)], "warehouse".to_string()));
    let p = parse("Please mine iron ore and carry it to base camp");
    assert_eq!(fetch(&p), (vec![("iron_ore".to_string(), 1)], "base_camp".to_string()));
}

#[test]
fn without_a_destination_the_items_stay_in_hand() {
    let p = parse("돌 하나 주워");
    match p.plan.unwrap() {
        Plan::Collect { targets } => assert_eq!((targets[0].resource.as_str(), targets[0].quantity), ("stone", 1)),
        other => panic!("{other:?}"),
    }
}

#[test]
fn bring_without_saying_where_asks_where() {
    let p = parse("철광석 가져와");
    assert!(p.plan.is_none());
    assert_eq!(p.problem.unwrap().code, "missing_destination");
}

#[test]
fn walking_to_a_place_and_stopping() {
    match parse("창고로 이동해").plan.unwrap() {
        Plan::Goto { place } => assert_eq!(place.place, "warehouse"),
        other => panic!("{other:?}"),
    }
    assert_eq!(parse("멈춰").plan, Some(Plan::Stop));
    assert_eq!(parse("stop").plan, Some(Plan::Stop));
}

#[test]
fn a_place_without_a_direction_marker_is_a_destination_with_a_warning() {
    let p = parse("창고 석탄 가져와");
    assert_eq!(fetch(&p).1, "warehouse");
    assert!(p.warnings.iter().any(|w| w.contains("방향 표지")));
}

#[test]
fn two_places_without_markers_are_refused() {
    let p = parse("창고 베이스캠프 나무 가져와");
    assert_eq!(p.problem.unwrap().code, "conflicting_destinations");
}

#[test]
fn an_origin_is_ignored_with_a_warning_when_a_destination_exists() {
    let p = parse("창고에서 석탄 베이스캠프로 가져와");
    assert_eq!(fetch(&p), (vec![("coal".to_string(), 1)], "base_camp".to_string()));
    assert!(p.warnings.iter().any(|w| w.contains("출발 지점")));
}

#[test]
fn nothing_understood_is_reported_not_guessed() {
    let p = parse("ㅋㅋㅋ 음");
    assert_eq!(p.problem.as_ref().unwrap().code, "no_target");
    let p = parse("   ");
    assert_eq!(p.problem.unwrap().code, "empty");
    // unknown words next to known ones are listed, and the rest still runs
    let p = parse("다이아몬드 나무 창고로 가져와");
    assert_eq!(fetch(&p).0, vec![("log".to_string(), 1)]);
    assert_eq!(p.unmatched, vec!["다이아몬드".to_string()]);
}

#[test]
fn quantity_limits() {
    let p = parse("나무 500개 창고로 가져와");
    assert_eq!(fetch(&p).0, vec![("log".to_string(), 99)]);
    assert!(p.warnings.iter().any(|w| w.contains("줄였")));
    let p = parse("나무 0개 창고로 가져와");
    assert_eq!(p.problem.unwrap().code, "zero_quantity");
}

#[test]
fn tails_must_agree_so_a_misspelt_particle_does_not_match() {
    // 창고을 is not Korean; the word stays unmatched rather than being repaired
    let p = parse("나무를 창고을 가져와");
    assert!(p.hits.iter().all(|h| h.category != Category::Place));
}

#[test]
fn every_parse_leaves_a_trace() {
    let p = parse("나무 두 개를 캐서 창고로 가져와");
    let stages: Vec<&str> = p.trace.iter().map(|t| t.stage.as_str()).collect();
    for s in ["tokenize", "quantity", "lexical", "roles", "plan"] {
        assert!(stages.contains(&s), "missing stage {s}: {stages:?}");
    }
}

/// Character n-gram stand-in for an embedding model.
struct Bigrams;

fn vector(text: &str) -> Vec<f32> {
    const DIM: usize = 2048;
    let mut v = vec![0.0f32; DIM];
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut bump = |s: String, w: f32| {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        v[(h % DIM as u64) as usize] += w;
    };
    for c in &chars {
        bump(c.to_string(), 0.3);
    }
    for w in chars.windows(2) {
        bump(w.iter().collect(), 1.0);
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

impl PhraseEmbedder for Bigrams {
    fn embed_phrases(&self, phrases: &[String]) -> anyhow::Result<Vec<Arc<[f32]>>> {
        Ok(phrases.iter().map(|p| vector(p).into()).collect())
    }
}

#[test]
fn an_unknown_compound_is_found_by_embedding_and_the_known_words_are_not_re_scored() {
    let mut p = parser();
    p.attach_embedder(&Bigrams).unwrap();
    assert!(p.has_embedder());
    let r = p.parse("장작더미 두 개를 물류창고로 가져와", Some(&Bigrams));
    // 물류창고 is an exact word, 장작더미 is not
    let (targets, dest) = fetch(&r);
    assert_eq!(targets, vec![("log".to_string(), 2)]);
    assert_eq!(dest, "warehouse");
    let by_embedding: Vec<_> = r.hits.iter().filter(|h| h.via == Via::Embedding).collect();
    assert_eq!(by_embedding.len(), 1, "{:?}", r.hits);
    assert_eq!(by_embedding[0].text, "장작더미");
    assert!(r.trace.iter().any(|t| t.stage == "semantic"));
}

#[test]
fn the_lexical_stage_works_without_any_embedder() {
    let p = parser();
    assert!(!p.has_embedder());
    let r = p.parse("장작더미 창고로 가져와", None);
    let problem = r.problem.unwrap();
    assert_eq!(problem.code, "unknown_item", "an unknown compound stays unknown without embeddings");
    assert!(problem.message.contains("장작더미"), "the message names the word it did not understand");
}

#[test]
fn vision_prompts_embed_and_score_through_the_same_dictionary() {
    let l = Lexicon::from_json(LEXICON).unwrap();
    let bank = VisionBank::build(&l, &Bigrams).unwrap();
    assert_eq!(bank.entities.len(), l.vision.len());
    let q = vector("an orange flag on a thin white pole");
    let s = bank.score(&q, &|c| c, 0.5);
    assert_eq!(s[0].id, "base_camp");
    let mut ids = bank.ids_for_resource("coal");
    ids.sort();
    assert_eq!(ids, vec!["coal", "coal_ore"]);
}
