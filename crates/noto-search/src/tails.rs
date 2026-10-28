use crate::text::{ends_with_final, ends_with_rieul};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TailKind {
    Bare,
    Object,
    Direction,
    Source,
    Link,
    Plural,
    Copula,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Agree {
    Any,
    Final,
    Open,
    FinalNotRieul,
    OpenOrRieul,
}

struct Tail {
    text: &'static str,
    kind: TailKind,
    agree: Agree,
}

const fn t(text: &'static str, kind: TailKind, agree: Agree) -> Tail {
    Tail { text, kind, agree }
}

use Agree::*;
use TailKind::*;

const NOMINAL: &[Tail] = &[
    t("에서는", Source, Any),
    t("에서도", Source, Any),
    t("에서의", Source, Any),
    t("에게서", Source, Any),
    t("한테서", Source, Any),
    t("으로까지", Direction, FinalNotRieul),
    t("로까지", Direction, OpenOrRieul),
    t("으로는", Direction, FinalNotRieul),
    t("으로도", Direction, FinalNotRieul),
    t("으로의", Direction, FinalNotRieul),
    t("로는", Direction, OpenOrRieul),
    t("로도", Direction, OpenOrRieul),
    t("로의", Direction, OpenOrRieul),
    t("쪽으로", Direction, Any),
    t("쪽에서", Source, Any),
    t("쪽에", Direction, Any),
    t("에서", Source, Any),
    t("부터의", Source, Any),
    t("부터", Source, Any),
    t("으로", Direction, FinalNotRieul),
    t("에게", Direction, Any),
    t("한테", Direction, Any),
    t("까지의", Direction, Any),
    t("까지", Direction, Any),
    t("에는", Direction, Any),
    t("에도", Direction, Any),
    t("보다", Object, Any),
    t("처럼", Object, Any),
    t("마다", Object, Any),
    t("조차", Object, Any),
    t("마저", Object, Any),
    t("밖에", Object, Any),
    t("이라도", Object, Final),
    t("라도", Object, Open),
    t("이든", Object, Final),
    t("든", Object, Open),
    t("이랑", Object, Final),
    t("이나", Object, Final),
    t("랑", Object, Open),
    t("나", Object, Open),
    t("하고", Object, Any),
    t("들을", Plural, Any),
    t("들의", Plural, Any),
    t("들이", Plural, Any),
    t("들은", Plural, Any),
    t("들는", Plural, Any),
    t("들도", Plural, Any),
    t("들", Plural, Any),
    t("과의", Link, Final),
    t("와의", Link, Open),
    t("만을", Object, Any),
    t("만", Object, Any),
    t("뿐", Object, Any),
    t("이", Object, Final),
    t("가", Object, Open),
    t("을", Object, Final),
    t("를", Object, Open),
    t("은", Object, Final),
    t("는", Object, Open),
    t("과", Object, Final),
    t("와", Object, Open),
    t("로", Direction, OpenOrRieul),
    t("의", Link, Any),
    t("에", Direction, Any),
    t("도", Object, Any),
];

const COPULA: &[Tail] = &[
    t("입니다", Copula, Any),
    t("이어야", Copula, Any),
    t("이라면", Copula, Any),
    t("이거나", Copula, Any),
    t("이고", Copula, Any),
    t("이며", Copula, Any),
    t("이면", Copula, Any),
    t("인데", Copula, Any),
    t("인지", Copula, Any),
    t("이다", Copula, Any),
];

pub const VERB_TAILS: &[&str] = &[
    "", "해", "해서", "해줘", "해와", "해와라", "해주세요", "해라", "해봐", "해보자", "해다", "해다가", "해다줘",
    "해다놔", "해오세요", "해오라", "해오고", "해놔", "해놓고", "하세요", "하라", "하고", "한", "할", "하기",
    "하자", "합니다", "하여", "했다", "했어", "했고", "하면", "하면서", "하다", "하는", "하며", "해둬", "해두세요",
    "세요", "십시오", "시오", "라", "렴", "렴아", "면", "고", "고서", "니", "는", "기", "자", "어", "아", "어서", "아서",
    "어라", "아라", "어줘", "아줘", "줘", "주세요", "서", "와", "와라", "와줘", "와주세요", "오세요", "오라", "오고",
    "다가", "다", "놔", "놔라", "놔줘", "놓아", "놓아라", "놓아줘", "두고", "둬", "두세요", "봐", "보자",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reading {
    pub stem: String,
    pub tail: String,
    pub kind: TailKind,
}

fn agrees(stem: &str, agree: Agree) -> bool {
    match agree {
        Any => true,
        Final => ends_with_final(stem) == Some(true),
        Open => ends_with_final(stem) == Some(false),
        FinalNotRieul => ends_with_final(stem) == Some(true) && !ends_with_rieul(stem),
        OpenOrRieul => ends_with_final(stem) == Some(false) || ends_with_rieul(stem),
    }
}

pub fn peel(word: &str) -> Vec<Reading> {
    let mut out = vec![Reading { stem: word.to_string(), tail: String::new(), kind: TailKind::Bare }];
    for tail in NOMINAL.iter().chain(COPULA.iter()) {
        let Some(stem) = word.strip_suffix(tail.text) else { continue };
        if stem.is_empty() {
            continue;
        }
        if !agrees(stem, tail.agree) {
            continue;
        }
        let reading = Reading { stem: stem.to_string(), tail: tail.text.to_string(), kind: tail.kind };
        if !out.contains(&reading) {
            out.push(reading);
        }
    }
    out
}

pub fn verb_remainder<'a>(word: &'a str, entry: &str) -> Option<&'a str> {
    let rest = word.strip_prefix(entry)?;
    VERB_TAILS.contains(&rest).then_some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stems(word: &str) -> Vec<(String, String)> {
        peel(word).into_iter().map(|r| (r.stem, r.tail)).collect()
    }

    #[test]
    fn object_particles_follow_batchim_agreement() {
        assert!(stems("석탄을").contains(&("석탄".into(), "을".into())));
        assert!(!stems("석탄를").contains(&("석탄".into(), "를".into())));
        assert!(stems("나무를").contains(&("나무".into(), "를".into())));
        assert!(!stems("나무을").contains(&("나무".into(), "을".into())));
    }

    #[test]
    fn direction_particle_ro_and_euro_agree_with_the_stem() {
        assert!(stems("창고로").contains(&("창고".into(), "로".into())));
        assert!(!stems("창고으로").iter().any(|(s, _)| s == "창고"));
        assert!(stems("석탄으로").contains(&("석탄".into(), "으로".into())));
        assert!(!stems("석탄로").iter().any(|(s, t)| s == "석탄" && t == "로"));
        assert!(stems("돌로").contains(&("돌".into(), "로".into())));
        assert!(!stems("돌으로").iter().any(|(s, t)| s == "돌" && t == "으로"));
    }

    #[test]
    fn the_longest_tail_is_listed_and_kinds_are_right() {
        let r = peel("창고에서");
        let hit = r.iter().find(|x| x.stem == "창고").unwrap();
        assert_eq!(hit.kind, TailKind::Source);
        let r = peel("창고까지");
        assert_eq!(r.iter().find(|x| x.stem == "창고").unwrap().kind, TailKind::Direction);
        let r = peel("통나무들을");
        assert!(r.iter().any(|x| x.stem == "통나무" && x.kind == TailKind::Plural));
    }

    #[test]
    fn the_bare_word_is_always_the_first_reading_and_stems_are_never_empty() {
        let r = peel("를");
        assert_eq!(r[0].tail, "");
        assert!(r.iter().all(|x| !x.stem.is_empty()));
    }

    #[test]
    fn non_hangul_stems_skip_agreement_tails() {
        assert!(!stems("logs이").iter().any(|(_, t)| t == "이"));
        assert!(stems("logs").len() == 1);
    }

    #[test]
    fn copula_forms_are_peeled() {
        assert!(stems("석탄이고").contains(&("석탄".into(), "이고".into())));
    }

    #[test]
    fn verb_endings_are_a_closed_list() {
        assert_eq!(verb_remainder("채굴해서", "채굴"), Some("해서"));
        assert_eq!(verb_remainder("채굴", "채굴"), Some(""));
        assert_eq!(verb_remainder("채굴장", "채굴"), None);
        assert_eq!(verb_remainder("가져오세요", "가져오"), Some("세요"));
    }
}
