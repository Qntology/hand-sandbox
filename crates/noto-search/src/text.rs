const HANGUL_BASE: u32 = 0xAC00;
const HANGUL_END: u32 = 0xD7A3;

pub fn is_hangul_syllable(c: char) -> bool {
    (HANGUL_BASE..=HANGUL_END).contains(&(c as u32))
}

pub fn final_index(c: char) -> Option<u32> {
    if !is_hangul_syllable(c) {
        return None;
    }
    Some((c as u32 - HANGUL_BASE) % 28)
}

pub const FINAL_RIEUL: u32 = 8;

pub fn ends_with_final(stem: &str) -> Option<bool> {
    stem.chars().last().and_then(final_index).map(|t| t != 0)
}

pub fn ends_with_rieul(stem: &str) -> bool {
    stem.chars().last().and_then(final_index) == Some(FINAL_RIEUL)
}

pub fn eul_reul(word: &str) -> &'static str {
    match ends_with_final(word) {
        Some(true) => "을",
        Some(false) => "를",
        None => "을(를)",
    }
}

pub fn i_ga(word: &str) -> &'static str {
    match ends_with_final(word) {
        Some(true) => "이",
        Some(false) => "가",
        None => "이(가)",
    }
}

pub fn euro(word: &str) -> &'static str {
    match ends_with_final(word) {
        Some(true) if ends_with_rieul(word) => "로",
        Some(true) => "으로",
        Some(false) => "로",
        None => "(으)로",
    }
}

pub fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn tokens(normalized: &str) -> Vec<String> {
    normalized.split_whitespace().map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn particles_follow_the_last_syllable() {
        use super::{eul_reul, euro};
        assert_eq!((eul_reul("원목"), eul_reul("개"), eul_reul("abc")), ("을", "를", "을(를)"));
        assert_eq!((euro("창고"), euro("베이스캠프"), euro("창"), euro("서울"), euro("x")), ("로", "로", "으로", "로", "(으)로"));
    }

    use super::*;

    #[test]
    fn final_consonant_detection() {
        assert_eq!(ends_with_final("나무"), Some(false));
        assert_eq!(ends_with_final("석탄"), Some(true));
        assert_eq!(ends_with_final("돌"), Some(true));
        assert!(ends_with_rieul("돌"));
        assert!(!ends_with_rieul("석탄"));
        assert_eq!(ends_with_final("log"), None);
        assert_eq!(ends_with_final(""), None);
    }

    #[test]
    fn normalize_strips_punctuation_and_case() {
        assert_eq!(normalize("  나무를,  Bring 2 LOGS!! "), "나무를 bring 2 logs");
        assert_eq!(tokens(&normalize("a, b")), vec!["a", "b"]);
    }
}
