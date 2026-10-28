use crate::lexicon::QuantityTable;
use crate::tails::peel;

pub const MAX_QUANTITY: u32 = 99;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quantity {
    pub value: u32,
    pub start: usize,
    pub end: usize,
}

impl QuantityTable {
    fn is_counter(&self, word: &str) -> bool {
        self.counters.iter().any(|c| c == word)
    }

    fn parse_native(&self, word: &str) -> Option<u32> {
        if let Some(v) = self.native_units.get(word) {
            return Some(*v);
        }
        for (tens, tv) in &self.native_tens {
            if let Some(rest) = word.strip_prefix(tens.as_str()) {
                if rest.is_empty() {
                    return Some(*tv);
                }
                if let Some(u) = self.native_units.get(rest) {
                    return Some(tv + u);
                }
            }
        }
        None
    }

    fn parse_sino(&self, word: &str) -> Option<u32> {
        if let Some(v) = self.sino_digits.get(word) {
            return Some(*v);
        }
        for (tens, tv) in &self.sino_tens {
            let Some(idx) = word.find(tens.as_str()) else { continue };
            let (head, rest) = (&word[..idx], &word[idx + tens.len()..]);
            let mult = if head.is_empty() { 1 } else { *self.sino_digits.get(head)? };
            let unit = if rest.is_empty() { 0 } else { *self.sino_digits.get(rest)? };
            return Some(mult * tv + unit);
        }
        None
    }

    fn parse_plain(&self, word: &str) -> Option<u32> {
        if !word.is_empty() && word.chars().all(|c| c.is_ascii_digit()) && word.len() <= 4 {
            return word.parse().ok();
        }
        self.words.get(word).copied()
    }

    fn parse_free(&self, word: &str) -> Option<u32> {
        self.parse_plain(word).or_else(|| self.parse_native(word))
    }

    fn parse_counted(&self, word: &str) -> Option<u32> {
        self.parse_free(word).or_else(|| self.parse_sino(word))
    }

    pub fn read(&self, tokens: &[String], i: usize) -> Option<Quantity> {
        let word = tokens.get(i)?;
        for reading in peel(word) {
            let stem = reading.stem.as_str();
            for counter in &self.counters {
                if let Some(num) = stem.strip_suffix(counter.as_str()) {
                    if !num.is_empty() {
                        if let Some(v) = self.parse_counted(num) {
                            return Some(Quantity { value: v, start: i, end: i + 1 });
                        }
                    }
                }
            }
            if let Some(v) = self.parse_counted(stem) {
                if let Some(next) = tokens.get(i + 1) {
                    if peel(next).iter().any(|r| self.is_counter(&r.stem)) {
                        return Some(Quantity { value: v, start: i, end: i + 2 });
                    }
                }
            }
            if let Some(v) = self.parse_free(stem) {
                return Some(Quantity { value: v, start: i, end: i + 1 });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn table() -> QuantityTable {
        let m = |pairs: &[(&str, u32)]| -> BTreeMap<String, u32> {
            pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        };
        QuantityTable {
            counters: vec!["개".into(), "조각".into(), "그루".into()],
            native_units: m(&[("한", 1), ("하나", 1), ("두", 2), ("둘", 2), ("세", 3), ("셋", 3), ("네", 4), ("다섯", 5), ("아홉", 9)]),
            native_tens: m(&[("열", 10), ("스물", 20), ("스무", 20)]),
            sino_digits: m(&[("일", 1), ("이", 2), ("삼", 3), ("오", 5)]),
            sino_tens: m(&[("십", 10)]),
            words: m(&[("one", 1), ("two", 2), ("three", 3)]),
        }
    }

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn read(s: &str, i: usize) -> Option<(u32, usize, usize)> {
        table().read(&toks(s), i).map(|q| (q.value, q.start, q.end))
    }

    #[test]
    fn digits_with_and_without_counters() {
        assert_eq!(read("나무 3개 가져와", 1), Some((3, 1, 2)));
        assert_eq!(read("나무 3 개 가져와", 1), Some((3, 1, 3)));
        assert_eq!(read("bring 2 logs", 1), Some((2, 1, 2)));
    }

    #[test]
    fn native_korean_numbers() {
        assert_eq!(read("두 개 가져와", 0), Some((2, 0, 2)));
        assert_eq!(read("세개를 가져와", 0), Some((3, 0, 1)));
        assert_eq!(read("열두 개", 0), Some((12, 0, 2)));
        assert_eq!(read("스무개", 0), Some((20, 0, 1)));
        assert_eq!(read("나무 하나 가져와", 1), Some((1, 1, 2)));
    }

    #[test]
    fn sino_numbers_need_a_counter() {
        assert_eq!(read("삼 개", 0), Some((3, 0, 2)));
        assert_eq!(read("이십오개", 0), Some((25, 0, 1)));
        assert_eq!(read("십 개", 0), Some((10, 0, 2)));
        assert_eq!(read("이 나무", 0), None);
        assert_eq!(read("오 나무", 0), None);
    }

    #[test]
    fn english_words_and_particles_after_counters() {
        assert_eq!(read("two logs", 0), Some((2, 0, 1)));
        assert_eq!(read("두 개의 나무", 0), Some((2, 0, 2)));
        assert_eq!(read("나무 두 그루를", 1), Some((2, 1, 3)));
    }

    #[test]
    fn plain_words_are_not_quantities() {
        assert_eq!(read("나무를 가져와", 0), None);
        assert_eq!(read("개 가져와", 0), None);
        assert_eq!(read("12345", 0), None, "five digits are an id, not a count");
    }
}
