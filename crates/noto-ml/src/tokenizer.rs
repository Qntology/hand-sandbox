//! Tokenizer wrapper: canonicalizes text the way the reference project does
//! (lowercase, punctuation to spaces), then pads or truncates to a fixed length.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
    pad_id: u32,
    max_len: usize,
}

impl TextTokenizer {
    pub fn from_file(path: &Path, pad_id: u32, max_len: usize) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))
            .with_context(|| "SigLIP2 tokenizer.json is required next to model.safetensors")?;
        Ok(TextTokenizer {
            inner,
            pad_id,
            max_len: max_len.max(1),
        })
    }

    /// Lowercases, turns every non-alphanumeric character into a space and
    /// collapses runs of spaces.
    pub fn canonicalize(text: &str) -> String {
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

    /// Ids and mask for every text, flattened to `(texts.len() * max_len)`.
    pub fn encode(&self, texts: &[&str]) -> Result<(Vec<u32>, Vec<f32>)> {
        let mut ids_out = Vec::with_capacity(texts.len() * self.max_len);
        let mut mask_out = Vec::with_capacity(texts.len() * self.max_len);
        for text in texts {
            let canon = Self::canonicalize(text);
            let enc = self
                .inner
                .encode(canon.as_str(), true)
                .map_err(|e| anyhow!("tokenizing {canon:?}: {e}"))?;
            let mut ids: Vec<u32> = enc.get_ids().to_vec();
            ids.truncate(self.max_len);
            let real = ids.len();
            ids.resize(self.max_len, self.pad_id);
            ids_out.extend(ids);
            mask_out.extend((0..self.max_len).map(|i| if i < real { 1.0 } else { 0.0 }));
        }
        Ok((ids_out, mask_out))
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_lowercases_and_strips_punctuation() {
        assert_eq!(
            TextTokenizer::canonicalize("A Photo of: Cardboard-Box!"),
            "a photo of cardboard box"
        );
        assert_eq!(TextTokenizer::canonicalize("  a   b  "), "a b");
    }
}
