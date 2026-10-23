//! Parallel embedding: a phrase bank that embeds every dictionary phrase once
//! (deduplicated, batched, cached, checkpointed to disk) and a tile pipeline
//! that embeds many crops of a frame in batched forward passes.
//!
//! The shape follows the logis-center embedding stack:
//!
//! * duplicates are removed before any forward pass;
//! * phrases are sorted by length and cut into micro-batches whose size is a
//!   function of the length only, so a phrase embeds the same way whichever
//!   other phrases ride in the same call;
//! * results are cached under a *space id* (model fingerprint + forward tag),
//!   so a different checkpoint never reads another checkpoint's vectors;
//! * zero or non-finite vectors are refused instead of cached, because a cached
//!   zero vector would silently remove that phrase from every later score;
//! * new vectors are appended to a checksummed file, and a torn tail is
//!   dropped on restore without losing the intact records before it.
//!
//! Nothing here depends on model weights: the bank works with any
//! [`TextEmbedder`], so tests use a deterministic stand-in.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use image::{DynamicImage, GenericImageView};
use rayon::prelude::*;

use crate::model::Siglip2;

pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64. A fixed hash, so values agree across processes and platforms.
pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Cache space for one model checkpoint and one forward path.
pub fn space_id(fingerprint: u64, forward_tag: &str) -> u64 {
    let h = fnv1a(FNV_OFFSET, &fingerprint.to_le_bytes());
    fnv1a(h, forward_tag.as_bytes())
}

/// Tag of the batched text forward. Raise it whenever a change alters the
/// numbers a text embeds to.
pub const FORWARD_TAG_TEXT: &str = "hand-ml/siglip2-text/batched/v1";

/// A vector is cacheable when it has numbers, all finite, and is not all zero.
pub fn is_cacheable(v: &[f32]) -> bool {
    !v.is_empty() && v.iter().all(|x| x.is_finite()) && v.iter().any(|&x| x != 0.0)
}

/// Micro-batch sizing. The row count depends only on the text length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchOpts {
    /// Largest number of texts in one forward pass.
    pub max_rows: usize,
    /// Largest `rows * length` product in one forward pass.
    pub token_budget: usize,
    /// Most entries the bank keeps in memory. Beyond this, new vectors are
    /// returned to the caller but not stored.
    pub max_entries: usize,
}

impl Default for BatchOpts {
    fn default() -> Self {
        BatchOpts { max_rows: 64, token_budget: 4096, max_entries: 50_000 }
    }
}

impl BatchOpts {
    /// Rows for texts of `len` tokens (or characters, as a proxy).
    pub fn rows_for(&self, len: usize) -> usize {
        (self.token_budget / len.max(1)).min(self.max_rows).max(1)
    }
}

/// Anything that turns phrases into unit-length vectors.
pub trait TextEmbedder {
    fn embedding_dim(&self) -> usize;
    /// Identifies the model. Vectors cached under one fingerprint are never
    /// returned for another.
    fn fingerprint(&self) -> u64;
    /// Embeds `texts` in one batch, in input order.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Anything that turns image crops into unit-length vectors.
pub trait ImageEmbedder {
    fn embedding_dim(&self) -> usize;
    /// Embeds `images`; output order matches input order.
    fn embed_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>>;
}

impl TextEmbedder for Siglip2 {
    fn embedding_dim(&self) -> usize {
        Siglip2::embedding_dim(self)
    }

    fn fingerprint(&self) -> u64 {
        Siglip2::fingerprint(self)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.embed_texts(&refs)
    }
}

impl ImageEmbedder for Siglip2 {
    fn embedding_dim(&self) -> usize {
        Siglip2::embedding_dim(self)
    }

    fn embed_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
        self.embed_images(images, 16)
    }
}

/// What one `embed` call did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BankRun {
    pub requested: usize,
    pub unique: usize,
    /// Unique phrases already in memory (or restored from disk).
    pub cached: usize,
    pub computed: usize,
    pub micro_batches: usize,
    /// Vectors that came back from disk on this call.
    pub restored: usize,
    /// Computed vectors that did not fit under `max_entries`.
    pub not_stored: usize,
    pub elapsed_ms: f64,
}

/// Counters over the whole life of a bank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BankStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub disk_records: u64,
    pub disk_repaired: u64,
}

#[derive(Default)]
struct Inner {
    spaces: HashMap<u64, HashMap<String, Arc<[f32]>>>,
    restored: HashMap<u64, bool>,
    hits: u64,
    misses: u64,
    disk_records: u64,
    disk_repaired: u64,
}

/// Phrase vectors, deduplicated and cached per model.
pub struct PhraseBank {
    inner: Mutex<Inner>,
    dir: Option<PathBuf>,
}

impl PhraseBank {
    /// A bank that lives in memory only.
    pub fn memory() -> Self {
        PhraseBank { inner: Mutex::new(Inner::default()), dir: None }
    }

    /// A bank that checkpoints new vectors under `dir`, one file per space.
    pub fn open(dir: impl Into<PathBuf>) -> Self {
        PhraseBank { inner: Mutex::new(Inner::default()), dir: Some(dir.into()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn path_of(&self, space: u64) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{space:016x}.bin")))
    }

    /// Phrases are keyed by their trimmed text with runs of whitespace collapsed.
    pub fn key_of(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Embeds `texts` and returns one vector per input, in input order.
    pub fn embed(
        &self,
        embedder: &dyn TextEmbedder,
        texts: &[String],
        opts: &BatchOpts,
    ) -> Result<(Vec<Arc<[f32]>>, BankRun)> {
        let started = Instant::now();
        let space = space_id(embedder.fingerprint(), FORWARD_TAG_TEXT);
        let dim = embedder.embedding_dim();
        let mut run = BankRun { requested: texts.len(), ..Default::default() };

        // 1. Deduplicate while keeping first-seen order.
        let mut unique: Vec<String> = Vec::new();
        let mut slot: HashMap<String, usize> = HashMap::new();
        let mut order: Vec<usize> = Vec::with_capacity(texts.len());
        for t in texts {
            let key = Self::key_of(t);
            let next = unique.len();
            let at = *slot.entry(key.clone()).or_insert_with(|| {
                unique.push(key);
                next
            });
            order.push(at);
        }
        run.unique = unique.len();

        // 2. Restore the space from disk once, then look everything up.
        run.restored = self.restore(space, dim);
        let mut vectors: Vec<Option<Arc<[f32]>>> = vec![None; unique.len()];
        {
            let mut g = self.lock();
            let (mut hits, mut misses) = (0u64, 0u64);
            if let Some(m) = g.spaces.get(&space) {
                for (i, k) in unique.iter().enumerate() {
                    match m.get(k) {
                        Some(v) if v.len() == dim => {
                            vectors[i] = Some(v.clone());
                            hits += 1;
                        }
                        _ => misses += 1,
                    }
                }
            } else {
                misses = unique.len() as u64;
            }
            g.hits += hits;
            g.misses += misses;
        }
        run.cached = vectors.iter().filter(|v| v.is_some()).count();

        // 3. Compute the missing phrases: shortest first, micro-batches sized by length.
        let mut missing: Vec<usize> = (0..unique.len()).filter(|&i| vectors[i].is_none()).collect();
        missing.sort_by_key(|&i| (unique[i].chars().count(), i));
        let mut fresh: Vec<(String, Arc<[f32]>)> = Vec::new();
        let mut at = 0;
        while at < missing.len() {
            // Lengths ascend, so the row budget only shrinks as a batch grows:
            // a candidate joins while the batch (with it) still fits the budget
            // set by the candidate's own length.
            let mut end = at + 1;
            while end < missing.len()
                && end - at + 1 <= opts.rows_for(unique[missing[end]].chars().count())
            {
                end += 1;
            }
            let group = &missing[at..end];
            let batch: Vec<String> = group.iter().map(|&i| unique[i].clone()).collect();
            let out = embedder.embed_batch(&batch)?;
            if out.len() != batch.len() {
                bail!("embedder returned {} vectors for {} texts", out.len(), batch.len());
            }
            run.micro_batches += 1;
            for (&i, v) in group.iter().zip(out) {
                if v.len() != dim {
                    bail!("embedder returned a {}-wide vector for {:?}, expected {dim}", v.len(), unique[i]);
                }
                if !is_cacheable(&v) {
                    return Err(anyhow!(
                        "embedder returned an unusable (zero or non-finite) vector for {:?}",
                        unique[i]
                    ));
                }
                let v: Arc<[f32]> = v.into();
                fresh.push((unique[i].clone(), v.clone()));
                vectors[i] = Some(v);
            }
            at = end;
        }
        run.computed = fresh.len();

        // 4. Store and checkpoint the new vectors.
        if !fresh.is_empty() {
            let mut stored: Vec<(String, Arc<[f32]>)> = Vec::new();
            {
                let mut g = self.lock();
                let m = g.spaces.entry(space).or_default();
                for (k, v) in &fresh {
                    if m.len() >= opts.max_entries {
                        run.not_stored += 1;
                        continue;
                    }
                    if m.insert(k.clone(), v.clone()).is_none() {
                        stored.push((k.clone(), v.clone()));
                    }
                }
            }
            if let Some(path) = self.path_of(space) {
                // A failed checkpoint only costs a recompute next start.
                let _ = append_records(&path, &stored);
            }
        }

        run.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let out = order
            .into_iter()
            .map(|i| vectors[i].clone().expect("every unique phrase was resolved"))
            .collect();
        Ok((out, run))
    }

    /// Reads this space's file once. Returns how many vectors it brought back.
    fn restore(&self, space: u64, dim: usize) -> usize {
        {
            let mut g = self.lock();
            if g.restored.get(&space).copied().unwrap_or(false) {
                return 0;
            }
            g.restored.insert(space, true);
        }
        let path = match self.path_of(space) {
            Some(p) => p,
            None => return 0,
        };
        let (records, good_len, broken) = read_records(&path);
        let mut repaired = false;
        if broken {
            // Keep the intact prefix: rewrite the file without the torn tail.
            repaired = rewrite_prefix(&path, good_len).is_ok();
        }
        let mut g = self.lock();
        let m = g.spaces.entry(space).or_default();
        let mut n = 0usize;
        for (k, v) in records {
            if v.len() == dim && is_cacheable(&v) {
                m.insert(k, v.into());
                n += 1;
            }
        }
        g.disk_records += n as u64;
        if repaired {
            g.disk_repaired += 1;
        }
        n
    }

    pub fn stats(&self) -> BankStats {
        let g = self.lock();
        BankStats {
            entries: g.spaces.values().map(HashMap::len).sum(),
            hits: g.hits,
            misses: g.misses,
            disk_records: g.disk_records,
            disk_repaired: g.disk_repaired,
        }
    }

    /// Whether every phrase is already cached for `embedder`.
    pub fn all_cached(&self, embedder: &dyn TextEmbedder, texts: &[String]) -> bool {
        let space = space_id(embedder.fingerprint(), FORWARD_TAG_TEXT);
        self.restore(space, embedder.embedding_dim());
        let g = self.lock();
        match g.spaces.get(&space) {
            Some(m) => texts.iter().all(|t| m.contains_key(&Self::key_of(t))),
            None => texts.is_empty(),
        }
    }

    /// Drops every cached vector and deletes the checkpoint files.
    pub fn clear(&self) {
        let mut g = self.lock();
        g.spaces.clear();
        g.restored.clear();
        drop(g);
        if let Some(dir) = &self.dir {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    if e.path().extension().map(|x| x == "bin").unwrap_or(false) {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint file: append-only, one checksummed record per phrase.
//   head:   magic "HPHR" u32 | version u32 | 0 u64
//   record: text length u32 | text UTF-8 | dim u32 | f32 * dim | checksum u64
// The checksum is FNV-1a over the text bytes and the vector bytes.
// ---------------------------------------------------------------------------

const DISK_MAGIC: u32 = 0x4850_4852;
const DISK_VERSION: u32 = 1;
const DISK_HEAD: usize = 16;

fn record_checksum(text: &[u8], vec: &[f32]) -> u64 {
    let mut h = fnv1a(FNV_OFFSET, text);
    for x in vec {
        h = fnv1a(h, &x.to_le_bytes());
    }
    h
}

fn disk_head() -> Vec<u8> {
    let mut h = Vec::with_capacity(DISK_HEAD);
    h.extend_from_slice(&DISK_MAGIC.to_le_bytes());
    h.extend_from_slice(&DISK_VERSION.to_le_bytes());
    h.extend_from_slice(&0u64.to_le_bytes());
    h
}

fn encode_records(items: &[(String, Arc<[f32]>)]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (text, vec) in items {
        let t = text.as_bytes();
        buf.extend_from_slice(&(t.len() as u32).to_le_bytes());
        buf.extend_from_slice(t);
        buf.extend_from_slice(&(vec.len() as u32).to_le_bytes());
        for x in vec.iter() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&record_checksum(t, vec).to_le_bytes());
    }
    buf
}

/// Intact records of a checkpoint file, the byte length of the intact prefix,
/// and whether anything after that prefix was damaged or cut off.
pub fn read_records(path: &Path) -> (Vec<(String, Vec<f32>)>, usize, bool) {
    let buf = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return (Vec::new(), 0, false),
    };
    if buf.len() < DISK_HEAD {
        return (Vec::new(), 0, !buf.is_empty());
    }
    let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    if u32_at(0) != DISK_MAGIC || u32_at(4) != DISK_VERSION {
        return (Vec::new(), 0, true);
    }
    let mut out = Vec::new();
    let mut off = DISK_HEAD;
    let mut broken = false;
    while off < buf.len() {
        let parsed = (|| -> Option<(String, Vec<f32>, usize)> {
            let mut o = off;
            if o + 4 > buf.len() {
                return None;
            }
            let tl = u32_at(o) as usize;
            o += 4;
            if o + tl > buf.len() {
                return None;
            }
            let text = std::str::from_utf8(&buf[o..o + tl]).ok()?.to_string();
            o += tl;
            if o + 4 > buf.len() {
                return None;
            }
            let dim = u32_at(o) as usize;
            o += 4;
            if dim == 0 || dim > 65_536 || o + dim * 4 + 8 > buf.len() {
                return None;
            }
            let vec: Vec<f32> = (0..dim)
                .map(|k| {
                    let p = o + k * 4;
                    f32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]])
                })
                .collect();
            o += dim * 4;
            let mut ck = [0u8; 8];
            ck.copy_from_slice(&buf[o..o + 8]);
            o += 8;
            if u64::from_le_bytes(ck) != record_checksum(text.as_bytes(), &vec) {
                return None;
            }
            Some((text, vec, o))
        })();
        match parsed {
            Some((text, vec, next)) => {
                out.push((text, vec));
                off = next;
            }
            None => {
                broken = true;
                break;
            }
        }
    }
    (out, off, broken)
}

fn append_records(path: &Path, items: &[(String, Arc<[f32]>)]) -> std::io::Result<()> {
    use std::io::Write;
    if items.is_empty() {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let is_new = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut buf = if is_new { disk_head() } else { Vec::new() };
    buf.extend_from_slice(&encode_records(items));
    f.write_all(&buf)
}

/// Keeps only the first `len` bytes of the file (atomically, via a temp file).
fn rewrite_prefix(path: &Path, len: usize) -> std::io::Result<()> {
    let buf = std::fs::read(path)?;
    let keep = if len >= DISK_HEAD { buf[..len.min(buf.len())].to_vec() } else { disk_head() };
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, keep)?;
    std::fs::rename(&tmp, path)
}

// ---------------------------------------------------------------------------
// Frame tiles
// ---------------------------------------------------------------------------

/// A rectangle of a frame, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Which layout level produced this tile: 0 is the whole frame.
    pub level: u8,
}

impl Tile {
    pub fn center(&self) -> (f32, f32) {
        (self.x as f32 + self.w as f32 / 2.0, self.y as f32 + self.h as f32 / 2.0)
    }

    pub fn area(&self) -> u32 {
        self.w * self.h
    }
}

/// Tiles for a frame: the whole frame first, then one grid per entry of
/// `grids` (`(columns, rows)`), each tile grown by `overlap` (a fraction of
/// its size) on every side and clamped to the frame.
pub fn tile_layout(width: u32, height: u32, grids: &[(u32, u32)], overlap: f32) -> Vec<Tile> {
    let mut tiles = vec![Tile { x: 0, y: 0, w: width, h: height, level: 0 }];
    for (level, &(cols, rows)) in grids.iter().enumerate() {
        let (cols, rows) = (cols.max(1), rows.max(1));
        if (cols, rows) == (1, 1) {
            continue;
        }
        let (tw, th) = (width as f32 / cols as f32, height as f32 / rows as f32);
        for r in 0..rows {
            for c in 0..cols {
                let x0 = (c as f32 * tw - overlap * tw).max(0.0);
                let y0 = (r as f32 * th - overlap * th).max(0.0);
                let x1 = ((c + 1) as f32 * tw + overlap * tw).min(width as f32);
                let y1 = ((r + 1) as f32 * th + overlap * th).min(height as f32);
                let (x, y) = (x0.round() as u32, y0.round() as u32);
                let (w, h) = ((x1.round() as u32).saturating_sub(x), (y1.round() as u32).saturating_sub(y));
                if w >= 4 && h >= 4 {
                    tiles.push(Tile { x, y, w, h, level: (level + 1).min(255) as u8 });
                }
            }
        }
    }
    tiles
}

/// Crops every tile out of `image`, in parallel.
pub fn crop_tiles(image: &DynamicImage, tiles: &[Tile]) -> Vec<DynamicImage> {
    let (iw, ih) = image.dimensions();
    tiles
        .par_iter()
        .map(|t| {
            let x = t.x.min(iw.saturating_sub(1));
            let y = t.y.min(ih.saturating_sub(1));
            image.crop_imm(x, y, t.w.min(iw - x), t.h.min(ih - y))
        })
        .collect()
}

/// Embeds all tiles of all `frames` through one embedder call, so the
/// embedder can batch across frames. Returns the per-frame tile vectors.
pub fn embed_frame_tiles(
    embedder: &dyn ImageEmbedder,
    frames: &[(&DynamicImage, &[Tile])],
) -> Result<Vec<Vec<Vec<f32>>>> {
    let crops: Vec<DynamicImage> = frames
        .iter()
        .flat_map(|(img, tiles)| crop_tiles(img, tiles))
        .collect();
    let all = embedder.embed_images_batch(&crops)?;
    if all.len() != crops.len() {
        bail!("image embedder returned {} vectors for {} crops", all.len(), crops.len());
    }
    let mut it = all.into_iter();
    Ok(frames
        .iter()
        .map(|(_, tiles)| it.by_ref().take(tiles.len()).collect())
        .collect())
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Cosine similarity between every row of `rows` and every vector of
/// `columns`, row-major `rows.len() x columns.len()`. Vectors are assumed to be
/// unit length, so this is a dot product. Rows are scored in parallel.
pub fn cosine_matrix(rows: &[Vec<f32>], columns: &[Arc<[f32]>]) -> Vec<f32> {
    let n = columns.len();
    let mut out = vec![0.0f32; rows.len() * n];
    out.par_chunks_mut(n.max(1)).zip(rows.par_iter()).for_each(|(dst, row)| {
        for (j, col) in columns.iter().enumerate() {
            dst[j] = row.iter().zip(col.iter()).map(|(a, b)| a * b).sum();
        }
    });
    out
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic stand-in: a unit vector derived from the text hash.
    pub struct HashEmbedder {
        pub dim: usize,
        pub fingerprint: u64,
        pub batches: AtomicUsize,
        pub rows: AtomicUsize,
        pub batch_sizes: Mutex<Vec<usize>>,
    }

    impl HashEmbedder {
        pub fn new(dim: usize, fingerprint: u64) -> Self {
            HashEmbedder {
                dim,
                fingerprint,
                batches: AtomicUsize::new(0),
                rows: AtomicUsize::new(0),
                batch_sizes: Mutex::new(Vec::new()),
            }
        }

        pub fn vector(&self, text: &str) -> Vec<f32> {
            let mut v: Vec<f32> = (0..self.dim)
                .map(|k| {
                    let h = fnv1a(fnv1a(FNV_OFFSET, text.as_bytes()), &(k as u32).to_le_bytes());
                    ((h >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
                })
                .collect();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            v
        }
    }

    impl TextEmbedder for HashEmbedder {
        fn embedding_dim(&self) -> usize {
            self.dim
        }
        fn fingerprint(&self) -> u64 {
            self.fingerprint
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.batches.fetch_add(1, Ordering::SeqCst);
            self.rows.fetch_add(texts.len(), Ordering::SeqCst);
            self.batch_sizes.lock().unwrap().push(texts.len());
            Ok(texts.iter().map(|t| self.vector(t)).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::HashEmbedder;
    use super::*;
    use std::sync::atomic::Ordering;

    fn texts(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hand-ml-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn duplicates_are_embedded_once_and_order_is_kept() {
        let e = HashEmbedder::new(16, 1);
        let bank = PhraseBank::memory();
        let input = texts(&["a tree", "a rock", "a tree", "  a   tree ", "a rock"]);
        let (out, run) = bank.embed(&e, &input, &BatchOpts::default()).unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(run.requested, 5);
        assert_eq!(run.unique, 2);
        assert_eq!(run.computed, 2);
        assert_eq!(e.rows.load(Ordering::SeqCst), 2);
        assert_eq!(out[0], out[2]);
        assert_eq!(out[0], out[3], "whitespace differences share one entry");
        assert_eq!(&out[1][..], &e.vector("a rock")[..]);
    }

    #[test]
    fn second_call_is_served_from_cache() {
        let e = HashEmbedder::new(16, 1);
        let bank = PhraseBank::memory();
        let input = texts(&["coal", "iron ore", "log"]);
        bank.embed(&e, &input, &BatchOpts::default()).unwrap();
        let before = e.batches.load(Ordering::SeqCst);
        let (_, run) = bank.embed(&e, &input, &BatchOpts::default()).unwrap();
        assert_eq!(run.computed, 0);
        assert_eq!(run.cached, 3);
        assert_eq!(e.batches.load(Ordering::SeqCst), before);
        assert!(bank.all_cached(&e, &input));
        assert!(!bank.all_cached(&e, &texts(&["coal", "diamond"])));
    }

    #[test]
    fn micro_batches_follow_the_row_budget() {
        let e = HashEmbedder::new(8, 1);
        let bank = PhraseBank::memory();
        let input: Vec<String> = (0..10).map(|i| format!("phrase {i}")).collect();
        let opts = BatchOpts { max_rows: 4, token_budget: 4096, max_entries: 100 };
        let (_, run) = bank.embed(&e, &input, &opts).unwrap();
        assert_eq!(run.micro_batches, 3);
        let sizes = e.batch_sizes.lock().unwrap().clone();
        assert_eq!(sizes, vec![4, 4, 2]);
    }

    #[test]
    fn long_phrases_get_smaller_batches() {
        let e = HashEmbedder::new(8, 1);
        let bank = PhraseBank::memory();
        let long = "x".repeat(40);
        let input: Vec<String> = (0..6).map(|i| format!("{long}{i}")).collect();
        let opts = BatchOpts { max_rows: 64, token_budget: 100, max_entries: 100 };
        bank.embed(&e, &input, &opts).unwrap();
        let sizes = e.batch_sizes.lock().unwrap().clone();
        assert!(sizes.iter().all(|&s| s <= 2), "{sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 6);
    }

    #[test]
    fn a_phrase_embeds_the_same_whichever_phrases_ride_with_it() {
        let e = HashEmbedder::new(16, 1);
        let alone = PhraseBank::memory();
        let (a, _) = alone.embed(&e, &texts(&["oak log"]), &BatchOpts::default()).unwrap();
        let together = PhraseBank::memory();
        let (b, _) = together
            .embed(&e, &texts(&["stone", "oak log", "coal", "iron ore"]), &BatchOpts::default())
            .unwrap();
        assert_eq!(a[0], b[1]);
    }

    #[test]
    fn different_models_never_share_cached_vectors() {
        let one = HashEmbedder::new(8, 1);
        let two = HashEmbedder::new(8, 2);
        let bank = PhraseBank::memory();
        let input = texts(&["tree"]);
        bank.embed(&one, &input, &BatchOpts::default()).unwrap();
        let (_, run) = bank.embed(&two, &input, &BatchOpts::default()).unwrap();
        assert_eq!(run.computed, 1, "a new fingerprint is a new space");
        assert_eq!(bank.stats().entries, 2);
    }

    struct ZeroEmbedder;
    impl TextEmbedder for ZeroEmbedder {
        fn embedding_dim(&self) -> usize {
            4
        }
        fn fingerprint(&self) -> u64 {
            9
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
        }
    }

    #[test]
    fn zero_vectors_are_refused_and_never_cached() {
        let bank = PhraseBank::memory();
        let err = bank.embed(&ZeroEmbedder, &texts(&["x"]), &BatchOpts::default()).unwrap_err();
        assert!(err.to_string().contains("unusable"));
        assert_eq!(bank.stats().entries, 0);
        assert!(is_cacheable(&[0.0, 1.0]));
        assert!(!is_cacheable(&[0.0, 0.0]));
        assert!(!is_cacheable(&[f32::NAN, 1.0]));
        assert!(!is_cacheable(&[]));
    }

    #[test]
    fn disk_checkpoint_round_trips_across_banks() {
        let dir = temp_dir("roundtrip");
        let e = HashEmbedder::new(12, 5);
        let input = texts(&["나무", "바위", "석탄"]);
        let first = PhraseBank::open(&dir);
        let (a, _) = first.embed(&e, &input, &BatchOpts::default()).unwrap();
        drop(first);

        let e2 = HashEmbedder::new(12, 5);
        let second = PhraseBank::open(&dir);
        let (b, run) = second.embed(&e2, &input, &BatchOpts::default()).unwrap();
        assert_eq!(run.restored, 3);
        assert_eq!(run.computed, 0);
        assert_eq!(e2.rows.load(Ordering::SeqCst), 0, "nothing was recomputed after restore");
        assert_eq!(a, b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_tail_is_dropped_but_the_intact_prefix_survives() {
        let dir = temp_dir("torn");
        let e = HashEmbedder::new(8, 7);
        let bank = PhraseBank::open(&dir);
        bank.embed(&e, &texts(&["one", "two", "three"]), &BatchOpts::default()).unwrap();
        drop(bank);

        let file = std::fs::read_dir(&dir).unwrap().flatten().next().unwrap().path();
        let mut bytes = std::fs::read(&file).unwrap();
        bytes.truncate(bytes.len() - 5); // cut the last record in the middle
        std::fs::write(&file, &bytes).unwrap();

        let e2 = HashEmbedder::new(8, 7);
        let again = PhraseBank::open(&dir);
        let (_, run) = again.embed(&e2, &texts(&["one", "two", "three"]), &BatchOpts::default()).unwrap();
        assert_eq!(run.restored, 2);
        assert_eq!(run.computed, 1, "only the damaged record is recomputed");
        assert_eq!(again.stats().disk_repaired, 1);
        // The repaired file now holds all three records and no damage.
        let (recs, _, broken) = read_records(&file);
        assert_eq!(recs.len(), 3);
        assert!(!broken);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_flipped_byte_fails_the_checksum() {
        let dir = temp_dir("flip");
        let e = HashEmbedder::new(8, 7);
        let bank = PhraseBank::open(&dir);
        bank.embed(&e, &texts(&["alpha", "beta"]), &BatchOpts::default()).unwrap();
        drop(bank);
        let file = std::fs::read_dir(&dir).unwrap().flatten().next().unwrap().path();
        let mut bytes = std::fs::read(&file).unwrap();
        let last = bytes.len() - 12; // inside the last vector
        bytes[last] ^= 0xFF;
        std::fs::write(&file, &bytes).unwrap();
        let (recs, _, broken) = read_records(&file);
        assert_eq!(recs.len(), 1);
        assert!(broken);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_entry_cap_returns_vectors_without_storing_them() {
        let e = HashEmbedder::new(8, 1);
        let bank = PhraseBank::memory();
        let input: Vec<String> = (0..5).map(|i| format!("p{i}")).collect();
        let opts = BatchOpts { max_rows: 64, token_budget: 4096, max_entries: 3 };
        let (out, run) = bank.embed(&e, &input, &opts).unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(run.not_stored, 2);
        assert_eq!(bank.stats().entries, 3);
    }

    #[test]
    fn clear_removes_memory_and_files() {
        let dir = temp_dir("clear");
        let e = HashEmbedder::new(8, 1);
        let bank = PhraseBank::open(&dir);
        bank.embed(&e, &texts(&["a"]), &BatchOpts::default()).unwrap();
        bank.clear();
        assert_eq!(bank.stats().entries, 0);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tile_layout_starts_with_the_whole_frame_and_stays_inside() {
        let tiles = tile_layout(384, 256, &[(2, 2), (4, 3)], 0.15);
        assert_eq!(tiles[0], Tile { x: 0, y: 0, w: 384, h: 256, level: 0 });
        assert_eq!(tiles.len(), 1 + 4 + 12);
        for t in &tiles {
            assert!(t.x + t.w <= 384 && t.y + t.h <= 256, "{t:?}");
            assert!(t.w >= 4 && t.h >= 4);
        }
        // Neighbouring tiles overlap.
        assert!(tiles[1].x + tiles[1].w > tiles[2].x);
        // A 1x1 grid adds nothing beyond the whole frame.
        assert_eq!(tile_layout(100, 100, &[(1, 1)], 0.0).len(), 1);
    }

    #[test]
    fn crops_match_their_tiles() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(40, 20, |x, y| image::Rgb([x as u8, y as u8, 0])));
        let tiles = tile_layout(40, 20, &[(2, 1)], 0.0);
        let crops = crop_tiles(&img, &tiles);
        assert_eq!(crops.len(), 3);
        assert_eq!(crops[0].dimensions(), (40, 20));
        assert_eq!(crops[1].dimensions(), (20, 20));
        assert_eq!(crops[2].to_rgb8().get_pixel(0, 0).0[0], 20);
    }

    struct MeanColour;
    impl ImageEmbedder for MeanColour {
        fn embedding_dim(&self) -> usize {
            3
        }
        fn embed_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
            Ok(images
                .iter()
                .map(|im| {
                    let rgb = im.to_rgb8();
                    let n = (rgb.width() * rgb.height()) as f32;
                    let mut s = [0.0f32; 3];
                    for p in rgb.pixels() {
                        for c in 0..3 {
                            s[c] += p.0[c] as f32;
                        }
                    }
                    s.iter().map(|v| v / n).collect()
                })
                .collect())
        }
    }

    #[test]
    fn tiles_of_several_frames_go_through_one_embedder_call() {
        let red = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb([200, 0, 0])));
        let blue = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb([0, 0, 200])));
        let t1 = tile_layout(32, 32, &[(2, 2)], 0.0);
        let t2 = tile_layout(32, 32, &[(2, 1)], 0.0);
        let out = embed_frame_tiles(&MeanColour, &[(&red, &t1), (&blue, &t2)]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), t1.len());
        assert_eq!(out[1].len(), t2.len());
        assert!((out[0][0][0] - 200.0).abs() < 1e-3);
        assert!((out[1][0][2] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn cosine_matrix_is_row_major_and_matches_dot_products() {
        let rows = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8]];
        let cols: Vec<Arc<[f32]>> = vec![vec![1.0f32, 0.0].into(), vec![0.0f32, 1.0].into()];
        let m = cosine_matrix(&rows, &cols);
        assert_eq!(m.len(), 6);
        assert_eq!(&m[0..2], &[1.0, 0.0]);
        assert_eq!(&m[2..4], &[0.0, 1.0]);
        assert!((m[4] - 0.6).abs() < 1e-6 && (m[5] - 0.8).abs() < 1e-6);
        assert!(cosine_matrix(&[], &cols).is_empty());
    }

    #[test]
    fn rows_for_never_returns_zero() {
        let o = BatchOpts { max_rows: 8, token_budget: 10, max_entries: 1 };
        assert_eq!(o.rows_for(0), 8);
        assert_eq!(o.rows_for(1000), 1);
        assert_eq!(o.rows_for(5), 2);
    }
}
