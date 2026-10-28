use std::sync::Arc;

use anyhow::{Context, Result};
use lancedb::arrow::arrow_array::{ArrayRef, BooleanArray, Float32Array, RecordBatch, StringArray};
use serde::Deserialize;

use crate::schema::dictionary_schema;

#[derive(Debug, Deserialize)]
struct DictFile {
    template: String,
    entries: Vec<DictJsonEntry>,
}

#[derive(Debug, Deserialize)]
struct DictJsonEntry {
    id: String,
    label_en: String,
    label_ko: String,
    category: String,
    size_m: [f32; 3],
    typical_mass_kg: f32,
    graspable: bool,
    source: String,
}

pub fn dictionary_batch(raw_json: &str) -> Result<RecordBatch> {
    let file: DictFile = serde_json::from_str(raw_json).context("parsing logistics dictionary")?;
    let n = file.entries.len();
    let pick = |f: fn(&DictJsonEntry) -> &str| -> StringArray {
        StringArray::from_iter_values(file.entries.iter().map(|e| f(e).to_string()))
    };
    let template: Vec<&str> = vec![file.template.as_str(); n];
    let size = |axis: usize| -> Float32Array {
        Float32Array::from(file.entries.iter().map(|e| e.size_m[axis]).collect::<Vec<f32>>())
    };

    let columns: Vec<ArrayRef> = vec![
        Arc::new(pick(|e| &e.id)),
        Arc::new(pick(|e| &e.label_en)),
        Arc::new(pick(|e| &e.label_ko)),
        Arc::new(pick(|e| &e.category)),
        Arc::new(StringArray::from(template)),
        Arc::new(size(0)),
        Arc::new(size(1)),
        Arc::new(size(2)),
        Arc::new(Float32Array::from(
            file.entries.iter().map(|e| e.typical_mass_kg).collect::<Vec<f32>>(),
        )),
        Arc::new(BooleanArray::from(
            file.entries.iter().map(|e| e.graspable).collect::<Vec<bool>>(),
        )),
        Arc::new(pick(|e| &e.source)),
    ];
    Ok(RecordBatch::try_new(dictionary_schema(), columns)?)
}
