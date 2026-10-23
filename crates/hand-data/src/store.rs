//! LanceDB store for the synthetic dataset. One Lance dataset per table, all
//! under one directory that can be copied or versioned as a unit.

use anyhow::Result;
use lancedb::arrow::arrow_array::RecordBatch;
use lancedb::arrow::arrow_schema::SchemaRef;
use lancedb::{connect, Connection, Table};

pub struct LanceStore {
    conn: Connection,
}

impl LanceStore {
    /// Opens (or creates) the store at `uri`, a local directory path.
    pub async fn open(uri: &str) -> Result<Self> {
        let conn = connect(uri).execute().await?;
        Ok(LanceStore { conn })
    }

    /// Returns the table, creating it empty with `schema` when missing.
    pub async fn ensure_table(&self, name: &str, schema: SchemaRef) -> Result<Table> {
        let existing = self.conn.table_names().execute().await?;
        if existing.iter().any(|n| n == name) {
            Ok(self.conn.open_table(name).execute().await?)
        } else {
            Ok(self.conn.create_empty_table(name, schema).execute().await?)
        }
    }

    /// Appends batches in one write. Empty batches are skipped. The table is
    /// created from the first batch's schema when missing.
    pub async fn append_all(&self, name: &str, batches: Vec<RecordBatch>) -> Result<()> {
        let batches: Vec<RecordBatch> = batches.into_iter().filter(|b| b.num_rows() > 0).collect();
        let Some(first) = batches.first() else {
            return Ok(());
        };
        let table = self.ensure_table(name, first.schema()).await?;
        table.add(batches).execute().await?;
        Ok(())
    }

    /// Drops the table if present, then writes `batches` as its only content.
    pub async fn replace_with(&self, name: &str, batches: Vec<RecordBatch>) -> Result<()> {
        let existing = self.conn.table_names().execute().await?;
        if existing.iter().any(|n| n == name) {
            self.conn.drop_table(name, &[]).await?;
        }
        self.append_all(name, batches).await
    }

    /// Appends one batch.
    pub async fn append(&self, name: &str, batch: RecordBatch) -> Result<()> {
        self.append_all(name, vec![batch]).await
    }

    pub async fn row_count(&self, name: &str) -> Result<usize> {
        let existing = self.conn.table_names().execute().await?;
        if !existing.iter().any(|n| n == name) {
            return Ok(0);
        }
        let table = self.conn.open_table(name).execute().await?;
        Ok(table.count_rows(None).await?)
    }
}
