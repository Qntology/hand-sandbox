//! Loads the shipped logistics dictionary into a Lance table.

use noto_data::{dictionary_batch, LanceStore};

#[tokio::test]
async fn shipped_dictionary_writes_to_lance() {
    let raw = include_str!("../../../data/logistics_dictionary.json");
    let batch = dictionary_batch(raw).unwrap();
    let expected = batch.num_rows();
    assert_eq!(expected, 28, "the shipped dictionary has 28 labels");

    let dir = std::env::temp_dir().join(format!("noto-dict-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = LanceStore::open(dir.to_str().unwrap()).await.unwrap();
    store.append("dictionary", batch).await.unwrap();
    assert_eq!(store.row_count("dictionary").await.unwrap(), expected);
    let _ = std::fs::remove_dir_all(&dir);
}
