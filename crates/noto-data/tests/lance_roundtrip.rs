use noto_data::{random_scene, simulate_scene, LanceStore};

#[tokio::test]
async fn episodes_round_trip_through_lance() {
    let dir = std::env::temp_dir().join(format!("noto-data-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let uri = dir.to_str().expect("utf-8 temp path");

    let store = LanceStore::open(uri).await.unwrap();
    let mut expected_contacts = 0;
    let mut expected_joints = 0;
    for seed in 0..5u64 {
        let scene = random_scene(seed, "carton");
        let rows = simulate_scene(&scene).unwrap();
        expected_contacts += rows.contacts.num_rows();
        expected_joints += rows.joint_states.num_rows();
        store.append("samples", rows.samples).await.unwrap();
        store.append("contacts", rows.contacts).await.unwrap();
        store.append("joint_states", rows.joint_states).await.unwrap();
    }

    assert_eq!(store.row_count("samples").await.unwrap(), 5);
    assert_eq!(store.row_count("contacts").await.unwrap(), expected_contacts);
    assert_eq!(store.row_count("joint_states").await.unwrap(), expected_joints);
    assert_eq!(store.row_count("missing").await.unwrap(), 0);

    let reopened = LanceStore::open(uri).await.unwrap();
    assert_eq!(reopened.row_count("samples").await.unwrap(), 5);

    let _ = std::fs::remove_dir_all(&dir);
}
