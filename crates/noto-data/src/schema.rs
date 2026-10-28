use std::sync::Arc;

use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};

pub const TABLE_SAMPLES: &str = "samples";
pub const TABLE_CONTACTS: &str = "contacts";
pub const TABLE_JOINT_STATES: &str = "joint_states";
pub const TABLE_POINT_CLOUD: &str = "point_cloud";
pub const TABLE_VISION: &str = "vision";
pub const TABLE_TACTILE: &str = "tactile";
pub const TABLE_EMBEDDINGS: &str = "embeddings";
pub const TABLE_DICTIONARY: &str = "dictionary";

fn f32_field(name: &str) -> Field {
    Field::new(name, DataType::Float32, false)
}

fn i32_field(name: &str) -> Field {
    Field::new(name, DataType::Int32, false)
}

fn str_field(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

fn bool_field(name: &str) -> Field {
    Field::new(name, DataType::Boolean, false)
}

pub fn samples_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        str_field("label"),
        str_field("material"),
        Field::new("seed", DataType::Int64, false),
        i32_field("finger_count"),
        i32_field("joint_segments"),
        str_field("base_kind"),
        str_field("joint_kinds"),
        str_field("actuation"),
        f32_field("size_w_m"),
        f32_field("size_h_m"),
        f32_field("size_d_m"),
        f32_field("mass_kg"),
        f32_field("friction"),
        f32_field("inertia_xx"),
        f32_field("inertia_yy"),
        f32_field("inertia_zz"),
        f32_field("com_x_m"),
        f32_field("com_y_m"),
        f32_field("com_z_m"),
        f32_field("object_x_m"),
        f32_field("object_y_m"),
        f32_field("object_z_m"),
        f32_field("object_roll_rad"),
        f32_field("object_pitch_rad"),
        f32_field("object_yaw_rad"),
        str_field("grasp_type"),
        i32_field("finger_contacts"),
        i32_field("contacts_total"),
        bool_field("force_closed"),
        bool_field("stable"),
        f32_field("friction_capacity_n"),
        f32_field("required_n"),
        f32_field("max_payload_kg"),
        str_field("layers"),
    ]))
}

pub fn contacts_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        i32_field("contact_index"),
        i32_field("source_finger"),
        f32_field("point_x_m"),
        f32_field("point_y_m"),
        f32_field("point_z_m"),
        f32_field("normal_x"),
        f32_field("normal_y"),
        f32_field("normal_z"),
        f32_field("force_n"),
        f32_field("friction"),
    ]))
}

pub fn joint_states_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        i32_field("finger_index"),
        str_field("finger_name"),
        bool_field("is_thumb"),
        i32_field("joint_index"),
        str_field("joint_kind"),
        f32_field("flex_rad"),
        f32_field("abd_rad"),
        f32_field("roll_rad"),
        f32_field("flex_min_rad"),
        f32_field("flex_max_rad"),
        f32_field("max_torque_nm"),
    ]))
}

pub fn point_cloud_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        i32_field("point_index"),
        str_field("owner"),
        f32_field("x_m"),
        f32_field("y_m"),
        f32_field("z_m"),
        f32_field("normal_x"),
        f32_field("normal_y"),
        f32_field("normal_z"),
    ]))
}

pub fn vision_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        str_field("view"),
        i32_field("width"),
        i32_field("height"),
        Field::new("rgb", DataType::Binary, true),
        Field::new("depth_m", DataType::Binary, true),
        Field::new("segmentation", DataType::Binary, true),
        str_field("lighting"),
        str_field("texture"),
    ]))
}

pub fn tactile_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        i32_field("finger_index"),
        i32_field("rows"),
        i32_field("cols"),
        Field::new("pressure_pa", DataType::Binary, false),
    ]))
}

pub fn embeddings_schema(dim: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("sample_id"),
        str_field("view"),
        str_field("model"),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim,
            ),
            false,
        ),
    ]))
}

pub fn dictionary_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        str_field("id"),
        str_field("label_en"),
        str_field("label_ko"),
        str_field("category"),
        str_field("template"),
        f32_field("size_w_m"),
        f32_field("size_h_m"),
        f32_field("size_d_m"),
        f32_field("typical_mass_kg"),
        bool_field("graspable"),
        str_field("source"),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_have_expected_key_columns() {
        assert!(samples_schema().field_with_name("sample_id").is_ok());
        assert!(contacts_schema().field_with_name("force_n").is_ok());
        assert!(joint_states_schema().field_with_name("max_torque_nm").is_ok());
        assert!(embeddings_schema(768).field_with_name("vector").is_ok());
        assert!(dictionary_schema().field_with_name("label_ko").is_ok());
    }
}
