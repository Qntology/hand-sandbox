//! Item and resource-node catalogue.
//!
//! The break times follow the Minecraft convention of a block hardness
//! multiplied by 1.5 when bare hands are the natural tool and by 5 when a tool
//! would be needed. Unlike Minecraft, no tool is required to get a drop: the
//! robot hand always yields the item, only slower. Masses come from typical
//! material densities times the item size, so they are assumptions, not
//! measurements (see `data/README.md`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Log,
    Stone,
    Coal,
    IronOre,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemSpec {
    pub kind: ItemKind,
    pub id: &'static str,
    pub label_ko: &'static str,
    pub label_en: &'static str,
    /// Bounding box edge lengths in metres.
    pub size_m: [f32; 3],
    pub mass_kg: f32,
    pub color: [u8; 3],
}

impl ItemKind {
    pub const ALL: [ItemKind; 4] = [ItemKind::Log, ItemKind::Stone, ItemKind::Coal, ItemKind::IronOre];

    pub fn spec(self) -> ItemSpec {
        match self {
            // 0.15 m cube, ~700 kg/m3 wood
            ItemKind::Log => ItemSpec {
                kind: self,
                id: "log",
                label_ko: "원목",
                label_en: "wooden log",
                size_m: [0.15, 0.15, 0.15],
                mass_kg: 2.4,
                color: [128, 92, 54],
            },
            // 0.10 m cube, ~2600 kg/m3 stone (a chipped piece, not a full block)
            ItemKind::Stone => ItemSpec {
                kind: self,
                id: "stone",
                label_ko: "돌",
                label_en: "stone piece",
                size_m: [0.10, 0.10, 0.10],
                mass_kg: 2.6,
                color: [128, 128, 132],
            },
            // 0.10 m cube, ~1400 kg/m3 coal
            ItemKind::Coal => ItemSpec {
                kind: self,
                id: "coal",
                label_ko: "석탄",
                label_en: "coal lump",
                size_m: [0.10, 0.10, 0.10],
                mass_kg: 1.4,
                color: [36, 36, 40],
            },
            // 0.10 m cube, ~4000 kg/m3 raw iron ore
            ItemKind::IronOre => ItemSpec {
                kind: self,
                id: "iron_ore",
                label_ko: "철광석",
                label_en: "raw iron ore",
                size_m: [0.10, 0.10, 0.10],
                mass_kg: 4.0,
                color: [186, 140, 112],
            },
        }
    }

    pub fn id(self) -> &'static str {
        self.spec().id
    }

    pub fn from_id(id: &str) -> Option<ItemKind> {
        ItemKind::ALL.into_iter().find(|k| k.id() == id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Tree,
    Boulder,
    CoalOre,
    IronOre,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec {
    pub kind: NodeKind,
    pub id: &'static str,
    pub label_ko: &'static str,
    pub label_en: &'static str,
    /// Minecraft-style block hardness.
    pub hardness: f32,
    /// True when bare hands are the natural tool (wood), false when a tool
    /// would be needed (stone and ores).
    pub hand_natural: bool,
    pub drop: ItemKind,
    pub drop_count: u32,
    pub color: [u8; 3],
}

impl NodeKind {
    pub const ALL: [NodeKind; 4] = [NodeKind::Tree, NodeKind::Boulder, NodeKind::CoalOre, NodeKind::IronOre];

    pub fn spec(self) -> NodeSpec {
        match self {
            NodeKind::Tree => NodeSpec {
                kind: self,
                id: "tree",
                label_ko: "나무",
                label_en: "tree",
                hardness: 2.0,
                hand_natural: true,
                drop: ItemKind::Log,
                drop_count: 3,
                color: [104, 74, 42],
            },
            NodeKind::Boulder => NodeSpec {
                kind: self,
                id: "boulder",
                label_ko: "바위",
                label_en: "boulder",
                hardness: 1.5,
                hand_natural: false,
                drop: ItemKind::Stone,
                drop_count: 2,
                color: [118, 118, 122],
            },
            NodeKind::CoalOre => NodeSpec {
                kind: self,
                id: "coal_ore",
                label_ko: "석탄 광석",
                label_en: "coal ore",
                hardness: 3.0,
                hand_natural: false,
                drop: ItemKind::Coal,
                drop_count: 2,
                color: [96, 96, 100],
            },
            NodeKind::IronOre => NodeSpec {
                kind: self,
                id: "iron_ore_node",
                label_ko: "철 광석",
                label_en: "iron ore",
                hardness: 3.0,
                hand_natural: false,
                drop: ItemKind::IronOre,
                drop_count: 1,
                color: [150, 118, 98],
            },
        }
    }

    pub fn id(self) -> &'static str {
        self.spec().id
    }

    /// Seconds of bare-hand work to break the node with a full five-finger hand.
    pub fn break_time_s(self) -> f32 {
        let s = self.spec();
        s.hardness * if s.hand_natural { 1.5 } else { 5.0 }
    }
}

/// Ground surface kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ground {
    Sand,
    Grass,
    Dirt,
    Rock,
}

impl Ground {
    pub fn color(self) -> [u8; 3] {
        match self {
            Ground::Sand => [214, 196, 140],
            Ground::Grass => [96, 156, 72],
            Ground::Dirt => [128, 96, 64],
            Ground::Rock => [140, 140, 146],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_are_unique() {
        for k in ItemKind::ALL {
            assert_eq!(ItemKind::from_id(k.id()), Some(k));
        }
        let mut ids: Vec<&str> = NodeKind::ALL.iter().map(|k| k.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), NodeKind::ALL.len());
        assert_eq!(ItemKind::from_id("nope"), None);
    }

    #[test]
    fn break_times_follow_the_hardness_convention() {
        assert!((NodeKind::Tree.break_time_s() - 3.0).abs() < 1e-6);
        assert!((NodeKind::Boulder.break_time_s() - 7.5).abs() < 1e-6);
        assert!((NodeKind::CoalOre.break_time_s() - 15.0).abs() < 1e-6);
        assert!((NodeKind::IronOre.break_time_s() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn every_node_drops_a_real_item() {
        for k in NodeKind::ALL {
            let s = k.spec();
            assert!(s.drop_count > 0);
            assert!(s.drop.spec().mass_kg > 0.0);
        }
    }
}
