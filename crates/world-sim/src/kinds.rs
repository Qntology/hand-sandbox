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
    /// A small corrugated parcel: no node yields it, it lies where the world
    /// placed it (`World::new`), so the robot finds it by looking.
    Parcel,
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
    /// Brittle lumps (stone, coal, ore) can be broken into smaller pieces by
    /// striking them; a log cannot be split by hand.
    pub divisible: bool,
}

/// Smallest piece a divisible item is broken into, as a share of one item.
pub const MIN_PIECE: f32 = 0.125;

impl ItemSpec {
    /// Edge lengths of a piece holding `amount` of one item (same shape, volume
    /// scales with the amount).
    pub fn piece_size_m(&self, amount: f32) -> [f32; 3] {
        let k = amount.clamp(MIN_PIECE, 8.0).cbrt();
        [self.size_m[0] * k, self.size_m[1] * k, self.size_m[2] * k]
    }
}

impl ItemKind {
    pub const ALL: [ItemKind; 5] = [ItemKind::Log, ItemKind::Stone, ItemKind::Coal, ItemKind::IronOre, ItemKind::Parcel];

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
                divisible: false,
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
                divisible: true,
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
                divisible: true,
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
                divisible: true,
            },
            // A small corrugated parcel: the median delivery weighs 0.7 kg (Amaral,
            // de Oliveira & Beckers 2026: half below 0.7 kg); 16 × 12 × 10 cm is
            // inside the small class of a locker study (cells about 5 in high,
            // Ranjbari et al. 2023) and the hand's span.
            ItemKind::Parcel => ItemSpec {
                kind: self,
                id: "parcel",
                label_ko: "택배 상자",
                label_en: "parcel box",
                size_m: [0.16, 0.10, 0.12],
                mass_kg: 0.7,
                color: [196, 158, 108],
                divisible: false,
            },
        }
    }

    /// Whether a node yields it (mined) or the world placed it (found by looking).
    pub fn from_node(self) -> bool {
        NodeKind::ALL.into_iter().any(|n| n.spec().drop == self)
    }

    pub fn id(self) -> &'static str {
        self.spec().id
    }

    pub fn from_id(id: &str) -> Option<ItemKind> {
        ItemKind::ALL.into_iter().find(|k| k.id() == id)
    }

    /// Seconds of striking to break one whole item in half with a full
    /// five-finger hand (est: half the work that freed one piece from its node,
    /// `break_time / drop_count / 2`). Infinite for an item that cannot be split.
    pub fn split_time_s(self) -> f32 {
        if !self.spec().divisible {
            return f32::INFINITY;
        }
        NodeKind::ALL
            .into_iter()
            .find(|n| n.spec().drop == self)
            .map_or(f32::INFINITY, |n| n.break_time_s() / n.spec().drop_count as f32 * 0.5)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ground {
    Sand,
    Grass,
    Dirt,
    Rock,
    /// Open water (the sea of the rugged map). No base drives or walks into it.
    Water,
    /// Sand the waves keep wet, along the shore.
    WetSand,
    /// Compacted gravel, the pads under the waypoints.
    Gravel,
    /// Concrete floor of a site's buildings, porches, stairs and ramps (`crate::site`).
    Floor,
}

impl Ground {
    pub const ALL: [Ground; 8] = [Ground::Sand, Ground::Grass, Ground::Dirt, Ground::Rock, Ground::Water, Ground::WetSand, Ground::Gravel, Ground::Floor];

    pub fn color(self) -> [u8; 3] {
        match self {
            Ground::Sand => [214, 196, 140],
            Ground::Grass => [96, 156, 72],
            Ground::Dirt => [128, 96, 64],
            Ground::Rock => [140, 140, 146],
            Ground::Water => [56, 112, 178],
            Ground::WetSand => [168, 150, 104],
            Ground::Gravel => [176, 168, 152],
            Ground::Floor => [206, 206, 200],
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Ground::Sand => "sand",
            Ground::Grass => "grass",
            Ground::Dirt => "dirt",
            Ground::Rock => "rock",
            Ground::Water => "water",
            Ground::WetSand => "wet_sand",
            Ground::Gravel => "gravel",
            Ground::Floor => "floor",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Ground::Sand => "모래",
            Ground::Grass => "풀밭",
            Ground::Dirt => "흙",
            Ground::Rock => "바위 지면",
            Ground::Water => "바다",
            Ground::WetSand => "젖은 모래",
            Ground::Gravel => "자갈 패드",
            Ground::Floor => "콘크리트 바닥",
        }
    }

    /// Code in [`crate::snapshot::TerrainGrid::ground`].
    pub fn code(self) -> u8 {
        match self {
            Ground::Sand => 0,
            Ground::Grass => 1,
            Ground::Dirt => 2,
            Ground::Rock => 3,
            Ground::Water => 4,
            Ground::WetSand => 5,
            Ground::Gravel => 6,
            Ground::Floor => 7,
        }
    }

    pub fn is_water(self) -> bool {
        self == Ground::Water
    }

    /// The physics of this ground (`noto_core::surface`).
    pub fn surface(self) -> noto_core::Surface {
        match self {
            Ground::Sand => noto_core::Surface::Sand,
            Ground::Grass => noto_core::Surface::Grass,
            Ground::Dirt => noto_core::Surface::Dirt,
            Ground::Rock => noto_core::Surface::Rock,
            Ground::Water => noto_core::Surface::Water,
            Ground::WetSand => noto_core::Surface::WetSand,
            Ground::Gravel => noto_core::Surface::Gravel,
            // Dry concrete: the firm floor of the laboratory tests.
            Ground::Floor => noto_core::Surface::Firm,
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

    #[test]
    fn brittle_items_split_and_a_log_does_not() {
        assert!(ItemKind::Log.split_time_s().is_infinite());
        // A boulder takes 7.5 s for two stones: halving one stone is a quarter of that.
        assert!((ItemKind::Stone.split_time_s() - 1.875).abs() < 1e-5);
        assert!(ItemKind::Coal.split_time_s().is_finite() && ItemKind::IronOre.split_time_s().is_finite());
        // A half piece has half the volume: edges shrink by the cube root of two.
        let half = ItemKind::Stone.spec().piece_size_m(0.5);
        assert!((half[0] - 0.10 / 2f32.cbrt()).abs() < 1e-5);
        let codes: std::collections::HashSet<u8> = Ground::ALL.iter().map(|g| g.code()).collect();
        assert_eq!(codes.len(), Ground::ALL.len());
    }
}
