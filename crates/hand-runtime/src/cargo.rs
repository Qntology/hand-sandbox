//! The cargo decision: what to do with a piece the robot is about to pick up.
//!
//! The structure follows the decision model of endmill-sandbox
//! (`src/decision.rs`): deterministic gates, each with a value, a limit, a
//! severity (0 pass, 1 watch, 2 over, 3 critical) and the action it asks for;
//! the rule action is the most conservative action among the gates of the
//! worst severity. A planning step then looks for the cheapest way that
//! passes every gate:
//!
//! 1. **take** the whole piece;
//! 2. **shuttle**: the arms and the hand could take it, but the bay is full
//!    of this quest's cargo — carry that to the destination, come back;
//! 3. **split**: the piece is too heavy for the arms, the hand or the empty
//!    bay — break it (stone, coal and ore are brittle) and take the largest
//!    part that passes, halving down to [`MIN_PIECE`];
//! 4. **refuse**, with the gate that blocks it.
//!
//! Every gate value comes from the physics the world already runs (the grasp
//! test, `Body::lift`, the drive limits for the load, the route for the
//! loaded base); nothing here is tuned to a scenario.

use hand_core::balance::{FeedForward, Source};
use serde::Serialize;
use world_sim::kinds::MIN_PIECE;
use world_sim::nav::Goal;
use world_sim::world::{DropItem, AMOUNT_EPS};
use world_sim::{amount_text, World};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CargoAction {
    Take,
    Shuttle,
    Split { portion: f32 },
    Refuse,
}

impl CargoAction {
    /// Order of caution, as endmill's `Action::conservativeness`.
    pub fn conservativeness(&self) -> u8 {
        match self {
            CargoAction::Take => 0,
            CargoAction::Shuttle => 1,
            CargoAction::Split { .. } => 2,
            CargoAction::Refuse => 3,
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            CargoAction::Take => "take",
            CargoAction::Shuttle => "shuttle",
            CargoAction::Split { .. } => "split",
            CargoAction::Refuse => "refuse",
        }
    }

    pub fn label_ko(&self) -> String {
        match self {
            CargoAction::Take => "그대로 들기".into(),
            CargoAction::Shuttle => "먼저 운반하고 돌아오기".into(),
            CargoAction::Split { portion } => format!("쪼개서 {:.0}%만 들기", portion * 100.0),
            CargoAction::Refuse => "들 수 없음".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CargoGate {
    pub id: &'static str,
    pub label_ko: &'static str,
    pub value: f32,
    pub limit: f32,
    pub unit: &'static str,
    pub severity: u8,
    pub action: CargoAction,
    pub message: String,
}

impl CargoGate {
    pub fn passed(&self) -> bool {
        self.severity < 2
    }
}

/// What the task knows that the world does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CargoCtx {
    /// Mass in the bay that belongs to this quest (what a trip would deliver), kg.
    pub quest_cargo_kg: f32,
    /// Where a trip would go; `None` for a quest that keeps its items.
    pub destination: Option<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CargoDecision {
    pub item: String,
    /// Items in the piece.
    pub amount: f32,
    /// Gates of the whole piece as it lies.
    pub gates: Vec<CargoGate>,
    /// The rule action of those gates (the most conservative of the worst).
    pub rule_action: CargoAction,
    /// What the planning step chose.
    pub action: CargoAction,
    /// Share of the piece to take (1 for take and shuttle).
    pub portion: f32,
    /// Messages of the gates at severity 2 or more.
    pub reasons: Vec<String>,
    pub summary_ko: String,
    /// Stable failure code for `Refuse` (`lift_overload`, `grasp_unstable`, `capacity_exceeded`, `route_loaded`).
    pub code: Option<&'static str>,
}

fn gate(id: &'static str, label_ko: &'static str, value: f32, limit: f32, unit: &'static str, severity: u8, action: CargoAction, message: String) -> CargoGate {
    CargoGate { id, label_ko, value, limit, unit, severity, action, message }
}

/// The gates for taking `amount` items of the piece with `cargo_kg` already in the bay.
fn gates_for(w: &World, d: &DropItem, amount: f32, cargo_kg: f32, ctx: &CargoCtx) -> Vec<CargoGate> {
    let spec = d.kind.spec();
    let kg = spec.mass_kg * amount;
    let split = if spec.divisible { CargoAction::Split { portion: 0.5 } } else { CargoAction::Refuse };
    let mut out = Vec::new();
    // 1. The hand: friction capacity over the force needed (grasp test, safety factor in).
    let g = w.grasp_for(d.kind, amount);
    let sev = if !g.stable { 3 } else if g.margin < 1.2 { 1 } else { 0 };
    out.push(gate("grip", "손 마찰", g.margin, 1.0, "×", sev, if sev >= 2 { split } else { CargoAction::Take }, format!("마찰 용량 / 필요 힘 {:.2} ({} {} kg)", g.margin, spec.label_ko, fmt_kg(kg))));
    // 2. The arms and the balance while lifting (one arm, else both).
    let ff = FeedForward { enabled: w.ff_enabled, gain: w.ff_gains.get(&Source::ArmLift).copied().unwrap_or(1.0) };
    let (rep, two) = w.body.lift(kg, cargo_kg, &ff, w.grip_capacity_for(d.kind, amount));
    let worst = rep.worst().map_or(0.0, |j| j.peak_ratio);
    let joint = rep.worst().map(|j| j.joint.clone()).unwrap_or_default();
    let sev = if !rep.feasible { 3 } else if worst > 0.9 || rep.margin < 0.1 { 1 } else { 0 };
    out.push(gate(
        "lift",
        "팔 토크·균형",
        worst,
        1.0,
        "피크비",
        sev,
        if sev >= 2 { split } else { CargoAction::Take },
        format!(
            "{}{} 피크 대비 {:.0}% · 균형 여유 {:.2}{}",
            if two { "두 팔 · " } else { "" },
            joint,
            worst * 100.0,
            rep.margin,
            rep.limiting.as_ref().filter(|_| !rep.feasible).map(|l| format!(" · 한계 {l}")).unwrap_or_default()
        ),
    ));
    // 3. The bay: what the legs or the chassis carry continuously.
    let cap = w.capacity_kg();
    let total = cargo_kg + kg;
    let foreign = (cargo_kg - ctx.quest_cargo_kg).max(0.0);
    let (sev, act) = if total <= cap + 1e-4 {
        (if total > 0.9 * cap { 1 } else { 0 }, CargoAction::Take)
    } else if kg + foreign <= cap + 1e-4 && ctx.quest_cargo_kg > AMOUNT_EPS && ctx.destination.is_some() {
        (2, CargoAction::Shuttle)
    } else if kg > cap - foreign + 1e-4 && foreign < 1e-3 {
        (3, split)
    } else {
        (3, CargoAction::Refuse)
    };
    out.push(gate("capacity", "적재 한도", total, cap, "kg", sev, act, format!("실은 {} + {} kg / 한도 {} kg{}", fmt_kg(cargo_kg), fmt_kg(kg), fmt_kg(cap), if foreign > 1e-3 { format!(" (이 퀘스트와 무관한 짐 {} kg)", fmt_kg(foreign)) } else { String::new() })));
    // 4. The way back with this load: the loaded base must still get to the destination.
    if let Some((x, z)) = ctx.destination {
        let empty = w.plan_route(Goal::Within { x, z, reach_m: 1.1 }).is_some();
        if empty {
            let spec = w.drive_spec_for(total);
            let blocked: std::collections::HashSet<(usize, usize)> = w.nodes.iter().filter_map(|n| w.terrain.cell_of(n.x, n.z)).collect();
            let start = w.terrain.cell_of(w.agent.x, w.agent.z).unwrap_or((0, 0));
            let loaded = world_sim::nav::plan(&spec, &w.terrain, &|a, b| blocked.contains(&(a, b)), start, Goal::Within { x, z, reach_m: 1.1 }).is_some();
            let sev = if loaded { 0 } else { 3 };
            let act = if loaded { CargoAction::Take } else if ctx.quest_cargo_kg > AMOUNT_EPS { CargoAction::Shuttle } else { split };
            out.push(gate("route", "짐 실은 경로", spec.max_slope_deg, 0.0, "°", sev, act, format!("{} kg를 싣고 갈 수 있는 경사 {:.0}°·턱 {:.0} cm (지면마다 마찰·구동력 한계 따로){}", fmt_kg(total), spec.max_slope_deg, spec.max_step_m * 100.0, if loaded { "" } else { " — 목적지까지 길이 없음" })));
        }
    }
    out
}

fn fmt_kg(kg: f32) -> String {
    format!("{kg:.1}")
}

/// The rule action of a gate list: the most conservative action among the worst gates.
pub fn rule_action(gates: &[CargoGate]) -> CargoAction {
    let worst = gates.iter().map(|g| g.severity).max().unwrap_or(0);
    if worst < 2 {
        return CargoAction::Take;
    }
    gates.iter().filter(|g| g.severity == worst).map(|g| g.action).max_by_key(|a| a.conservativeness()).unwrap_or(CargoAction::Take)
}

/// Decides what to do with piece `d`.
pub fn decide(w: &World, d: &DropItem, ctx: &CargoCtx) -> CargoDecision {
    let cargo = w.carried_kg();
    let gates = gates_for(w, d, d.amount, cargo, ctx);
    let rule = rule_action(&gates);
    let reasons: Vec<String> = gates.iter().filter(|g| g.severity >= 2).map(|g| format!("[{}] {}", g.label_ko, g.message)).collect();
    let spec = d.kind.spec();
    let name = spec.label_ko;
    let mut out = CargoDecision { item: d.kind.id().into(), amount: d.amount, gates: gates.clone(), rule_action: rule, action: CargoAction::Take, portion: 1.0, reasons, summary_ko: String::new(), code: None };
    if rule == CargoAction::Take {
        out.summary_ko = format!("{name} {}개를 그대로 듭니다", amount_text(d.amount));
        return out;
    }
    let passes = |gs: &[CargoGate]| gs.iter().all(|g| g.passed());
    // The whole piece passes with an empty bay of this quest's cargo: make a trip first.
    let foreign = (cargo - ctx.quest_cargo_kg).max(0.0);
    if ctx.destination.is_some() && ctx.quest_cargo_kg > AMOUNT_EPS && passes(&gates_for(w, d, d.amount, foreign, ctx)) {
        out.action = CargoAction::Shuttle;
        out.summary_ko = format!("{name} {}개를 더 실을 수 없어 먼저 실은 짐을 운반하고 돌아옵니다", amount_text(d.amount));
        return out;
    }
    // Too heavy for the arms, the hand or the empty bay: break it.
    let blocked_by_foreign = gates.iter().any(|g| g.id == "capacity" && g.severity >= 2 && g.action == CargoAction::Refuse);
    if spec.divisible && !blocked_by_foreign {
        let mut portion = 0.5f32;
        while d.amount * portion >= MIN_PIECE - AMOUNT_EPS {
            let part = d.amount * portion;
            if passes(&gates_for(w, d, part, cargo, ctx)) {
                out.action = CargoAction::Split { portion };
                out.portion = portion;
                out.summary_ko = format!("{name} {}개는 {} — 쪼개서 {:.0}%({} kg)만 들고 나릅니다", amount_text(d.amount), gates.iter().filter(|g| g.severity >= 2).map(|g| g.label_ko).collect::<Vec<_>>().join("·") + " 초과", portion * 100.0, fmt_kg(spec.mass_kg * part));
                return out;
            }
            // A part that would pass with an empty bay: carry this quest's cargo first.
            if ctx.destination.is_some() && ctx.quest_cargo_kg > AMOUNT_EPS && passes(&gates_for(w, d, part, foreign, ctx)) {
                out.action = CargoAction::Shuttle;
                out.summary_ko = format!("{name}을(를) 쪼개도 지금은 실을 자리가 없어 먼저 운반하고 돌아옵니다");
                return out;
            }
            portion *= 0.5;
        }
    }
    out.action = CargoAction::Refuse;
    let worst = gates.iter().filter(|g| g.severity >= 2).max_by_key(|g| g.severity);
    out.code = worst.map(|g| match g.id {
        "grip" => "grasp_unstable",
        "lift" => "lift_overload",
        "route" => "route_loaded",
        _ => "capacity_exceeded",
    });
    out.summary_ko = match worst {
        Some(g) if g.id == "capacity" && foreign > 1e-3 => format!("적재 한도를 넘습니다 — {} (먼저 '내려놔' 명령으로 무관한 짐을 내려놓으세요)", g.message),
        Some(g) if g.id == "capacity" => format!("적재 한도를 넘습니다 — {}", g.message),
        Some(g) => format!("{name}을(를) 들 수 없습니다 — [{}] {}{}", g.label_ko, g.message, if spec.divisible { " (가장 작은 조각도 안 됨)" } else { " (쪼갤 수 없는 물건)" }),
        None => format!("{name}을(를) 들 수 없습니다"),
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hand_core::{BaseKind, BodyPlan, FingerCount, HandConfig, Tier};
    use world_sim::{ItemKind, WorldConfig};

    fn world(body: BodyPlan, base: BaseKind, fingers: FingerCount) -> World {
        let mut hand = HandConfig::preset(fingers);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, hand, body, ..WorldConfig::default() }).unwrap()
    }

    fn piece(w: &World, kind: ItemKind, amount: f32) -> DropItem {
        DropItem { id: 900, kind, x: w.agent.x + 0.4, z: w.agent.z, amount }
    }

    #[test]
    fn a_light_piece_is_taken_whole() {
        let w = world(BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let d = decide(&w, &piece(&w, ItemKind::Coal, 1.0), &CargoCtx { quest_cargo_kg: 0.0, destination: None });
        assert_eq!(d.action, CargoAction::Take, "{d:?}");
        assert!(d.gates.iter().all(|g| g.passed()));
    }

    #[test]
    fn a_stone_too_heavy_for_budget_arms_is_split_and_a_log_is_refused() {
        let w = world(BodyPlan { arm: Tier::Budget, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
        let stone = decide(&w, &piece(&w, ItemKind::Stone, 1.0), &CargoCtx { quest_cargo_kg: 0.0, destination: None });
        match stone.action {
            CargoAction::Split { portion } => assert!(portion <= 0.5 && portion >= MIN_PIECE),
            other => panic!("{other:?}: {stone:?}"),
        }
        assert!(stone.gates.iter().any(|g| g.id == "lift" && g.severity >= 2), "the arms are the reason");
        let log = decide(&w, &piece(&w, ItemKind::Log, 1.0), &CargoCtx { quest_cargo_kg: 0.0, destination: None });
        assert_eq!(log.action, CargoAction::Refuse);
        assert_eq!(log.code, Some("lift_overload"));
    }

    #[test]
    fn a_full_bay_of_quest_cargo_makes_a_trip_and_foreign_cargo_refuses() {
        let mut w = world(BodyPlan::default(), BaseKind::Track, FingerCount::Five);
        let cap = w.capacity_kg();
        let log = ItemKind::Log.spec().mass_kg;
        // As many logs as fit: one more does not.
        let n = (cap / log).floor();
        assert!(n >= 1.0, "a tracked mid robot carries a log ({cap} kg)");
        w.agent.carried.push((ItemKind::Log, n));
        let dest = Some((w.waypoints[1].x, w.waypoints[1].z));
        let d = decide(&w, &piece(&w, ItemKind::Log, 1.0), &CargoCtx { quest_cargo_kg: n * log, destination: dest });
        assert_eq!(d.action, CargoAction::Shuttle, "{d:?}");
        // The same bay holding someone else's cargo: nothing to deliver, so no trip.
        let d = decide(&w, &piece(&w, ItemKind::Log, 1.0), &CargoCtx { quest_cargo_kg: 0.0, destination: dest });
        assert_eq!(d.action, CargoAction::Refuse);
        assert_eq!(d.code, Some("capacity_exceeded"));
        assert!(d.summary_ko.contains("적재 한도"), "{}", d.summary_ko);
    }
}
