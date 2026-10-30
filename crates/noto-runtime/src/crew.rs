//! The crew dispatcher (작업자 배차): sub-workers (`world_sim::crew`) added around the first worker, and the two ways
//! a command runs over them.
//!
//! 개별 트랙 (solo): the command's count is split into shares, one quest per worker. Worker 1 (the first worker) runs
//! its share with the full pipeline — vision, precision gates, the fork procedure. Each sub-worker runs its share as
//! a lane of its own: the nearest free tree → fell it → pick its drops → carry → deliver.
//!
//! 협업 트랙 (team): one quest whose steps are the collaboration flow, roles by form, a loading bay between them:
//!   ① 편성     humanoids harvest and stage, lift trucks load and haul.
//!   ② 상차장   a bay on ground a lift truck can reach, between the nearest trees and the destination; lift trucks
//!              park behind it, facing its row of slots.
//!   ③ 채취     each humanoid fells a tree of its own (the first worker as a humanoid: its own vision finds one).
//!   ④ 적치     humanoids carry the logs to the bay and put them down in its row.
//!   ⑤ 상차     a lift truck takes a batch onto its tines. The first worker as a lift truck runs its fork procedure on
//!              them — rider look → carriage camera → cross-check → insert → re-check — with SGBM or TIPSv2 by its
//!              cameras.
//!   ⑥ 운반 → ⑦ 인도   it drives to the destination and delivers, and comes back until the count is reached.
//! A team without a lift truck, or without a humanoid, runs the solo track instead (and says so).
//!
//! The heavy vision runs for the first worker only; sub-workers plan, move and work on the shared world (one robot's
//! vision for the crew, as the sequential cooperation before it).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use noto_quest::{QuestResult, State, StepKind, FIRST_WORKER, TEAM};
use noto_sds::Level;
use noto_search::{PlaceRef, Plan, Target};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use world_sim::crew::{Form, Job, CREW_MAX};
use world_sim::nav::Goal;
use world_sim::snapshot::{drop_id, waypoint_id};
use world_sim::world::{DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};
use world_sim::{Action, ItemKind, TimedEvent, WorldEvent};

use crate::runtime::Runtime;

/// How far behind its bay a compact lift truck waits (m): the tines' length and a little.
pub const PARK_BEHIND_M: f32 = 1.4;
/// A lift truck goes for the staged items within this of them (m): its tines reach that far ahead.
pub const FORK_REACH_M: f32 = 1.1;
/// A team quest that has not moved on for this long fails (s).
pub const STALL_S: f64 = 420.0;
const NOTES_MAX: usize = 60;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackMode {
    #[default]
    Solo,
    Team,
}

impl TrackMode {
    pub fn from_id(s: &str) -> Option<TrackMode> {
        match s {
            "solo" | "individual" | "개별" => Some(TrackMode::Solo),
            "team" | "collab" | "협업" => Some(TrackMode::Team),
            _ => None,
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            TrackMode::Solo => "개별 트랙",
            TrackMode::Team => "협업 트랙",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A share of its own, start to end (개별 트랙).
    Solo,
    /// Fells and stages at the bay (협업: 휴머노이드).
    Harvester,
    /// Takes staged items on its tines and hauls them (협업: 지게차).
    Hauler,
}

impl Role {
    pub fn label_ko(self) -> &'static str {
        match self {
            Role::Solo => "개별 운반",
            Role::Harvester => "채취 · 적치",
            Role::Hauler => "상차 · 운반",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Stage {
    Idle,
    ToNode { node: u32 },
    Mining { node: u32 },
    ToDrop { drop: u32 },
    Picking { drop: u32 },
    ToBay,
    Staging,
    ToDest,
    Delivering,
    ToPark,
    Parked,
    Done,
}

impl Stage {
    pub fn label_ko(&self) -> String {
        match self {
            Stage::Idle => "대기".into(),
            Stage::ToNode { node } => format!("n{node}로 이동"),
            Stage::Mining { node } => format!("n{node} 채취"),
            Stage::ToDrop { drop } => format!("{}로 이동", drop_id(*drop)),
            Stage::Picking { drop } => format!("{} 줍기", drop_id(*drop)),
            Stage::ToBay => "상차장으로 운반".into(),
            Stage::Staging => "상차장 적치".into(),
            Stage::ToDest => "목적지로 운반".into(),
            Stage::Delivering => "인도".into(),
            Stage::ToPark => "상차장 집결".into(),
            Stage::Parked => "상차 대기".into(),
            Stage::Done => "완료".into(),
        }
    }
}

/// What a sub-worker is doing for its track.
#[derive(Clone, Debug, Serialize)]
pub struct Lane {
    pub worker: u32,
    pub form: Form,
    pub role: Role,
    pub quest: Option<String>,
    pub stage: Stage,
    pub item: Option<ItemKind>,
    /// Its share (개별 트랙), and what it has delivered (개별) or staged / hauled (협업).
    pub want: u32,
    pub done: f32,
    pub dest: Option<u32>,
    /// Drops it has claimed and still means to pick up.
    pub picks: Vec<u32>,
    #[serde(skip)]
    avoid: HashSet<u32>,
    pub note: String,
}

/// A command run over the team (협업 트랙): its counts along the flow.
#[derive(Clone, Debug, Serialize)]
pub struct TeamRun {
    pub quest: String,
    pub item: ItemKind,
    pub resource: String,
    pub want: u32,
    pub dest: u32,
    pub dest_ko: String,
    pub bay: u32,
    pub humanoids: Vec<u32>,
    pub forklifts: Vec<u32>,
    /// The first worker's role, and the role quest it runs now.
    pub main_role: Role,
    pub main_quest: Option<String>,
    /// Cumulative counts along the flow: picked up by the harvesters, put down at the bay, taken onto tines, delivered.
    pub harvested: f32,
    pub staged: f32,
    pub loaded: f32,
    pub delivered: f32,
    pub felled: u32,
    pub parked: bool,
    pub done: bool,
    started_tick: u64,
    started_time_s: f64,
    started_energy_j: f64,
    last_progress_s: f64,
    /// Drops the first worker (as a lift truck) was given to take from the bay.
    main_claim: Vec<u32>,
    /// What the first worker as a harvester has put down at the bay in its current role quest.
    main_progress: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlowNote {
    pub tick: u64,
    pub time_s: f64,
    pub worker: u32,
    pub step: String,
    pub text: String,
}

#[derive(Default)]
pub struct Crew {
    pub mode: TrackMode,
    pub lanes: BTreeMap<u32, Lane>,
    pub team: Option<TeamRun>,
    pub notes: VecDeque<FlowNote>,
    /// Role quests of the first worker as a lift truck: the staged drops each is to take, set on the task when it
    /// starts (`driver::begin_next`).
    pub(crate) staged_pick: HashMap<String, Vec<u32>>,
}

impl Lane {
    pub fn idle(worker: u32, form: Form) -> Lane {
        Lane { worker, form, role: Role::Solo, quest: None, stage: Stage::Idle, item: None, want: 0, done: 0.0, dest: None, picks: Vec::new(), avoid: HashSet::new(), note: String::new() }
    }
}

fn plan_for(target: &Target, quantity: u32, dest: Option<&PlaceRef>) -> Plan {
    let t = Target { quantity, ..target.clone() };
    match dest {
        Some(d) => Plan::Fetch { targets: vec![t], destination: d.clone() },
        None => Plan::Collect { targets: vec![t] },
    }
}

/// How near a worker goes to a node it fells: a humanoid within arm's reach, a lift truck where it stops for its
/// rider to step down and walk the rest.
fn mine_reach(w: &world_sim::Worker) -> f32 {
    match w.form {
        Form::Humanoid => MINE_REACH_M - 0.1,
        Form::Forklift => world_sim::world::traffic::rider_stop_m(&w.body),
    }
}

/// How near a lift truck stops to an item to take it: the tines' length in front of the item, plus the gap
/// (`FORK_REACH_M` on a compact truck).
fn fork_reach(w: &world_sim::Worker) -> f32 {
    FORK_REACH_M.max(noto_core::act::fork::standoff_x(&w.body, 0.5) + 0.1)
}

fn parse_drops(text: &str) -> Vec<u32> {
    text.split(',').filter_map(|s| s.trim().strip_prefix('d')?.parse().ok()).collect()
}

impl Runtime {
    pub fn crew(&self) -> &Crew {
        &self.crew
    }

    /// Adds `n` sub-workers around the first worker: "humanoid", "forklift", or "mixed" (one lift truck, the rest
    /// humanoids). Their cameras follow the first worker's unless given.
    pub fn crew_add(&mut self, n: usize, form: &str, cameras: Option<u8>) -> Result<Vec<u32>, String> {
        if n == 0 {
            return Err("추가할 작업자 수는 1 이상이어야 합니다".into());
        }
        let room = CREW_MAX.saturating_sub(self.world.crew.len());
        if room == 0 {
            return Err(format!("서브 작업자는 최대 {CREW_MAX}명입니다"));
        }
        let has_forklift = self.world.crew.iter().any(|w| w.form == Form::Forklift) || Form::of_base(self.world.hand.base.kind) == Form::Forklift;
        let forms: Vec<Form> = match form {
            "mixed" | "mix" | "혼합" => (0..n.min(room)).map(|i| if i == 0 && !has_forklift { Form::Forklift } else { Form::Humanoid }).collect(),
            f => {
                let f = Form::from_id(f).ok_or_else(|| format!("알 수 없는 작업자 형태 '{f}' (humanoid · forklift · mixed)"))?;
                vec![f; n.min(room)]
            }
        };
        let cameras = cameras.unwrap_or(self.cameras());
        let ids = self.world.add_crew(&forms, cameras);
        if ids.is_empty() {
            return Err("최초 작업자 주위에 빈 자리가 없습니다".into());
        }
        let list = ids
            .iter()
            .filter_map(|id| self.world.crew_worker(*id))
            .map(|w| format!("작업자 {} {} (카메라 {}대)", w.id, w.form.label_ko(), w.cameras))
            .collect::<Vec<_>>()
            .join(", ");
        self.log("crew", Level::Info, format!("서브 작업자 {}명 추가: {list} — 최초 작업자 주위", ids.len()), Some(json!({"ids": ids, "cameras": cameras})));
        for id in &ids {
            if let Some(w) = self.world.crew_worker(*id) {
                self.crew.lanes.insert(*id, Lane::idle(*id, w.form));
            }
        }
        Ok(ids)
    }

    /// Takes every sub-worker off the map; their quests end.
    pub fn crew_clear(&mut self) -> usize {
        self.crew_stop("작업자를 모두 뺐습니다");
        self.crew.lanes.clear();
        let n = self.world.remove_crew();
        if n > 0 {
            self.log("crew", Level::Info, format!("서브 작업자 {n}명을 뺌"), None);
        }
        n
    }

    pub fn set_track(&mut self, mode: TrackMode) {
        if self.crew.mode != mode {
            self.crew.mode = mode;
            self.log("crew", Level::Info, format!("작업 방식: {} — 다음 명령부터", mode.label_ko()), None);
        }
    }

    /// Stops every sub-worker and ends the crew's quests (`reason` on each).
    pub(crate) fn crew_stop(&mut self, reason: &str) {
        let ids: Vec<u32> = self.world.crew.iter().map(|w| w.id).collect();
        for id in ids {
            let _ = self.world.crew_command(id, Job::Idle);
            let _ = self.world.crew_take_outcome(id);
        }
        let mine: Vec<String> = self.quests.all().iter().filter(|q| q.owner != FIRST_WORKER && !q.is_finished()).map(|q| q.id.clone()).collect();
        for id in mine {
            if let Some(q) = self.quests.get_mut(&id) {
                let _ = q.fail("crew_stopped", reason, None);
            }
        }
        for lane in self.crew.lanes.values_mut() {
            lane.stage = Stage::Idle;
            lane.quest = None;
            lane.picks.clear();
            lane.role = Role::Solo;
        }
        self.crew.team = None;
        self.crew.staged_pick.clear();
    }

    fn note(&mut self, worker: u32, step: &str, text: impl Into<String>) {
        let text = text.into();
        self.log("crew", Level::Info, format!("[작업자 {worker}] {step}: {text}"), Some(json!({"worker": worker, "step": step})));
        self.crew.notes.push_back(FlowNote { tick: self.world.tick, time_s: self.world.time_s, worker, step: step.into(), text });
        while self.crew.notes.len() > NOTES_MAX {
            self.crew.notes.pop_front();
        }
    }

    /// A command over the crew, by the track: None when there is no crew or the command is not one the crew splits
    /// (it then runs on the first worker alone).
    pub(crate) fn crew_submit(&mut self, plan: &Plan) -> Option<Result<String, String>> {
        if self.world.crew.is_empty() {
            return None;
        }
        let (target, dest) = match plan {
            Plan::Fetch { targets, destination } if targets.len() == 1 => (targets[0].clone(), Some(destination.clone())),
            Plan::Collect { targets } if targets.len() == 1 => (targets[0].clone(), None),
            _ => return None,
        };
        let item = ItemKind::from_id(&target.resource)?;
        let dest_id = match &dest {
            Some(d) => Some(self.world.waypoint_by_key(&d.place)?.id),
            None => None,
        };
        // Leftovers of an earlier command (a lane mid-way, a team flow) end first.
        if self.crew.team.is_some() || self.crew.lanes.values().any(|l| l.quest.is_some()) {
            self.crew_stop("새 명령으로 바뀜");
        }
        Some(match (self.crew.mode, dest.clone(), dest_id) {
            (TrackMode::Team, Some(d), Some(did)) => match self.team_submit(&target, item, &d, did) {
                Ok(id) => Ok(id),
                Err(why) => {
                    self.note(0, "편성", format!("협업 트랙으로 나눌 수 없어 개별 트랙으로 진행: {why}"));
                    self.solo_submit(&target, item, dest.as_ref(), dest_id)
                }
            },
            (TrackMode::Team, None, _) => {
                self.note(0, "편성", "목적지가 없는 채취는 개별 트랙으로 진행 (적치·운반할 곳이 없음)");
                self.solo_submit(&target, item, None, None)
            }
            (_, d, did) => self.solo_submit(&target, item, d.as_ref(), did),
        })
    }

    /// 개별 트랙: shares of the count, one quest per worker.
    fn solo_submit(&mut self, target: &Target, item: ItemKind, dest: Option<&PlaceRef>, dest_id: Option<u32>) -> Result<String, String> {
        let workers: Vec<u32> = std::iter::once(FIRST_WORKER).chain(self.world.crew.iter().map(|w| w.id)).collect();
        let n = target.quantity.max(1);
        let k = workers.len() as u32;
        let tick = self.world.tick;
        let mut first = None;
        let mut shares = Vec::new();
        for (i, &w) in workers.iter().enumerate() {
            let share = n / k + u32::from((i as u32) < n % k);
            if share == 0 {
                continue;
            }
            let plan = plan_for(target, share, dest);
            let tag = format!("[개별 · 작업자 {w}]");
            let id = self.quests.add_owned(&plan, &crate::runtime::world_labels(&self.world), tick, w, "solo", None, &tag).map_err(|e| e.to_string())?;
            first.get_or_insert(id.clone());
            shares.push(format!("작업자 {w} {share}개"));
            if w != FIRST_WORKER {
                if let Some(q) = self.quests.get_mut(&id) {
                    let _ = q.start(tick);
                }
                if let Some(lane) = self.crew.lanes.get_mut(&w) {
                    *lane = Lane { role: Role::Solo, quest: Some(id.clone()), stage: Stage::Idle, item: Some(item), want: share, done: 0.0, dest: dest_id, picks: Vec::new(), avoid: HashSet::new(), note: String::new(), ..lane.clone() };
                }
            }
        }
        self.note(0, "편성", format!("개별 트랙: {} {n}개를 {}명이 나눔 — {}", item.spec().label_ko, shares.len(), shares.join(", ")));
        first.ok_or_else(|| "나눌 몫이 없습니다".into())
    }

    /// 협업 트랙: the team quest, the bay, the roles.
    fn team_submit(&mut self, target: &Target, item: ItemKind, dest: &PlaceRef, dest_id: u32) -> Result<String, String> {
        let main_form = Form::of_base(self.world.hand.base.kind);
        let mut humanoids: Vec<u32> = self.world.crew.iter().filter(|w| w.form == Form::Humanoid).map(|w| w.id).collect();
        let mut forklifts: Vec<u32> = self.world.crew.iter().filter(|w| w.form == Form::Forklift).map(|w| w.id).collect();
        match main_form {
            Form::Humanoid => humanoids.insert(0, FIRST_WORKER),
            Form::Forklift => forklifts.insert(0, FIRST_WORKER),
        }
        if humanoids.is_empty() || forklifts.is_empty() {
            return Err(format!("휴머노이드 {}명 · 지게차 {}대 — 둘 다 있어야 채취와 운반을 나눔", humanoids.len(), forklifts.len()));
        }
        if !item.from_node() {
            return Err(format!("{}은(는) 캐는 자원이 아님", item.spec().label_ko));
        }
        let bay_at = self.bay_site(item, dest_id, &forklifts).ok_or("지게차가 닿는 곳에 상차장을 둘 자리가 없음")?;
        let bay = self.world.add_bay(bay_at.0, bay_at.1, bay_at.2, "상차장");
        let n = target.quantity.max(1);
        let dest_ko = self.world.waypoint(dest_id).map(|p| p.label_ko.clone()).unwrap_or_default();
        let name = item.spec().label_ko;
        let steps = [
            (StepKind::Assign, format!("① 편성 · 역할 배정 (휴머노이드 {}: 채취·적치 / 지게차 {}: 상차·운반)", humanoids.len(), forklifts.len())),
            (StepKind::Goto, "② 상차장 설치 · 지게차 집결".to_string()),
            (StepKind::Mine, format!("③ {name} 채취 (휴머노이드)")),
            (StepKind::Stage, "④ 상차장 적치".to_string()),
            (StepKind::Load, "⑤ 지게차 상차 (포크)".to_string()),
            (StepKind::Carry, format!("⑥ {dest_ko}{} 운반 (지게차)", noto_search::text::euro(&dest_ko))),
            (StepKind::Deliver, format!("⑦ {dest_ko}에 인도 {n}개")),
        ];
        let plan = plan_for(target, n, Some(dest));
        let tick = self.world.tick;
        let title = format!("[협업] {name} {n}개 → {dest_ko} (휴머노이드 {} · 지게차 {})", humanoids.len(), forklifts.len());
        let qid = self.quests.add_custom(title, &steps, &plan, tick, TEAM, "team");
        if let Some(q) = self.quests.get_mut(&qid) {
            let _ = q.start(tick);
            let _ = q.advance_to(1);
        }
        let energy = self.world.agent.energy_used_j + self.world.crew.iter().map(|w| w.energy_j).sum::<f64>();
        let main_role = if main_form == Form::Humanoid { Role::Harvester } else { Role::Hauler };
        self.crew.team = Some(TeamRun {
            quest: qid.clone(),
            item,
            resource: target.resource.clone(),
            want: n,
            dest: dest_id,
            dest_ko,
            bay,
            humanoids: humanoids.clone(),
            forklifts: forklifts.clone(),
            main_role,
            main_quest: None,
            harvested: 0.0,
            staged: 0.0,
            loaded: 0.0,
            delivered: 0.0,
            felled: 0,
            parked: false,
            done: false,
            started_tick: tick,
            started_time_s: self.world.time_s,
            started_energy_j: energy,
            last_progress_s: self.world.time_s,
            main_claim: Vec::new(),
            main_progress: 0.0,
        });
        for lane in self.crew.lanes.values_mut() {
            let role = if lane.form == Form::Humanoid { Role::Harvester } else { Role::Hauler };
            let stage = if role == Role::Hauler { Stage::ToPark } else { Stage::Idle };
            *lane = Lane { role, quest: Some(qid.clone()), stage, item: Some(item), want: 0, done: 0.0, dest: Some(dest_id), picks: Vec::new(), avoid: HashSet::new(), note: String::new(), ..lane.clone() };
        }
        let ids = self.world.crew.iter().map(|w| w.id).collect::<Vec<_>>();
        for id in ids {
            if self.crew.lanes.get(&id).is_some_and(|l| l.stage == Stage::ToPark) {
                self.send_to_park(id);
            }
        }
        // The first worker as a humanoid harvests with its own pipeline: a role quest to the bay.
        if main_role == Role::Harvester {
            let share = n.div_ceil(humanoids.len() as u32).max(1);
            self.main_harvest(share);
        }
        self.note(0, "① 편성", format!("휴머노이드 {:?} 채취·적치, 지게차 {:?} 상차·운반 · 작업자 1은 {}", humanoids, forklifts, main_role.label_ko()));
        self.note(0, "② 상차장", format!("상차장 {} @ ({:.1}, {:.1}) — 나무 쪽을 향한 적치 슬롯 {}칸", waypoint_id(bay), bay_at.0, bay_at.1, world_sim::crew::BAY_SLOTS));
        Ok(qid)
    }

    /// Where a bay goes: near the source trees the team can reach, on the destination's side, on cells clear for its
    /// row of slots, reachable by the first lift truck. (x, z, the point it faces).
    fn bay_site(&self, item: ItemKind, dest: u32, forklifts: &[u32]) -> Option<(f32, f32, (f32, f32))> {
        let (dx, dz) = self.world.waypoint(dest).map(|p| (p.x, p.z))?;
        let from = match forklifts.first() {
            Some(&FIRST_WORKER) | None => (self.world.agent.x, self.world.agent.z),
            Some(id) => self.world.crew_worker(*id).map(|w| (w.x, w.z))?,
        };
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        let mut sources: Vec<(f32, f32, f32)> = self.world.nodes.iter().filter(|n| n.kind.spec().drop == item).map(|n| (n.x, n.z, (n.x - ax).hypot(n.z - az))).collect();
        sources.sort_by(|a, b| a.2.total_cmp(&b.2));
        let blocked = self.world.blocked_cells();
        let spec = noto_core::BaseKind::Wheel.spec();
        let start = self.world.terrain.cell_of(from.0, from.1)?;
        let clear = |x: f32, z: f32| {
            self.world.terrain.cell_of(x, z).is_some_and(|c| !blocked.contains(&c))
                && self.world.nodes.iter().all(|n| (n.x - x).hypot(n.z - z) > 0.8)
                && self.world.ground_at(self.world.terrain.cell_of(x, z).unwrap_or((0, 0))) != world_sim::Ground::Water
        };
        for &(nx, nz, _) in sources.iter().take(6) {
            let toward = (dz - nz).atan2(dx - nx);
            for r in [2.4f32, 3.0, 3.6] {
                for turn in [0.0f32, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6] {
                    let a = toward + turn;
                    let (bx, bz) = (nx + r * a.cos(), nz + r * a.sin());
                    let face = (nz - bz).atan2(nx - bx);
                    let (fx, fz) = (face.cos(), face.sin());
                    let (rx, rz) = (-fz, fx);
                    let row = (0..world_sim::crew::BAY_SLOTS).all(|k| {
                        let off = (k as f32 - (world_sim::crew::BAY_SLOTS as f32 - 1.0) * 0.5) * world_sim::crew::BAY_SLOT_M;
                        clear(bx + fx * world_sim::crew::BAY_ROW_AHEAD_M + rx * off, bz + fz * world_sim::crew::BAY_ROW_AHEAD_M + rz * off)
                    });
                    let behind = self.park_behind();
                    let park = (bx - fx * behind, bz - fz * behind);
                    if !(clear(bx, bz) && row && clear(park.0, park.1)) {
                        continue;
                    }
                    let goal = Goal::Within { x: park.0, z: park.1, reach_m: 0.4 };
                    if world_sim::nav::plan(&spec, &self.world.terrain, &|cx, cz| blocked.contains(&(cx, cz)), start, goal).is_some() {
                        return Some((bx, bz, (nx, nz)));
                    }
                }
            }
        }
        None
    }

    /// How far behind its bay a lift truck waits, so that its tines stop short of the row of staged items.
    fn park_behind(&self) -> f32 {
        let body = self.world.crew.iter().find(|w| w.form == Form::Forklift).map_or(&self.world.body, |w| &w.body);
        PARK_BEHIND_M.max(noto_core::act::fork::standoff_x(body, world_sim::ItemKind::Log.spec().size_m[0]) - world_sim::crew::BAY_ROW_AHEAD_M + 0.1)
    }

    fn bay_frame(&self, bay: u32) -> Option<((f32, f32), (f32, f32))> {
        let p = self.world.waypoint(bay)?;
        let mid = self.world.bay_slot(bay, world_sim::crew::BAY_SLOTS / 2)?;
        let (fx, fz) = (mid.0 - p.x, mid.1 - p.z);
        let n = fx.hypot(fz).max(1e-3);
        Some(((p.x, p.z), (fx / n, fz / n)))
    }

    /// Where a lift truck waits for a batch: behind the bay, facing its row.
    fn park_spot(&self, bay: u32) -> Option<((f32, f32), (f32, f32))> {
        let ((bx, bz), (fx, fz)) = self.bay_frame(bay)?;
        let row = self.world.bay_slot(bay, world_sim::crew::BAY_SLOTS / 2)?;
        let behind = self.park_behind();
        Some(((bx - fx * behind, bz - fz * behind), row))
    }

    fn send_to_park(&mut self, id: u32) {
        let Some(bay) = self.crew.team.as_ref().map(|t| t.bay) else { return };
        let Some(((px, pz), _)) = self.park_spot(bay) else { return };
        match self.world.crew_command(id, Job::Go { x: px, z: pz, reach_m: 0.35 }) {
            Ok(()) => self.set_stage(id, Stage::ToPark),
            Err(e) => self.note(id, "② 집결", format!("상차장으로 갈 수 없음: {e}")),
        }
    }

    fn set_stage(&mut self, id: u32, stage: Stage) {
        if let Some(l) = self.crew.lanes.get_mut(&id) {
            l.stage = stage;
        }
    }

    /// A role quest for the first worker as a humanoid: harvest `n` and put them down at the bay, with its own
    /// pipeline (vision finds the tree, the gaze check before each pick).
    fn main_harvest(&mut self, n: u32) {
        let Some(team) = self.crew.team.as_ref() else { return };
        let Some(key) = self.world.waypoint(team.bay).map(|p| p.key.clone()) else { return };
        let target = Target { resource: team.resource.clone(), quantity: n, via: noto_search::Via::Exact, score: 1.0, text: String::new(), origin: Vec::new() };
        let place = PlaceRef { place: key, via: noto_search::Via::Exact, score: 1.0, text: "상차장".into() };
        let plan = plan_for(&target, n, Some(&place));
        let parent = team.quest.clone();
        let tick = self.world.tick;
        match self.quests.add_owned(&plan, &crate::runtime::world_labels(&self.world), tick, FIRST_WORKER, "team", Some(&parent), "[협업 · 작업자 1 · 채취·적치]") {
            Ok(id) => {
                if let Some(t) = self.crew.team.as_mut() {
                    t.main_quest = Some(id.clone());
                    t.main_progress = 0.0;
                }
                self.note(FIRST_WORKER, "③ 채취", format!("작업자 1(휴머노이드)이 {} {n}개를 캐서 상차장에 적치 — 퀘스트 {id}", self.crew.team.as_ref().map_or("", |t| t.item.spec().label_ko)));
            }
            Err(e) => self.note(FIRST_WORKER, "③ 채취", format!("역할 퀘스트를 만들 수 없음: {e}")),
        }
    }

    /// A role quest for the first worker as a lift truck: take these staged drops onto its tines with its fork
    /// procedure and deliver them.
    fn main_haul(&mut self, drops: Vec<u32>) {
        let Some(team) = self.crew.team.as_ref() else { return };
        let Some(dest_key) = self.world.waypoint(team.dest).map(|p| p.key.clone()) else { return };
        let n = drops.len() as u32;
        let target = Target { resource: team.resource.clone(), quantity: n, via: noto_search::Via::Exact, score: 1.0, text: String::new(), origin: Vec::new() };
        let place = PlaceRef { place: dest_key, via: noto_search::Via::Exact, score: 1.0, text: team.dest_ko.clone() };
        let plan = plan_for(&target, n, Some(&place));
        let parent = team.quest.clone();
        let tick = self.world.tick;
        match self.quests.add_owned(&plan, &crate::runtime::world_labels(&self.world), tick, FIRST_WORKER, "team", Some(&parent), "[협업 · 작업자 1 · 상차·운반]") {
            Ok(id) => {
                let list = drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", ");
                self.crew.staged_pick.insert(id.clone(), drops.clone());
                if let Some(t) = self.crew.team.as_mut() {
                    t.main_quest = Some(id.clone());
                    t.main_claim = drops;
                }
                let how = if self.stereo() { "SGBM 스테레오" } else { "TIPSv2 단안" };
                self.note(FIRST_WORKER, "⑤ 상차", format!("작업자 1(지게차)이 상차장의 {list}을(를) 포크 절차({how} 깊이)로 실음 — 퀘스트 {id}"));
            }
            Err(e) => self.note(FIRST_WORKER, "⑤ 상차", format!("역할 퀘스트를 만들 수 없음: {e}")),
        }
    }

    /// Nodes and drops someone is already working on: the lanes', the first worker's, and (for a harvester) the
    /// bay's staged ones.
    fn claims(&self) -> (HashSet<u32>, HashSet<u32>) {
        let mut nodes = HashSet::new();
        let mut drops = HashSet::new();
        for l in self.crew.lanes.values() {
            match l.stage {
                Stage::ToNode { node } | Stage::Mining { node } => {
                    nodes.insert(node);
                }
                Stage::ToDrop { drop } | Stage::Picking { drop } => {
                    drops.insert(drop);
                }
                _ => {}
            }
            drops.extend(l.picks.iter().copied());
        }
        if let Some(t) = &self.task {
            drops.extend(t.drops.iter().copied());
            if let Some(id) = t.target_lm.and_then(|id| self.gates.landmark(id)).map(|l| (l.x, l.z)).and_then(|(x, z)| {
                let (tx, tz) = self.world.true_of(x, z);
                self.world.nodes.iter().filter(|n| (n.x - tx).hypot(n.z - tz) < 1.2).min_by(|a, b| (a.x - tx).hypot(a.z - tz).total_cmp(&(b.x - tx).hypot(b.z - tz))).map(|n| n.id)
            }) {
                nodes.insert(id);
            }
        }
        if let Some(a) = self.world.current_action() {
            if let Some(n) = a.strip_prefix("mine(n").and_then(|s| s.strip_suffix(')')).and_then(|s| s.parse().ok()) {
                nodes.insert(n);
            }
        }
        if let Some(team) = &self.crew.team {
            drops.extend(self.world.staged_at(team.bay));
            drops.extend(team.main_claim.iter().copied());
        }
        (nodes, drops)
    }

    fn nearest_node(&self, from: (f32, f32), item: ItemKind, avoid: &HashSet<u32>) -> Option<(u32, f32, f32)> {
        let (nodes, _) = self.claims();
        self.world
            .nodes
            .iter()
            .filter(|n| n.kind.spec().drop == item && !nodes.contains(&n.id) && !avoid.contains(&n.id))
            .min_by(|a, b| (a.x - from.0).hypot(a.z - from.1).total_cmp(&(b.x - from.0).hypot(b.z - from.1)))
            .map(|n| (n.id, n.x, n.z))
    }

    fn nearest_loose_drop(&self, from: (f32, f32), item: ItemKind) -> Option<u32> {
        let (_, drops) = self.claims();
        self.world
            .drops
            .iter()
            .filter(|d| d.kind == item && d.support.is_ground() && !drops.contains(&d.id))
            .min_by(|a, b| (a.x - from.0).hypot(a.z - from.1).total_cmp(&(b.x - from.0).hypot(b.z - from.1)))
            .map(|d| d.id)
    }

    /// Advances the crew once a tick: the team's counts from the first worker's events, each lane, the first
    /// worker's role, the team quest's flow.
    pub(crate) fn crew_tick(&mut self, events: &[TimedEvent]) {
        if self.world.crew.is_empty() && self.crew.team.is_none() {
            return;
        }
        self.team_events(events);
        let ids: Vec<u32> = self.crew.lanes.keys().copied().collect();
        for id in ids {
            if self.world.crew_worker(id).is_none() {
                self.crew.lanes.remove(&id);
                continue;
            }
            self.lane_step(id);
        }
        self.main_role_step();
        self.team_flow();
    }

    /// The first worker's part, read off its own events: what it picked up, put down at the bay, delivered.
    fn team_events(&mut self, events: &[TimedEvent]) {
        let Some(team) = self.crew.team.as_ref() else { return };
        let (bay, dest, item, role) = (waypoint_id(team.bay), waypoint_id(team.dest), team.item, team.main_role);
        // The first worker's own role quest (it may have finished this very tick: its task is gone already).
        let main_on = team.main_quest.is_some();
        let mut moved = false;
        for te in events {
            match &te.event {
                WorldEvent::ItemPicked { item: k, amount, .. } if *k == item && main_on => {
                    let t = self.crew.team.as_mut().expect("checked");
                    match role {
                        Role::Harvester => t.harvested += amount,
                        _ => t.loaded += amount,
                    }
                    moved = true;
                }
                WorldEvent::Delivered { waypoint, item: k, amount } if *k == item && *waypoint == bay => {
                    let t = self.crew.team.as_mut().expect("checked");
                    t.staged += amount;
                    t.main_progress += amount;
                    moved = true;
                }
                WorldEvent::Delivered { waypoint, item: k, amount } if *k == item && *waypoint == dest && main_on => {
                    self.crew.team.as_mut().expect("checked").delivered += amount;
                    moved = true;
                }
                _ => {}
            }
        }
        if moved {
            self.crew.team.as_mut().expect("checked").last_progress_s = self.world.time_s;
        }
    }

    fn lane_step(&mut self, id: u32) {
        let Some(lane) = self.crew.lanes.get(&id).cloned() else { return };
        if lane.quest.is_none() || lane.stage == Stage::Done {
            return;
        }
        let outcome = self.world.crew_take_outcome(id);
        match lane.role {
            Role::Solo | Role::Harvester => self.harvest_lane(id, lane, outcome),
            Role::Hauler => self.haul_lane(id, lane, outcome),
        }
    }

    fn worker_at(&self, id: u32) -> (f32, f32) {
        self.world.crew_worker(id).map_or((0.0, 0.0), |w| (w.x, w.z))
    }

    /// A humanoid's lane (and a solo lift truck's): fell, pick, then carry — to the bay (협업) or the destination (개별).
    fn harvest_lane(&mut self, id: u32, mut lane: Lane, outcome: Option<(bool, String)>) {
        let item = lane.item.unwrap_or(ItemKind::Log);
        let team = self.crew.team.is_some();
        let step = if team { "협업" } else { "개별" };
        let reach_pick = if lane.form == Form::Forklift { self.world.crew_worker(id).map_or(FORK_REACH_M, fork_reach) } else { PICKUP_REACH_M - 0.1 };
        match (lane.stage.clone(), outcome) {
            (Stage::ToNode { node }, Some((true, _))) => match self.world.crew_command(id, Job::Mine { node }) {
                Ok(()) => {
                    lane.stage = Stage::Mining { node };
                    self.lane_quest_to(&lane, StepKind::Mine, None);
                }
                Err(e) => {
                    lane.avoid.insert(node);
                    lane.stage = Stage::Idle;
                    lane.note = e;
                }
            },
            (Stage::ToNode { node }, Some((false, why))) | (Stage::Mining { node }, Some((false, why))) => {
                lane.avoid.insert(node);
                lane.stage = Stage::Idle;
                lane.note = why;
            }
            (Stage::Mining { node }, Some((true, drops))) => {
                let mine: Vec<u32> = parse_drops(&drops).into_iter().filter(|d| self.world.drop_item(*d).is_some_and(|x| x.kind == item)).collect();
                self.note(id, if team { "③ 채취" } else { "채취" }, format!("n{node}를 쓰러뜨림 → {}", mine.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", ")));
                if let Some(t) = self.crew.team.as_mut() {
                    t.felled += 1;
                    t.last_progress_s = self.world.time_s;
                }
                lane.picks.extend(mine);
                lane.stage = Stage::Idle;
            }
            (Stage::ToDrop { drop }, Some((true, _))) => match self.world.crew_command(id, Job::Pick { drop }) {
                Ok(()) => {
                    lane.stage = Stage::Picking { drop };
                    self.lane_quest_to(&lane, StepKind::Collect, None);
                }
                Err(e) => {
                    lane.picks.retain(|d| *d != drop);
                    lane.stage = Stage::Idle;
                    lane.note = e;
                }
            },
            (Stage::ToDrop { drop }, Some((false, why))) | (Stage::Picking { drop }, Some((false, why))) => {
                lane.picks.retain(|d| *d != drop);
                lane.stage = Stage::Idle;
                lane.note = why;
            }
            (Stage::Picking { drop }, Some((true, _))) => {
                lane.picks.retain(|d| *d != drop);
                if let Some(t) = self.crew.team.as_mut() {
                    t.harvested += 1.0;
                    t.last_progress_s = self.world.time_s;
                }
                lane.stage = Stage::Idle;
            }
            (Stage::ToBay, Some((true, _))) => {
                let bay = self.crew.team.as_ref().map(|t| t.bay);
                match bay.map(|b| self.world.crew_command(id, Job::Stage { bay: b })) {
                    Some(Ok(())) => lane.stage = Stage::Staging,
                    _ => lane.stage = Stage::Idle,
                }
            }
            (Stage::Staging, Some((true, ids))) => {
                let n = parse_drops(&ids).len() as f32;
                lane.done += n;
                if let Some(t) = self.crew.team.as_mut() {
                    t.staged += n;
                    t.last_progress_s = self.world.time_s;
                }
                self.note(id, "④ 적치", format!("상차장에 {} {}개 내려놓음", item.spec().label_ko, n));
                lane.stage = Stage::Idle;
            }
            (Stage::ToDest, Some((true, _))) => match lane.dest.map(|w| self.world.crew_command(id, Job::Deliver { waypoint: w })) {
                Some(Ok(())) => {
                    lane.stage = Stage::Delivering;
                    self.lane_quest_to(&lane, StepKind::Deliver, None);
                }
                _ => lane.stage = Stage::Idle,
            },
            (Stage::Delivering, Some((true, place))) => {
                lane.done += self.moved_by(id);
                self.note(id, "인도", format!("{place}에 {} 인도 ({}/{})", item.spec().label_ko, lane.done, lane.want));
                lane.stage = Stage::Idle;
            }
            (Stage::ToBay, Some((false, why))) | (Stage::ToDest, Some((false, why))) | (Stage::Staging, Some((false, why))) | (Stage::Delivering, Some((false, why))) => {
                lane.note = why;
                lane.stage = Stage::Idle;
            }
            _ => {}
        }
        if lane.stage == Stage::Idle {
            self.lane_next(id, &mut lane, step, reach_pick);
        }
        self.crew.lanes.insert(id, lane);
    }

    /// How much a sub-worker's last delivery or staging put down.
    fn moved_by(&self, id: u32) -> f32 {
        self.world.crew_worker(id).map_or(0.0, |w| w.last_moved)
    }

    /// What an idle harvesting lane does next.
    fn lane_next(&mut self, id: u32, lane: &mut Lane, step: &str, reach_pick: f32) {
        let item = lane.item.unwrap_or(ItemKind::Log);
        let (carried, room) = self.world.crew_worker(id).map_or((0.0, false), |w| (w.carried_count(), w.room_for(item)));
        let me = self.worker_at(id);
        if let Some(team) = self.crew.team.clone() {
            if team.done {
                lane.stage = Stage::Done;
                return;
            }
            let more = self.team_needs_harvest();
            // Carry what it has to the bay once full, out of drops to pick, or once the team has enough.
            if carried > 0.0 && (!room || lane.picks.is_empty() || !more) {
                if let Some((bx, bz)) = self.world.waypoint(team.bay).map(|p| (p.x, p.z)) {
                    if self.world.crew_command(id, Job::Go { x: bx, z: bz, reach_m: DELIVER_REACH_M - 0.4 }).is_ok() {
                        lane.stage = Stage::ToBay;
                        return;
                    }
                }
            }
            lane.picks.retain(|d| self.world.drop_item(*d).is_some());
            // Pick up what its tree dropped only while the team is short of the count; the rest stays where it fell.
            let short = self.team_secured() < team.want as f32 - 1e-3;
            if !short && !lane.picks.is_empty() {
                lane.picks.clear();
            }
            if let Some(&d) = lane.picks.first().filter(|_| room) {
                if let Some((x, z)) = self.world.drop_item(d).map(|d| (d.x, d.z)) {
                    if self.world.crew_command(id, Job::Go { x, z, reach_m: reach_pick }).is_ok() {
                        lane.stage = Stage::ToDrop { drop: d };
                        return;
                    }
                }
                lane.picks.retain(|x| *x != d);
            }
            if !more {
                return;
            }
            let near = self.world.waypoint(team.bay).map_or(me, |p| (p.x, p.z));
            if let Some((node, x, z)) = self.nearest_node(near, item, &lane.avoid) {
                match self.world.crew_command(id, Job::Go { x, z, reach_m: self.world.crew_worker(id).map_or(MINE_REACH_M, mine_reach) }) {
                    Ok(()) => {
                        lane.stage = Stage::ToNode { node };
                        self.note(id, "③ 채취", format!("n{node}로 감"));
                    }
                    Err(e) => {
                        lane.avoid.insert(node);
                        lane.note = e;
                    }
                }
            }
            return;
        }
        // 개별 트랙.
        if lane.done >= lane.want as f32 - 1e-3 {
            lane.stage = Stage::Done;
            self.finish_lane(id, lane);
            return;
        }
        let need = lane.want as f32 - lane.done;
        if carried > 0.0 && (carried >= need - 1e-3 || !room || lane.picks.is_empty()) {
            if let Some((x, z)) = lane.dest.and_then(|w| self.world.waypoint(w)).map(|p| (p.x, p.z)) {
                match self.world.crew_command(id, Job::Go { x, z, reach_m: DELIVER_REACH_M - 0.4 }) {
                    Ok(()) => {
                        lane.stage = Stage::ToDest;
                        self.lane_quest_to(lane, StepKind::Carry, None);
                        return;
                    }
                    Err(e) => lane.note = e,
                }
            }
        }
        lane.picks.retain(|d| self.world.drop_item(*d).is_some());
        if let Some(&d) = lane.picks.first().filter(|_| room && carried < need) {
            if let Some((x, z)) = self.world.drop_item(d).map(|d| (d.x, d.z)) {
                if self.world.crew_command(id, Job::Go { x, z, reach_m: reach_pick }).is_ok() {
                    lane.stage = Stage::ToDrop { drop: d };
                    return;
                }
            }
            lane.picks.retain(|x| *x != d);
        }
        if item.from_node() {
            match self.nearest_node(me, item, &lane.avoid) {
                Some((node, x, z)) => match self.world.crew_command(id, Job::Go { x, z, reach_m: self.world.crew_worker(id).map_or(MINE_REACH_M, mine_reach) }) {
                    Ok(()) => {
                        lane.stage = Stage::ToNode { node };
                        self.lane_quest_to(lane, StepKind::Approach, Some(format!("n{node}")));
                        self.note(id, step, format!("n{node}로 감 ({}/{})", lane.done, lane.want));
                    }
                    Err(e) => {
                        lane.avoid.insert(node);
                        lane.note = e;
                    }
                },
                None if lane.avoid.len() > 12 || self.world.nodes.iter().all(|n| n.kind.spec().drop != item) => self.fail_lane(id, lane, "캘 수 있는 대상이 남아 있지 않음"),
                None => {}
            }
        } else if let Some(d) = self.nearest_loose_drop(me, item) {
            if let Some((x, z)) = self.world.drop_item(d).map(|d| (d.x, d.z)) {
                if self.world.crew_command(id, Job::Go { x, z, reach_m: reach_pick }).is_ok() {
                    lane.picks.push(d);
                    lane.stage = Stage::ToDrop { drop: d };
                    self.lane_quest_to(lane, StepKind::Approach, Some(drop_id(d)));
                }
            }
        } else {
            self.fail_lane(id, lane, "주울 물건이 남아 있지 않음");
        }
    }

    /// A lift truck's lane in the team: park at the bay, take a batch onto the tines, haul it, come back.
    fn haul_lane(&mut self, id: u32, mut lane: Lane, outcome: Option<(bool, String)>) {
        let Some(team) = self.crew.team.clone() else {
            self.crew.lanes.insert(id, lane);
            return;
        };
        match (lane.stage.clone(), outcome) {
            (Stage::ToPark, Some((true, _))) => {
                if let Some((_, row)) = self.park_spot(team.bay) {
                    let _ = self.world.crew_command(id, Job::Park { x: row.0, z: row.1 });
                }
                lane.stage = Stage::Parked;
                if let Some(t) = self.crew.team.as_mut() {
                    t.parked = true;
                }
                self.note(id, "② 집결", "상차장 뒤에 정차, 적치 슬롯을 향함");
            }
            (Stage::ToPark, Some((false, why))) => {
                lane.note = why;
                lane.stage = Stage::ToPark;
                self.crew.lanes.insert(id, lane);
                self.send_to_park(id);
                return;
            }
            (Stage::ToDrop { drop }, Some((true, _))) => match self.world.crew_command(id, Job::Pick { drop }) {
                Ok(()) => lane.stage = Stage::Picking { drop },
                Err(e) => {
                    lane.picks.retain(|d| *d != drop);
                    lane.note = e;
                    lane.stage = Stage::Parked;
                }
            },
            (Stage::Picking { drop }, Some((true, _))) => {
                lane.picks.retain(|d| *d != drop);
                if let Some(t) = self.crew.team.as_mut() {
                    t.loaded += 1.0;
                    t.last_progress_s = self.world.time_s;
                }
                self.note(id, "⑤ 상차", format!("{}를 포크로 실음", drop_id(drop)));
                lane.stage = Stage::Parked;
            }
            (Stage::ToDrop { drop }, Some((false, why))) | (Stage::Picking { drop }, Some((false, why))) => {
                lane.picks.retain(|d| *d != drop);
                lane.note = why;
                lane.stage = Stage::Parked;
            }
            (Stage::ToDest, Some((true, _))) => {
                if self.world.crew_command(id, Job::Deliver { waypoint: team.dest }).is_ok() {
                    lane.stage = Stage::Delivering;
                }
            }
            (Stage::Delivering, Some((true, _))) => {
                let n = self.moved_by(id);
                lane.done += n;
                if let Some(t) = self.crew.team.as_mut() {
                    t.delivered += n;
                    t.last_progress_s = self.world.time_s;
                }
                self.note(id, "⑦ 인도", format!("{}에 {} {n}개 인도", team.dest_ko, team.item.spec().label_ko));
                lane.stage = Stage::ToPark;
                self.crew.lanes.insert(id, lane);
                if !self.crew.team.as_ref().is_some_and(|t| t.done || t.delivered >= t.want as f32) {
                    self.send_to_park(id);
                }
                return;
            }
            (Stage::ToDest, Some((false, why))) | (Stage::Delivering, Some((false, why))) => {
                lane.note = why;
                lane.stage = Stage::Parked;
            }
            _ => {}
        }
        if lane.stage == Stage::Parked {
            let carried = self.world.crew_worker(id).map_or(0.0, |w| w.carried_count());
            let room = self.world.crew_worker(id).is_some_and(|w| w.room_for(team.item));
            lane.picks.retain(|d| self.world.drop_item(*d).is_some());
            if let Some(&d) = lane.picks.first().filter(|_| room) {
                if let Some((x, z)) = self.world.drop_item(d).map(|d| (d.x, d.z)) {
                    if self.world.crew_command(id, Job::Go { x, z, reach_m: self.world.crew_worker(id).map_or(FORK_REACH_M, fork_reach) }).is_ok() {
                        lane.stage = Stage::ToDrop { drop: d };
                    }
                }
            } else if carried > 0.0 && (lane.picks.is_empty() || !room) {
                let (x, z) = self.world.waypoint(team.dest).map(|p| (p.x, p.z)).unwrap_or_default();
                if self.world.crew_command(id, Job::Go { x, z, reach_m: DELIVER_REACH_M - 0.4 }).is_ok() {
                    lane.stage = Stage::ToDest;
                    self.note(id, "⑥ 운반", format!("{} {carried}개를 싣고 {}로", team.item.spec().label_ko, team.dest_ko));
                }
            } else if let Some(batch) = self.batch_for(id) {
                self.note(id, "⑤ 상차", format!("적치분 {}개 상차 시작", batch.len()));
                lane.picks = batch;
            }
        }
        self.crew.lanes.insert(id, lane);
    }

    /// Items the team has made sure of: staged, loaded or delivered, in a harvester's hands, or still to come from the
    /// first worker as a harvester.
    fn team_secured(&self) -> f32 {
        let Some(t) = &self.crew.team else { return 0.0 };
        let held: f32 = self.crew.lanes.values().filter(|l| l.role == Role::Harvester).filter_map(|l| self.world.crew_worker(l.worker)).map(|w| w.carried_count()).sum();
        let main_left = match (&t.main_quest, t.main_role) {
            (Some(q), Role::Harvester) => self.quests.get(q).filter(|q| !q.is_finished()).map_or(0.0, |q| match &q.plan {
                Plan::Fetch { targets, .. } => (targets.first().map_or(0.0, |x| x.quantity as f32) - t.main_progress).max(0.0),
                _ => 0.0,
            }),
            _ => 0.0,
        };
        t.staged.max(t.loaded).max(t.delivered) + held + main_left
    }

    /// Items the team still needs harvested: the count less everything already picked up, staged, loaded or
    /// delivered, and less what the first worker as a harvester still means to bring.
    fn team_needs_harvest(&self) -> bool {
        let Some(t) = &self.crew.team else { return false };
        let lying: f32 = self.crew.lanes.values().filter(|l| l.role == Role::Harvester).map(|l| l.picks.len() as f32).sum();
        self.team_secured() + lying < t.want as f32 - 1e-3
    }

    /// The staged drops a lift truck may take now: a full batch, or what is left when nobody harvests any more.
    fn batch_for(&self, id: u32) -> Option<Vec<u32>> {
        let t = self.crew.team.as_ref()?;
        let mut taken: HashSet<u32> = self.crew.lanes.values().filter(|l| l.worker != id).flat_map(|l| l.picks.iter().copied()).collect();
        taken.extend(t.main_claim.iter().copied());
        let staged: Vec<u32> = self.world.staged_at(t.bay).into_iter().filter(|d| !taken.contains(d)).collect();
        if staged.is_empty() {
            return None;
        }
        let cap = match self.world.crew_worker(id) {
            Some(w) => (w.capacity_kg / t.item.spec().mass_kg).floor().max(1.0) as usize,
            None => world_sim::crew::BAY_SLOTS,
        };
        let in_transit: f32 = self.crew.lanes.values().filter(|l| l.role == Role::Hauler).filter_map(|l| self.world.crew_worker(l.worker)).map(|w| w.carried_count()).sum::<f32>()
            + if t.main_role == Role::Hauler { self.world.agent.carried.iter().map(|(_, n)| n).sum::<f32>() } else { 0.0 };
        let remaining = (t.want as f32 - t.delivered - in_transit).ceil().max(0.0) as usize;
        let batch = cap.min(remaining).min(world_sim::crew::BAY_SLOTS).max(1);
        let harvesting = self.team_needs_harvest() || self.crew.lanes.values().any(|l| l.role == Role::Harvester && !matches!(l.stage, Stage::Idle | Stage::Done)) || t.main_quest.as_ref().is_some_and(|q| t.main_role == Role::Harvester && self.quests.get(q).is_some_and(|q| !q.is_finished()));
        (remaining > 0 && (staged.len() >= batch || !harvesting)).then(|| staged.into_iter().take(batch).collect())
    }

    /// The first worker's part in the team: as a humanoid, its next harvest quest; as a lift truck, waiting at the
    /// bay and then a staged pickup with its own fork procedure.
    fn main_role_step(&mut self) {
        let Some(team) = self.crew.team.clone() else { return };
        if team.done {
            return;
        }
        if let Some(q) = &team.main_quest {
            match self.quests.get(q).map(|q| q.state) {
                Some(State::Done) | Some(State::Failed) | None => {
                    let failed = self.quests.get(q).is_some_and(|q| q.state == State::Failed);
                    if failed {
                        let why = self.quests.get(q).and_then(|q| q.failure.as_ref()).map(|f| f.message_ko.clone()).unwrap_or_default();
                        self.note(FIRST_WORKER, "작업자 1", format!("역할 퀘스트 {q} 실패: {why} — 서브 작업자가 이어 받음"));
                    }
                    if let Some(t) = self.crew.team.as_mut() {
                        t.main_quest = None;
                        t.main_claim.clear();
                    }
                }
                _ => return,
            }
        }
        if self.task.is_some() || self.quests.current_id().is_some() || self.world.busy() {
            return;
        }
        match team.main_role {
            Role::Harvester => {
                if self.team_needs_harvest() {
                    let cap = (self.world.capacity_kg() / team.item.spec().mass_kg).floor().max(1.0) as u32;
                    let left = (team.want as f32 - team.staged.max(team.loaded).max(team.delivered)).ceil().max(1.0) as u32;
                    self.main_harvest(cap.min(left));
                }
            }
            Role::Hauler => {
                let Some(((px, pz), _)) = self.park_spot(team.bay) else { return };
                let far = (self.world.agent.x - px).hypot(self.world.agent.z - pz) > 0.8;
                if let Some(batch) = self.main_batch() {
                    self.main_haul(batch);
                } else if far {
                    if self.world.command(Action::MoveTo { x: px, z: pz, reach_m: 0.4 }).is_ok() {
                        self.note(FIRST_WORKER, "② 집결", "작업자 1(지게차)이 상차장 뒤로 감");
                    }
                } else if !team.parked {
                    if let Some(t) = self.crew.team.as_mut() {
                        t.parked = true;
                    }
                    self.note(FIRST_WORKER, "② 집결", "작업자 1(지게차)이 상차장 뒤에 정차");
                }
            }
            Role::Solo => {}
        }
    }

    /// The staged drops the first worker as a lift truck takes next (a batch by its own capacity).
    fn main_batch(&self) -> Option<Vec<u32>> {
        let t = self.crew.team.as_ref()?;
        let taken: HashSet<u32> = self.crew.lanes.values().flat_map(|l| l.picks.iter().copied()).collect();
        let staged: Vec<u32> = self.world.staged_at(t.bay).into_iter().filter(|d| !taken.contains(d)).collect();
        if staged.is_empty() {
            return None;
        }
        let cap = (self.world.capacity_kg() / t.item.spec().mass_kg).floor().max(1.0) as usize;
        let in_transit: f32 = self.crew.lanes.values().filter(|l| l.role == Role::Hauler).filter_map(|l| self.world.crew_worker(l.worker)).map(|w| w.carried_count()).sum();
        let remaining = (t.want as f32 - t.delivered - in_transit).ceil().max(0.0) as usize;
        let batch = cap.min(remaining).min(world_sim::crew::BAY_SLOTS);
        let harvesting = self.team_needs_harvest() || self.crew.lanes.values().any(|l| l.role == Role::Harvester && !matches!(l.stage, Stage::Idle | Stage::Done));
        (batch > 0 && (staged.len() >= batch || !harvesting)).then(|| staged.into_iter().take(batch).collect())
    }

    /// The team quest's steps follow the counts; the quest finishes when the count is delivered.
    fn team_flow(&mut self) {
        let Some(t) = self.crew.team.clone() else { return };
        if t.done {
            return;
        }
        let n = t.want as f32;
        let (at, progress, note) = if !t.parked {
            (1, 0.5, "지게차가 상차장으로 가는 중".to_string())
        } else if t.harvested.max(t.staged) < n - 1e-3 {
            (2, t.harvested.max(t.staged) / n, format!("채취 {}/{} (쓰러뜨린 나무 {})", t.harvested.max(t.staged), t.want, t.felled))
        } else if t.staged < n - 1e-3 {
            (3, t.staged / n, format!("적치 {}/{}", t.staged, t.want))
        } else if t.loaded < n - 1e-3 {
            (4, t.loaded / n, format!("상차 {}/{}", t.loaded, t.want))
        } else if t.delivered < n - 1e-3 {
            (5, t.delivered / n, format!("운반 중 · 인도 {}/{}", t.delivered, t.want))
        } else {
            (7, 1.0, String::new())
        };
        if let Some(q) = self.quests.get_mut(&t.quest) {
            if at < 7 {
                let _ = q.advance_to(at);
                let _ = q.set_progress(at, progress, Some(note));
            }
        }
        if t.delivered >= n - 1e-3 {
            let energy = self.world.agent.energy_used_j + self.world.crew.iter().map(|w| w.energy_j).sum::<f64>() - t.started_energy_j;
            let summary = format!(
                "협업 완료: {} {}개를 {}에 인도 — 쓰러뜨린 나무 {} · 적치 {} · 상차 {} · 휴머노이드 {} · 지게차 {}",
                t.item.spec().label_ko,
                t.delivered,
                t.dest_ko,
                t.felled,
                t.staged,
                t.loaded,
                t.humanoids.len(),
                t.forklifts.len()
            );
            let result = QuestResult {
                ticks: self.world.tick.saturating_sub(t.started_tick),
                time_s: self.world.time_s - t.started_time_s,
                energy_j: energy,
                precision: None,
                summary: summary.clone(),
                items: vec![(t.item.id().to_string(), t.delivered.round() as u32)],
                wasted_actions: 0,
                trips: 0,
            };
            if let Some(q) = self.quests.get_mut(&t.quest) {
                let _ = q.finish_all(result);
            }
            if let Some(team) = self.crew.team.as_mut() {
                team.done = true;
            }
            for lane in self.crew.lanes.values_mut() {
                lane.stage = Stage::Done;
            }
            self.note(0, "⑦ 인도", summary.clone());
            self.log("quest", Level::Info, format!("퀘스트 {} 완료: {summary}", t.quest), Some(json!({"quest": t.quest, "time_s": self.world.time_s - t.started_time_s})));
            return;
        }
        if self.world.time_s - t.last_progress_s > STALL_S {
            let why = format!("{}초 동안 흐름이 나아가지 않음 (채취 {} · 적치 {} · 상차 {} · 인도 {})", STALL_S as u32, t.harvested, t.staged, t.loaded, t.delivered);
            if let Some(q) = self.quests.get_mut(&t.quest) {
                let _ = q.fail("team_stalled", &why, None);
            }
            self.note(0, "협업", format!("중단: {why}"));
            self.crew_stop(&why);
        }
    }

    /// Moves a lane's own quest (개별 트랙) on to the step of `kind`, with a note.
    fn lane_quest_to(&mut self, lane: &Lane, kind: StepKind, note: Option<String>) {
        if self.crew.team.is_some() {
            return;
        }
        let Some(q) = lane.quest.as_ref().and_then(|id| self.quests.get_mut(id)) else { return };
        let Some(i) = q.steps.iter().position(|s| s.kind == kind) else { return };
        if q.current().is_some_and(|at| at > i) {
            // A further round (the share is more than one trip): back to that step.
            let _ = q.rewind(i, "다음 회차");
        } else {
            let _ = q.advance_to(i);
        }
        let text = format!("{}/{}{}", lane.done, lane.want, note.map(|n| format!(" · {n}")).unwrap_or_default());
        let _ = q.set_progress(i, (lane.done / lane.want.max(1) as f32).clamp(0.0, 1.0), Some(text));
    }

    fn finish_lane(&mut self, id: u32, lane: &Lane) {
        let Some(qid) = lane.quest.clone() else { return };
        let energy = self.world.crew_worker(id).map_or(0.0, |w| w.energy_j);
        let item = lane.item.map_or("", |k| k.spec().label_ko);
        let summary = format!("작업자 {id}({}): {item} {}개 인도", lane.form.label_ko(), lane.done);
        let started = self.quests.get(&qid).and_then(|q| q.started_tick).unwrap_or(self.world.tick);
        let result = QuestResult {
            ticks: self.world.tick.saturating_sub(started),
            time_s: self.world.tick.saturating_sub(started) as f64 * world_sim::TICK_S as f64,
            energy_j: energy,
            precision: None,
            summary: summary.clone(),
            items: vec![(lane.item.map_or("", |k| k.id()).to_string(), lane.done.round() as u32)],
            wasted_actions: 0,
            trips: 0,
        };
        if let Some(q) = self.quests.get_mut(&qid) {
            let _ = q.finish_all(result);
        }
        self.note(id, "완료", summary);
    }

    fn fail_lane(&mut self, id: u32, lane: &mut Lane, why: &str) {
        if let Some(q) = lane.quest.as_ref().and_then(|id| self.quests.get_mut(id)) {
            let _ = q.fail("crew_failed", why, None);
        }
        lane.stage = Stage::Done;
        self.note(id, "실패", why.to_string());
    }

    pub fn crew_json(&self) -> Value {
        let d = self.world.terrain.size_m().1;
        let at = |x: f32, z: f32| json!([(x * 1000.0).round() / 1000.0, ((d - z) * 1000.0).round() / 1000.0]);
        let main_form = Form::of_base(self.world.hand.base.kind);
        let mut workers = vec![json!({
            "id": FIRST_WORKER, "form": main_form, "form_ko": main_form.label_ko(), "first": true, "cameras": self.cameras(),
            "pos": at(self.world.agent.x, self.world.agent.z),
            "role": self.crew.team.as_ref().map(|t| t.main_role),
            "role_ko": self.crew.team.as_ref().map_or("최초 작업자 (전체 비전)", |t| t.main_role.label_ko()),
            "status_ko": self.world.current_action().unwrap_or_else(|| "대기".into()),
            "carried": self.world.agent.carried.iter().map(|(_, n)| n).sum::<f32>(),
            "quest": self.quests.current_id(),
        })];
        for w in &self.world.crew {
            let lane = self.crew.lanes.get(&w.id);
            workers.push(json!({
                "id": w.id, "form": w.form, "form_ko": w.form.label_ko(), "first": false, "cameras": w.cameras,
                "pos": at(w.x, w.z), "yaw_deg": w.yaw.to_degrees(),
                "role": lane.map(|l| l.role), "role_ko": lane.map_or("대기", |l| if l.quest.is_some() { l.role.label_ko() } else { "대기" }),
                "stage": lane.map(|l| &l.stage), "stage_ko": lane.map(|l| l.stage.label_ko()),
                "status_ko": w.status_ko(), "carried": w.carried_count(), "carried_kg": w.carried_kg(), "capacity_kg": w.capacity_kg,
                "quest": lane.and_then(|l| l.quest.clone()), "done": lane.map(|l| l.done), "want": lane.map(|l| l.want),
                "note": lane.map(|l| l.note.clone()), "energy_kj": w.energy_j / 1000.0,
            }));
        }
        let team = self.crew.team.as_ref().map(|t| {
            json!({
                "quest": t.quest, "item": t.item.id(), "item_ko": t.item.spec().label_ko, "want": t.want, "dest": waypoint_id(t.dest), "dest_ko": t.dest_ko,
                "bay": waypoint_id(t.bay), "bay_pos": self.world.waypoint(t.bay).map(|p| at(p.x, p.z)),
                "staged_now": self.world.staged_at(t.bay).len(),
                "humanoids": t.humanoids, "forklifts": t.forklifts, "main_role": t.main_role, "main_quest": t.main_quest,
                "counts": {"felled": t.felled, "harvested": t.harvested, "staged": t.staged, "loaded": t.loaded, "delivered": t.delivered},
                "parked": t.parked, "done": t.done,
                "depth_procedure": if self.stereo() { "sgbm" } else { "tipsv2" },
            })
        });
        json!({
            "mode": self.crew.mode, "mode_ko": self.crew.mode.label_ko(), "max": CREW_MAX, "count": self.world.crew.len(),
            "workers": workers, "team": team,
            "notes": self.crew.notes.iter().rev().take(24).map(|n| json!({"tick": n.tick, "time_s": (n.time_s * 10.0).round() / 10.0, "worker": n.worker, "step": n.step, "text": n.text})).collect::<Vec<_>>(),
        })
    }
}
