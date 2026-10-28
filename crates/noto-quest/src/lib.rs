use noto_search::Plan;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    Locate,
    Approach,
    Mine,
    Collect,
    Carry,
    Deliver,
    Goto,
    Unload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Pending,
    Active,
    Done,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Step {
    pub kind: StepKind,
    pub label_ko: String,
    pub state: State,
    pub progress: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
    #[serde(skip_serializing_if = "is_zero")]
    pub retries: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Failure {
    pub code: String,
    pub message_ko: String,
    pub step: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuestResult {
    pub ticks: u64,
    pub time_s: f64,
    pub energy_j: f64,
    pub precision: Option<f64>,
    pub summary: String,
    pub items: Vec<(String, u32)>,
    pub wasted_actions: u32,
    pub trips: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Quest {
    pub id: String,
    pub title: String,
    pub state: State,
    pub created_tick: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_tick: Option<u64>,
    pub steps: Vec<Step>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<QuestResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<Failure>,
    pub plan: Plan,
}

pub trait Labels {
    fn resource_ko(&self, id: &str) -> String;
    fn place_ko(&self, id: &str) -> String;
    fn resource_mined(&self, id: &str) -> bool {
        let _ = id;
        true
    }
    fn origin_ko(&self, id: &str) -> String {
        match id {
            "house" => "집".into(),
            "doorstep" => "현관 앞".into(),
            "shelf" => "선반".into(),
            "table" => "포장 테이블".into(),
            other => self.place_ko(other),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuestError {
    NotPending,
    NotActive,
    BadStep(usize),
    StepNotActive(usize),
    NoItems,
}

impl std::fmt::Display for QuestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuestError::NotPending => write!(f, "the quest is not pending"),
            QuestError::NotActive => write!(f, "the quest is not active"),
            QuestError::BadStep(i) => write!(f, "no step {i}"),
            QuestError::StepNotActive(i) => write!(f, "step {i} is not active"),
            QuestError::NoItems => write!(f, "this plan has no quest"),
        }
    }
}

impl std::error::Error for QuestError {}

fn mine_label(labels: &dyn Labels, resource: &str, name: &str) -> String {
    if labels.resource_mined(resource) {
        format!("{name} 채취")
    } else {
        format!("{name} 내려다보며 확인")
    }
}

fn gather_steps(steps: &mut Vec<Step>, t: &noto_search::Target, labels: &dyn Labels) {
    let name = labels.resource_ko(&t.resource);
    let at: Vec<String> = t.origin.iter().map(|p| labels.origin_ko(&p.place)).collect();
    let locate = if at.is_empty() { format!("{name} 탐색") } else { format!("{} {name} 탐색", at.join(" ")) };
    steps.push(step(StepKind::Locate, locate, Some(&t.resource)));
    steps.push(step(StepKind::Approach, format!("{name} 가까이 접근"), Some(&t.resource)));
    steps.push(step(StepKind::Mine, mine_label(labels, &t.resource, &name), Some(&t.resource)));
    let collect = match at.last() {
        Some(place) if !labels.resource_mined(&t.resource) => format!("{place}에서 {name} 집기 {}개", t.quantity),
        _ => format!("{name} 줍기 {}개", t.quantity),
    };
    steps.push(step(StepKind::Collect, collect, Some(&t.resource)));
}

fn step(kind: StepKind, label: String, resource: Option<&str>) -> Step {
    Step { kind, label_ko: label, state: State::Pending, progress: 0.0, note: None, resource: resource.map(String::from), retries: 0 }
}

impl Quest {
    pub fn from_plan(id: String, plan: &Plan, labels: &dyn Labels, tick: u64) -> Result<Quest, QuestError> {
        let mut steps = Vec::new();
        let title;
        match plan {
            Plan::Fetch { targets, destination } => {
                for t in targets {
                    gather_steps(&mut steps, t, labels);
                }
                let place = labels.place_ko(&destination.place);
                let carry = if destination.implied() { "복귀" } else { "운반" };
                steps.push(step(StepKind::Carry, format!("{place}{} {carry}", noto_search::text::euro(&place)), None));
                steps.push(step(StepKind::Deliver, format!("{place}에 인도"), None));
                title = format!("{} → {place} 운반", item_list(targets, labels));
            }
            Plan::Collect { targets } => {
                for t in targets {
                    gather_steps(&mut steps, t, labels);
                }
                title = format!("{} 채취", item_list(targets, labels));
            }
            Plan::Goto { place } => {
                let p = labels.place_ko(&place.place);
                steps.push(step(StepKind::Goto, format!("{p}{} 이동", noto_search::text::euro(&p)), None));
                title = format!("{p}{} 이동", noto_search::text::euro(&p));
            }
            Plan::Drop { items, place } => {
                let what = if items.is_empty() {
                    "적재함 전부".to_string()
                } else {
                    items
                        .iter()
                        .map(|t| match t.quantity {
                            Some(n) => format!("{} {n}개", labels.resource_ko(&t.resource)),
                            None => format!("{} 전부", labels.resource_ko(&t.resource)),
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                match place {
                    Some(pl) => {
                        let p = labels.place_ko(&pl.place);
                        steps.push(step(StepKind::Goto, format!("{p}{} 이동", noto_search::text::euro(&p)), None));
                        steps.push(step(StepKind::Unload, format!("{p}에 {what} 내려놓기"), None));
                        title = format!("{what} → {p}에 내려놓기");
                    }
                    None => {
                        steps.push(step(StepKind::Unload, format!("{what} 내려놓기"), None));
                        title = format!("{what} 내려놓기");
                    }
                }
            }
            Plan::Stop => return Err(QuestError::NoItems),
        }
        Ok(Quest {
            id,
            title,
            state: State::Pending,
            created_tick: tick,
            started_tick: None,
            steps,
            result: None,
            failure: None,
            plan: plan.clone(),
        })
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, State::Done | State::Failed)
    }

    pub fn current(&self) -> Option<usize> {
        self.steps.iter().position(|s| s.state == State::Active)
    }

    pub fn start(&mut self, tick: u64) -> Result<(), QuestError> {
        if self.state != State::Pending {
            return Err(QuestError::NotPending);
        }
        self.state = State::Active;
        self.started_tick = Some(tick);
        self.steps[0].state = State::Active;
        Ok(())
    }

    fn active_step(&mut self, i: usize) -> Result<&mut Step, QuestError> {
        if self.state != State::Active {
            return Err(QuestError::NotActive);
        }
        let s = self.steps.get_mut(i).ok_or(QuestError::BadStep(i))?;
        if s.state != State::Active {
            return Err(QuestError::StepNotActive(i));
        }
        Ok(s)
    }

    pub fn set_progress(&mut self, i: usize, progress: f32, note: Option<String>) -> Result<(), QuestError> {
        let s = self.active_step(i)?;
        s.progress = progress.clamp(0.0, 1.0);
        if note.is_some() {
            s.note = note;
        }
        Ok(())
    }

    pub fn complete_step(&mut self, i: usize) -> Result<bool, QuestError> {
        let s = self.active_step(i)?;
        s.state = State::Done;
        s.progress = 1.0;
        match self.steps.get_mut(i + 1) {
            Some(next) => {
                next.state = State::Active;
                Ok(false)
            }
            None => Ok(true),
        }
    }

    pub fn rewind(&mut self, to: usize, note: &str) -> Result<(), QuestError> {
        if self.state != State::Active {
            return Err(QuestError::NotActive);
        }
        let at = self.current().ok_or(QuestError::NotActive)?;
        if to > at {
            return Err(QuestError::BadStep(to));
        }
        for s in &mut self.steps[to..] {
            s.state = State::Pending;
            s.progress = 0.0;
            s.note = None;
        }
        let s = &mut self.steps[to];
        s.state = State::Active;
        s.retries += 1;
        s.note = Some(note.to_string());
        Ok(())
    }

    pub fn finish(&mut self, result: QuestResult) -> Result<(), QuestError> {
        if self.state != State::Active {
            return Err(QuestError::NotActive);
        }
        if self.steps.iter().any(|s| s.state != State::Done) {
            return Err(QuestError::StepNotActive(self.current().unwrap_or(0)));
        }
        self.state = State::Done;
        self.result = Some(result);
        Ok(())
    }

    pub fn fail(&mut self, code: &str, message_ko: &str, result: Option<QuestResult>) -> Result<(), QuestError> {
        if self.is_finished() {
            return Err(QuestError::NotActive);
        }
        let at = self.current().unwrap_or(0);
        if let Some(s) = self.steps.get_mut(at) {
            s.state = State::Failed;
            s.note = Some(message_ko.to_string());
        }
        self.state = State::Failed;
        self.failure = Some(Failure { code: code.to_string(), message_ko: message_ko.to_string(), step: at });
        self.result = result;
        Ok(())
    }
}

fn item_list(targets: &[noto_search::Target], labels: &dyn Labels) -> String {
    targets
        .iter()
        .map(|t| format!("{} {}개", labels.resource_ko(&t.resource), t.quantity))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Clone, Debug, Default)]
pub struct QuestBook {
    quests: Vec<Quest>,
    next: u32,
}

impl QuestBook {
    pub fn new() -> QuestBook {
        QuestBook { quests: Vec::new(), next: 1 }
    }

    pub fn add(&mut self, plan: &Plan, labels: &dyn Labels, tick: u64) -> Result<String, QuestError> {
        let id = format!("q{}", self.next.max(1));
        let q = Quest::from_plan(id.clone(), plan, labels, tick)?;
        self.next = self.next.max(1) + 1;
        self.quests.push(q);
        Ok(id)
    }

    pub fn all(&self) -> &[Quest] {
        &self.quests
    }

    pub fn get(&self, id: &str) -> Option<&Quest> {
        self.quests.iter().find(|q| q.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Quest> {
        self.quests.iter_mut().find(|q| q.id == id)
    }

    pub fn current_id(&self) -> Option<String> {
        self.quests
            .iter()
            .find(|q| q.state == State::Active)
            .or_else(|| self.quests.iter().find(|q| q.state == State::Pending))
            .map(|q| q.id.clone())
    }

    pub fn clear(&mut self) {
        self.quests.clear();
        self.next = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_search::{PlaceRef, Target, Via};

    struct Ko;
    impl Labels for Ko {
        fn resource_ko(&self, id: &str) -> String {
            match id {
                "log" => "원목".into(),
                "stone" => "돌".into(),
                other => other.into(),
            }
        }
        fn place_ko(&self, id: &str) -> String {
            match id {
                "warehouse" => "창고".into(),
                other => other.into(),
            }
        }
    }

    fn target(r: &str, q: u32) -> Target {
        Target { resource: r.into(), quantity: q, via: Via::Exact, score: 1.0, text: r.into(), origin: vec![] }
    }

    fn place(p: &str) -> PlaceRef {
        PlaceRef { place: p.into(), via: Via::Exact, score: 1.0, text: p.into() }
    }

    fn fetch() -> Plan {
        Plan::Fetch { targets: vec![target("log", 2)], destination: place("warehouse") }
    }

    #[test]
    fn a_fetch_plan_has_the_six_steps_in_order() {
        let q = Quest::from_plan("q1".into(), &fetch(), &Ko, 5).unwrap();
        let kinds: Vec<StepKind> = q.steps.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![StepKind::Locate, StepKind::Approach, StepKind::Mine, StepKind::Collect, StepKind::Carry, StepKind::Deliver]
        );
        assert_eq!(q.title, "원목 2개 → 창고 운반");
        assert_eq!(q.steps[3].label_ko, "원목 줍기 2개");
        assert_eq!(q.state, State::Pending);
        assert!(q.steps.iter().all(|s| s.state == State::Pending));
    }

    #[test]
    fn several_resources_repeat_the_gather_steps_and_share_one_delivery() {
        let plan = Plan::Fetch { targets: vec![target("log", 2), target("stone", 3)], destination: place("warehouse") };
        let q = Quest::from_plan("q1".into(), &plan, &Ko, 0).unwrap();
        assert_eq!(q.steps.len(), 4 + 4 + 2);
        assert_eq!(q.steps[4].resource.as_deref(), Some("stone"));
        assert_eq!(q.steps.last().unwrap().kind, StepKind::Deliver);
        assert_eq!(q.title, "원목 2개, 돌 3개 → 창고 운반");
    }

    struct Parcels;
    impl Labels for Parcels {
        fn resource_ko(&self, id: &str) -> String {
            if id == "parcel" { "택배".into() } else { id.into() }
        }
        fn place_ko(&self, id: &str) -> String {
            match id {
                "base_camp" => "베이스캠프".into(),
                "warehouse" => "창고".into(),
                other => other.into(),
            }
        }
        fn resource_mined(&self, id: &str) -> bool {
            id != "parcel"
        }
    }

    #[test]
    fn a_parcel_is_looked_down_at_picked_where_it_lies_and_brought_back_to_the_base() {
        let mut t = target("parcel", 1);
        t.origin = vec![place("warehouse"), place("shelf")];
        let home = PlaceRef { place: "base_camp".into(), via: Via::Exact, score: 1.0, text: String::new() };
        let q = Quest::from_plan("q".into(), &Plan::Fetch { targets: vec![t], destination: home.clone() }, &Parcels, 0).unwrap();
        let labels: Vec<&str> = q.steps.iter().map(|s| s.label_ko.as_str()).collect();
        assert_eq!(labels, vec!["창고 선반 택배 탐색", "택배 가까이 접근", "택배 내려다보며 확인", "선반에서 택배 집기 1개", "베이스캠프로 복귀", "베이스캠프에 인도"]);
        let said = PlaceRef { text: "베이스캠프로".into(), ..home };
        let q = Quest::from_plan("q".into(), &Plan::Fetch { targets: vec![target("parcel", 2)], destination: said }, &Parcels, 0).unwrap();
        assert_eq!(q.steps[0].label_ko, "택배 탐색");
        assert_eq!(q.steps[3].label_ko, "택배 줍기 2개");
        assert_eq!(q.steps[4].label_ko, "베이스캠프로 운반");
        let mut t = target("parcel", 1);
        t.origin = vec![place("doorstep")];
        let q = Quest::from_plan("q".into(), &Plan::Collect { targets: vec![t] }, &Parcels, 0).unwrap();
        assert_eq!(q.steps[3].label_ko, "현관 앞에서 택배 집기 1개");
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["plan"]["targets"][0]["origin"][0]["place"], "doorstep");
    }

    #[test]
    fn collect_and_goto_plans() {
        let c = Quest::from_plan("q".into(), &Plan::Collect { targets: vec![target("stone", 1)] }, &Ko, 0).unwrap();
        assert_eq!(c.steps.len(), 4);
        assert_eq!(c.title, "돌 1개 채취");
        let g = Quest::from_plan("q".into(), &Plan::Goto { place: place("warehouse") }, &Ko, 0).unwrap();
        assert_eq!(g.steps.len(), 1);
        assert_eq!(g.steps[0].kind, StepKind::Goto);
        assert_eq!(Quest::from_plan("q".into(), &Plan::Stop, &Ko, 0).unwrap_err(), QuestError::NoItems);
    }

    #[test]
    fn a_drop_plan_unloads_here_or_after_going_there() {
        use noto_search::DropTarget;
        let one = DropTarget { resource: "log".into(), quantity: Some(1), via: Via::Exact, score: 1.0, text: "원목".into() };
        let q = Quest::from_plan("q".into(), &Plan::Drop { items: vec![one.clone()], place: None }, &Ko, 0).unwrap();
        assert_eq!(q.steps.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StepKind::Unload]);
        assert_eq!(q.title, "원목 1개 내려놓기");
        let all = Quest::from_plan("q".into(), &Plan::Drop { items: vec![], place: Some(place("warehouse")) }, &Ko, 0).unwrap();
        assert_eq!(all.steps.iter().map(|s| s.kind).collect::<Vec<_>>(), vec![StepKind::Goto, StepKind::Unload]);
        assert_eq!(all.title, "적재함 전부 → 창고에 내려놓기");
        let v = serde_json::to_value(&all).unwrap();
        assert_eq!(v["steps"][1]["kind"], "unload");
        assert_eq!(v["plan"]["kind"], "drop");
    }

    #[test]
    fn steps_move_one_at_a_time_to_done() {
        let mut q = Quest::from_plan("q1".into(), &fetch(), &Ko, 0).unwrap();
        q.start(3).unwrap();
        assert_eq!(q.current(), Some(0));
        q.set_progress(0, 0.5, Some("한 바퀴".into())).unwrap();
        assert_eq!(q.steps[0].progress, 0.5);
        for i in 0..5 {
            assert!(!q.complete_step(i).unwrap());
            assert_eq!(q.current(), Some(i + 1));
        }
        assert!(q.complete_step(5).unwrap(), "the last step reports that the quest can finish");
        let r = QuestResult { ticks: 10, time_s: 1.0, energy_j: 2.0, precision: Some(1.0), summary: "ok".into(), items: vec![], wasted_actions: 0, trips: 0 };
        q.finish(r).unwrap();
        assert_eq!(q.state, State::Done);
        assert!(q.is_finished());
    }

    #[test]
    fn illegal_transitions_are_refused() {
        let mut q = Quest::from_plan("q1".into(), &fetch(), &Ko, 0).unwrap();
        assert_eq!(q.complete_step(0).unwrap_err(), QuestError::NotActive);
        q.start(0).unwrap();
        assert_eq!(q.start(1).unwrap_err(), QuestError::NotPending);
        assert_eq!(q.complete_step(2).unwrap_err(), QuestError::StepNotActive(2));
        assert_eq!(q.complete_step(99).unwrap_err(), QuestError::BadStep(99));
        let r = QuestResult { ticks: 0, time_s: 0.0, energy_j: 0.0, precision: None, summary: String::new(), items: vec![], wasted_actions: 0, trips: 0 };
        assert!(q.finish(r).is_err(), "unfinished steps block finish");
    }

    #[test]
    fn rewinding_reopens_earlier_steps_and_counts_the_retry() {
        let mut q = Quest::from_plan("q1".into(), &fetch(), &Ko, 0).unwrap();
        q.start(0).unwrap();
        for i in 0..3 {
            q.complete_step(i).unwrap();
        }
        assert_eq!(q.current(), Some(3));
        q.rewind(0, "한 그루 더 필요").unwrap();
        assert_eq!(q.current(), Some(0));
        assert!(q.steps[1..].iter().all(|s| s.state == State::Pending));
        assert_eq!(q.steps[0].retries, 1);
        assert_eq!(q.steps[0].note.as_deref(), Some("한 그루 더 필요"));
        assert_eq!(q.rewind(2, "x").unwrap_err(), QuestError::BadStep(2), "cannot rewind forward");
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["steps"][0]["retries"], 1);
        assert!(v["steps"][1].get("retries").is_none());
    }

    #[test]
    fn failing_marks_the_active_step_and_keeps_the_reason() {
        let mut q = Quest::from_plan("q1".into(), &fetch(), &Ko, 0).unwrap();
        q.start(0).unwrap();
        q.complete_step(0).unwrap();
        q.fail("no_path", "경로가 없습니다", None).unwrap();
        assert_eq!(q.state, State::Failed);
        assert_eq!(q.steps[1].state, State::Failed);
        assert_eq!(q.failure.as_ref().unwrap().code, "no_path");
        assert_eq!(q.failure.as_ref().unwrap().step, 1);
        assert!(q.fail("again", "x", None).is_err(), "a finished quest cannot fail twice");
    }

    #[test]
    fn the_book_numbers_quests_and_picks_the_current_one() {
        let mut b = QuestBook::new();
        let a = b.add(&fetch(), &Ko, 0).unwrap();
        let c = b.add(&Plan::Collect { targets: vec![target("stone", 1)] }, &Ko, 1).unwrap();
        assert_eq!((a.as_str(), c.as_str()), ("q1", "q2"));
        assert_eq!(b.current_id().as_deref(), Some("q1"));
        b.get_mut("q1").unwrap().start(0).unwrap();
        b.get_mut("q1").unwrap().fail("x", "y", None).unwrap();
        assert_eq!(b.current_id().as_deref(), Some("q2"));
        assert!(b.add(&Plan::Stop, &Ko, 0).is_err());
        assert_eq!(b.all().len(), 2);
    }

    #[test]
    fn json_matches_what_the_ui_reads() {
        let mut q = Quest::from_plan("q1".into(), &fetch(), &Ko, 0).unwrap();
        q.start(0).unwrap();
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["state"], "active");
        assert_eq!(v["steps"][0]["kind"], "locate");
        assert_eq!(v["steps"][0]["state"], "active");
        assert!(v["steps"][0].get("note").is_none());
        assert_eq!(v["created_tick"], 0);
    }
}
