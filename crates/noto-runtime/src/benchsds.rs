//! Live hooks of the v1.0 benchmark areas in the SDS `act` track, called by
//! one line in `Runtime::tick` so `actsds.rs` and `mlsds.rs` do not grow:
//!
//! * [`Runtime::record_contact_sds`] — collisions, falls and touch
//!   (`act|collide:<kind>|<form>`, `act|tactile:<kind>|<form>`), the live
//!   counterparts of the collision `[CX]` and tactile `[TX]` benchmarks;
//! * [`Runtime::record_coop_sds`] — several robots carrying one object
//!   (`act|coop:<kind>|<form>`), the cooperation `[KX]` benchmark's.
//!
//! Today they are wired to the event stream and record each moment's outcome
//! and number; the physics that emits these moments comes with the
//! collision, tactile and cooperation models (`noto_core::act`), and the
//! benchmark-matched layers grow here then. Primaries of the `act` track are
//! used instead of new tracks so no ledger layout changes. The engine never
//! reads any of this back.

use noto_sds::{Scope, Track};
use world_sim::{TimedEvent, WorldEvent};

use crate::runtime::Runtime;

/// Contact moments and the area each belongs to: a body meeting a wall or a
/// curb, tipping, being hit, falling off an edge (`collide`), and the
/// fingers' and toes' slip and roll (`tactile`).
pub const CONTACT_KINDS: [(&str, &str); 7] = [("bump", "collide"), ("tip", "collide"), ("hit", "collide"), ("edge_fall", "collide"), ("incipient_slip", "tactile"), ("heel_off", "tactile"), ("toe_push", "tactile")];

/// Cooperative-carry moments: a carrier giving way, the object dropped, a
/// carrier over its share.
pub const COOP_KINDS: [&str; 3] = ["yield", "co_drop", "overload_share"];

impl Runtime {
    /// The bench-side live recorders of one tick (the one line in `tick`).
    pub(crate) fn record_bench_sds(&mut self, events: &[TimedEvent]) {
        self.record_contact_sds(events);
        self.record_coop_sds(events);
    }

    /// Collisions, falls and touch as they happen.
    pub(crate) fn record_contact_sds(&mut self, events: &[TimedEvent]) {
        for e in events {
            let WorldEvent::Act { kind, ok, value, .. } = &e.event else { continue };
            let Some((_, area)) = CONTACT_KINDS.iter().find(|(k, _)| k == kind) else { continue };
            self.record_moment(&format!("{area}:{kind}"), *ok, *value);
        }
    }

    /// Several robots carrying one object, as it happens.
    pub(crate) fn record_coop_sds(&mut self, events: &[TimedEvent]) {
        for e in events {
            let WorldEvent::Act { kind, ok, value, .. } = &e.event else { continue };
            if COOP_KINDS.contains(&kind.as_str()) {
                self.record_moment(&format!("coop:{kind}"), *ok, *value);
            }
        }
    }

    /// One moment: its outcome by robot form, its number by ground.
    fn record_moment(&mut self, primary: &str, ok: bool, value: Option<f32>) {
        let form = noto_core::bench::form_id(&self.world.body);
        self.sds.record_agreement(&Scope::new(Track::Act, primary, &form), "ok", ok);
        if let Some(v) = value {
            let ground = self.world.ground_here().id();
            self.sds.record_baseline(&Scope::new(Track::Act, &format!("{primary}:ts"), ground), "value", v as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;

    fn rt() -> Runtime {
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        Runtime::new(RuntimeConfig::default(), &lex, None).unwrap()
    }

    fn act(kind: &str, ok: bool, value: Option<f32>) -> TimedEvent {
        TimedEvent { tick: 1, time_s: 0.1, event: WorldEvent::Act { act: "walk".into(), kind: kind.into(), label: String::new(), ok, value, parts: vec![] } }
    }

    #[test]
    fn contact_and_coop_moments_reach_their_scopes() {
        let mut rt = rt();
        let form = noto_core::bench::form_id(&rt.world().body);
        rt.record_bench_sds(&[act("bump", true, Some(310.0)), act("tip", false, None), act("toe_push", true, Some(0.4)), act("co_drop", false, Some(12.0)), act("contact", true, Some(1.0))]);
        let sds = rt.sds();
        assert_eq!(sds.agreement_rate(&Scope::new(Track::Act, "collide:bump", &form), "ok"), None, "one observation is not trusted yet");
        let st = sds.status(500);
        let has = |scope: &str, axis: &str| st.axes.iter().any(|a| a.scope == scope && a.axis == axis);
        assert!(has(&format!("act|collide:bump|{form}"), "ok"));
        assert!(has(&format!("act|collide:tip|{form}"), "ok"));
        assert!(has(&format!("act|tactile:toe_push|{form}"), "ok"));
        assert!(has(&format!("act|coop:co_drop|{form}"), "ok"));
        let ground = rt.world().ground_here().id();
        assert!(has(&format!("act|collide:bump:ts|{ground}"), "value"));
        assert!(!st.axes.iter().any(|a| a.scope.starts_with("act|collide:contact")), "a grasp contact is not a collision");
    }

    #[test]
    fn a_runtime_on_disk_stamps_its_ledgers_with_the_engine_revision() {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("noto-benchsds-{}-{nanos}", std::process::id()));
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        {
            let mut rt = Runtime::new(RuntimeConfig { parcels: Some(0), ..RuntimeConfig::default() }, &lex, Some(&dir)).unwrap();
            assert_eq!(rt.sds().engine_rev(), noto_core::bench::ENGINE_REV);
            rt.record_bench_sds(&[act("bump", true, Some(310.0))]);
            rt.sds_mut().flush();
        }
        // The act ledger lists the revision its observations came from, and so does the run file.
        let ledger: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("sds").join(Track::Act.ledger().file_name())).unwrap()).unwrap();
        assert!(ledger["engine_revs"].as_array().unwrap().iter().any(|r| r == noto_core::bench::ENGINE_REV), "{}", ledger["engine_revs"]);
        let runs = dir.join("sds").join("runs").join(Track::Act.ledger().as_str());
        let run = std::fs::read_dir(&runs).unwrap().filter_map(|e| e.ok()).next().expect("a run file").path();
        let run: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run).unwrap()).unwrap();
        assert_eq!(run["engine_rev"], noto_core::bench::ENGINE_REV);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
