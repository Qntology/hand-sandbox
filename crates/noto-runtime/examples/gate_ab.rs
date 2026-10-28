use noto_quest::State;
use noto_runtime::{GateCfg, NoiseCfg, Runtime, RuntimeConfig};

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

#[derive(Default)]
struct Row {
    done: usize,
    runs: usize,
    wasted: u32,
    ticks: u64,
    energy: f64,
    rp: f64,
    rr: f64,
    gp: f64,
    gr: f64,
}

fn run(gates: bool, seeds: &[u64], noise: &NoiseCfg, command: &str) -> Row {
    let mut row = Row::default();
    for &seed in seeds {
        let cfg = RuntimeConfig {
            seed,
            noise: NoiseCfg { seed: seed + 100, ..noise.clone() },
            gates: GateCfg { enabled: gates, ..GateCfg::default() },
            ..RuntimeConfig::default()
        };
        let mut rt = Runtime::new(cfg, LEXICON, None).expect("runtime");
        let s = rt.submit(command);
        let Some(id) = s.quest_id else {
            eprintln!("command not understood: {:?}", s.error);
            std::process::exit(1);
        };
        rt.run_until_idle(60_000);
        let q = rt.quest_book().get(&id).unwrap();
        row.runs += 1;
        row.done += (q.state == State::Done) as usize;
        row.wasted += q.result.as_ref().map_or(0, |r| r.wasted_actions);
        row.energy += q.result.as_ref().map_or(0.0, |r| r.energy_j);
        row.ticks += rt.world().tick;
        let e = rt.eval().summary();
        row.rp += e.raw_precision.unwrap_or(0.0);
        row.gp += e.gated_precision.unwrap_or(0.0);
        row.rr += e.raw_recall.unwrap_or(0.0);
        row.gr += e.gated_recall.unwrap_or(0.0);
    }
    let n = row.runs.max(1) as f64;
    row.rp /= n;
    row.gp /= n;
    row.rr /= n;
    row.gr /= n;
    row
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: u64 = a.first().and_then(|s| s.parse().ok()).unwrap_or(12);
    let confusion: f32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.12);
    let fp: f32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.30);
    let command = a.get(3).cloned().unwrap_or_else(|| "돌 2개를 창고로 가져와".into());
    let seeds: Vec<u64> = (1..=n).map(|i| i * 2 + 1).collect();
    let noise = NoiseCfg { confusion_rate: confusion, false_positive_rate: fp, ..NoiseCfg::demo() };
    println!("command: {command}   seeds: {}   confusion {confusion}  false positives/scan {fp}", seeds.len());
    println!("{:<10} {:>6} {:>7} {:>8} {:>9} {:>10} {:>10} {:>10} {:>10}", "gates", "done", "wasted", "ticks", "energy J", "prec raw", "prec used", "recall raw", "recall used");
    for gates in [false, true] {
        let r = run(gates, &seeds, &noise, &command);
        println!(
            "{:<10} {:>3}/{:<2} {:>7} {:>8} {:>9.0} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            if gates { "on" } else { "off" },
            r.done,
            r.runs,
            r.wasted,
            r.ticks,
            r.energy,
            r.rp,
            r.gp,
            r.rr,
            r.gr
        );
    }
}
