//! solve_r — solve the canonical hard `R` board with a trained value net via
//! **anytime weighted-A\*** (a descending weight ladder with incumbent pruning),
//! to get the shortest solution the current solver can find without retraining.
//!
//!   cargo run --release --features ml --bin solve_r -- \
//!       --checkpoint data/ml24_wds3/value_latest.safetensors [--hidden 1024] \
//!       [--blocks 6] [--weights 2.5,2.0,1.5,1.2] [--budget 5000000] [--batch 2000] [--cpu]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::bwas::{anytime_search, BwasOutcome};
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::r_board;
use puzzle8::puzzle24::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::search::{Heuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle24::state::State;

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let checkpoint: Option<PathBuf> = argv
        .iter()
        .position(|a| a == "--checkpoint")
        .and_then(|i| argv.get(i + 1))
        .map(PathBuf::from);
    let hidden: usize = arg(&argv, "--hidden", DEFAULT_HIDDEN);
    let blocks: usize = arg(&argv, "--blocks", DEFAULT_BLOCKS);
    let budget: u64 = arg(&argv, "--budget", 5_000_000);
    let batch: usize = arg(&argv, "--batch", 2000);
    let weights: Vec<f32> = arg(&argv, "--weights", "2.5,2.0,1.5,1.2".to_string())
        .split(',')
        .filter_map(|w| w.trim().parse().ok())
        .collect();

    print!("warming up Walking Distance table... ");
    let t = Instant::now();
    WalkingDistanceHeuristic::warm_up();
    println!("ready in {:.1}s", t.elapsed().as_secs_f64());

    let device = if argv.iter().any(|a| a == "--cpu") {
        candle_core::Device::Cpu
    } else {
        match pick_device() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not init device: {}", e);
                return ExitCode::FAILURE;
            }
        }
    };

    let mut varmap = VarMap::new();
    let mut net =
        match ValueNet::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), hidden, blocks) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("error building net: {}", e);
                return ExitCode::FAILURE;
            }
        };
    match &checkpoint {
        Some(path) => match varmap.load(path) {
            Ok(()) => println!("loaded checkpoint: {}", path.display()),
            Err(e) => {
                eprintln!("error loading checkpoint {}: {}", path.display(), e);
                return ExitCode::FAILURE;
            }
        },
        None => {
            eprintln!("--checkpoint required");
            return ExitCode::FAILURE;
        }
    }
    if argv.iter().any(|a| a == "--residual") {
        net.set_residual(true); // V = WD + raw (must match how the net was trained)
        println!("residual mode: V = WD + raw");
    }
    println!(
        "device: {}, weights: {:?}, budget/weight: {}",
        device_kind(&device),
        weights,
        budget
    );

    let r = r_board();
    let value_of = |states: &[State]| net.values(states, &device).expect("value net forward");
    let t = Instant::now();
    let outcome = anytime_search(&r, &weights, batch, budget, value_of, |s: &State| {
        WalkingDistanceHeuristic.h(s)
    });
    let secs = t.elapsed().as_secs_f64();

    match outcome {
        BwasOutcome::Solved { moves, nodes_expanded } => {
            // Self-certify: replay reaches GOAL.
            let mut s = r;
            for &m in &moves {
                s = s.apply(m);
            }
            let ok = s == puzzle8::puzzle24::state::GOAL;
            println!(
                "R solved: {} moves (WD LB 140, known-best 156, LB 152), {} nodes, {:.0}s, replay_ok={}",
                moves.len(),
                nodes_expanded,
                secs,
                ok
            );
        }
        BwasOutcome::BudgetExceeded { nodes_expanded } => {
            println!("R unsolved ({} nodes, {:.0}s)", nodes_expanded, secs);
        }
    }
    ExitCode::SUCCESS
}
