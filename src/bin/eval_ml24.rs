//! eval_ml24 — evaluate a trained (or random-init) 24-puzzle value net.
//!
//! Loads a value-network checkpoint (safetensors) and runs the mid-depth eval:
//! a fixed random-walk holdout labeled with true optimal by admissible `idastar`,
//! reporting solve rate and mean excess-over-optimal.
//!
//!   cargo run --release --features ml --bin eval_ml24 -- \
//!       [--checkpoint value_latest.safetensors] [--hidden 1024] [--blocks 6] \
//!       [--weight 2.0] [--batch 2000] [--budget 1000000] \
//!       [--holdout 100] [--depth-min 20] [--depth-max 50] \
//!       [--label lcwd|lc] [--seed ...] [--metal|--cpu]

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::bwas::BwasConfig;
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::{run, EvalConfig, LabelHeuristic};
use puzzle8::puzzle24::ml::profile;
use puzzle8::puzzle24::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle24::search::WalkingDistanceHeuristic;
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
    let weight: f32 = arg(&argv, "--weight", 2.0);
    let batch: usize = arg(&argv, "--batch", 2000);
    let budget: u64 = arg(&argv, "--budget", 1_000_000);
    let holdout_n: usize = arg(&argv, "--holdout", 100);
    let depth_min: u32 = arg(&argv, "--depth-min", 20);
    let depth_max: u32 = arg(&argv, "--depth-max", 50);
    let seed: u64 = arg(&argv, "--seed", 0x24_5EED);
    let label = match arg(&argv, "--label", "lcwd".to_string()).as_str() {
        "lc" => LabelHeuristic::Lc,
        _ => LabelHeuristic::LcWd,
    };
    if argv.iter().any(|a| a == "--profile") {
        profile::set_enabled(true);
    }

    if label == LabelHeuristic::LcWd {
        print!("warming up Walking Distance table... ");
        let t = std::time::Instant::now();
        WalkingDistanceHeuristic::warm_up();
        println!("ready in {:.1}s", t.elapsed().as_secs_f64());
    }

    // Default to the GPU (Metal-if-available, CPU fallback); `--cpu` forces CPU.
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
    println!("device: {}, hidden: {}, blocks: {}", device_kind(&device), hidden, blocks);

    let mut varmap = VarMap::new();
    let net =
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
        None => println!("no --checkpoint given: using random-init net (I/O path check only)"),
    }

    let value_of = |states: &[State]| net.values(states, &device).expect("value net forward");

    let cfg = EvalConfig {
        bwas: BwasConfig { weight, batch_size: batch, node_budget: budget },
        holdout_n,
        depth_min,
        depth_max,
        seed,
        optimal_max_bound: (depth_max as u8).saturating_add(5),
        label_heuristic: label,
    };
    let report = run(value_of, &cfg);
    report.print();
    if profile::is_enabled() {
        print!("\n{}", profile::report());
    }
    ExitCode::SUCCESS
}
