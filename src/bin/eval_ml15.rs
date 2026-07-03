//! eval_ml15 — evaluate a trained (or randomly-initialized) 15-puzzle value net.
//!
//! Loads a value-network checkpoint (safetensors) and runs the evaluation
//! harness: a fixed random holdout and the 17 known depth-80 antipodes,
//! reporting mean solution length and excess over the optimal 80. With no
//! checkpoint it uses a random net (useful for validating the I/O path).
//!
//!   cargo run --release --features ml --bin eval_ml15 -- \
//!       [--checkpoint value_net.safetensors] [--hidden 512] \
//!       [--antipodes data/pdb15_antipodes.txt] \
//!       [--weight 2.0] [--batch 1000] [--budget 1000000] \
//!       [--holdout 100] [--holdout-k 60] [--seed 12345]

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle15::ml::bwas::BwasConfig;
use puzzle8::puzzle15::ml::device::{device_kind, pick_device};
use puzzle8::puzzle15::ml::eval::{run, EvalConfig};
use puzzle8::puzzle15::ml::profile;
use puzzle8::puzzle15::ml::value_net::{ValueNet, DEFAULT_BLOCKS, DEFAULT_HIDDEN};
use puzzle8::puzzle15::state::State;

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
    let antipodes: String = arg(&argv, "--antipodes", "data/pdb15_antipodes.txt".to_string());
    let weight: f32 = arg(&argv, "--weight", 2.0);
    let batch: usize = arg(&argv, "--batch", 1000);
    let budget: u64 = arg(&argv, "--budget", 1_000_000);
    let holdout_n: usize = arg(&argv, "--holdout", 100);
    let holdout_k_max: u32 = arg(&argv, "--holdout-k", 60);
    let holdout_seed: u64 = arg(&argv, "--seed", 0xE_15_5EED);
    if argv.iter().any(|a| a == "--profile") {
        profile::set_enabled(true);
    }

    // Default to the GPU (Metal-if-available, CPU fallback; see train_ml15).
    // `--cpu` forces the CPU backend.
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
    let net = match ValueNet::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), hidden, blocks) {
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
        antipodes_path: PathBuf::from(antipodes),
        holdout_n,
        holdout_k_max,
        holdout_seed,
    };
    let report = run(value_of, &cfg);
    report.print();
    if profile::is_enabled() {
        print!("\n{}", profile::report());
    }
    ExitCode::SUCCESS
}
