//! eval_ml24 — evaluate a trained (or random-init) 24-puzzle value net.
//!
//! Loads a value-network checkpoint (safetensors) and runs one of two evals:
//!   --mode middepth (default): random-walk holdout labeled with true optimal by
//!     admissible `idastar`; reports solve rate + mean excess-over-optimal.
//!   --mode deep: deep boards (past ~depth 50, no optimal) graded vs the beam
//!     baseline; reports solve rates, mean excess-over-beam, and the `R` board.
//!
//!   cargo run --release --features ml --bin eval_ml24 -- \
//!       [--checkpoint value_latest.safetensors] [--hidden 1024] [--blocks 6] \
//!       [--weight 2.0] [--batch 2000] [--budget 1000000] \
//!       [--holdout 100] [--depth-min 20] [--depth-max 50] \
//!       [--label lcwd|lc] [--seed ...] [--metal|--cpu] \
//!       [--mode middepth|deep] [--baseline wd|manhattan] [--with-r] \
//!       [--beam-width 1000] [--beam-budget 2000000]

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

use puzzle8::puzzle24::ml::beam::BeamConfig;
use puzzle8::puzzle24::ml::bwas::BwasConfig;
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::{
    run, run_deep, DeepEvalConfig, DeepHoldout, EvalConfig, LabelHeuristic,
};
use puzzle8::puzzle24::ml::generator::BaselineHeuristic;
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
    // Deep-mode holdout source: `walk` (folding walks, blind past ~optimal-60) or
    // `wdsearch` (genuinely-deep WD-boards — the real deep-regime test).
    let holdout = match arg(&argv, "--holdout-source", "walk".to_string()).as_str() {
        "wdsearch" => DeepHoldout::WdSearch {
            width: arg(&argv, "--search-width", 4000),
            depth: arg(&argv, "--search-depth", 120),
        },
        _ => DeepHoldout::Walk,
    };
    let seed: u64 = arg(&argv, "--seed", 0x24_5EED);
    let label = match arg(&argv, "--label", "lcwd".to_string()).as_str() {
        "lc" => LabelHeuristic::Lc,
        _ => LabelHeuristic::LcWd,
    };
    // Deep mode: grade the learned solver vs the beam baseline (no optimal past
    // ~depth 50). Reuses --depth-min/--depth-max as the deep walk-length range.
    let mode = arg(&argv, "--mode", "middepth".to_string());
    let with_r = argv.iter().any(|a| a == "--with-r");
    let beam_width: usize = arg(&argv, "--beam-width", 1000);
    let beam_budget: u64 = arg(&argv, "--beam-budget", 2_000_000);
    let baseline = match arg(&argv, "--baseline", "wd".to_string()).as_str() {
        "manhattan" => BaselineHeuristic::Manhattan,
        _ => BaselineHeuristic::Wd,
    };
    if argv.iter().any(|a| a == "--profile") {
        profile::set_enabled(true);
    }

    let need_wd = (mode != "deep" && label == LabelHeuristic::LcWd)
        || (mode == "deep" && (baseline == BaselineHeuristic::Wd || with_r))
        || argv.iter().any(|a| a == "--residual"); // residual V = WD + raw
    if need_wd {
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
                eprintln!("error: could not init device: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    println!(
        "device: {}, hidden: {}, blocks: {}",
        device_kind(&device),
        hidden,
        blocks
    );

    let mut varmap = VarMap::new();
    let mut net = match ValueNet::new(
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
        hidden,
        blocks,
    ) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("error building net: {e}");
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
    if argv.iter().any(|a| a == "--residual") {
        net.set_residual(true); // V = WD + raw (must match how the net was trained)
        println!("residual mode: V = WD + raw");
    }

    let value_of = |states: &[State]| net.values(states, &device).expect("value net forward");

    if mode == "deep" {
        let cfg = DeepEvalConfig {
            bwas: BwasConfig {
                weight,
                batch_size: batch,
                node_budget: budget,
            },
            beam: BeamConfig {
                width: beam_width,
                max_depth: 220,
                node_budget: beam_budget,
            },
            baseline,
            holdout_n,
            depth_min,
            depth_max,
            seed,
            include_r: with_r,
            holdout,
        };
        run_deep(value_of, &cfg).print();
    } else {
        let cfg = EvalConfig {
            bwas: BwasConfig {
                weight,
                batch_size: batch,
                node_budget: budget,
            },
            holdout_n,
            depth_min,
            depth_max,
            seed,
            optimal_max_bound: (depth_max as u8).saturating_add(5),
            label_heuristic: label,
        };
        run(value_of, &cfg).print();
    }
    if profile::is_enabled() {
        print!("\n{}", profile::report());
    }
    ExitCode::SUCCESS
}
