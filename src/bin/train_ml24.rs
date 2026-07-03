//! train_ml24 — adversarial generator/solver co-training for the 24-puzzle.
//!
//! Trains a DAVI value-net solver and a REINFORCE board generator against each
//! other, with the generator rewarded by regret vs a fast beam-search suboptimal
//! baseline. Validated on mid-depth boards (excess-over-optimal via idastar).
//!
//!   cargo run --release --features ml --bin train_ml24 -- \
//!       [--rounds 40] [--solver-steps 3000] [--gen-steps 400] \
//!       [--batch 4096] [--gen-frac 0.5] [--hidden 1024] [--blocks 6] \
//!       [--solver-k 50] [--gen-k 40] \
//!       [--solver-weight 2.5] [--solver-sbatch 2000] \
//!       [--solver-budget 30000 (generator reward)] [--eval-budget 300000] \
//!       [--baseline wd|manhattan] [--beam-width 1000] [--beam-budget 2000000] \
//!       [--eval-every 2] [--holdout 100] [--depth-min 20] [--depth-max 50] \
//!       [--eval-mode middepth|deep] [--eval-depth-min 60] [--eval-depth-max 120] [--eval-with-r] \
//!       [--out data/ml24] [--seed 1] [--metal|--cpu] [--resume] [--quiet]

use std::path::PathBuf;
use std::process::ExitCode;

use puzzle8::puzzle24::ml::alternate::{run, AlternationConfig, EvalSpec};
use puzzle8::puzzle24::ml::beam::BeamConfig;
use puzzle8::puzzle24::ml::bwas::BwasConfig;
use puzzle8::puzzle24::ml::davi::DaviConfig;
use puzzle8::puzzle24::ml::device::{device_kind, pick_device};
use puzzle8::puzzle24::ml::eval::{DeepEvalConfig, EvalConfig, LabelHeuristic};
use puzzle8::puzzle24::ml::generator::{BaselineHeuristic, GeneratorConfig};
use puzzle8::puzzle24::ml::profile;
use puzzle8::puzzle24::search::WalkingDistanceHeuristic;

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();

    let rounds: u32 = arg(&argv, "--rounds", 40);
    let solver_steps: u32 = arg(&argv, "--solver-steps", 3000);
    // Defaults reflect the 15-puzzle lessons: batch 1024 (the GPU saturates
    // there — a bigger batch is more compute for the same throughput and fewer
    // updates); gen-steps kept modest (each runs the learned BWAS solve, which is
    // the gen-phase cost).
    let gen_steps: u32 = arg(&argv, "--gen-steps", 100);
    let batch: usize = arg(&argv, "--batch", 1024);
    let gen_frac: f32 = arg(&argv, "--gen-frac", 0.5);
    let hidden: usize = arg(&argv, "--hidden", 1024);
    let blocks: usize = arg(&argv, "--blocks", 6);
    let solver_k: u32 = arg(&argv, "--solver-k", 50);
    let gen_k: u32 = arg(&argv, "--gen-k", 40);
    let solver_weight: f32 = arg(&argv, "--solver-weight", 2.5);
    let solver_sbatch: usize = arg(&argv, "--solver-sbatch", 2000);
    // The generator-reward BWAS wants a LOW budget (cheap, fails fast); the eval
    // BWAS wants a HIGH budget (else the metric is search-limited and reads ~0
    // even when the value function is good — the 15-puzzle lesson). Keep them
    // separate.
    let solver_budget: u64 = arg(&argv, "--solver-budget", 30_000);
    let eval_budget: u64 = arg(&argv, "--eval-budget", 300_000);
    let beam_width: usize = arg(&argv, "--beam-width", 1000);
    let beam_budget: u64 = arg(&argv, "--beam-budget", 2_000_000);
    let eval_every: u32 = arg(&argv, "--eval-every", 2);
    let holdout: usize = arg(&argv, "--holdout", 100);
    let depth_min: u32 = arg(&argv, "--depth-min", 20);
    let depth_max: u32 = arg(&argv, "--depth-max", 50);
    // In-loop eval mode: mid-depth (excess-over-optimal via IDA*, ≤~d50) or deep
    // (excess-over-beam) for the scale-to-R regime, where the mid-depth holdout
    // saturates and stops being informative. Deep reuses the generator beam +
    // baseline; --eval-depth-* set the deep holdout walk-length range.
    let eval_mode = arg(&argv, "--eval-mode", "middepth".to_string());
    let eval_depth_min: u32 = arg(&argv, "--eval-depth-min", 60);
    let eval_depth_max: u32 = arg(&argv, "--eval-depth-max", 120);
    let eval_with_r = argv.iter().any(|a| a == "--eval-with-r");
    let out: String = arg(&argv, "--out", "data/ml24".to_string());
    let seed: u64 = arg(&argv, "--seed", 1);
    let baseline = match arg(&argv, "--baseline", "wd".to_string()).as_str() {
        "manhattan" => BaselineHeuristic::Manhattan,
        _ => BaselineHeuristic::Wd,
    };
    let verbose = !argv.iter().any(|a| a == "--quiet");
    let resume = argv.iter().any(|a| a == "--resume");
    if argv.iter().any(|a| a == "--profile") {
        profile::set_enabled(true);
    }

    // Warm up Walking Distance up front (the eval labels and the WD beam baseline
    // both use it) so the ~590 MB load / BFS is honest, not hidden mid-training.
    print!("warming up Walking Distance table... ");
    let t = std::time::Instant::now();
    WalkingDistanceHeuristic::warm_up();
    println!("ready in {:.1}s", t.elapsed().as_secs_f64());

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
    println!(
        "device: {}, rounds: {}, solver-steps/round: {}, gen-steps/round: {}, batch: {}, hidden: {}, blocks: {}",
        device_kind(&device),
        rounds,
        solver_steps,
        gen_steps,
        batch,
        hidden,
        blocks,
    );

    let solver_bwas =
        BwasConfig { weight: solver_weight, batch_size: solver_sbatch, node_budget: solver_budget };
    let beam = BeamConfig { width: beam_width, max_depth: 220, node_budget: beam_budget };
    // High-budget eval, decoupled from the cheap generator-reward budget so the
    // metric reflects the value function, not the search budget.
    let eval_bwas =
        BwasConfig { weight: solver_weight, batch_size: solver_sbatch, node_budget: eval_budget };
    let eval_spec = if eval_mode == "deep" {
        EvalSpec::Deep(DeepEvalConfig {
            bwas: eval_bwas,
            beam,
            baseline,
            holdout_n: holdout,
            depth_min: eval_depth_min,
            depth_max: eval_depth_max,
            seed: 0x24_5EED,
            include_r: eval_with_r,
        })
    } else {
        EvalSpec::MidDepth(EvalConfig {
            bwas: eval_bwas,
            holdout_n: holdout,
            depth_min,
            depth_max,
            seed: 0x24_5EED,
            optimal_max_bound: (depth_max as u8).saturating_add(5),
            label_heuristic: LabelHeuristic::LcWd,
        })
    };
    let cfg = AlternationConfig {
        rounds,
        solver_steps_per_round: solver_steps,
        generator_steps_per_round: gen_steps,
        solver_batch: batch,
        generator_frac: gen_frac,
        davi: DaviConfig { k_max: solver_k, hidden, blocks, lr: 1e-3, target_sync_every: 1000 },
        generator: GeneratorConfig {
            k_max: gen_k,
            hidden,
            lr: 1e-3,
            baseline_decay: 0.95,
            solver_bwas,
            beam,
            baseline,
            fail_penalty: 400.0,
        },
        eval_every,
        eval: eval_spec,
        checkpoint_dir: PathBuf::from(out),
        seed,
        verbose,
        resume,
    };

    match run(&cfg, device) {
        Ok(()) => {
            println!("training complete; checkpoints + metrics in {}", cfg.checkpoint_dir.display());
            if profile::is_enabled() {
                print!("\n{}", profile::report());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("training error: {}", e);
            ExitCode::FAILURE
        }
    }
}
