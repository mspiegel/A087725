//! train_ml15 — run the adversarial generator/solver co-training loop.
//!
//! Trains a DAVI value-net solver and a REINFORCE board generator against each
//! other on the 15-puzzle, checkpointing to `--out` and logging metrics.
//! See TRAINING.md.
//!
//!   cargo run --release --features ml --bin train_ml15 -- \
//!       [--rounds 15] [--solver-steps 2000] [--gen-steps 400] \
//!       [--batch 4096] [--gen-frac 0.5] [--hidden 512] \
//!       [--solver-k 40] [--gen-k 25] \
//!       [--solver-weight 2.0] [--solver-sbatch 1000] [--solver-budget 100000] \
//!       [--eval-every 1] [--holdout 100] [--holdout-k 60] \
//!       [--out data/ml15] [--seed 1] [--quiet]

use std::path::PathBuf;
use std::process::ExitCode;

use puzzle8::puzzle15::ml::alternate::{run, AlternationConfig};
use puzzle8::puzzle15::ml::bwas::BwasConfig;
use puzzle8::puzzle15::ml::davi::DaviConfig;
use puzzle8::puzzle15::ml::device::{device_kind, pick_device};
use puzzle8::puzzle15::ml::eval::EvalConfig;
use puzzle8::puzzle15::ml::generator::GeneratorConfig;
use puzzle8::puzzle15::ml::profile;

fn arg<T: std::str::FromStr>(argv: &[String], flag: &str, default: T) -> T {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();

    let rounds: u32 = arg(&argv, "--rounds", 15);
    let solver_steps: u32 = arg(&argv, "--solver-steps", 2000);
    let gen_steps: u32 = arg(&argv, "--gen-steps", 400);
    let batch: usize = arg(&argv, "--batch", 4096);
    let gen_frac: f32 = arg(&argv, "--gen-frac", 0.5);
    let hidden: usize = arg(&argv, "--hidden", 512);
    let blocks: usize = arg(&argv, "--blocks", 4);
    let solver_k: u32 = arg(&argv, "--solver-k", 40);
    let gen_k: u32 = arg(&argv, "--gen-k", 25);
    let solver_weight: f32 = arg(&argv, "--solver-weight", 2.0);
    let solver_sbatch: usize = arg(&argv, "--solver-sbatch", 1000);
    let solver_budget: u64 = arg(&argv, "--solver-budget", 100_000);
    let eval_every: u32 = arg(&argv, "--eval-every", 1);
    let holdout: usize = arg(&argv, "--holdout", 100);
    let holdout_k: u32 = arg(&argv, "--holdout-k", 60);
    let out: String = arg(&argv, "--out", "data/ml15".to_string());
    let seed: u64 = arg(&argv, "--seed", 1);
    let verbose = !argv.iter().any(|a| a == "--quiet");
    let resume = argv.iter().any(|a| a == "--resume");
    if argv.iter().any(|a| a == "--profile") {
        profile::set_enabled(true);
    }

    // Default to CPU: measured ~7x faster than Metal at this PoC net size
    // (hidden 256, small batches) — Metal's per-dispatch latency + one-time
    // shader compilation dominate, and candle's CPU gemm is multi-threaded.
    // `--metal` opts into the GPU (worthwhile only for a much larger net/batch).
    let device = if argv.iter().any(|a| a == "--metal") {
        match pick_device() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: could not init Metal device: {}", e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        candle_core::Device::Cpu
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

    let solver_bwas = BwasConfig { weight: solver_weight, batch_size: solver_sbatch, node_budget: solver_budget };
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
            fail_penalty: 200.0,
        },
        eval_every,
        eval: EvalConfig {
            bwas: solver_bwas,
            antipodes_path: PathBuf::from("data/pdb15_antipodes.txt"),
            holdout_n: holdout,
            holdout_k_max: holdout_k,
            holdout_seed: 0xE_15_5EED,
        },
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
