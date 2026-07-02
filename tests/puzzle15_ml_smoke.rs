//! Integration smoke test for the 15-puzzle adversarial ML PoC.
//!
//! Runs a tiny end-to-end co-training loop through the crate's public API and
//! asserts it completes without panics/NaNs and writes a loadable checkpoint.
//! The whole file is empty unless the `ml` feature is on (the `ml` module does
//! not exist otherwise).
#![cfg(feature = "ml")]

use std::path::PathBuf;

use candle_core::Device;
use puzzle8::puzzle15::ml::alternate::{run, AlternationConfig};
use puzzle8::puzzle15::ml::bwas::BwasConfig;
use puzzle8::puzzle15::ml::checkpoint;
use puzzle8::puzzle15::ml::davi::DaviConfig;
use puzzle8::puzzle15::ml::eval::EvalConfig;
use puzzle8::puzzle15::ml::generator::GeneratorConfig;

#[test]
fn end_to_end_tiny_cotraining_run() {
    let dir = std::env::temp_dir().join(format!("ml15_smoke_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let bwas = BwasConfig { weight: 2.0, batch_size: 16, node_budget: 15_000 };
    let cfg = AlternationConfig {
        rounds: 2,
        solver_steps_per_round: 4,
        generator_steps_per_round: 4,
        solver_batch: 32,
        generator_frac: 0.5,
        davi: DaviConfig { k_max: 8, hidden: 32, lr: 1e-3, target_sync_every: 8 },
        generator: GeneratorConfig {
            k_max: 8,
            hidden: 32,
            lr: 1e-3,
            baseline_decay: 0.9,
            solver_bwas: bwas,
            fail_penalty: 200.0,
        },
        eval_every: 2,
        eval: EvalConfig {
            bwas,
            antipodes_path: PathBuf::from("data/pdb15_antipodes.txt"),
            holdout_n: 5,
            holdout_k_max: 8,
            holdout_seed: 1,
        },
        checkpoint_dir: dir.clone(),
        seed: 1,
        verbose: false,
        resume: false,
    };

    // Runs on CPU for determinism/portability in CI.
    run(&cfg, Device::Cpu).expect("co-training loop failed");
    assert!(checkpoint::value_latest_path(&dir).exists(), "no checkpoint written");
    assert!(dir.join("metrics.tsv").exists(), "no metrics written");

    let _ = std::fs::remove_dir_all(&dir);
}
