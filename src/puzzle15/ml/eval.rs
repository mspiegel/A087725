//! Evaluation harness for the trained solver.
//!
//! Two metrics, both self-certifying (a returned move sequence is checked by
//! construction inside BWAS's `g`, and can be replayed):
//!
//! - **Random holdout:** a fixed set of scrambled boards (fixed seed, so the set
//!   is identical across evals) — average solution length + failure rate. This
//!   is the "does the solver improve over training" trend signal.
//! - **Antipodes:** the 15-puzzle's 17 known depth-80 boards
//!   (`data/pdb15_antipodes.txt`). Reports mean excess over the optimal 80 —
//!   directly comparable to DeepCubeA's reported +2.8 on this exact experiment.

use std::path::PathBuf;

use super::bwas::{search, BwasConfig, BwasOutcome};
use super::scramble::{scramble, Rng};
use crate::puzzle15::enumerate::antipodes::load_ranks;
use crate::puzzle15::rank::unrank;
use crate::puzzle15::state::{State, DIAMETER};

pub struct EvalConfig {
    pub bwas: BwasConfig,
    /// Path to the pinned antipodes file (`data/pdb15_antipodes.txt`).
    pub antipodes_path: PathBuf,
    /// Number of random holdout boards.
    pub holdout_n: usize,
    /// Scramble depth range for the holdout set.
    pub holdout_k_max: u32,
    /// Fixed seed so the holdout set is identical across evaluations.
    pub holdout_seed: u64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            bwas: BwasConfig { weight: 2.0, batch_size: 1000, node_budget: 1_000_000 },
            antipodes_path: PathBuf::from("data/pdb15_antipodes.txt"),
            holdout_n: 100,
            holdout_k_max: 60,
            holdout_seed: 0xE_15_5EED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalReport {
    pub holdout_n: usize,
    pub holdout_solved: usize,
    pub holdout_mean_len: Option<f32>,
    pub holdout_fail_rate: f32,
    pub antipode_n: usize,
    pub antipode_solved: usize,
    pub antipode_mean_len: Option<f32>,
    /// mean(len) − 80 over solved antipodes (DeepCubeA benchmark: +2.8).
    pub antipode_mean_excess: Option<f32>,
}

impl EvalReport {
    pub fn print(&self) {
        println!("── evaluation ──");
        println!(
            "  holdout: solved {}/{} ({:.1}% fail), mean len {}",
            self.holdout_solved,
            self.holdout_n,
            self.holdout_fail_rate * 100.0,
            self.holdout_mean_len.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into()),
        );
        println!(
            "  antipodes(80): solved {}/{}, mean len {}, mean excess over 80 {} (DeepCubeA: +2.8)",
            self.antipode_solved,
            self.antipode_n,
            self.antipode_mean_len.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into()),
            self.antipode_mean_excess
                .map(|v| format!("{v:+.2}"))
                .unwrap_or_else(|| "-".into()),
        );
    }
}

/// Run both metrics using `value_of` (the solver's batched cost-to-go) as the
/// BWAS heuristic. Returns a report; never panics on a missing antipodes file
/// (reports `antipode_n = 0` and logs a warning instead).
pub fn run<F>(value_of: F, cfg: &EvalConfig) -> EvalReport
where
    F: Fn(&[State]) -> Vec<f32>,
{
    // Fixed holdout set.
    let mut rng = Rng::new(cfg.holdout_seed);
    let holdout: Vec<State> =
        (0..cfg.holdout_n).map(|_| scramble(&mut rng, cfg.holdout_k_max).0).collect();

    let mut holdout_solved = 0usize;
    let mut holdout_len_sum = 0u64;
    for s in &holdout {
        if let BwasOutcome::Solved { moves, .. } = search(s, &cfg.bwas, &value_of) {
            holdout_solved += 1;
            holdout_len_sum += moves.len() as u64;
        }
    }
    let holdout_mean_len =
        (holdout_solved > 0).then(|| holdout_len_sum as f32 / holdout_solved as f32);
    let holdout_fail_rate = if cfg.holdout_n > 0 {
        (cfg.holdout_n - holdout_solved) as f32 / cfg.holdout_n as f32
    } else {
        0.0
    };

    // Antipodes.
    let antipodes: Vec<State> = match load_ranks(&cfg.antipodes_path) {
        Ok(ranks) => ranks.into_iter().map(unrank).collect(),
        Err(e) => {
            eprintln!("warning: could not load antipodes {}: {}", cfg.antipodes_path.display(), e);
            Vec::new()
        }
    };
    let mut antipode_solved = 0usize;
    let mut antipode_len_sum = 0u64;
    for s in &antipodes {
        if let BwasOutcome::Solved { moves, .. } = search(s, &cfg.bwas, &value_of) {
            antipode_solved += 1;
            antipode_len_sum += moves.len() as u64;
        }
    }
    let antipode_mean_len =
        (antipode_solved > 0).then(|| antipode_len_sum as f32 / antipode_solved as f32);
    let antipode_mean_excess = antipode_mean_len.map(|m| m - DIAMETER as f32);

    EvalReport {
        holdout_n: cfg.holdout_n,
        holdout_solved,
        holdout_mean_len,
        holdout_fail_rate,
        antipode_n: antipodes.len(),
        antipode_solved,
        antipode_mean_len,
        antipode_mean_excess,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle15::search::{Heuristic, ManhattanHeuristic};

    fn manhattan_batch(states: &[State]) -> Vec<f32> {
        states.iter().map(|s| ManhattanHeuristic.h(s) as f32).collect()
    }

    #[test]
    fn harness_io_and_reporting_path() {
        // Shallow holdout solves under a modest budget; antipodes (depth 80) will
        // NOT solve under this small budget with Manhattan — that's fine, this
        // test validates the I/O + reporting path, not solver quality.
        let cfg = EvalConfig {
            bwas: BwasConfig { weight: 1.0, batch_size: 8, node_budget: 200_000 },
            antipodes_path: PathBuf::from("data/pdb15_antipodes.txt"),
            holdout_n: 12,
            holdout_k_max: 8,
            holdout_seed: 123,
        };
        let rep = run(manhattan_batch, &cfg);
        // Antipodes file must load and contain the 17 known boards.
        assert_eq!(rep.antipode_n, 17, "expected 17 antipodes loaded");
        // Shallow holdout should mostly solve.
        assert!(rep.holdout_solved >= 10, "holdout solved too few: {}", rep.holdout_solved);
        assert!(rep.holdout_mean_len.is_some());
    }
}
