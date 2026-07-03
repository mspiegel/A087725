//! Evaluation harness for the trained 24-puzzle solver — **mid-depth mode**.
//!
//! Unlike the 15-puzzle (17 known depth-80 antipodes), the 24-puzzle has no
//! enumerated deep-board ground truth and no feasible optimal solver past
//! ~depth 50. So we evaluate on a fixed holdout of mid-depth random walks whose
//! **true optimal length** is obtained by the admissible `idastar` (feasible in
//! this range), and report the learned solver's mean **excess over optimal** —
//! the same quantity that measured the 15-puzzle solver's quality.
//!
//! Boards whose optimal exceeds the `idastar` bound are counted as "unlabeled"
//! and excluded from the excess statistic (but still counted for solve rate).

use super::bwas::{search, BwasConfig, BwasOutcome};
use super::scramble::{scramble_exact, Rng};
use crate::puzzle24::search::{
    idastar_inc_mut_bounded_with_stats, BoundedOutcome, IncHeuristicMut, LinearConflictInc, MaxInc,
    WalkingDistanceHeuristic, WalkingDistanceInc,
};
use crate::puzzle24::state::State;

/// Admissible heuristic used to compute the true-optimal labels via `idastar`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelHeuristic {
    /// Linear Conflict only — table-free (used by tests; fine for shallow boards).
    Lc,
    /// `max(Linear Conflict, Walking Distance)` — the strong cheap combo used in
    /// production (loads `data/wd24.bin`; much faster on mid-depth boards).
    LcWd,
}

pub struct EvalConfig {
    /// BWAS config for the learned solver.
    pub bwas: BwasConfig,
    /// Number of holdout boards.
    pub holdout_n: usize,
    /// Inclusive walk-length range the holdout is drawn from.
    pub depth_min: u32,
    pub depth_max: u32,
    /// Fixed seed so the holdout set is identical across evaluations.
    pub seed: u64,
    /// `idastar` threshold cap for the true-optimal label (guards runtime; a
    /// board whose optimal exceeds this is left "unlabeled").
    pub optimal_max_bound: u8,
    /// Which admissible heuristic labels the optimal.
    pub label_heuristic: LabelHeuristic,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            bwas: BwasConfig { weight: 2.0, batch_size: 1000, node_budget: 1_000_000 },
            holdout_n: 100,
            depth_min: 20,
            depth_max: 50,
            seed: 0x24_5EED,
            optimal_max_bound: 55,
            label_heuristic: LabelHeuristic::LcWd,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalReport {
    pub holdout_n: usize,
    pub holdout_solved: usize,
    pub holdout_mean_len: Option<f32>,
    pub holdout_fail_rate: f32,
    /// Boards for which `idastar` returned the true optimal within the bound.
    pub optimal_labeled_n: usize,
    /// Mean(learned_len − optimal) over boards that are both labeled and solved.
    pub mean_excess_over_optimal: Option<f32>,
}

impl EvalReport {
    pub fn print(&self) {
        // eprintln (stderr, unbuffered) so eval stats stream live in a captured
        // log, matching the round/solver lines — not block-buffered like stdout.
        eprintln!("── evaluation (mid-depth) ──");
        eprintln!(
            "  holdout: solved {}/{} ({:.1}% fail), mean len {}",
            self.holdout_solved,
            self.holdout_n,
            self.holdout_fail_rate * 100.0,
            self.holdout_mean_len.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
        );
        eprintln!(
            "  optimal-labeled: {}/{}, mean excess over optimal: {}",
            self.optimal_labeled_n,
            self.holdout_n,
            self.mean_excess_over_optimal
                .map(|v| format!("{:+.2}", v))
                .unwrap_or_else(|| "-".into()),
        );
    }
}

/// Run the mid-depth eval using `value_of` (the solver's batched cost-to-go) as
/// the BWAS heuristic. Dispatches on `cfg.label_heuristic` for the optimal labels.
pub fn run<F>(value_of: F, cfg: &EvalConfig) -> EvalReport
where
    F: Fn(&[State]) -> Vec<f32>,
{
    match cfg.label_heuristic {
        LabelHeuristic::Lc => run_with(value_of, cfg, &LinearConflictInc),
        LabelHeuristic::LcWd => {
            WalkingDistanceHeuristic::warm_up();
            run_with(value_of, cfg, &MaxInc::new(LinearConflictInc, WalkingDistanceInc))
        }
    }
}

fn run_with<F, E>(value_of: F, cfg: &EvalConfig, opt_h: &E) -> EvalReport
where
    F: Fn(&[State]) -> Vec<f32>,
    E: IncHeuristicMut,
{
    // Fixed holdout: walk lengths spread over [depth_min, depth_max].
    let mut rng = Rng::new(cfg.seed);
    let dmax = cfg.depth_max.max(cfg.depth_min);
    let holdout: Vec<State> = (0..cfg.holdout_n)
        .map(|_| {
            let k = rng.gen_range(cfg.depth_min.max(1), dmax);
            scramble_exact(&mut rng, k)
        })
        .collect();

    let mut solved = 0usize;
    let mut len_sum = 0u64;
    let mut labeled = 0usize;
    let mut excess_sum = 0i64;
    let mut excess_n = 0usize;

    for board in &holdout {
        // True optimal (admissible idastar, bounded).
        let optimal: Option<u32> =
            match idastar_inc_mut_bounded_with_stats(board, opt_h, cfg.optimal_max_bound).0 {
                BoundedOutcome::Solved(moves) => Some(moves.len() as u32),
                _ => None, // ProvedAtLeast / Unsolvable → unlabeled
            };
        if optimal.is_some() {
            labeled += 1;
        }

        // Learned solver.
        let learned: Option<u32> = match search(board, &cfg.bwas, &value_of) {
            BwasOutcome::Solved { moves, .. } => Some(moves.len() as u32),
            BwasOutcome::BudgetExceeded { .. } => None,
        };
        if let Some(l) = learned {
            solved += 1;
            len_sum += l as u64;
            if let Some(o) = optimal {
                excess_sum += l as i64 - o as i64;
                excess_n += 1;
            }
        }
    }

    EvalReport {
        holdout_n: cfg.holdout_n,
        holdout_solved: solved,
        holdout_mean_len: (solved > 0).then(|| len_sum as f32 / solved as f32),
        holdout_fail_rate: if cfg.holdout_n > 0 {
            (cfg.holdout_n - solved) as f32 / cfg.holdout_n as f32
        } else {
            0.0
        },
        optimal_labeled_n: labeled,
        mean_excess_over_optimal: (excess_n > 0).then(|| excess_sum as f32 / excess_n as f32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::{Heuristic, ManhattanHeuristic};

    fn manhattan_batch(states: &[State]) -> Vec<f32> {
        states.iter().map(|s| ManhattanHeuristic.h(s) as f32).collect()
    }

    #[test]
    fn harness_io_and_reporting_path() {
        // Shallow holdout + LC-only labeling => fast and table-free (no WD load).
        let cfg = EvalConfig {
            bwas: BwasConfig { weight: 1.0, batch_size: 8, node_budget: 500_000 },
            holdout_n: 12,
            depth_min: 4,
            depth_max: 10,
            seed: 123,
            optimal_max_bound: 20,
            label_heuristic: LabelHeuristic::Lc,
        };
        let rep = run(manhattan_batch, &cfg);
        assert_eq!(rep.holdout_n, 12);
        assert!(rep.holdout_solved >= 8, "shallow holdout solved too few: {}", rep.holdout_solved);
        assert!(rep.holdout_mean_len.is_some());
        // Shallow boards must all get a true-optimal label within bound 20.
        assert_eq!(rep.optimal_labeled_n, 12, "all shallow boards should be labeled");
        assert!(rep.mean_excess_over_optimal.is_some());
    }
}
