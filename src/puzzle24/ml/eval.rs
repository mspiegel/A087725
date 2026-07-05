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

use super::beam::{beam_search, BeamConfig};
use super::bwas::{search, BwasConfig, BwasOutcome};
use super::generator::BaselineHeuristic;
use super::scramble::{scramble_exact, Rng};
use super::wdsearch::{construct_deep_boards, Diversity, WdSearchConfig};
use crate::puzzle24::search::{
    idastar_inc_mut_bounded_with_stats, BoundedOutcome, Heuristic, IncHeuristicMut,
    LinearConflictInc, ManhattanHeuristic, MaxInc, WalkingDistanceHeuristic, WalkingDistanceInc,
};
use crate::puzzle24::state::{State, DIAMETER_LOWER, N_CELLS};

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

// ─────────────────────────── deep mode ───────────────────────────
//
// Past ~depth 50 there is no feasible optimal solver, so the mid-depth
// excess-over-optimal metric is unavailable. Instead we grade the learned solver
// against the fast suboptimal **beam baseline** (the same reference the generator
// trains against), on deep random-walk boards and optionally the `R` board. The
// headline number is `mean_excess_over_beam` (learned − beam, over boards both
// solve) — **negative means the learned solver beats beam** — plus the WD lower
// bound and `DIAMETER_LOWER` for absolute context on `R`.

/// The `R` board — 180° rotation (blank fixed at cell 0, tile `25-i` at cell `i`);
/// the canonical hard 24-puzzle board (Rokicki LB 152, WD 140).
pub fn r_board() -> State {
    let mut c = [0u8; N_CELLS];
    for i in 1..N_CELLS {
        c[i] = (25 - i) as u8;
    }
    State(c)
}

/// Source of the deep holdout boards.
#[derive(Clone, Copy, Debug)]
pub enum DeepHoldout {
    /// Folding random walks over `[depth_min, depth_max]`. WARNING: a folding walk
    /// reaches optimal ≈ 0.5·length, so even a walk-150 holdout tops out around
    /// **optimal-60** — it does NOT test the genuinely-deep (WD-120) regime.
    Walk,
    /// WD-search constructed boards (`super::wdsearch`) — genuinely deep
    /// (WD ≈ `depth`, optimal ≥ WD). The holdout that actually probes the deep
    /// regime the deep-board training targets.
    WdSearch { width: usize, depth: usize },
}

pub struct DeepEvalConfig {
    /// BWAS config for the learned solver.
    pub bwas: BwasConfig,
    /// Beam config for the baseline reference solver.
    pub beam: BeamConfig,
    /// Admissible heuristic for the beam baseline (Wd in production).
    pub baseline: BaselineHeuristic,
    /// Number of holdout boards.
    pub holdout_n: usize,
    /// Inclusive walk-length range (used only by `DeepHoldout::Walk`).
    pub depth_min: u32,
    pub depth_max: u32,
    pub seed: u64,
    /// Also evaluate the canonical `R` board and report it explicitly.
    pub include_r: bool,
    /// Where the holdout boards come from (walk vs WD-search).
    pub holdout: DeepHoldout,
}

/// The `R`-board line of a deep eval (reported separately for context).
#[derive(Debug, Clone)]
pub struct RBoardResult {
    pub learned: Option<u32>,
    pub beam: Option<u32>,
    /// Walking-Distance lower bound on `R` (should be 140).
    pub wd_lb: u8,
}

#[derive(Debug, Clone)]
pub struct DeepEvalReport {
    pub n: usize,
    pub learned_solved: usize,
    pub beam_solved: usize,
    /// Boards solved by *both* (the fair-comparison set for excess/wins).
    pub both_solved: usize,
    pub mean_learned_len: Option<f32>,
    pub mean_beam_len: Option<f32>,
    /// Mean(learned − beam) over both-solved boards; **negative = learned better**.
    pub mean_excess_over_beam: Option<f32>,
    /// Count of both-solved boards where learned is strictly shorter than beam.
    pub learned_wins: usize,
    pub r_line: Option<RBoardResult>,
}

impl DeepEvalReport {
    pub fn print(&self) {
        eprintln!("── evaluation (deep, vs beam baseline) ──");
        eprintln!(
            "  solved: learned {}/{}, beam {}/{}, both {}/{}",
            self.learned_solved, self.n, self.beam_solved, self.n, self.both_solved, self.n,
        );
        eprintln!(
            "  mean len: learned {}, beam {}",
            self.mean_learned_len.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
            self.mean_beam_len.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
        );
        eprintln!(
            "  mean excess over beam (both-solved): {}  [negative = learned beats beam], learned wins {}/{}",
            self.mean_excess_over_beam
                .map(|v| format!("{:+.2}", v))
                .unwrap_or_else(|| "-".into()),
            self.learned_wins,
            self.both_solved,
        );
        if let Some(r) = &self.r_line {
            eprintln!(
                "  R board: learned {}, beam {}  (WD LB {}, DIAMETER_LOWER {})",
                r.learned.map(|v| v.to_string()).unwrap_or_else(|| "unsolved".into()),
                r.beam.map(|v| v.to_string()).unwrap_or_else(|| "unsolved".into()),
                r.wd_lb,
                DIAMETER_LOWER,
            );
        }
    }
}

/// Deep-mode eval: grade `value_of` (the learned solver) against the beam
/// baseline on deep boards. No optimal labels (infeasible past ~depth 50).
pub fn run_deep<F>(value_of: F, cfg: &DeepEvalConfig) -> DeepEvalReport
where
    F: Fn(&[State]) -> Vec<f32>,
{
    let beam_h: Box<dyn Heuristic + Send + Sync> = match cfg.baseline {
        BaselineHeuristic::Wd => {
            WalkingDistanceHeuristic::warm_up();
            Box::new(WalkingDistanceHeuristic)
        }
        BaselineHeuristic::Manhattan => Box::new(ManhattanHeuristic),
    };

    // Deep holdout (+ optional R board).
    let mut rng = Rng::new(cfg.seed);
    let mut boards: Vec<(bool, State)> = match cfg.holdout {
        DeepHoldout::Walk => {
            let dmax = cfg.depth_max.max(cfg.depth_min);
            (0..cfg.holdout_n)
                .map(|_| {
                    let k = rng.gen_range(cfg.depth_min.max(1), dmax);
                    (false, scramble_exact(&mut rng, k))
                })
                .collect()
        }
        DeepHoldout::WdSearch { width, depth } => {
            WalkingDistanceHeuristic::warm_up();
            let sc = WdSearchConfig {
                width,
                target_depth: depth,
                node_budget: 0,
                diversity: Diversity::Stochastic {
                    random_slots: (width / 4).max(1),
                    temperature: 8.0,
                },
            };
            construct_deep_boards(cfg.holdout_n, &sc, &mut rng)
                .into_iter()
                .map(|(s, _wd)| (false, s))
                .collect()
        }
    };
    if cfg.include_r {
        boards.push((true, r_board()));
    }

    let mut learned_solved = 0usize;
    let mut beam_solved = 0usize;
    let mut both_solved = 0usize;
    let mut learned_len_sum = 0u64;
    let mut beam_len_sum = 0u64;
    let mut excess_sum = 0i64;
    let mut learned_wins = 0usize;
    let mut r_line = None;

    for (is_r, board) in &boards {
        let learned = match search(board, &cfg.bwas, &value_of) {
            BwasOutcome::Solved { moves, .. } => Some(moves.len() as u32),
            BwasOutcome::BudgetExceeded { .. } => None,
        };
        let beam = beam_search(board, beam_h.as_ref(), &cfg.beam).map(|m| m.len() as u32);

        // The R board is reported separately and excluded from the aggregate
        // holdout statistics (it is not a random-walk sample).
        if *is_r {
            let wd_lb = {
                WalkingDistanceHeuristic::warm_up();
                WalkingDistanceHeuristic.h(board)
            };
            r_line = Some(RBoardResult { learned, beam, wd_lb });
            continue;
        }

        if let Some(l) = learned {
            learned_solved += 1;
            learned_len_sum += l as u64;
        }
        if let Some(b) = beam {
            beam_solved += 1;
            beam_len_sum += b as u64;
        }
        if let (Some(l), Some(b)) = (learned, beam) {
            both_solved += 1;
            excess_sum += l as i64 - b as i64;
            if l < b {
                learned_wins += 1;
            }
        }
    }

    DeepEvalReport {
        n: cfg.holdout_n,
        learned_solved,
        beam_solved,
        both_solved,
        mean_learned_len: (learned_solved > 0)
            .then(|| learned_len_sum as f32 / learned_solved as f32),
        mean_beam_len: (beam_solved > 0).then(|| beam_len_sum as f32 / beam_solved as f32),
        mean_excess_over_beam: (both_solved > 0)
            .then(|| excess_sum as f32 / both_solved as f32),
        learned_wins,
        r_line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn deep_harness_grades_vs_beam() {
        // Manhattan baseline (table-free) + moderate walks. The "learned solver"
        // here is Manhattan-BWAS, so it and the beam should both solve most
        // boards; this checks the deep grading path is populated and consistent.
        let cfg = DeepEvalConfig {
            bwas: BwasConfig { weight: 2.0, batch_size: 16, node_budget: 300_000 },
            beam: BeamConfig { width: 200, max_depth: 80, node_budget: 500_000 },
            baseline: BaselineHeuristic::Manhattan,
            holdout_n: 8,
            depth_min: 12,
            depth_max: 24,
            seed: 55,
            include_r: false,
            holdout: DeepHoldout::Walk,
        };
        let rep = run_deep(manhattan_batch, &cfg);
        assert_eq!(rep.n, 8);
        assert!(rep.learned_solved >= 6, "learned solved too few: {}", rep.learned_solved);
        assert!(rep.beam_solved >= 6, "beam solved too few: {}", rep.beam_solved);
        assert!(rep.both_solved >= 6);
        assert!(rep.mean_learned_len.is_some() && rep.mean_beam_len.is_some());
        assert!(rep.mean_excess_over_beam.is_some());
        assert!(rep.learned_wins <= rep.both_solved);
        assert!(rep.r_line.is_none());
    }

    #[test]
    fn r_board_is_valid() {
        let r = r_board();
        assert!(r.is_solvable(), "R board must be solvable");
        assert_eq!(r.0[0], 0, "R blank at cell 0");
        for i in 1..N_CELLS {
            assert_eq!(r.0[i], (25 - i) as u8, "R cell {i}");
        }
    }
}
