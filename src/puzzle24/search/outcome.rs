//! Result types shared by both search engines.
//!
//! These live apart from either engine on purpose. Both
//! [`engine`](super::engine) — the cWD lower-bound prover the `R` program runs —
//! and [`recursive`](super::recursive) — the generic heuristic-driven ladder the
//! ML and corridor tooling runs — return them, and neither owns the other. They
//! were originally defined inside the recursive module, which left the primary
//! engine importing its own return types from the module it replaced.

use crate::puzzle24::state::Move;

/// Search-effort statistics for a single IDA\* solve.
///
/// `nodes` counts every node visited (one search step, equivalently one
/// heuristic evaluation), summed across all threshold iterations.
/// `iterations` counts the IDA\* deepening passes. Both are the natural levers
/// for benchmarking: incremental-heuristic and codegen changes move wall-clock
/// at fixed `nodes`, while move-ordering and duplicate-pruning changes move
/// `nodes` directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Total nodes visited across all iterations.
    pub nodes: u64,
    /// Number of IDA\* threshold iterations performed.
    pub iterations: u32,
}

impl SearchStats {
    /// Component-wise add: merge per-solve stats into a running total.
    pub fn add(&mut self, other: &SearchStats) {
        self.nodes += other.nodes;
        self.iterations += other.iterations;
    }
}

/// Outcome of a bounded (lower-bound) search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundedOutcome {
    /// An optimal solution was found within the bound; its length is the true
    /// optimal distance.
    Solved(Vec<Move>),
    /// Every IDA\* iteration up to and including threshold `max_bound` was
    /// exhausted without finding the goal. Since the heuristics are consistent,
    /// this *proves* `dist(start) ≥ K` (the next threshold the search would have
    /// tried).
    ProvedAtLeast(u8),
    /// `start` is unreachable from `GOAL` (impossible for solvable states).
    Unsolvable,
    /// The search was cut short by an explicit node budget while inside
    /// threshold `.0`. **This is not a proof of anything** — the threshold was
    /// not exhausted, so no lower bound follows. It exists so that a truncated
    /// benchmarking run cannot be mistaken for, or silently folded into, a real
    /// result: every `match` on this enum has to name it.
    BudgetExhausted(u8),
}
