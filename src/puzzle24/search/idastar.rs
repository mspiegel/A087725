//! Iterative-deepening A* over the 24-puzzle [`State`].
//!
//! Standard IDA*: at iteration `k`, depth-first search with `f = g + h` cut off
//! at threshold `bound_k`. With an admissible heuristic, the first solution
//! found is optimal. We prune immediate-undo moves (never apply `m.inverse()`
//! right after `m`), halving the branching factor without losing optimality.
//!
//! Costs stay `u8`: the 24-puzzle diameter is ≤ 205, below 255. The
//! [`IncHeuristic`] path threads a small per-node context through the recursion
//! so PDB heuristics advance in O(k) per move instead of re-projecting — and,
//! for the zero-aware 1-bit PDBs, it is *required* (their entries are deltas, so
//! the running `h` must be maintained incrementally; see `docs/zpdb-codec-spec.md`).
//!
//! Two driver families share this module:
//! - the *optimal-solve* drivers ([`idastar_inc_with_stats`] and friends) return
//!   the first (hence optimal) solution found;
//! - the *bounded* / lower-bound driver ([`idastar_inc_bounded_with_stats`])
//!   caps the threshold at `max_bound`. Because every heuristic here is
//!   *consistent*, an IDA\* iteration that exhausts threshold `b` is a proof
//!   that `dist(start) > b`, i.e. `dist(start) ≥ b+1`. Capping the iteration
//!   turns that into a usable lower bound — exactly how Rokicki proved the
//!   24-puzzle diameter ≥ 152.

use crate::puzzle24::state::{Move, State, GOAL};
#[cfg(feature = "parallel")]
use crate::puzzle24::symmetry;

use super::heuristic::Heuristic;
use super::move_dfa::{MovePruner, NullPruner};

/// Search-effort statistics for a single IDA\* solve.
///
/// `nodes` counts every node visited (one [`search`] invocation, equivalently
/// one heuristic evaluation), summed across all threshold iterations.
/// `iterations` counts the IDA\* deepening passes. Both are the natural levers
/// for benchmarking: incremental-heuristic and codegen changes move wall-clock
/// at fixed `nodes`, while move-ordering and duplicate-pruning changes move
/// `nodes` directly.
///
/// Under the `verifier-stats` feature, additional per-component counters
/// (`*_advances`, `proj_applies`, `zpdb_rank_calls`) are populated from inside
/// each [`IncHeuristic`] impl. Combined with samply self-time percentages this
/// gives true ns/call per component, immune to inlining attribution. The
/// feature is off by default — the per-node bumps cost ~5–10% wall time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Total nodes visited across all iterations.
    pub nodes: u64,
    /// Number of IDA\* threshold iterations performed.
    pub iterations: u32,
    /// Calls to `LinearConflictInc::advance`.
    #[cfg(feature = "verifier-stats")]
    pub lc_advances: u64,
    /// Calls to `WalkingDistanceInc::advance`.
    #[cfg(feature = "verifier-stats")]
    pub wd_advances: u64,
    /// Calls to `ZpdbInc::advance` (one per node).
    #[cfg(feature = "verifier-stats")]
    pub zpdb_advances: u64,
    /// Calls to `ZpdbLayout::rank` (only on cost-1 projected edges).
    #[cfg(feature = "verifier-stats")]
    pub zpdb_rank_calls: u64,
    /// Calls to `ProjectedState::apply` inside `ZpdbInc::advance` (= `2*N`
    /// per `zpdb_advances` — normal + reflected, per pattern).
    #[cfg(feature = "verifier-stats")]
    pub proj_applies: u64,
}

impl SearchStats {
    /// Component-wise add: merge per-solve stats into a running total.
    pub fn add(&mut self, other: &SearchStats) {
        self.nodes += other.nodes;
        self.iterations += other.iterations;
        #[cfg(feature = "verifier-stats")]
        {
            self.lc_advances += other.lc_advances;
            self.wd_advances += other.wd_advances;
            self.zpdb_advances += other.zpdb_advances;
            self.zpdb_rank_calls += other.zpdb_rank_calls;
            self.proj_applies += other.proj_applies;
        }
    }
}

/// Return an optimal move sequence from `start` to [`GOAL`].
///
/// `Some(vec![])` if `start == GOAL`; `None` only if `start` is unreachable
/// (impossible for solvable states).
pub fn idastar<H: Heuristic>(start: &State, h: &H) -> Option<Vec<Move>> {
    idastar_with_stats(start, h).0
}

/// Like [`idastar`], but also returns [`SearchStats`].
pub fn idastar_with_stats<H: Heuristic>(start: &State, h: &H) -> (Option<Vec<Move>>, SearchStats) {
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (Some(Vec::new()), stats);
    }
    let mut bound = h.h(start);
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    loop {
        stats.iterations += 1;
        match search(start, blank, 0, bound, &mut path, None, h, &mut stats) {
            Step::Found => return (Some(path), stats),
            Step::Aborted => unreachable!("no deadline set"),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (None, stats);
                }
                bound = next;
            }
        }
    }
}

enum Step {
    Found,
    /// Smallest `f` seen this iteration that exceeded `bound`.
    Bound(u8),
    /// The optional deadline elapsed mid-iteration; unwind and report a timeout.
    Aborted,
}

#[allow(clippy::too_many_arguments)]
fn search<H: Heuristic>(
    s: &State,
    blank: u8,
    g: u8,
    bound: u8,
    path: &mut Vec<Move>,
    last: Option<Move>,
    h: &H,
    stats: &mut SearchStats,
) -> Step {
    stats.nodes += 1;
    let f = g.saturating_add(h.h(s));
    if f > bound {
        return Step::Bound(f);
    }
    if s == &GOAL {
        return Step::Found;
    }

    let mut min_next = u8::MAX;
    for m in State::legal_moves_at(blank).iter() {
        if let Some(prev) = last {
            if m == prev.inverse() {
                continue;
            }
        }
        let (s_next, next_blank) = s.apply_at(m, blank);
        path.push(m);
        match search(&s_next, next_blank, g + 1, bound, path, Some(m), h, stats) {
            Step::Found => return Step::Found,
            Step::Aborted => unreachable!("no deadline in plain search"),
            Step::Bound(n) => {
                if n < min_next {
                    min_next = n;
                }
            }
        }
        path.pop();
    }
    Step::Bound(min_next)
}

/// A heuristic that can be advanced incrementally along a single move.
///
/// The plain [`Heuristic`] recomputes `h` from scratch at every node. For the
/// PDB heuristics that means re-projecting the full board (an O(25) scan per
/// pattern, plus a `reflect` for the diagonal view) on every one of millions of
/// nodes. An `IncHeuristic` instead carries a small per-node context that
/// [`advance`](IncHeuristic::advance) updates in O(k) when a move is applied —
/// and for the zero-aware 1-bit PDBs it is *required* (their stored entries are
/// deltas, so the absolute `h` must be threaded forward).
///
/// `Ctx` is `Copy` so the search threads it by value down the recursion;
/// backtracking is automatic (each stack frame owns its copy).
pub trait IncHeuristic {
    /// Per-node carried state.
    type Ctx: Copy;
    /// Evaluate at the search root: `(h(start), ctx0)`. `stats` is bumped for
    /// the sub-components this heuristic evaluates (e.g. one
    /// `zpdb_advances`/`lc_advances`/`wd_advances` increment per `root` call).
    fn root(&self, s: &State, stats: &mut SearchStats) -> (u8, Self::Ctx);
    /// Given the parent's context and the move `m` just applied to reach
    /// `child`, return `(h(child), child_ctx)`. Implementations bump the
    /// corresponding `SearchStats` counters so wall-time-per-call analysis has
    /// accurate call counts regardless of inlining decisions.
    fn advance(
        &self,
        parent: &Self::Ctx,
        child: &State,
        m: Move,
        stats: &mut SearchStats,
    ) -> (u8, Self::Ctx);
}

/// [`idastar_with_stats`] driven by an [`IncHeuristic`]. Identical results to
/// the plain version with an equivalent heuristic — only per-node cost differs.
pub fn idastar_inc_with_stats<E: IncHeuristic>(
    start: &State,
    e: &E,
) -> (Option<Vec<Move>>, SearchStats) {
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (Some(Vec::new()), stats);
    }
    let (h0, ctx0) = e.root(start, &mut stats);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    // Private, never-read cell so the shared `search_inc` signature is uniform;
    // in this sequential driver it is only set at the goal and then discarded.
    let found = std::sync::atomic::AtomicBool::new(false);
    loop {
        stats.iterations += 1;
        match search_inc(start, blank, ctx0, h0, 0, bound, None, &found, &mut path, None, e, &mut stats) {
            Step::Found => return (Some(path), stats),
            Step::Aborted => unreachable!("no deadline set"),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (None, stats);
                }
                bound = next;
            }
        }
    }
}

/// Convenience wrapper discarding stats.
pub fn idastar_inc<E: IncHeuristic>(start: &State, e: &E) -> Option<Vec<Move>> {
    idastar_inc_with_stats(start, e).0
}

/// Check the deadline at most once per this many nodes (cheap amortized cost;
/// `Instant::now()` is not free, so we don't call it per node).
const DEADLINE_CHECK_MASK: u64 = 0x3FFF; // every 16384 nodes

#[allow(clippy::too_many_arguments)]
fn search_inc<E: IncHeuristic>(
    s: &State,
    blank: u8,
    ctx: E::Ctx,
    h_val: u8,
    g: u8,
    bound: u8,
    deadline: Option<std::time::Instant>,
    found: &std::sync::atomic::AtomicBool,
    path: &mut Vec<Move>,
    last: Option<Move>,
    e: &E,
    stats: &mut SearchStats,
) -> Step {
    stats.nodes += 1;
    // Every ~16k nodes, check for a deadline or a sibling worker having already
    // found the goal (parallel solving early-exit). Both surface as `Aborted`;
    // the caller distinguishes them (found ⇒ another worker's `Found` wins). In
    // the sequential drivers `found` is a private cell nobody else reads, so the
    // load is a harmless no-op there.
    if stats.nodes & DEADLINE_CHECK_MASK == 0 {
        if found.load(std::sync::atomic::Ordering::Relaxed) {
            return Step::Aborted;
        }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                return Step::Aborted;
            }
        }
    }
    let f = g.saturating_add(h_val);
    if f > bound {
        return Step::Bound(f);
    }
    if s == &GOAL {
        // Signal any parallel siblings to stop — every goal at this threshold is
        // optimal-length, so the first one found is a valid optimal solution.
        found.store(true, std::sync::atomic::Ordering::Relaxed);
        return Step::Found;
    }

    let mut min_next = u8::MAX;
    for m in State::legal_moves_at(blank).iter() {
        if let Some(prev) = last {
            if m == prev.inverse() {
                continue;
            }
        }
        let (s_next, next_blank) = s.apply_at(m, blank);
        let (child_h, child_ctx) = e.advance(&ctx, &s_next, m, stats);
        path.push(m);
        match search_inc(&s_next, next_blank, child_ctx, child_h, g + 1, bound, deadline, found, path, Some(m), e, stats) {
            Step::Found => return Step::Found,
            Step::Aborted => return Step::Aborted,
            Step::Bound(n) => {
                if n < min_next {
                    min_next = n;
                }
            }
        }
        path.pop();
    }
    Step::Bound(min_next)
}

/// Outcome of a bounded ([lower-bound](idastar_inc_bounded_with_stats)) search.
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
}

/// Bounded / lower-bound IDA\* driven by an [`IncHeuristic`].
///
/// Runs the same deepening loop as [`idastar_inc_with_stats`], but refuses to
/// start any iteration whose threshold exceeds `max_bound`. If the goal is found
/// at or below that threshold the result is [`BoundedOutcome::Solved`] with the
/// optimal path; otherwise it is [`BoundedOutcome::ProvedAtLeast`]`(K)` where
/// `K` is the threshold the search would next have attempted — a proven lower
/// bound on the optimal distance (consistency ⇒ each exhausted threshold is a
/// valid lower bound). `K > max_bound` always.
pub fn idastar_inc_bounded_with_stats<E: IncHeuristic>(
    start: &State,
    e: &E,
    max_bound: u8,
) -> (BoundedOutcome, SearchStats) {
    idastar_inc_bounded_telemetry(start, e, max_bound, |_, _, _| {})
}

/// [`idastar_inc_bounded_with_stats`] with a per-iteration telemetry callback.
///
/// `on_iter(threshold, cumulative_stats, iter_elapsed)` fires once after each
/// completed IDA\* iteration that did not find the goal — the calibration signal
/// for the ladder harness (how node count and wall-clock grow with the
/// threshold). The callback is outside the node hot path, so it adds nothing to
/// per-node cost.
pub fn idastar_inc_bounded_telemetry<E, F>(
    start: &State,
    e: &E,
    max_bound: u8,
    mut on_iter: F,
) -> (BoundedOutcome, SearchStats)
where
    E: IncHeuristic,
    F: FnMut(u8, &SearchStats, std::time::Duration),
{
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (BoundedOutcome::Solved(Vec::new()), stats);
    }
    let (h0, ctx0) = e.root(start, &mut stats);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    let found = std::sync::atomic::AtomicBool::new(false); // private, never read here
    loop {
        if bound > max_bound {
            return (BoundedOutcome::ProvedAtLeast(bound), stats);
        }
        stats.iterations += 1;
        let iter_start = std::time::Instant::now();
        match search_inc(start, blank, ctx0, h0, 0, bound, None, &found, &mut path, None, e, &mut stats) {
            Step::Found => return (BoundedOutcome::Solved(path), stats),
            Step::Aborted => unreachable!("no deadline set"),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (BoundedOutcome::Unsolvable, stats);
                }
                on_iter(bound, &stats, iter_start.elapsed());
                bound = next;
            }
        }
    }
}

/// Outcome of a deadline-aware ([ladder](idastar_inc_ladder)) search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LadderOutcome {
    /// Optimal solution found within both the bound and the deadline.
    Solved(Vec<Move>),
    /// Every threshold up to `max_bound` was exhausted in time: proves `depth ≥ K`.
    ProvedAtLeast(u8),
    /// The deadline elapsed mid-iteration at threshold `K`. Since the previous
    /// iteration completed, `depth ≥ K` is still proven (`K` = the deepest
    /// threshold reached); the search just couldn't finish threshold `K`.
    TimedOut(u8),
    /// `start` is unreachable from `GOAL`.
    Unsolvable,
}

/// Unified deadline-aware IDA\* driver for the calibration harness.
///
/// Subsumes both measurement modes:
/// - **optimal-solve**: pass `max_bound = u8::MAX` and `deadline = Some(t)` — get
///   `Solved(len)` or `TimedOut(K)`.
/// - **bounded lower-bound**: pass a finite `max_bound` (and optional deadline) —
///   get `Solved`, `ProvedAtLeast(K)`, or `TimedOut(K)`.
///
/// `deadline = None` makes it a pure bounded search (never times out). The
/// deadline is polled every ~16k nodes, so the actual stop can overshoot `t` by
/// one such batch — fine for minute/hour-scale budgets.
pub fn idastar_inc_ladder<E: IncHeuristic>(
    start: &State,
    e: &E,
    max_bound: u8,
    deadline: Option<std::time::Instant>,
) -> (LadderOutcome, SearchStats) {
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (LadderOutcome::Solved(Vec::new()), stats);
    }
    let (h0, ctx0) = e.root(start, &mut stats);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    let found = std::sync::atomic::AtomicBool::new(false); // private, never read here
    loop {
        if bound > max_bound {
            return (LadderOutcome::ProvedAtLeast(bound), stats);
        }
        stats.iterations += 1;
        match search_inc(start, blank, ctx0, h0, 0, bound, deadline, &found, &mut path, None, e, &mut stats) {
            Step::Found => return (LadderOutcome::Solved(path), stats),
            // `bound` is the deepest threshold reached; the prior iteration
            // proved depth ≥ bound, so that is the lower bound at timeout.
            Step::Aborted => return (LadderOutcome::TimedOut(bound), stats),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (LadderOutcome::Unsolvable, stats);
                }
                bound = next;
            }
        }
    }
}

/// Parallel (shared-memory) variant of [`idastar_inc_ladder`] — same semantics
/// and identical results, only faster on a multi-core machine (Phase 2 "2P").
///
/// Each threshold iteration is parallelized by **tree-splitting + work-stealing**:
/// a cheap sequential breadth-first expansion grows a *frontier* of subtree roots
/// (≫ #cores), then `rayon` runs the unmodified sequential [`search_inc`] on each
/// subtree concurrently, and the per-subtree results are reduced — OR the "found"
/// flags, MIN the smallest-`f`-over-threshold (the next bound), SUM the stats.
///
/// Correctness is unchanged from the sequential driver: exhausting a threshold in
/// parallel still proves `dist ≥ next`, the MIN reduction is order-independent, and
/// the first solution found at the successful threshold is optimal (any worker's is
/// optimal-length). IDA\* stays memory-light — each worker holds one path + a
/// `Copy` context — so this does not blow up like parallel A\*. The bounded
/// lower-bound case (no early exit) has *zero* redundant work.
///
/// Bounds: `E: Sync` (the heuristic is shared) and `E::Ctx: Send + Sync` (the
/// frontier of `Copy` contexts is shared across workers) — satisfied by every
/// heuristic here. Build with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn idastar_inc_bounded_parallel<E>(
    start: &State,
    e: &E,
    max_bound: u8,
    deadline: Option<std::time::Instant>,
) -> (LadderOutcome, SearchStats)
where
    E: IncHeuristic + Sync,
    E::Ctx: Send + Sync,
{
    use rayon::prelude::*;
    use std::collections::VecDeque;

    /// One subtree-root work unit: a node plus the move-prefix that reaches it,
    /// so a worker that finds the goal can return the full path from `start`.
    struct PUnit<C> {
        state: State,
        blank: u8,
        g: u8,
        ctx: C,
        h: u8,
        last: Option<Move>,
        prefix: Vec<Move>,
    }

    /// Grow the frontier to ~this many subtree roots before going parallel — far
    /// more than #cores so work-stealing balances the wildly-varying subtree sizes.
    const SPLIT_TARGET: usize = 4096;

    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (LadderOutcome::Solved(Vec::new()), stats);
    }
    let (h0, ctx0) = e.root(start, &mut stats);
    let blank0 = start.blank_pos();
    let mut bound = h0;

    loop {
        if bound > max_bound {
            return (LadderOutcome::ProvedAtLeast(bound), stats);
        }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                return (LadderOutcome::TimedOut(bound), stats);
            }
        }
        stats.iterations += 1;

        // ---- split: sequential BFS growing a frontier of subtree roots ----
        let mut frontier: VecDeque<PUnit<E::Ctx>> = VecDeque::new();
        frontier.push_back(PUnit {
            state: *start, blank: blank0, g: 0, ctx: ctx0, h: h0, last: None, prefix: Vec::new(),
        });
        let mut min_f_split = u8::MAX;
        let mut found_in_split: Option<Vec<Move>> = None;
        while frontier.len() < SPLIT_TARGET {
            let u = match frontier.pop_front() {
                Some(u) => u,
                None => break, // tree exhausted during split
            };
            stats.nodes += 1;
            if u.state == GOAL {
                found_in_split = Some(u.prefix);
                break;
            }
            for m in State::legal_moves_at(u.blank).iter() {
                if let Some(prev) = u.last {
                    if m == prev.inverse() {
                        continue;
                    }
                }
                let (s_next, nb) = u.state.apply_at(m, u.blank);
                let (ch, cctx) = e.advance(&u.ctx, &s_next, m, &mut stats);
                let cf = (u.g + 1).saturating_add(ch);
                if cf > bound {
                    // Count the boundary-pruned child to match `search_inc`, which
                    // increments `nodes` at entry *before* the `f > bound` prune.
                    // (Non-pruned children are counted instead when popped/searched,
                    // so every node is tallied exactly once.)
                    stats.nodes += 1;
                    if cf < min_f_split {
                        min_f_split = cf;
                    }
                    continue;
                }
                let mut prefix = u.prefix.clone();
                prefix.push(m);
                frontier.push_back(PUnit {
                    state: s_next, blank: nb, g: u.g + 1, ctx: cctx, h: ch, last: Some(m), prefix,
                });
            }
        }
        if let Some(path) = found_in_split {
            return (LadderOutcome::Solved(path), stats);
        }
        if frontier.is_empty() {
            // Whole threshold exhausted during the split (small tree).
            if min_f_split == u8::MAX {
                return (LadderOutcome::Unsolvable, stats);
            }
            bound = min_f_split;
            continue;
        }

        // ---- parallel: run sequential search_inc on each subtree, then reduce ----
        enum Wr {
            Found(Vec<Move>),
            Aborted,
            Bound(u8),
        }
        let units: Vec<PUnit<E::Ctx>> = frontier.into_iter().collect();
        // Shared early-exit flag: the first worker to reach the goal sets it, and
        // the others bail (their goals would be the same optimal length). Only ever
        // set in a *solving* iteration; in a bounded exhaust it stays false, so the
        // proven LB is identical to the sequential driver (no early exit).
        let found_flag = std::sync::atomic::AtomicBool::new(false);
        let results: Vec<(Wr, SearchStats)> = units
            .par_iter()
            .map(|u| {
                let mut st = SearchStats::default();
                let mut path: Vec<Move> = Vec::new();
                let step =
                    search_inc(&u.state, u.blank, u.ctx, u.h, u.g, bound, deadline, &found_flag, &mut path, u.last, e, &mut st);
                let wr = match step {
                    Step::Found => {
                        let mut full = u.prefix.clone();
                        full.extend(path);
                        Wr::Found(full)
                    }
                    Step::Aborted => Wr::Aborted,
                    Step::Bound(n) => Wr::Bound(n),
                };
                (wr, st)
            })
            .collect();

        let mut found: Option<Vec<Move>> = None;
        let mut aborted = false;
        let mut min_f = min_f_split;
        for (wr, st) in results {
            stats.add(&st);
            match wr {
                Wr::Found(p) => {
                    if found.is_none() {
                        found = Some(p);
                    }
                }
                Wr::Aborted => aborted = true,
                Wr::Bound(n) => {
                    if n < min_f {
                        min_f = n;
                    }
                }
            }
        }
        if let Some(p) = found {
            return (LadderOutcome::Solved(p), stats);
        }
        if aborted {
            return (LadderOutcome::TimedOut(bound), stats);
        }
        if min_f == u8::MAX {
            return (LadderOutcome::Unsolvable, stats);
        }
        bound = min_f;
    }
}

/// Make/unmake counterpart of [`idastar_inc_bounded_parallel`]: identical
/// tree-splitting and identical results, but each worker searches its subtree
/// with the allocation-free [`IncHeuristicMut`] make/unmake path instead of the
/// `Copy` [`IncHeuristic`] path — cutting the per-node cost that dominates deep
/// runs (profiled ~50% context copying).
///
/// The shallow split still uses the `Copy` [`IncHeuristic`] (it is cheap and
/// keeps the frontier/node-count logic byte-identical to the copy driver); each
/// worker then re-seeds a fresh mutable context from its subtree root via
/// [`IncHeuristicMut::root`] and runs [`search_inc_mut`]. Node counts therefore
/// match the copy parallel driver exactly. Requires the heuristic to implement
/// *both* traits (as [`ZpdbInc`](crate::puzzle24::pdb::ZpdbInc) does).
///
/// Bounds: `E: IncHeuristic + IncHeuristicMut + Sync`; the split frontier of
/// `Copy` contexts needs `<E as IncHeuristic>::Ctx: Send + Sync`. Each worker's
/// mutable context is created and dropped inside its closure, never shared, so
/// it needs no extra bound. Build with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn idastar_inc_bounded_parallel_mut<E>(
    start: &State,
    e: &E,
    max_bound: u8,
    deadline: Option<std::time::Instant>,
    orbit_split: bool,
) -> (LadderOutcome, SearchStats)
where
    E: IncHeuristic + IncHeuristicMut + Sync,
    <E as IncHeuristic>::Ctx: Send + Sync,
{
    parallel_mut_core(start, e, &NullPruner, max_bound, deadline, orbit_split)
}

/// [`idastar_inc_bounded_parallel_mut`] with a [`MovePruner`] that skips provably
/// redundant move continuations (duplicate subtrees). Sound as a lower-bound proof:
/// a pruned board is reachable by a shorter/equal path, so no board — and no `f`
/// threshold — is lost. With [`NullPruner`] it is byte-identical to the un-pruned
/// driver (node counts unchanged); with a [`MoveDfa`](super::move_dfa::MoveDfa) it
/// removes the Taylor–Korf duplicates. The pruner is shared read-only across
/// workers; each carries its own `P::St` down its subtree.
#[cfg(feature = "parallel")]
pub fn idastar_inc_bounded_parallel_mut_pruned<E, P>(
    start: &State,
    e: &E,
    pruner: &P,
    max_bound: u8,
    deadline: Option<std::time::Instant>,
    orbit_split: bool,
) -> (LadderOutcome, SearchStats)
where
    E: IncHeuristic + IncHeuristicMut + Sync,
    <E as IncHeuristic>::Ctx: Send + Sync,
    P: MovePruner,
{
    parallel_mut_core(start, e, pruner, max_bound, deadline, orbit_split)
}

/// `orbit_split`: when `true` **and** `start` is σ-symmetric (`reflect(start) ==
/// start`), the split searches only one representative per σ-orbit of the *root's*
/// children (see [`symmetry::is_orbit_representative`]) — a sound ~2× reduction for
/// lower-bound proofs. The filter is applied at the true root only (`g == 0`);
/// applying it deeper would drop real boards. Callers are responsible for the
/// symmetry precondition; passing `true` on a non-symmetric board is unsound.
#[cfg(feature = "parallel")]
fn parallel_mut_core<E, P>(
    start: &State,
    e: &E,
    pruner: &P,
    max_bound: u8,
    deadline: Option<std::time::Instant>,
    orbit_split: bool,
) -> (LadderOutcome, SearchStats)
where
    E: IncHeuristic + IncHeuristicMut + Sync,
    <E as IncHeuristic>::Ctx: Send + Sync,
    P: MovePruner,
{
    use rayon::prelude::*;
    use std::collections::VecDeque;

    /// Subtree-root unit for the mut driver. Unlike the copy driver it does NOT
    /// carry a per-node context: each worker re-roots a fresh mutable context
    /// from `state`. It keeps the split's `Copy` context only transiently. `pst`
    /// is the pruner state at this subtree root, seeded during the split.
    struct PUnit<S> {
        state: State,
        blank: u8,
        g: u8,
        last: Option<Move>,
        prefix: Vec<Move>,
        pst: S,
    }

    const SPLIT_TARGET: usize = 4096;

    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (LadderOutcome::Solved(Vec::new()), stats);
    }
    let (h0, ctx0) = IncHeuristic::root(e, start, &mut stats);
    let blank0 = start.blank_pos();
    let pst0 = pruner.root_state(blank0);
    let mut bound = h0;

    loop {
        if bound > max_bound {
            return (LadderOutcome::ProvedAtLeast(bound), stats);
        }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                return (LadderOutcome::TimedOut(bound), stats);
            }
        }
        stats.iterations += 1;

        // ---- split: sequential BFS (Copy heuristic), same as the copy driver ----
        struct SplitUnit<C, S> {
            state: State,
            blank: u8,
            g: u8,
            ctx: C,
            last: Option<Move>,
            prefix: Vec<Move>,
            pst: S,
        }
        let mut frontier: VecDeque<SplitUnit<<E as IncHeuristic>::Ctx, P::St>> = VecDeque::new();
        frontier.push_back(SplitUnit {
            state: *start, blank: blank0, g: 0, ctx: ctx0, last: None, prefix: Vec::new(), pst: pst0,
        });
        let mut min_f_split = u8::MAX;
        let mut found_in_split: Option<Vec<Move>> = None;
        while frontier.len() < SPLIT_TARGET {
            let u = match frontier.pop_front() {
                Some(u) => u,
                None => break,
            };
            stats.nodes += 1;
            if u.state == GOAL {
                found_in_split = Some(u.prefix);
                break;
            }
            for m in State::legal_moves_at(u.blank).iter() {
                // Root-orbit-split (true root only): on a σ-symmetric board the
                // root's children fall into σ-orbits {m, transpose_move(m)}; search
                // one representative per orbit. Applied at `g == 0` only — deeper
                // filtering would drop real boards. A dropped child then adds to
                // neither `stats.nodes` nor `min_f_split` (its kept mirror carries
                // the same `f`), so the LB proof is unaffected.
                if orbit_split && u.g == 0 && !symmetry::is_orbit_representative(m) {
                    continue;
                }
                if let Some(prev) = u.last {
                    if m == prev.inverse() {
                        continue;
                    }
                }
                // Duplicate-subtree pruning, seeded consistently with the workers so
                // each surviving unit carries the right pruner state. No-op under
                // `NullPruner`, keeping the un-pruned frontier/counts identical.
                if pruner.is_pruned(u.pst, m) {
                    continue;
                }
                let (s_next, nb) = u.state.apply_at(m, u.blank);
                let (ch, cctx) = e.advance(&u.ctx, &s_next, m, &mut stats);
                let cf = (u.g + 1).saturating_add(ch);
                if cf > bound {
                    stats.nodes += 1; // match search_inc's count-at-entry (see copy driver)
                    if cf < min_f_split {
                        min_f_split = cf;
                    }
                    continue;
                }
                let mut prefix = u.prefix.clone();
                prefix.push(m);
                frontier.push_back(SplitUnit {
                    state: s_next, blank: nb, g: u.g + 1, ctx: cctx, last: Some(m), prefix,
                    pst: pruner.advance(u.pst, m),
                });
            }
        }
        if let Some(path) = found_in_split {
            return (LadderOutcome::Solved(path), stats);
        }
        if frontier.is_empty() {
            if min_f_split == u8::MAX {
                return (LadderOutcome::Unsolvable, stats);
            }
            bound = min_f_split;
            continue;
        }

        // ---- parallel: run search_inc_mut on each subtree, re-rooting the ctx ----
        enum Wr {
            Found(Vec<Move>),
            Aborted,
            Bound(u8),
        }
        // Drop the split's Copy contexts; workers re-root their own mutable ones.
        let units: Vec<PUnit<P::St>> = frontier
            .into_iter()
            .map(|u| PUnit { state: u.state, blank: u.blank, g: u.g, last: u.last, prefix: u.prefix, pst: u.pst })
            .collect();
        let found_flag = std::sync::atomic::AtomicBool::new(false);
        let results: Vec<(Wr, SearchStats)> = units
            .par_iter()
            .map(|u| {
                let mut st = SearchStats::default();
                let mut path: Vec<Move> = Vec::new();
                let (h_u, mut ctx) = IncHeuristicMut::root(e, &u.state);
                let step = search_inc_mut(
                    &u.state, u.blank, &mut ctx, h_u, u.g, bound, deadline, &found_flag,
                    &mut path, u.last, e, pruner, u.pst, false, &mut st,
                );
                let wr = match step {
                    Step::Found => {
                        let mut full = u.prefix.clone();
                        full.extend(path);
                        Wr::Found(full)
                    }
                    Step::Aborted => Wr::Aborted,
                    Step::Bound(n) => Wr::Bound(n),
                };
                (wr, st)
            })
            .collect();

        let mut found: Option<Vec<Move>> = None;
        let mut aborted = false;
        let mut min_f = min_f_split;
        for (wr, st) in results {
            stats.add(&st);
            match wr {
                Wr::Found(p) => {
                    if found.is_none() {
                        found = Some(p);
                    }
                }
                Wr::Aborted => aborted = true,
                Wr::Bound(n) => {
                    if n < min_f {
                        min_f = n;
                    }
                }
            }
        }
        if let Some(p) = found {
            return (LadderOutcome::Solved(p), stats);
        }
        if aborted {
            return (LadderOutcome::TimedOut(bound), stats);
        }
        if min_f == u8::MAX {
            return (LadderOutcome::Unsolvable, stats);
        }
        bound = min_f;
    }
}

/// Make/unmake variant of [`IncHeuristic`]: instead of producing a fresh `Copy`
/// context per node, the search threads a single `&mut Ctx`, mutating it in
/// place before recursing and undoing on backtrack. This avoids the per-node
/// context copy (profiled at ~21% on puzzle15; the payoff is *larger* here,
/// where a context holds four PDBs × normal+reflected `ProjectedState` plus the
/// `LcCtx`/`WdCtx`).
pub trait IncHeuristicMut {
    /// Per-node carried state (mutated in place; need not be `Copy`).
    type Ctx;
    /// Evaluate at the search root: `(h(start), ctx0)`.
    fn root(&self, s: &State) -> (u8, Self::Ctx);
    /// Mutate `ctx` by applying move `m`, returning `h` of the resulting child.
    /// `child` is the post-move state (the search already computed it), so
    /// positional heuristics can read the moved tile directly — mirroring
    /// [`IncHeuristic::advance`]. Purely projection-based heuristics ignore it.
    fn make(&self, ctx: &mut Self::Ctx, child: &State, m: Move) -> u8;
    /// Undo the most recent [`make`](Self::make) of `m`, restoring `ctx`.
    /// Implementations restore via an undo record or a self-inverse operation,
    /// so no state argument is needed.
    fn unmake(&self, ctx: &mut Self::Ctx, m: Move);

    /// Optional cheap, admissible **lower bound** on the `h` of the child reached
    /// by applying `m` at `blank` in `s`, computed WITHOUT mutating `ctx` and
    /// WITHOUT the (cache-missing) table probe `make` would do. `None` (default)
    /// ⇒ the caller must `make` to learn the child's `h`. When `Some(lb)`, the
    /// search may skip the child entirely if `g + 1 + lb > bound` — it would be
    /// pruned at first touch anyway, so no board/threshold is lost.
    #[inline(always)]
    fn child_h_lb(&self, _ctx: &Self::Ctx, _s: &State, _blank: u8, _m: Move) -> Option<u8> {
        None
    }

    /// [`make`](Self::make) with a pruning budget: `budget` is the largest `h`
    /// that keeps the child inside the current bound (`g + 1 + h ≤ bound`). A
    /// combiner may stop after a cheap component already exceeds `budget` and
    /// return that component's (admissible) value without evaluating the
    /// expensive ones — the child is pruned at first touch either way, so the
    /// search tree is identical to the eager evaluation. The returned value must
    /// always be admissible, and `ctx` must stay consistent for the matching
    /// [`unmake`](Self::unmake). Default: plain `make` (budget ignored).
    #[inline(always)]
    fn make_bounded(&self, ctx: &mut Self::Ctx, child: &State, m: Move, _budget: u8) -> u8 {
        self.make(ctx, child, m)
    }
}

/// [`idastar_inc_with_stats`] using a mutable make/unmake context.
pub fn idastar_inc_mut_with_stats<E: IncHeuristicMut>(
    start: &State,
    e: &E,
) -> (Option<Vec<Move>>, SearchStats) {
    idastar_inc_mut_orbit_with_stats(start, e, false)
}

/// [`idastar_inc_mut_with_stats`] with sequential root-orbit-split. When
/// `orbit_split` is `true` the search keeps one representative per σ-orbit of the
/// **root's** children — the single-threaded counterpart of
/// [`idastar_inc_bounded_parallel_mut`]'s split filter, giving the same ~2×
/// lower-bound reduction and identical node counts, without work-stealing
/// nondeterminism. **Sound only on a σ-fixed board** (`reflect(start) == start`);
/// the returned solution stays optimal because a dropped child's mirror reaches
/// the reflected (equal-length) optimum. Debug-asserts the precondition.
pub fn idastar_inc_mut_orbit_with_stats<E: IncHeuristicMut>(
    start: &State,
    e: &E,
    orbit_split: bool,
) -> (Option<Vec<Move>>, SearchStats) {
    idastar_inc_mut_pruned_orbit_with_stats(start, e, &NullPruner, orbit_split)
}

/// [`idastar_inc_mut_orbit_with_stats`] with a [`MovePruner`] (e.g. a
/// [`MoveDfa`](super::move_dfa::MoveDfa) for Taylor–Korf duplicate elimination) —
/// the sequential counterpart of [`idastar_inc_bounded_parallel_mut_pruned`].
/// With [`NullPruner`] it is byte-identical to the un-pruned driver; with a real
/// pruner it removes the same duplicate subtrees, node-identically to the
/// parallel pruned driver. Sound as a lower-bound proof (a pruned board is
/// reachable by a shorter/equal path, so no threshold is lost).
pub fn idastar_inc_mut_pruned_orbit_with_stats<E: IncHeuristicMut, P: MovePruner>(
    start: &State,
    e: &E,
    pruner: &P,
    orbit_split: bool,
) -> (Option<Vec<Move>>, SearchStats) {
    debug_assert!(
        !orbit_split || symmetry::is_symmetric(start),
        "orbit_split is unsound on a non-σ-symmetric board"
    );
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (Some(Vec::new()), stats);
    }
    let (h0, mut ctx) = e.root(start);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    let pst0 = pruner.root_state(blank);
    // Private, never-shared flag so `search_inc_mut` can take a non-optional
    // `&AtomicBool`; in this sequential driver nothing else reads it.
    let found = std::sync::atomic::AtomicBool::new(false);
    loop {
        stats.iterations += 1;
        // `ctx` is restored to the root projection after each iteration because
        // `search_inc_mut` unmakes every move it makes.
        match search_inc_mut(start, blank, &mut ctx, h0, 0, bound, None, &found, &mut path, None, e, pruner, pst0, orbit_split, &mut stats) {
            Step::Found => return (Some(path), stats),
            Step::Aborted => unreachable!("no deadline set"),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (None, stats);
                }
                bound = next;
            }
        }
    }
}

/// Convenience wrapper discarding stats.
pub fn idastar_inc_mut<E: IncHeuristicMut>(start: &State, e: &E) -> Option<Vec<Move>> {
    idastar_inc_mut_with_stats(start, e).0
}

/// Make/unmake counterpart of [`idastar_inc_bounded_with_stats`]: bounded IDA\*
/// that either solves optimally within `max_bound` or proves `dist ≥ K`. Same
/// semantics and identical node counts as the copy driver (deterministic — no
/// early-exit flag in play), just cheaper per node.
pub fn idastar_inc_mut_bounded_with_stats<E: IncHeuristicMut>(
    start: &State,
    e: &E,
    max_bound: u8,
) -> (BoundedOutcome, SearchStats) {
    idastar_inc_mut_bounded_orbit_with_stats(start, e, max_bound, false)
}

/// [`idastar_inc_mut_bounded_with_stats`] with sequential root-orbit-split (see
/// [`idastar_inc_mut_orbit_with_stats`]). `orbit_split = true` keeps one σ-orbit
/// representative of the root's children — the single-threaded counterpart of the
/// parallel driver's split filter, node-identical to it. **Sound only on a
/// σ-fixed board**; debug-asserts the precondition.
pub fn idastar_inc_mut_bounded_orbit_with_stats<E: IncHeuristicMut>(
    start: &State,
    e: &E,
    max_bound: u8,
    orbit_split: bool,
) -> (BoundedOutcome, SearchStats) {
    idastar_inc_mut_bounded_pruned_orbit_with_stats(start, e, &NullPruner, max_bound, orbit_split)
}

/// [`idastar_inc_mut_bounded_orbit_with_stats`] with a [`MovePruner`] — the
/// sequential counterpart of [`idastar_inc_bounded_parallel_mut_pruned`] (see
/// [`idastar_inc_mut_pruned_orbit_with_stats`]). Node-identical to the parallel
/// pruned driver with the same pruner and orbit flag.
pub fn idastar_inc_mut_bounded_pruned_orbit_with_stats<E: IncHeuristicMut, P: MovePruner>(
    start: &State,
    e: &E,
    pruner: &P,
    max_bound: u8,
    orbit_split: bool,
) -> (BoundedOutcome, SearchStats) {
    debug_assert!(
        !orbit_split || symmetry::is_symmetric(start),
        "orbit_split is unsound on a non-σ-symmetric board"
    );
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (BoundedOutcome::Solved(Vec::new()), stats);
    }
    let (h0, mut ctx) = e.root(start);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    let pst0 = pruner.root_state(blank);
    let found = std::sync::atomic::AtomicBool::new(false); // private, never read here
    loop {
        if bound > max_bound {
            return (BoundedOutcome::ProvedAtLeast(bound), stats);
        }
        stats.iterations += 1;
        // `ctx` is restored after each iteration — `search_inc_mut` unmakes every
        // move it makes on a `Bound` return.
        match search_inc_mut(start, blank, &mut ctx, h0, 0, bound, None, &found, &mut path, None, e, pruner, pst0, orbit_split, &mut stats) {
            Step::Found => return (BoundedOutcome::Solved(path), stats),
            Step::Aborted => unreachable!("no deadline set"),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (BoundedOutcome::Unsolvable, stats);
                }
                bound = next;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn search_inc_mut<E: IncHeuristicMut, P: MovePruner>(
    s: &State,
    blank: u8,
    ctx: &mut E::Ctx,
    h_val: u8,
    g: u8,
    bound: u8,
    deadline: Option<std::time::Instant>,
    found: &std::sync::atomic::AtomicBool,
    path: &mut Vec<Move>,
    last: Option<Move>,
    e: &E,
    p: &P,
    pst: P::St,
    orbit_split: bool,
    stats: &mut SearchStats,
) -> Step {
    stats.nodes += 1;
    // Same periodic deadline / sibling-found early-exit as `search_inc`. On
    // `Aborted` the whole search unwinds and `ctx` is discarded, so — like the
    // `Found` path — we return without unmaking. In the sequential driver
    // `found` is a private cell and `deadline` is `None`, so this never fires.
    if stats.nodes & DEADLINE_CHECK_MASK == 0 {
        if found.load(std::sync::atomic::Ordering::Relaxed) {
            return Step::Aborted;
        }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                return Step::Aborted;
            }
        }
    }
    let f = g.saturating_add(h_val);
    if f > bound {
        return Step::Bound(f);
    }
    if s == &GOAL {
        found.store(true, std::sync::atomic::Ordering::Relaxed);
        return Step::Found;
    }

    let mut min_next = u8::MAX;
    for m in State::legal_moves_at(blank).iter() {
        // Root-orbit-split (true root only): on a σ-symmetric board the root's
        // children fall into σ-orbits {m, transpose_move(m)}; search one
        // representative per orbit. Sequential counterpart of the parallel split's
        // `g == 0` filter — identical to it and to the parallel worker trees. Only
        // sound on a σ-fixed board (caller's precondition; debug-asserted in the
        // driver). Applied at `g == 0` only; deeper filtering would drop real
        // boards. The dropped child's kept mirror carries the same `f`, so neither
        // `stats.nodes` nor the next-threshold `min_next` is affected.
        if orbit_split && g == 0 && !symmetry::is_orbit_representative(m) {
            continue;
        }
        if let Some(prev) = last {
            if m == prev.inverse() {
                continue;
            }
        }
        // Skip provably-redundant continuations (duplicate subtrees). Sound: the
        // pruned board is reachable by a shorter/equal path, so no new board — and
        // no threshold — is lost. `NullPruner` makes this a no-op.
        if p.is_pruned(pst, m) {
            continue;
        }
        // Prune the child before generating/probing it, when the heuristic can
        // cheaply lower-bound its `h` from the parent's cached state. Sound: the
        // child's true `f ≥ g+1+lb > bound`, so `search_inc_mut` would return
        // `Bound` at its first line anyway; feeding `f_child` into `min_next`
        // preserves the next IDA* threshold. No-op for heuristics that return
        // `None` (the default), which monomorphizes the branch away.
        if let Some(lb) = e.child_h_lb(ctx, s, blank, m) {
            let f_child = (g + 1).saturating_add(lb);
            if f_child > bound {
                if f_child < min_next {
                    min_next = f_child;
                }
                continue;
            }
        }
        let (s_next, next_blank) = s.apply_at(m, blank);
        let child_h = e.make_bounded(ctx, &s_next, m, bound.saturating_sub(g + 1));
        path.push(m);
        match search_inc_mut(&s_next, next_blank, ctx, child_h, g + 1, bound, deadline, found, path, Some(m), e, p, p.advance(pst, m), orbit_split, stats) {
            // On Found/Aborted the whole search terminates so `ctx` is discarded
            // (no need to unmake); keep the accumulated path.
            Step::Found => return Step::Found,
            Step::Aborted => return Step::Aborted,
            Step::Bound(n) => {
                path.pop();
                e.unmake(ctx, m); // restore path + ctx for the next sibling
                if n < min_next {
                    min_next = n;
                }
            }
        }
    }

    Step::Bound(min_next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::heuristic::ManhattanHeuristic;
    use crate::puzzle24::search::tests_util::bfs_distances;

    #[test]
    fn solves_goal_with_empty_path() {
        assert!(idastar(&GOAL, &ManhattanHeuristic).unwrap().is_empty());
    }

    /// Deterministic pseudo-random scramble from GOAL (no immediate-undo moves).
    #[cfg(feature = "parallel")]
    fn scramble(seed: u64, steps: u32) -> State {
        let mut s = GOAL;
        let mut blank = GOAL.blank_pos();
        let mut last: Option<Move> = None;
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        for _ in 0..steps {
            let mut legal: Vec<Move> = Vec::new();
            for m in State::legal_moves_at(blank).iter() {
                if last.map_or(true, |p| m != p.inverse()) {
                    legal.push(m);
                }
            }
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let m = legal[(x as usize) % legal.len()];
            let (ns, nb) = s.apply_at(m, blank);
            s = ns;
            blank = nb;
            last = Some(m);
        }
        s
    }

    /// The move-pruning DFA must be a *sound optimization*: skipping duplicate
    /// subtrees may reduce work but must never change the optimal solution length
    /// (admissibility/completeness preserved). We also assert it actually fires
    /// (strictly fewer nodes on deeper bounded exhausts), so the guard is not vacuous.
    #[cfg(feature = "parallel")]
    #[test]
    fn move_dfa_parallel_preserves_optimality() {
        use crate::puzzle24::search::move_dfa::MoveDfa;
        use crate::puzzle24::search::IncManhattan;

        let dfa = MoveDfa::build_default();

        // (1) Soundness: on shallow boards, pruned and un-pruned optimal solves must
        // agree on optimal length, and the pruned solution must reach GOAL.
        for (seed, steps) in [(1u64, 10u32), (7, 12), (23, 14)] {
            let s = scramble(seed, steps);
            let (p_out, _) =
                idastar_inc_bounded_parallel_mut_pruned(&s, &IncManhattan, &dfa, u8::MAX, None, false);
            let (n_out, _) = idastar_inc_bounded_parallel_mut(&s, &IncManhattan, u8::MAX, None, false);
            match (&p_out, &n_out) {
                (LadderOutcome::Solved(a), LadderOutcome::Solved(b)) => {
                    assert_eq!(a.len(), b.len(), "DFA pruning changed optimal length (seed {seed})");
                    let mut t = s;
                    for &mv in a {
                        t = t.apply(mv);
                    }
                    assert_eq!(t, GOAL, "pruned solution invalid (seed {seed})");
                }
                _ => panic!("expected both drivers to solve (seed {seed})"),
            }
        }

        // (2) Anti-vacuous: on deeper boards, a bounded exhaust (which explores the
        // low-`f` loops where duplicates live) must expand strictly fewer nodes with
        // the DFA. `bound = h0 + 6` stays below optimal, so it exhausts a fat shell.
        let mut p_total = 0u64;
        let mut n_total = 0u64;
        for (seed, steps) in [(5u64, 30u32), (77, 40)] {
            let s = scramble(seed, steps);
            let h0 = IncHeuristicMut::root(&IncManhattan, &s).0;
            let bound = h0 + 6;
            let (_, p_stats) =
                idastar_inc_bounded_parallel_mut_pruned(&s, &IncManhattan, &dfa, bound, None, false);
            let (_, n_stats) = idastar_inc_bounded_parallel_mut(&s, &IncManhattan, bound, None, false);
            p_total += p_stats.nodes;
            n_total += n_stats.nodes;
        }
        assert!(
            p_total < n_total,
            "DFA pruning never fired ({p_total} vs {n_total}) — guard is vacuous"
        );
    }

    /// Root-orbit-split must be a *sound optimization*: on a σ-symmetric board it
    /// must reach the identical lower-bound outcome as the full search, while
    /// expanding ~half the nodes (one representative per σ-orbit of the root's
    /// children). Uses a σ-symmetric fixture found by BFS from GOAL — depth 16 with
    /// Manhattan 4, blank at the centre cell 12 (interior, 4 legal root moves → 2
    /// orbits → keep 2), so there is a wide exhaust window below the solution and
    /// the tree genuinely expands. Manhattan (σ-symmetric, weak) + `NullPruner` (the
    /// plain `_mut` driver) so the ~2× is not muddied by the move-DFA's asymmetric
    /// pruning.
    #[cfg(feature = "parallel")]
    #[test]
    fn root_orbit_split_parallel_sound_and_halves_nodes() {
        use crate::puzzle24::search::IncManhattan;
        use crate::puzzle24::symmetry::is_symmetric;

        // Fixture: `1 2 3 4 5 6 7 8 9 10 11 12 _ 14 15 16 17 18 13 20 21 22 23 24 19`
        let fix = State([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 14, 15, 16, 17, 18, 13, 20, 21, 22, 23, 24,
            19,
        ]);
        assert!(is_symmetric(&fix), "fixture must be σ-symmetric");
        assert!(fix.is_solvable(), "fixture must be solvable");
        assert_eq!(fix.blank_pos(), 12, "fixture blank on the centre diagonal");

        // Bounded exhaust well below the solution depth (16): a path-free
        // `ProvedAtLeast(_)` outcome that expands a real tree, so outcomes compare
        // by exact equality (no σ-mirror path ambiguity).
        let (o_off, s_off) = idastar_inc_bounded_parallel_mut(&fix, &IncManhattan, 10, None, false);
        let (o_on, s_on) = idastar_inc_bounded_parallel_mut(&fix, &IncManhattan, 10, None, true);

        assert_eq!(o_off, o_on, "orbit-split changed the lower-bound outcome");
        assert!(
            matches!(o_on, LadderOutcome::ProvedAtLeast(_)),
            "expected a path-free ProvedAtLeast, got {:?}",
            o_on
        );
        assert!(
            s_on.nodes < s_off.nodes,
            "orbit-split did not reduce nodes ({} vs {})",
            s_on.nodes,
            s_off.nodes
        );
        // ~2×: the root itself and its shared work are not halved, so allow slack.
        assert!(
            s_on.nodes * 2 >= s_off.nodes.saturating_sub(8),
            "reduction implausibly large ({} on vs {} off)",
            s_on.nodes,
            s_off.nodes
        );

        // Sanity: with a bound at/above the depth, both find an optimal-length
        // solution (path may be a σ-mirror, so compare lengths not moves).
        let (sol_off, _) = idastar_inc_bounded_parallel_mut(&fix, &IncManhattan, 30, None, false);
        let (sol_on, _) = idastar_inc_bounded_parallel_mut(&fix, &IncManhattan, 30, None, true);
        match (sol_off, sol_on) {
            (LadderOutcome::Solved(a), LadderOutcome::Solved(b)) => {
                assert_eq!(a.len(), b.len(), "orbit-split changed optimal length");
            }
            (a, b) => panic!("expected both Solved, got {:?} / {:?}", a, b),
        }
    }

    /// Sequential root-orbit-split (the single-threaded `_mut` driver's `g == 0`
    /// filter) must be a faithful counterpart of the parallel split filter: on a
    /// σ-symmetric board it produces **exactly the same node count** as the
    /// parallel orbit driver, reduces vs the un-split sequential run (~2×), and
    /// keeps the optimal length. Ties to the R use case by asserting R itself is
    /// σ-symmetric (R's own exhaust is proof-scale, so the real-tree checks run on
    /// a shallow σ-symmetric fixture — depth 16, the same one the parallel test
    /// uses — with Manhattan + `NullPruner` so the ~2× isn't muddied by move-DFA).
    #[cfg(feature = "parallel")]
    #[test]
    fn root_orbit_split_sequential_matches_parallel_on_symmetric_board() {
        use crate::puzzle24::search::IncManhattan;
        use crate::puzzle24::symmetry::is_symmetric;

        // The R use case: the 180° board is a σ-fixpoint, so orbit-split is sound
        // on it — the motivation for a sequential reference. (Its tree is too deep
        // to exhaust in a unit test, hence the shallow fixture below.)
        let r = State([
            0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
            1,
        ]);
        assert!(is_symmetric(&r), "board R must be σ-symmetric");

        // Shallow σ-symmetric fixture with a real exhaust window (blank on the
        // centre diagonal → 4 root moves → 2 orbits).
        let fix = State([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 14, 15, 16, 17, 18, 13, 20, 21, 22, 23, 24,
            19,
        ]);
        assert!(is_symmetric(&fix), "fixture must be σ-symmetric");

        // Bounded exhaust below the solution depth (16): path-free ProvedAtLeast.
        let (seq_off, s_seq_off) = idastar_inc_mut_bounded_orbit_with_stats(&fix, &IncManhattan, 10, false);
        let (seq_on, s_seq_on) = idastar_inc_mut_bounded_orbit_with_stats(&fix, &IncManhattan, 10, true);
        let (par_on, s_par_on) = idastar_inc_bounded_parallel_mut(&fix, &IncManhattan, 10, None, true);

        // (1) The headline cross-check: sequential orbit-split is node-identical to
        // the parallel orbit-split, and proves the same bound. (The two drivers
        // return different outcome enums — BoundedOutcome vs LadderOutcome — so
        // compare the proved bound explicitly.)
        assert_eq!(
            s_seq_on.nodes, s_par_on.nodes,
            "sequential orbit nodes {} != parallel orbit nodes {}",
            s_seq_on.nodes, s_par_on.nodes
        );
        let seq_bound = match seq_on {
            BoundedOutcome::ProvedAtLeast(k) => k,
            other => panic!("expected sequential path-free ProvedAtLeast, got {:?}", other),
        };
        let par_bound = match par_on {
            LadderOutcome::ProvedAtLeast(k) => k,
            other => panic!("expected parallel path-free ProvedAtLeast, got {:?}", other),
        };
        assert_eq!(seq_bound, par_bound, "sequential vs parallel proved-bound differs");
        // (2) The split actually reduces work vs the un-split sequential run…
        assert_eq!(seq_off, seq_on, "orbit-split changed the sequential lower-bound outcome");
        assert!(
            s_seq_on.nodes < s_seq_off.nodes,
            "sequential orbit-split did not reduce nodes ({} vs {})",
            s_seq_on.nodes, s_seq_off.nodes
        );
        // …by roughly 2× (root + shared work not halved, so allow slack).
        assert!(
            s_seq_on.nodes * 2 >= s_seq_off.nodes.saturating_sub(8),
            "reduction implausibly large ({} on vs {} off)",
            s_seq_on.nodes, s_seq_off.nodes
        );

        // (3) With a bound above the depth, orbit-split still finds an optimal-
        // length solution (path may be a σ-mirror, so compare lengths).
        let (sol_off, _) = idastar_inc_mut_bounded_orbit_with_stats(&fix, &IncManhattan, 30, false);
        let (sol_on, _) = idastar_inc_mut_bounded_orbit_with_stats(&fix, &IncManhattan, 30, true);
        match (sol_off, sol_on) {
            (BoundedOutcome::Solved(a), BoundedOutcome::Solved(b)) => {
                assert_eq!(a.len(), b.len(), "orbit-split changed optimal length");
            }
            (a, b) => panic!("expected both Solved, got {:?} / {:?}", a, b),
        }
    }

    /// Sequential move-DFA pruning (the `_mut` driver with a real [`MoveDfa`]) must
    /// match the parallel pruned driver exactly and actually remove duplicate
    /// subtrees. Cross-checks, on general scrambles (the DFA needs no symmetry):
    /// (1) sequential-pruned nodes == parallel-pruned nodes (same outcome), and
    /// (2) sequential-pruned nodes ≤ sequential-unpruned, strictly across the suite.
    #[cfg(feature = "parallel")]
    #[test]
    fn move_dfa_sequential_matches_parallel_and_prunes() {
        use crate::puzzle24::search::move_dfa::MoveDfa;
        use crate::puzzle24::search::IncManhattan;

        fn scramble(seed: u64, steps: u32) -> State {
            let mut s = GOAL;
            let mut last: Option<Move> = None;
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            for _ in 0..steps {
                let legal: Vec<Move> = s
                    .legal_moves()
                    .iter()
                    .filter(|&m| last.map_or(true, |p| m != p.inverse()))
                    .collect();
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let m = legal[(x as usize) % legal.len()];
                s = s.apply(m);
                last = Some(m);
            }
            s
        }

        let dfa = MoveDfa::build_default();
        let (mut total_pruned, mut total_plain) = (0u64, 0u64);
        for seed in 0..16u64 {
            let start = scramble(seed, 20);
            // Bound at optimal − 2: strictly path-free (no early exit, so seq/par
            // node counts are comparable) yet ≥ h0 for a real tree.
            let opt = idastar_inc_mut(&start, &IncManhattan).expect("solvable").len() as u8;
            let bound = opt.saturating_sub(2);
            // orbit_split = false (general boards); pruner = the move-DFA.
            let (seq_out, seq_st) =
                idastar_inc_mut_bounded_pruned_orbit_with_stats(&start, &IncManhattan, &dfa, bound, false);
            let (par_out, par_st) =
                idastar_inc_bounded_parallel_mut_pruned(&start, &IncManhattan, &dfa, bound, None, false);
            // Un-pruned sequential baseline (NullPruner) for the same tree.
            let (_, plain_st) =
                idastar_inc_mut_bounded_orbit_with_stats(&start, &IncManhattan, bound, false);

            assert_eq!(
                seq_st.nodes, par_st.nodes,
                "seed {seed}: sequential pruned nodes {} != parallel pruned {}",
                seq_st.nodes, par_st.nodes
            );
            // Both prove the same lower bound (the exhaust is path-free at bound 12).
            let seq_k = match seq_out {
                BoundedOutcome::ProvedAtLeast(k) => k,
                other => panic!("seed {seed}: expected ProvedAtLeast, got {:?}", other),
            };
            let par_k = match par_out {
                LadderOutcome::ProvedAtLeast(k) => k,
                other => panic!("seed {seed}: expected ProvedAtLeast, got {:?}", other),
            };
            assert_eq!(seq_k, par_k, "seed {seed}: proved-bound differs seq vs parallel");
            assert!(
                seq_st.nodes <= plain_st.nodes,
                "seed {seed}: pruned {} did MORE work than un-pruned {}",
                seq_st.nodes, plain_st.nodes
            );
            total_pruned += seq_st.nodes;
            total_plain += plain_st.nodes;
        }
        // Across the suite the DFA must actually remove work.
        assert!(
            total_pruned < total_plain,
            "move-DFA never pruned: {total_pruned} vs {total_plain}"
        );
    }

    #[test]
    fn stats_match_plain_idastar_and_count_work() {
        // GOAL: solved before any search node is visited.
        let (sol, stats) = idastar_with_stats(&GOAL, &ManhattanHeuristic);
        assert_eq!(sol.unwrap().len(), 0);
        assert_eq!(stats.nodes, 0);
        assert_eq!(stats.iterations, 0);

        // A non-trivial state: at least one iteration, at least one node, and
        // the path must match the stats-free wrapper exactly.
        let s = GOAL.apply(Move::Up).apply(Move::Left).apply(Move::Up);
        let (sol_s, stats_s) = idastar_with_stats(&s, &ManhattanHeuristic);
        let plain = idastar(&s, &ManhattanHeuristic).unwrap();
        assert_eq!(sol_s.unwrap(), plain);
        assert!(stats_s.iterations >= 1);
        assert!(stats_s.nodes >= 1);
    }

    #[test]
    fn one_step_neighbors_solved_in_one_move() {
        for m in Move::ALL {
            if GOAL.legal_moves().contains(m) {
                let s = GOAL.apply(m);
                let sol = idastar(&s, &ManhattanHeuristic).unwrap();
                assert_eq!(sol.len(), 1);
                let mut cur = s;
                for mv in &sol {
                    cur = cur.apply(*mv);
                }
                assert_eq!(cur, GOAL);
            }
        }
    }

    #[test]
    fn applying_solution_reaches_goal_on_shallow_walks() {
        // For every state at known true distance ≤ 8, IDA* returns a path of
        // exactly that length and that path reaches GOAL.
        let table = bfs_distances(8);
        for (raw, &truth) in &table {
            let s = State(*raw);
            let sol = idastar(&s, &ManhattanHeuristic).expect("solver should solve");
            assert_eq!(sol.len() as u8, truth, "non-optimal for {:?}", raw);
            let mut cur = s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL, "solution doesn't reach goal from {:?}", raw);
        }
    }

    #[test]
    fn random_walk_solvable_optimally() {
        // A random walk gives an upper bound; IDA* must return ≤ walk length.
        let mut rng: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = GOAL;
        let mut last: Option<Move> = None;
        let mut walk_len = 0u8;
        for _ in 0..18 {
            let opts: Vec<Move> = s
                .legal_moves()
                .iter()
                .filter(|&m| last.map_or(true, |p: Move| m != p.inverse()))
                .collect();
            let m = opts[(next() as usize) % opts.len()];
            s = s.apply(m);
            last = Some(m);
            walk_len += 1;
        }
        let sol = idastar(&s, &ManhattanHeuristic).expect("should solve");
        assert!((sol.len() as u8) <= walk_len, "len {} > walk {}", sol.len(), walk_len);
        let mut cur = s;
        for m in &sol {
            cur = cur.apply(*m);
        }
        assert_eq!(cur, GOAL);
    }

    #[test]
    fn inc_matches_plain_on_shallow_states() {
        // The incremental driver must return the same optimal length as plain
        // IDA* for a Manhattan-equivalent incremental heuristic.
        let table = bfs_distances(7);
        let manh = super::super::heuristic::IncManhattan;
        for (raw, &truth) in table.iter().take(2000) {
            let s = State(*raw);
            let sol = idastar_inc(&s, &manh).expect("inc solver should solve");
            assert_eq!(sol.len() as u8, truth, "inc non-optimal for {:?}", raw);
        }
    }

    // ---------- bounded / lower-bound mode -----------------------------------

    #[test]
    fn bounded_solves_within_budget_matches_optimal() {
        // With max_bound = true depth, the bounded search must find the optimal
        // solution and never exceed the unbounded optimum length.
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(9);
        for (raw, &truth) in table.iter().take(3000) {
            let s = State(*raw);
            let (outcome, _) = idastar_inc_bounded_with_stats(&s, &IncManhattan, truth);
            match outcome {
                BoundedOutcome::Solved(sol) => {
                    assert_eq!(sol.len() as u8, truth, "non-optimal for {:?}", raw);
                    let mut cur = s;
                    for m in &sol {
                        cur = cur.apply(*m);
                    }
                    assert_eq!(cur, GOAL, "solution doesn't reach goal from {:?}", raw);
                }
                other => panic!("expected Solved at max_bound==depth, got {:?} for {:?}", other, raw),
            }
        }
    }

    #[test]
    fn bounded_one_below_depth_proves_at_least_depth() {
        // max_bound = depth - 1 must fall short and prove dist >= depth exactly.
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(9);
        for (raw, &truth) in table.iter() {
            if truth == 0 {
                continue; // GOAL solves immediately regardless of bound
            }
            let s = State(*raw);
            let (outcome, _) = idastar_inc_bounded_with_stats(&s, &IncManhattan, truth - 1);
            assert_eq!(
                outcome,
                BoundedOutcome::ProvedAtLeast(truth),
                "expected ProvedAtLeast({}) for {:?}",
                truth,
                raw
            );
        }
    }

    #[test]
    fn bounded_below_root_h_proves_at_least_root_h() {
        // If max_bound < h(start), the search proves dist >= h(start) without
        // expanding any nodes past the root threshold.
        use super::super::heuristic::IncManhattan;
        let s = GOAL.apply(Move::Up).apply(Move::Left).apply(Move::Up).apply(Move::Left);
        let h0 = ManhattanHeuristic.h(&s);
        assert!(h0 >= 2);
        let (outcome, _) = idastar_inc_bounded_with_stats(&s, &IncManhattan, h0 - 1);
        assert_eq!(outcome, BoundedOutcome::ProvedAtLeast(h0));
    }

    #[test]
    fn bounded_goal_solved_with_empty_path() {
        use super::super::heuristic::IncManhattan;
        let (outcome, stats) = idastar_inc_bounded_with_stats(&GOAL, &IncManhattan, 0);
        assert_eq!(outcome, BoundedOutcome::Solved(Vec::new()));
        assert_eq!(stats.nodes, 0);
    }

    #[test]
    fn bounded_never_exceeds_unbounded_optimum() {
        // A generous bound must reproduce exactly the unbounded optimal length.
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(8);
        for (raw, &truth) in table.iter().take(1500) {
            let s = State(*raw);
            let (outcome, _) = idastar_inc_bounded_with_stats(&s, &IncManhattan, 200);
            match outcome {
                BoundedOutcome::Solved(sol) => assert_eq!(sol.len() as u8, truth),
                other => panic!("expected Solved for {:?}, got {:?}", raw, other),
            }
        }
    }

    #[test]
    fn ladder_unbounded_no_deadline_solves_optimally() {
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(8);
        for (raw, &truth) in table.iter().take(1500) {
            let s = State(*raw);
            let (outcome, _) = idastar_inc_ladder(&s, &IncManhattan, u8::MAX, None);
            match outcome {
                LadderOutcome::Solved(sol) => assert_eq!(sol.len() as u8, truth, "for {:?}", raw),
                other => panic!("expected Solved for {:?}, got {:?}", raw, other),
            }
        }
    }

    #[test]
    fn ladder_finite_bound_proves_lower_bound() {
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(8);
        for (raw, &truth) in table.iter() {
            if truth == 0 {
                continue;
            }
            let s = State(*raw);
            let (outcome, _) = idastar_inc_ladder(&s, &IncManhattan, truth - 1, None);
            assert_eq!(outcome, LadderOutcome::ProvedAtLeast(truth), "for {:?}", raw);
        }
    }

    #[test]
    fn ladder_elapsed_deadline_times_out_with_valid_lower_bound() {
        // A deep board with an already-elapsed deadline: the search must abort at
        // the first deadline check and report TimedOut(K) with K ≥ root h (a
        // valid admissible lower bound).
        use super::super::heuristic::IncManhattan;
        let r = State([0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert!(r.is_solvable());
        let h0 = ManhattanHeuristic.h(&r);
        let past = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let (outcome, _) = idastar_inc_ladder(&r, &IncManhattan, u8::MAX, Some(past));
        match outcome {
            LadderOutcome::TimedOut(k) => assert!(k >= h0, "timeout LB {} < root h {}", k, h0),
            other => panic!("expected TimedOut, got {:?}", other),
        }
    }

    #[test]
    fn bounded_telemetry_fires_once_per_failed_iteration() {
        use super::super::heuristic::IncManhattan;
        let s = GOAL.apply(Move::Up).apply(Move::Left).apply(Move::Up).apply(Move::Left);
        let truth = idastar(&s, &ManhattanHeuristic).unwrap().len() as u8;
        let mut thresholds = Vec::new();
        let (outcome, _) = idastar_inc_bounded_telemetry(&s, &IncManhattan, truth, |t, _, _| {
            thresholds.push(t);
        });
        assert!(matches!(outcome, BoundedOutcome::Solved(_)));
        // Telemetry fires only on failed iterations; thresholds are strictly
        // increasing and all strictly below the final (solving) threshold.
        for w in thresholds.windows(2) {
            assert!(w[0] < w[1], "thresholds not increasing: {:?}", thresholds);
        }
        for &t in &thresholds {
            assert!(t < truth, "telemetry fired at solving threshold {} (depth {})", t, truth);
        }
    }

    // ---------- parallel driver (2P) -----------------------------------------

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_solves_optimally_and_matches_sequential() {
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(8);
        for (raw, &truth) in table.iter().take(3000) {
            let s = State(*raw);
            // Unbounded parallel solve must return the optimal length and a path
            // that replays to GOAL — identical outcome to the sequential ladder.
            let (par, _) = idastar_inc_bounded_parallel(&s, &IncManhattan, u8::MAX, None);
            let par_sol = match &par {
                LadderOutcome::Solved(sol) => {
                    assert_eq!(sol.len() as u8, truth, "parallel non-optimal for {:?}", raw);
                    let mut cur = s;
                    for m in sol {
                        cur = cur.apply(*m);
                    }
                    assert_eq!(cur, GOAL, "parallel path doesn't reach GOAL for {:?}", raw);
                    sol.clone()
                }
                other => panic!("expected Solved for {:?}, got {:?}", raw, other),
            };
            // Same optimal length as the sequential ladder driver.
            let (seq, _) = idastar_inc_ladder(&s, &IncManhattan, u8::MAX, None);
            match seq {
                LadderOutcome::Solved(a) => {
                    assert_eq!(a.len(), par_sol.len(), "parallel vs sequential length mismatch for {:?}", raw);
                }
                other => panic!("expected sequential Solved, got {:?}", other),
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_bounded_proves_same_lower_bound_as_sequential() {
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(8);
        for (raw, &truth) in table.iter() {
            if truth == 0 {
                continue;
            }
            let s = State(*raw);
            // max_bound = depth-1: parallel must prove depth >= truth, matching
            // the sequential ladder exactly.
            let (par, _) = idastar_inc_bounded_parallel(&s, &IncManhattan, truth - 1, None);
            assert_eq!(par, LadderOutcome::ProvedAtLeast(truth), "parallel LB wrong for {:?}", raw);
            let (seq, _) = idastar_inc_ladder(&s, &IncManhattan, truth - 1, None);
            assert_eq!(seq, par, "parallel vs sequential LB mismatch for {:?}", raw);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_exhaust_node_count_matches_sequential() {
        // For a bounded *exhaust* (no early exit), parallel and sequential explore
        // the identical node set, so the node counters must agree exactly — guards
        // the split's boundary-prune counting against regressions.
        use super::super::heuristic::IncManhattan;
        let table = bfs_distances(8);
        for (raw, &truth) in table.iter() {
            if truth < 2 {
                continue;
            }
            let s = State(*raw);
            let mb = truth - 1; // exhausts every threshold up to the proof of `truth`
            let (_, seq_stats) = idastar_inc_bounded_with_stats(&s, &IncManhattan, mb);
            let (_, par_stats) = idastar_inc_bounded_parallel(&s, &IncManhattan, mb, None);
            assert_eq!(
                seq_stats.nodes, par_stats.nodes,
                "node count mismatch (seq {} vs par {}) for {:?}",
                seq_stats.nodes, par_stats.nodes, raw
            );
        }
    }
}
