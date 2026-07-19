//! Iterative-deepening A* over the 15-puzzle [`State`].
//!
//! Standard IDA*: at iteration `k`, depth-first search with `f = g + h` cut
//! off at threshold `bound_k`. If the goal is found, return the path; else
//! `bound_{k+1}` is the minimum `f` value that exceeded `bound_k`. Repeat.
//!
//! With an admissible heuristic, the first solution found is optimal. We
//! additionally prune immediate-undo moves (never apply `m.inverse()`
//! immediately after `m`), which roughly halves the branching factor without
//! sacrificing optimality.
//!
//! Cost types stay `u8` — the 15-puzzle diameter is 80, well below 255.

use crate::puzzle15::state::{Move, State, GOAL};

use super::heuristic::Heuristic;

/// Search-effort statistics for a single IDA\* solve.
///
/// `nodes` counts every node visited (one [`search`] invocation, equivalently
/// one heuristic evaluation), summed across all threshold iterations.
/// `iterations` counts the IDA\* deepening passes. Both are the natural levers
/// for benchmarking: incremental-heuristic and codegen changes move wall-clock
/// at fixed `nodes`, while move-ordering and duplicate-pruning changes move
/// `nodes` directly (see `OPTIMIZATION.md`).
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
/// Returns `Some(vec![])` if `start == GOAL`. Returns `None` only if `start`
/// is unreachable from `GOAL`, which is impossible for solvable states.
///
/// Thin wrapper over [`idastar_with_stats`] that discards the statistics.
pub fn idastar<H: Heuristic>(start: &State, h: &H) -> Option<Vec<Move>> {
    idastar_with_stats(start, h).0
}

/// Like [`idastar`], but also returns the [`SearchStats`] for the run.
///
/// The node counter adds one `u64` increment per node — negligible against the
/// per-node heuristic evaluation and successor generation — so callers that
/// don't need stats can use [`idastar`] without measurable cost.
pub fn idastar_with_stats<H: Heuristic>(start: &State, h: &H) -> (Option<Vec<Move>>, SearchStats) {
    let mut stats = SearchStats::default();

    if start == &GOAL {
        return (Some(Vec::new()), stats);
    }

    let mut bound = h.h(start);
    let mut path: Vec<Move> = Vec::with_capacity(96);
    let blank = start.blank_pos();

    loop {
        stats.iterations += 1;
        match search(start, blank, 0, bound, &mut path, None, h, &mut stats) {
            Step::Found => return (Some(path), stats),
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
    /// Smallest `f` value seen at this iteration that exceeded `bound`.
    Bound(u8),
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
/// PDB heuristic that means re-projecting the full board (an O(16) scan per
/// pattern, plus a `reflect` for the Korf view) on every one of millions of
/// nodes — wasteful, since a move changes exactly one tile. An `IncHeuristic`
/// instead carries a small per-node context (e.g. the projected boards) that
/// [`advance`](IncHeuristic::advance) updates in O(k) when a move is applied.
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
    /// corresponding `SearchStats` counters so wall-time-per-call analysis
    /// has accurate call counts regardless of inlining decisions.
    fn advance(
        &self,
        parent: &Self::Ctx,
        child: &State,
        m: Move,
        stats: &mut SearchStats,
    ) -> (u8, Self::Ctx);
}

/// [`idastar_with_stats`] driven by an [`IncHeuristic`]. Produces identical
/// results to the plain version with an equivalent heuristic — only the
/// per-node heuristic cost differs.
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
    let mut path: Vec<Move> = Vec::with_capacity(96);
    let blank = start.blank_pos();

    loop {
        stats.iterations += 1;
        match search_inc(
            start, blank, ctx0, h0, 0, bound, &mut path, None, e, &mut stats,
        ) {
            Step::Found => return (Some(path), stats),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (None, stats);
                }
                bound = next;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn search_inc<E: IncHeuristic>(
    s: &State,
    blank: u8,
    ctx: E::Ctx,
    h_val: u8,
    g: u8,
    bound: u8,
    path: &mut Vec<Move>,
    last: Option<Move>,
    e: &E,
    stats: &mut SearchStats,
) -> Step {
    stats.nodes += 1;
    let f = g.saturating_add(h_val);
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
        let (child_h, child_ctx) = e.advance(&ctx, &s_next, m, stats);
        path.push(m);
        match search_inc(
            &s_next,
            next_blank,
            child_ctx,
            child_h,
            g + 1,
            bound,
            path,
            Some(m),
            e,
            stats,
        ) {
            Step::Found => return Step::Found,
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

/// Make/unmake variant of [`IncHeuristic`] (OPTIMIZATION.md prototype): instead
/// of producing a fresh `Copy` context per node, the search threads a single
/// `&mut Ctx`, mutating it in place before recursing and undoing on backtrack.
/// This avoids the per-node context copy that profiling attributed ~21% of
/// solve time to (`ProjectedState::apply`'s array copies).
pub trait IncHeuristicMut {
    /// Per-node carried state (mutated in place; need not be `Copy`).
    type Ctx;
    /// Evaluate at the search root: `(h(start), ctx0)`.
    fn root(&self, s: &State) -> (u8, Self::Ctx);
    /// Mutate `ctx` by applying move `m`; return `h` of the resulting child.
    fn make(&self, ctx: &mut Self::Ctx, m: Move) -> u8;
    /// Undo the most recent [`make`](Self::make) of `m`, restoring `ctx`.
    fn unmake(&self, ctx: &mut Self::Ctx, m: Move);
}

/// [`idastar_inc_with_stats`] using a mutable make/unmake context.
pub fn idastar_inc_mut_with_stats<E: IncHeuristicMut>(
    start: &State,
    e: &E,
) -> (Option<Vec<Move>>, SearchStats) {
    let mut stats = SearchStats::default();
    if start == &GOAL {
        return (Some(Vec::new()), stats);
    }
    let (h0, mut ctx) = e.root(start);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(96);
    let blank = start.blank_pos();
    loop {
        stats.iterations += 1;
        // `ctx` is restored to the root projection after each iteration because
        // `search_inc_mut` unmakes every move it makes.
        match search_inc_mut(
            start, blank, &mut ctx, h0, 0, bound, &mut path, None, e, &mut stats,
        ) {
            Step::Found => return (Some(path), stats),
            Step::Bound(next) => {
                if next == u8::MAX {
                    return (None, stats);
                }
                bound = next;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn search_inc_mut<E: IncHeuristicMut>(
    s: &State,
    blank: u8,
    ctx: &mut E::Ctx,
    h_val: u8,
    g: u8,
    bound: u8,
    path: &mut Vec<Move>,
    last: Option<Move>,
    e: &E,
    stats: &mut SearchStats,
) -> Step {
    stats.nodes += 1;
    let f = g.saturating_add(h_val);
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
        let child_h = e.make(ctx, m);
        path.push(m);
        match search_inc_mut(
            &s_next,
            next_blank,
            ctx,
            child_h,
            g + 1,
            bound,
            path,
            Some(m),
            e,
            stats,
        ) {
            // On Found, keep the accumulated path and return; the whole search
            // terminates so `ctx` is discarded (no need to unmake).
            Step::Found => return Step::Found,
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
    use crate::puzzle15::search::heuristic::ManhattanHeuristic;
    use crate::puzzle15::search::tests_util::bfs_distances;

    #[test]
    fn solves_goal_with_empty_path() {
        let sol = idastar(&GOAL, &ManhattanHeuristic).unwrap();
        assert!(sol.is_empty());
    }

    #[test]
    fn stats_match_plain_idastar_and_count_work() {
        // GOAL: solved before any search node is visited.
        let (sol, stats) = idastar_with_stats(&GOAL, &ManhattanHeuristic);
        assert_eq!(sol.unwrap().len(), 0);
        assert_eq!(stats.nodes, 0);
        assert_eq!(stats.iterations, 0);

        // A non-trivial state: at least one iteration, at least one node, and
        // the returned path must match the stats-free wrapper exactly.
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
                // Applying solution must reach GOAL.
                let mut cur = s;
                for mv in &sol {
                    cur = cur.apply(*mv);
                }
                assert_eq!(cur, GOAL);
            }
        }
    }

    #[test]
    fn applying_solution_always_reaches_goal_on_shallow_walks() {
        // For each state at known true distance ≤ 10, IDA* must:
        //   1. return a solution of exactly that length (optimality), and
        //   2. that solution must apply to the state and reach GOAL.
        let table = bfs_distances(10);
        for (raw, &truth) in &table {
            let s = State(*raw);
            let sol = idastar(&s, &ManhattanHeuristic).expect("solver should solve");
            assert_eq!(
                sol.len() as u8,
                truth,
                "IDA* returned len {} but true dist is {} for {:?}",
                sol.len(),
                truth,
                raw
            );
            let mut cur = s;
            for m in &sol {
                cur = cur.apply(*m);
            }
            assert_eq!(cur, GOAL, "solution doesn't reach goal from {raw:?}");
        }
    }

    #[test]
    fn random_walk_of_depth_25_solvable_optimally() {
        // A random walk gives an upper bound; IDA* must return ≤ walk length
        // and applying the solution must reach GOAL.
        let pseudo = |i: u32| -> Move { Move::ALL[(i.wrapping_mul(2654435761) % 4) as usize] };
        // Construct a position by walking 25 random steps from GOAL.
        let mut s = GOAL;
        let mut walk_len = 0u8;
        for i in 0u32..25 {
            for k in 0u32..4 {
                let m = pseudo(i.wrapping_add(k));
                if s.legal_moves().contains(m) {
                    s = s.apply(m);
                    walk_len += 1;
                    break;
                }
            }
        }
        let sol = idastar(&s, &ManhattanHeuristic).expect("should solve");
        assert!(
            (sol.len() as u8) <= walk_len,
            "solution len {} exceeds walk len {}",
            sol.len(),
            walk_len
        );
        let mut cur = s;
        for m in &sol {
            cur = cur.apply(*m);
        }
        assert_eq!(cur, GOAL);
    }
}
