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

use crate::puzzle24::state::{Move, State, GOAL};

use super::heuristic::Heuristic;

/// Search-effort statistics for a single IDA\* solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Total nodes visited across all iterations.
    pub nodes: u64,
    /// Number of IDA\* threshold iterations performed.
    pub iterations: u32,
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
/// `Ctx` is `Copy` so the search threads it by value down the recursion;
/// backtracking is automatic (each stack frame owns its copy).
pub trait IncHeuristic {
    /// Per-node carried state.
    type Ctx: Copy;
    /// Evaluate at the search root: `(h(start), ctx0)`.
    fn root(&self, s: &State) -> (u8, Self::Ctx);
    /// Given the parent's context and the move `m` just applied to reach
    /// `child`, return `(h(child), child_ctx)`.
    fn advance(&self, parent: &Self::Ctx, child: &State, m: Move) -> (u8, Self::Ctx);
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
    let (h0, ctx0) = e.root(start);
    let mut bound = h0;
    let mut path: Vec<Move> = Vec::with_capacity(220);
    let blank = start.blank_pos();
    loop {
        stats.iterations += 1;
        match search_inc(start, blank, ctx0, h0, 0, bound, &mut path, None, e, &mut stats) {
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

/// Convenience wrapper discarding stats.
pub fn idastar_inc<E: IncHeuristic>(start: &State, e: &E) -> Option<Vec<Move>> {
    idastar_inc_with_stats(start, e).0
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
        let (child_h, child_ctx) = e.advance(&ctx, &s_next, m);
        path.push(m);
        match search_inc(&s_next, next_blank, child_ctx, child_h, g + 1, bound, path, Some(m), e, stats) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::search::heuristic::ManhattanHeuristic;
    use crate::puzzle24::search::tests_util::bfs_distances;

    #[test]
    fn solves_goal_with_empty_path() {
        assert!(idastar(&GOAL, &ManhattanHeuristic).unwrap().is_empty());
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
}
